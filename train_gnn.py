"""
train_gnn.py  —  Engine 1 training & evaluation
=================================================================
Semi-supervised training of the ward-level Graph Transformer.

The setup (semi-supervised node regression)
-------------------------------------------
Every hour we run ONE forward over all 289 wards, but the loss is computed only
at the 49 CPCB-labelled wards. The other 240 wards receive their prediction
*through the graph* — message passing from labelled neighbours makes the GNN the
spatial interpolator. This is why a graph model beats a per-cell GBDT here.

Guardrails honoured (PLAN.md)
-----------------------------
* Chronological splits from `split_lab` — never random.
* Target & persistence stay RAW AQI; only inputs are scaled (by preprocessing).
* Every number is reported through `metrics.py` as skill vs persistence.
* `--no-temporal` runs the no-history ablation (must still beat persistence).

    python train_gnn.py --horizon 24 --epochs 30
    python train_gnn.py --horizon 24 --epochs 30 --no-temporal   # ablation
=================================================================
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from stgnn_data import load_stgnn, build_supervised_pairs
from models.gnn_forecast import WardGraphTransformer, pinball_loss
from metrics import scoreboard, format_scoreboard, interval_coverage, pinball_loss as np_pinball

BASE = Path(__file__).resolve().parent
CKPT_DIR = BASE / "models" / "checkpoints"


def group_by_timestep(pairs):
    """t_input -> (node_idx[], y[], persist[]) so each graph forward serves all
    labelled wards at that hour at once."""
    order = np.argsort(pairs["t_input"])
    t = pairs["t_input"][order]
    node = pairs["node"][order]
    y = pairs["y"][order]
    yp = pairs["y_persist"][order]
    out = {}
    uniq, starts = np.unique(t, return_index=True)
    starts = list(starts) + [len(t)]
    for i, tt in enumerate(uniq):
        sl = slice(starts[i], starts[i + 1])
        out[int(tt)] = (node[sl], y[sl], yp[sl])
    return out


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available()
                                   or args.device == "cpu" else "cpu")
        self.temporal = not args.no_temporal
        self.d = load_stgnn(horizon=args.horizon)
        d = self.d

        # static graph tensors -> device
        self.x_static = torch.as_tensor(d.X_static, device=self.device)
        self.X_dyn = torch.as_tensor(d.X_dyn, device=self.device)          # [T,C,Fd]
        self.cell_of_node = torch.as_tensor(d.cell_of_node, device=self.device)
        self.edge_index = torch.as_tensor(d.edge_index, device=self.device)
        self.edge_attr3 = torch.as_tensor(d.edge_attr, device=self.device)  # [E,3]
        self.bearing = torch.as_tensor(np.deg2rad(d.edge_bearing_deg), device=self.device)
        self.src_cells = self.cell_of_node[self.edge_index[0]]
        self.w_sin_col, self.w_cos_col = d._wind_sin_col, d._wind_cos_col

        # supervised pairs per split, grouped by timestep
        self.train_g = group_by_timestep(build_supervised_pairs(d, "train"))
        self.val_pairs = build_supervised_pairs(d, "val")
        self.test_pairs = build_supervised_pairs(d, "test")

        # target standardisation from TRAIN labels only (leak-free)
        ally = np.concatenate([g[1] for g in self.train_g.values()])
        self.y_mu, self.y_sd = float(ally.mean()), float(ally.std() + 1e-6)

        # dense CAMS grid target for optional pretraining (289-ward supervision).
        # y_grid[t, cell] broadcast to wards -> a target at EVERY ward, every hour,
        # which is why it cures the 49-ward overfitting before station fine-tuning.
        self.y_grid = torch.as_tensor(d.y_grid, device=self.device)         # [T, C]
        tr_t = d.t_indices("train")
        gvals = d.y_grid[tr_t]
        gvals = gvals[np.isfinite(gvals)]
        self.g_mu, self.g_sd = float(gvals.mean()), float(gvals.std() + 1e-6)
        self.train_ts_all = tr_t

        # persistence broadcast to wards (for residual target + dense aux loss)
        self.persist_grid = torch.as_tensor(d.persist_grid, device=self.device)  # [T,C]

        # RESIDUAL MODE: predict the *correction to persistence* (y - persist).
        # Skill IS the ability to beat persistence, so learning the residual puts
        # the model's capacity exactly where it counts. One unified scale is used
        # for both the station loss and the dense-grid aux loss, because a residual
        # over persistence is the same physical quantity (an AQI correction) at
        # either reference point.
        resid = np.concatenate([g[1] - g[2] for g in self.train_g.values()])
        self.r_mu, self.r_sd = float(resid.mean()), float(resid.std() + 1e-6)

        self.model = WardGraphTransformer(
            n_static=len(d.static_names), n_dyn=len(d.dyn_names),
            hidden=args.hidden, heads=args.heads, n_layers=args.layers,
            dropout=args.dropout, edge_dim=4, temporal=self.temporal,
            temporal_kind=getattr(args, "temporal_kind", "gru"), lookback=args.lookback,
        ).to(self.device)
        self.quantiles = self.model.quantiles
        self.qmid = self.model.median_index()

    # ---- batched forward over many timesteps (block-diagonal graph) -------
    def forward_batch(self, ts: torch.Tensor) -> torch.Tensor:
        """ts: [B] timestep indices -> predictions [B, N, Q] (normalised).

        B independent 289-ward graphs are stacked into ONE disjoint graph so a
        single TransformerConv call serves the whole batch. The GRU window is
        gathered vectorised over the batch (no python loop).
        """
        B = ts.shape[0]
        N, E = self.d.n_nodes, self.edge_index.shape[1]

        if self.temporal:
            L = self.args.lookback
            offs = torch.arange(L - 1, -1, -1, device=self.device)        # L..0
            idx = (ts.view(B, 1) - offs.view(1, L)).clamp_min(0)          # [B, L]
            seq = self.X_dyn[idx]                                         # [B, L, C, Fd]
            Bc = B * self.d.n_cells
            seq = seq.permute(0, 2, 1, 3).reshape(Bc, L, -1)             # [B*C, L, Fd]
            h = self.model.encode_dynamic(seq)                           # [B*C, hidden]
            dyn_block = h.view(B, self.d.n_cells, -1)                    # [B, C, hidden]
        else:
            dyn_block = self.X_dyn[ts]                                    # [B, C, Fd]

        dyn_nodes = dyn_block[:, self.cell_of_node, :]                    # [B, N, dyn_in]
        stat = self.x_static.unsqueeze(0).expand(B, -1, -1)              # [B, N, Fs]
        big_x = torch.cat([stat, dyn_nodes], dim=2).reshape(B * N, -1)

        node_off = (torch.arange(B, device=self.device) * N).view(B, 1, 1)
        big_ei = (self.edge_index.unsqueeze(0) + node_off).permute(1, 0, 2).reshape(2, B * E)

        w_sin = self.X_dyn[ts][:, self.src_cells, self.w_sin_col]        # [B, E]
        w_cos = self.X_dyn[ts][:, self.src_cells, self.w_cos_col]
        align = torch.cos(self.bearing) * w_cos + torch.sin(self.bearing) * w_sin
        big_attr = torch.cat(
            [self.edge_attr3.unsqueeze(0).expand(B, -1, -1), align.unsqueeze(2)], dim=2
        ).reshape(B * E, 4)

        out = self.model(big_x, big_ei, big_attr)                        # [B*N, Q]
        return out.view(B, N, -1)

    # ---- target helpers (residual vs absolute) ---------------------------
    @property
    def _mu(self): return self.r_mu if self.args.residual else self.y_mu

    @property
    def _sd(self): return self.r_sd if self.args.residual else self.y_sd

    # ---- train / eval -----------------------------------------------------
    def train_epoch(self):
        self.model.train()
        ts_all = np.array(list(self.train_g.keys()))
        np.random.shuffle(ts_all)
        bs = self.args.batch_ts
        N = self.d.n_nodes
        aux_w = self.args.aux_grid_weight
        total, nb = 0.0, 0
        for i in range(0, len(ts_all), bs):
            batch = ts_all[i:i + bs]
            ts = torch.as_tensor(batch, device=self.device)
            pred = self.forward_batch(ts)                                 # [B,N,Q]
            flat = pred.reshape(-1, pred.shape[-1])

            # --- station loss (at the 49 labelled wards) ---
            gidx, ys = [], []
            for b, t in enumerate(batch):
                node, y, ypers = self.train_g[int(t)]
                tgt = (y - ypers) if self.args.residual else y
                gidx.append(b * N + node)
                ys.append((tgt - self._mu) / self._sd)
            gidx = torch.as_tensor(np.concatenate(gidx), device=self.device)
            ys = torch.as_tensor(np.concatenate(ys), device=self.device)
            loss = pinball_loss(flat[gidx], ys, self.quantiles)

            # --- dense CAMS-grid aux loss (at ALL 289 wards, every step) ---
            if aux_w > 0:
                gt = self.y_grid[ts][:, self.cell_of_node]                # [B,N]
                if self.args.residual:
                    gt = gt - self.persist_grid[ts][:, self.cell_of_node]
                mask = torch.isfinite(gt).reshape(-1)
                gy = ((gt.reshape(-1) - self._mu) / self._sd)[mask]
                loss = loss + aux_w * pinball_loss(flat[mask], gy, self.quantiles)

            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            total += loss.item(); nb += 1
        return total / max(nb, 1)

    def pretrain_epoch(self):
        """One epoch of dense CAMS-grid supervision over ALL 289 wards.

        Every ward gets its cell's `target_aqi_t{h}` as a (weak but dense) label,
        so the spatial+temporal encoder sees the full data distribution and stops
        memorising the 49 stations. Cheap: reuses the same batched graph forward.
        """
        self.model.train()
        ts_all = self.train_ts_all.copy()
        np.random.shuffle(ts_all)
        bs, N = self.args.batch_ts, self.d.n_nodes
        total, nb = 0.0, 0
        for i in range(0, len(ts_all), bs):
            batch = ts_all[i:i + bs]
            ts = torch.as_tensor(batch, device=self.device)
            pred = self.forward_batch(ts)                                 # [B,N,Q]
            tgt = self.y_grid[ts][:, self.cell_of_node]                   # [B,N]
            mask = torch.isfinite(tgt)
            tgt = (tgt - self.g_mu) / self.g_sd
            p = pred.reshape(-1, pred.shape[-1])[mask.reshape(-1)]
            y = tgt.reshape(-1)[mask.reshape(-1)]
            loss = pinball_loss(p, y, self.quantiles)
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            total += loss.item(); nb += 1
        return total / max(nb, 1)

    @torch.no_grad()
    def predict_pairs(self, pairs):
        """Deterministic per-example predictions for a split. Returns
        (y_true, y_persist, p10, p50, p90) in a stable order so several models'
        outputs can be averaged element-wise for an ensemble."""
        self.model.eval()
        g = group_by_timestep(pairs)
        keys = np.array(list(g.keys()))
        bs, N = self.args.batch_ts, self.d.n_nodes
        yt, p50, p10, p90, yp = [], [], [], [], []
        for i in range(0, len(keys), bs):
            batch = keys[i:i + bs]
            pred = self.forward_batch(torch.as_tensor(batch, device=self.device))  # [B,N,Q]
            pred = pred.reshape(-1, pred.shape[-1]).cpu().numpy() * self._sd + self._mu
            for b, t in enumerate(batch):
                node, y, ypers = g[int(t)]
                out = pred[b * N + node]
                if self.args.residual:                       # AQI = persistence + correction
                    out = out + ypers[:, None]
                yt.append(y); yp.append(ypers)
                p10.append(out[:, 0]); p50.append(out[:, self.qmid]); p90.append(out[:, -1])
        return (np.concatenate(yt), np.concatenate(yp),
                np.concatenate(p10), np.concatenate(p50), np.concatenate(p90))

    def evaluate(self, pairs, label):
        yt, yp, p10, p50, p90 = self.predict_pairs(pairs)
        sb = scoreboard(yt, p50, yp, label)
        sb["coverage_10_90"] = interval_coverage(yt, p10, p90)
        sb["pinball_p90"] = np_pinball(yt, p90, 0.9)
        return sb

    def fit(self):
        a = self.args
        print(f"device={self.device} temporal={self.temporal} "
              f"train_ts={len(self.train_g)} y_mu={self.y_mu:.1f} y_sd={self.y_sd:.1f}")

        # ---- optional dense CAMS-grid pretraining (warm start) -----------
        if a.pretrain_epochs > 0:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
            for ep in range(1, a.pretrain_epochs + 1):
                t0 = time.time()
                pl = self.pretrain_epoch()
                vg = self.evaluate(self.val_pairs, "val(grid-warmstart)")
                print(f"pre{ep:>3} ({time.time()-t0:4.0f}s) grid_pinball={pl:.4f}  {format_scoreboard(vg)}")

        # ---- station fine-tuning ----------------------------------------
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=a.lr * (0.3 if a.pretrain_epochs > 0 else 1.0),
                                    weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=a.epochs)
        best_rmse, best_state, bad = float("inf"), None, 0
        for ep in range(1, a.epochs + 1):
            t0 = time.time()
            tr = self.train_epoch()
            sched.step()
            val = self.evaluate(self.val_pairs, "val")
            flag = ""
            if val["rmse"] < best_rmse - 1e-3:
                best_rmse = val["rmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0; flag = "  *best"
            else:
                bad += 1
            print(f"ep{ep:>3} ({time.time()-t0:4.0f}s) train_pinball={tr:.4f}  "
                  f"{format_scoreboard(val)}  cov={val['coverage_10_90']:.2f}{flag}")
            if bad >= a.patience:
                print(f"early stop (no val improvement in {a.patience})"); break
        if best_state:
            self.model.load_state_dict(best_state)
        return best_rmse

    def save(self, suffix: str = ""):
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        tag = "notemporal" if not self.temporal else "temporal"
        path = CKPT_DIR / f"gnn_forecast_h{self.args.horizon}_{tag}{suffix}.pt"
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": vars(self.args),
            "y_mu": self.y_mu, "y_sd": self.y_sd,
            "r_mu": self.r_mu, "r_sd": self.r_sd, "residual": self.args.residual,
            "static_names": self.d.static_names, "dyn_names": self.d.dyn_names,
            "quantiles": self.quantiles,
        }, path)
        return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=24, choices=[24, 48, 72])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dropout", type=float, default=0.3)  # 2.4 sweep: better calibration
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pretrain-epochs", type=int, default=0,
                   help="epochs of dense CAMS-grid warm start before station fine-tuning")
    p.add_argument("--residual", action="store_true",
                   help="predict the correction to persistence (usually more accurate)")
    p.add_argument("--aux-grid-weight", type=float, default=0.0,
                   help="weight of the dense CAMS-grid aux loss added EVERY step (0=off)")
    p.add_argument("--seed", type=int, default=0, help="for ensembling")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-ts", type=int, default=64, help="timesteps per graph batch")
    p.add_argument("--no-temporal", action="store_true", help="no-history ablation")
    p.add_argument("--temporal-kind", default="gru", choices=["gru", "transformer", "tcn"],
                   help="2.5: temporal encoder type (temporal mode only)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-train-ts", type=int, default=0,
                   help="debug: cap number of train timesteps (0 = all)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tr = Trainer(args)
    if args.max_train_ts:                       # smoke-test knob
        keys = list(tr.train_g.keys())[:args.max_train_ts]
        tr.train_g = {k: tr.train_g[k] for k in keys}
    tr.fit()
    test = tr.evaluate(tr.test_pairs, "test")
    print("\n== FINAL ==")
    print(format_scoreboard(test), f" cov={test['coverage_10_90']:.2f}")
    path = tr.save()
    print("saved:", path)


if __name__ == "__main__":
    main()
