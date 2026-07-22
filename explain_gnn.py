"""
explain_gnn.py  —  Engine 1 explainability: SHAP + GNNExplainer
=================================================================
Two complementary "why" views on a ward's AQI forecast:

  SHAP (feature attribution)   -> WHICH input features pushed this ward's p50 up
                                  or down (pm2_5, boundary_layer_height, wind, ...).
  GNNExplainer (graph masks)   -> WHICH neighbouring wards' messages and WHICH
                                  features the model actually relied on (a sparse
                                  edge_mask + node-feature mask).

Both run on the **snapshot model** (`--no-temporal` checkpoint), because its
`forward(x, edge_index, edge_attr)` takes the RAW, interpretable node features
(`[static | dynamic]`) — so an explanation is stated in real physical columns,
not GRU hidden state. The snapshot model is therefore both the no-history
ablation AND the explainable model — one artefact, two jobs.

    python explain_gnn.py --ward <ward_id> --time-index <t>
    python explain_gnn.py --global-shap 300     # global importance over 300 queries
=================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stgnn_data import load_stgnn
from models.gnn_forecast import WardGraphTransformer

BASE = Path(__file__).resolve().parent
CKPT_DIR = BASE / "models" / "checkpoints"
OUT_DIR = BASE / "data" / "explain"


class ExplainContext:
    """Loads the snapshot checkpoint + data and builds (x, edge_index, edge_attr)
    for any timestep, plus name lookups. All the glue the explainers share."""

    def __init__(self, horizon: int = 24, device: str = "cpu", realtime: bool | None = None):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        ckpt_path = CKPT_DIR / f"gnn_forecast_h{horizon}_notemporal.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"{ckpt_path} not found — train the snapshot model first:\n"
                f"  python train_gnn.py --horizon {horizon} --no-temporal")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.d = load_stgnn(horizon=horizon, realtime=realtime)
        self.y_mu, self.y_sd = ckpt["y_mu"], ckpt["y_sd"]
        self.feature_names = self.d.static_names + self.d.dyn_names

        self.model = WardGraphTransformer(
            n_static=len(self.d.static_names), n_dyn=len(self.d.dyn_names),
            hidden=ckpt["config"]["hidden"], heads=ckpt["config"]["heads"],
            n_layers=ckpt["config"]["layers"], dropout=0.0,
            edge_dim=4, temporal=False, quantiles=ckpt["quantiles"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.qmid = self.model.median_index()

        d = self.d
        self.x_static = torch.as_tensor(d.X_static, device=self.device)
        self.X_dyn = torch.as_tensor(d.X_dyn, device=self.device)
        self.cell_of_node = torch.as_tensor(d.cell_of_node, device=self.device)
        self.edge_index = torch.as_tensor(d.edge_index, device=self.device)
        self.edge_attr3 = torch.as_tensor(d.edge_attr, device=self.device)
        self.bearing = torch.as_tensor(np.deg2rad(d.edge_bearing_deg), device=self.device)
        self.src_cells = self.cell_of_node[d.edge_index[0]]
        self.w_sin_col, self.w_cos_col = d._wind_sin_col, d._wind_cos_col

    def node_x(self, t: int) -> torch.Tensor:
        """[N, F] snapshot node features (raw, interpretable) at timestep t."""
        dyn_nodes = self.X_dyn[t][self.cell_of_node]
        return torch.cat([self.x_static, dyn_nodes], dim=1)

    def edge_attr(self, t: int) -> torch.Tensor:
        w_sin = self.X_dyn[t, self.src_cells, self.w_sin_col]
        w_cos = self.X_dyn[t, self.src_cells, self.w_cos_col]
        align = torch.cos(self.bearing) * w_cos + torch.sin(self.bearing) * w_sin
        return torch.cat([self.edge_attr3, align.unsqueeze(1)], dim=1)

    def ward_to_node(self, ward_id) -> int:
        row = self.d.nodes[self.d.nodes["ward_id"].astype(str) == str(ward_id)]
        if len(row) == 0:
            raise ValueError(f"ward_id {ward_id!r} not found")
        return int(row["node_idx"].iloc[0])

    def predict_aqi(self, t: int, node: int) -> float:
        with torch.no_grad():
            out = self.model(self.node_x(t), self.edge_index, self.edge_attr(t))
        return float(out[node, self.qmid].item() * self.y_sd + self.y_mu)


# --------------------------------------------------------------------------
# SHAP — gradient-based feature attribution on the graph
# --------------------------------------------------------------------------
class _MedianAtNode(torch.nn.Module):
    """Wraps the GNN so SHAP sees f(X) -> p50 at ONE target node, graph fixed."""
    def __init__(self, model, edge_index, edge_attr, node, qmid, n_nodes, n_feat):
        super().__init__()
        self.model, self.edge_index, self.edge_attr = model, edge_index, edge_attr
        self.node, self.qmid, self.N, self.F = node, qmid, n_nodes, n_feat

    def forward(self, X_flat):
        B = X_flat.shape[0]
        outs = []
        for b in range(B):                     # SHAP passes small batches
            x = X_flat[b].view(self.N, self.F)
            out = self.model(x, self.edge_index, self.edge_attr)
            outs.append(out[self.node, self.qmid])
        return torch.stack(outs).view(B, 1)


def shap_ward(ctx: ExplainContext, node: int, t: int, n_background: int = 40, seed: int = 0):
    """Local SHAP for one ward at one hour. Returns a DataFrame of the target
    ward's OWN-feature contributions (AQI points), most influential first.

    Background = a random sample of other timesteps' full graph states, so a
    feature's SHAP value answers 'vs a typical hour, how much did today's value
    of this feature move THIS ward's forecast'.
    """
    import shap
    rng = np.random.default_rng(seed)
    N, F = ctx.d.n_nodes, len(ctx.feature_names)
    ei, ea = ctx.edge_index, ctx.edge_attr(t)
    wrapper = _MedianAtNode(ctx.model, ei, ea, node, ctx.qmid, N, F).to(ctx.device)

    train_ts = ctx.d.t_indices("train")
    bg_ts = rng.choice(train_ts, size=min(n_background, len(train_ts)), replace=False)
    bg = torch.stack([ctx.node_x(int(tt)).reshape(-1) for tt in bg_ts]).to(ctx.device)
    query = ctx.node_x(t).reshape(1, -1).to(ctx.device)

    expl = shap.GradientExplainer(wrapper, bg)
    sv = expl.shap_values(query)                # [1, N*F] (or list)
    sv = np.asarray(sv).reshape(N, F) * ctx.y_sd  # de-normalise to AQI points

    own = sv[node]                              # target ward's own feature attribution
    # spatial (neighbour) contribution: everything that isn't the target node row
    neigh_total = float(sv.sum() - own.sum())
    df = (pd.DataFrame({"feature": ctx.feature_names, "shap_aqi": own})
          .assign(abs=lambda x: x["shap_aqi"].abs())
          .sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True))
    return df, neigh_total


def shap_global(ctx: ExplainContext, n_queries: int = 300, seed: int = 0) -> pd.DataFrame:
    """Global importance: mean |SHAP| per feature over many labelled ward-hours.

    Uses a fast single-baseline (expected-gradients-lite) so hundreds of queries
    are cheap: attribution = (x - x_ref) * d p50 / d x, averaged. This is the
    gradient-SHAP estimator with one reference = train-mean graph state.
    """
    rng = np.random.default_rng(seed)
    N, F = ctx.d.n_nodes, len(ctx.feature_names)
    train_ts = ctx.d.t_indices("train")
    x_ref = torch.stack([ctx.node_x(int(tt)) for tt in
                         rng.choice(train_ts, size=min(64, len(train_ts)), replace=False)]).mean(0)

    lab = ctx.d.labels
    lab = lab[lab["split_lab"] == "val"]
    idx = rng.choice(len(lab), size=min(n_queries, len(lab)), replace=False)
    rows = lab.iloc[idx]

    acc = np.zeros(F)
    n = 0
    for node, t in zip(rows["node_idx"].to_numpy(), rows["t_idx"].to_numpy()):
        node, t = int(node), int(t)
        x = ctx.node_x(t).clone().requires_grad_(True)
        out = ctx.model(x, ctx.edge_index, ctx.edge_attr(t))[node, ctx.qmid]
        ctx.model.zero_grad()
        out.backward()
        contrib = ((x[node] - x_ref[node]) * x.grad[node]).detach().cpu().numpy()
        acc += np.abs(contrib) * ctx.y_sd
        n += 1
    imp = acc / max(n, 1)
    return (pd.DataFrame({"feature": ctx.feature_names, "mean_abs_shap_aqi": imp})
            .sort_values("mean_abs_shap_aqi", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------
# GNNExplainer — sparse edge + feature masks (which neighbours matter)
# --------------------------------------------------------------------------
def gnn_explain_ward(ctx: ExplainContext, node: int, t: int, epochs: int = 200):
    """PyG GNNExplainer for one ward-hour. Returns (edges_df, feat_df):
      edges_df : neighbouring wards ranked by learned edge importance
      feat_df  : node features ranked by learned feature-mask importance
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ModelConfig

    x = ctx.node_x(t)
    ei = ctx.edge_index
    ea = ctx.edge_attr(t)

    explainer = Explainer(
        model=ctx.model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="regression", task_level="node", return_type="raw"),
    )
    expl = explainer(x, ei, index=node, edge_attr=ea, target=None)

    em = expl.edge_mask.detach().cpu().numpy()
    src = ei[0].cpu().numpy(); dst = ei[1].cpu().numpy()
    # edges that deliver a message TO the target node
    incoming = np.where(dst == node)[0]
    node2ward = dict(zip(ctx.d.nodes["node_idx"], ctx.d.nodes["ward_id"]))
    node2name = dict(zip(ctx.d.nodes["node_idx"], ctx.d.nodes["ward_name"]))
    edges_df = (pd.DataFrame({
        "neighbour_node": src[incoming],
        "neighbour_ward": [node2ward.get(s) for s in src[incoming]],
        "neighbour_name": [node2name.get(s) for s in src[incoming]],
        "importance": em[incoming],
    }).sort_values("importance", ascending=False).reset_index(drop=True))

    nm = expl.node_mask.detach().cpu().numpy()[node]      # target node's feature mask
    feat_df = (pd.DataFrame({"feature": ctx.feature_names, "importance": nm})
               .sort_values("importance", ascending=False).reset_index(drop=True))
    return edges_df, feat_df


def gnn_stability(ctx: ExplainContext, node: int, t: int, seeds: int = 3,
                  top: int = 5, epochs: int = 100) -> dict:
    """3.5 — run GNNExplainer with several seeds and report how consistent the
    top-neighbour set is (mean pairwise Jaccard). High agreement = trustworthy
    explanation; low = the explanation is seed-sensitive, treat with caution."""
    import itertools
    sets = []
    for s in range(seeds):
        torch.manual_seed(s)
        edges_df, _ = gnn_explain_ward(ctx, node, t, epochs=epochs)
        sets.append(set(edges_df.head(top)["neighbour_ward"].astype(str)))
    jac = []
    for a, b in itertools.combinations(sets, 2):
        u = a | b
        jac.append(len(a & b) / len(u) if u else 1.0)
    agree = float(np.mean(jac)) if jac else 1.0
    return {"seeds": seeds, "top": top, "top_neighbour_agreement": round(agree, 3),
            "verdict": "stable" if agree >= 0.6 else "seed-sensitive"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--ward", default=None, help="ward_id to explain")
    p.add_argument("--time-index", type=int, default=None, help="timestep index")
    p.add_argument("--global-shap", type=int, default=0, help="N queries for global importance")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ctx = ExplainContext(horizon=args.horizon, device=args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.global_shap:
        g = shap_global(ctx, n_queries=args.global_shap)
        g.to_csv(OUT_DIR / "shap_global_importance.csv", index=False)
        print("Global feature importance (mean |SHAP|, AQI points) — top 15:")
        print(g.head(15).to_string(index=False))

    if args.ward is not None and args.time_index is not None:
        node = ctx.ward_to_node(args.ward)
        t = args.time_index
        pred = ctx.predict_aqi(t, node)
        print(f"\nWard {args.ward} (node {node}) @ t={t} "
              f"[{str(ctx.d.times[t])[:16]}]  predicted AQI(p50) = {pred:.1f}")

        df, neigh = shap_ward(ctx, node, t)
        print("\nSHAP local — top drivers (AQI points):")
        print(df.head(12).to_string(index=False))
        print(f"neighbour (spatial) net contribution: {neigh:+.1f} AQI")

        edges_df, feat_df = gnn_explain_ward(ctx, node, t)
        print("\nGNNExplainer — most influential neighbouring wards:")
        print(edges_df.head(8).to_string(index=False))
        edges_df.to_csv(OUT_DIR / f"gnnexplainer_edges_ward{args.ward}_t{t}.csv", index=False)
        df.to_csv(OUT_DIR / f"shap_ward{args.ward}_t{t}.csv", index=False)


if __name__ == "__main__":
    main()
