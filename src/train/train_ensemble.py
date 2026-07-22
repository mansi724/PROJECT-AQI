"""
train_ensemble.py  —  higher accuracy by averaging several models
=================================================================
Trains K copies of the forecaster with different random seeds and averages their
predictions. Ensembling is the most reliable accuracy lever there is: independent
models make independent mistakes, so the average cancels noise. It also widens
and calibrates the p10/p90 band (each member disagrees a little).

Every member uses the winning single-model recipe (residual target + dense-grid
aux loss). Reported through metrics.py as skill vs persistence, same as always.

    python -m src.train.train_ensemble --k 4 --residual --aux-grid-weight 0.3 --no-temporal --epochs 45
=================================================================
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from src.train.train_gnn import Trainer
from src.metrics import scoreboard, format_scoreboard, interval_coverage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=4, help="ensemble size")
    p.add_argument("--horizon", type=int, default=24, choices=[24, 48, 72])
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--dropout", type=float, default=0.3)  # 2.4 sweep: better calibration
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pretrain-epochs", type=int, default=0)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--aux-grid-weight", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-ts", type=int, default=64)
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-train-ts", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    val_preds, test_preds = [], []
    val_ref = test_ref = None
    for k in range(args.k):
        a = copy.deepcopy(args)
        a.seed = 1000 + k
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        print(f"\n########## ensemble member {k+1}/{args.k} (seed {a.seed}) ##########")
        tr = Trainer(a)
        if a.max_train_ts:
            keys = list(tr.train_g.keys())[:a.max_train_ts]
            tr.train_g = {kk: tr.train_g[kk] for kk in keys}
        tr.fit()
        ckpt = tr.save(suffix=f"_ens{k}")            # persist this member for serving
        print("  saved member ->", ckpt.name)
        yv, ypv, v10, v50, v90 = tr.predict_pairs(tr.val_pairs)
        yt, ypt, t10, t50, t90 = tr.predict_pairs(tr.test_pairs)
        val_ref, test_ref = (yv, ypv), (yt, ypt)
        val_preds.append((v10, v50, v90)); test_preds.append((t10, t50, t90))
        # per-member scoreboards
        print("  member val :", format_scoreboard(scoreboard(yv, v50, ypv, "val")))
        print("  member test:", format_scoreboard(scoreboard(yt, t50, ypt, "test")))
        del tr; torch.cuda.empty_cache()

    def blend(preds):
        arr = np.stack([np.stack(p) for p in preds])   # [K,3,M]
        return arr.mean(0)                               # [3,M]

    for name, ref, preds in [("val", val_ref, val_preds), ("test", test_ref, test_preds)]:
        y, yp = ref
        p10, p50, p90 = blend(preds)
        sb = scoreboard(y, p50, yp, f"ENSEMBLE-{name}")
        sb["coverage_10_90"] = interval_coverage(y, p10, p90)
        print(f"\n== {format_scoreboard(sb)}  cov={sb['coverage_10_90']:.2f} ==")


if __name__ == "__main__":
    main()
