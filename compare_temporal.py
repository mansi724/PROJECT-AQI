"""
compare_temporal.py — improvement 2.5: compare temporal encoders.

Trains the TEMPORAL model with each encoder (GRU / Transformer / TCN) under
identical settings and reports val (winter) + test skill vs persistence. The bar
to beat is the current serving **snapshot ensemble** (test +23.6% / val +10.4%).
Reuses `train_gnn.Trainer` — no new training logic.

    python compare_temporal.py
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import torch

from train_gnn import Trainer

BASE = dict(horizon=24, epochs=40, hidden=64, heads=4, layers=2, lookback=24,
            lr=2e-3, dropout=0.3, weight_decay=1e-4, pretrain_epochs=10,
            residual=False, aux_grid_weight=0.0, seed=0, patience=8, batch_ts=64,
            no_temporal=False, device="cuda", max_train_ts=0)

KINDS = ["gru", "transformer", "tcn"]


def run(kind: str) -> dict:
    args = SimpleNamespace(**{**BASE, "temporal_kind": kind})
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tr = Trainer(args)
    tr.fit()
    val = tr.evaluate(tr.val_pairs, "val")
    test = tr.evaluate(tr.test_pairs, "test")
    n_params = sum(p.numel() for p in tr.model.parameters())
    del tr; torch.cuda.empty_cache()
    return {"kind": kind, "params": n_params,
            "val_skill": val["skill_%"], "val_cov": val["coverage_10_90"],
            "test_skill": test["skill_%"], "test_cov": test["coverage_10_90"]}


def main():
    t0 = time.time()
    rows = []
    for k in KINDS:
        print(f"\n===== temporal encoder: {k} =====")
        r = run(k)
        rows.append(r)
        print(f"  -> val {r['val_skill']:+.1f}% (cov {r['val_cov']:.2f}) | "
              f"test {r['test_skill']:+.1f}% (cov {r['test_cov']:.2f})")

    rows.sort(key=lambda r: -r["val_skill"])
    print(f"\n{'='*64}\nTEMPORAL ENCODER COMPARISON ({time.time()-t0:.0f}s)")
    print(f"{'encoder':14} {'params':>8} {'val_skill':>9} {'val_cov':>7} {'test_skill':>10} {'test_cov':>8}")
    for r in rows:
        print(f"{r['kind']:14} {r['params']:>8,} {r['val_skill']:+8.1f}% {r['val_cov']:6.2f} "
              f"{r['test_skill']:+9.1f}% {r['test_cov']:7.2f}")
    print("\nBar to beat (snapshot ensemble, now serving): val +10.4% / test +23.6% / cov ~0.65")


if __name__ == "__main__":
    main()
