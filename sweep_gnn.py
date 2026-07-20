"""
sweep_gnn.py — improvement 2.4: hyperparameter search for the forecaster recipe.

Reuses the existing `train_gnn.Trainer` (no new training logic). Trains one
SNAPSHOT model per config and selects on the **val split (winter) skill vs
persistence** — chronological, never random-split. The **test split is never used
for selection** (summer; reported only for reference). The winning config is
written to `models/checkpoints/best_config.json` to feed h48/h72 (2.2) and the
final ensemble retrain (2.1).

    python sweep_gnn.py            # full sweep (~35-45 min)
    python sweep_gnn.py --quick    # 2 configs, short — smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_gnn import Trainer
from metrics import format_scoreboard

CKPT_DIR = Path(__file__).resolve().parent / "models" / "checkpoints"

# Base recipe = what currently serves. Each config overrides a few knobs.
BASE = dict(horizon=24, epochs=40, hidden=64, heads=4, layers=2, lookback=24,
            lr=2e-3, dropout=0.2, weight_decay=1e-4, pretrain_epochs=10,
            residual=False, aux_grid_weight=0.0, seed=0, patience=8, batch_ts=64,
            no_temporal=True, device="cuda", max_train_ts=0)

# Focused grid (one/two knobs off the base) — snapshot ignores `lookback`.
GRID = [
    {"name": "base"},
    {"name": "hidden96", "hidden": 96},
    {"name": "hidden48", "hidden": 48},
    {"name": "layers3", "layers": 3},
    {"name": "heads8", "heads": 8},
    {"name": "lr1e-3", "lr": 1e-3},
    {"name": "lr3e-3", "lr": 3e-3},
    {"name": "dropout0.3", "dropout": 0.3},
    {"name": "dropout0.1", "dropout": 0.1},
    {"name": "wd3e-4", "weight_decay": 3e-4},
    {"name": "hidden96_layers3", "hidden": 96, "layers": 3},
    {"name": "hidden96_heads8_lr1e-3", "hidden": 96, "heads": 8, "lr": 1e-3},
]


def run_config(cfg: dict) -> dict:
    args = SimpleNamespace(**{**BASE, **{k: v for k, v in cfg.items() if k != "name"}})
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tr = Trainer(args)
    tr.fit()
    val = tr.evaluate(tr.val_pairs, "val")
    test = tr.evaluate(tr.test_pairs, "test")
    del tr; torch.cuda.empty_cache()
    return {"name": cfg["name"], "config": {k: getattr(args, k) for k in
            ("hidden", "heads", "layers", "lr", "dropout", "weight_decay", "pretrain_epochs")},
            "val_skill": val["skill_%"], "val_rmse": val["rmse"], "val_cov": val["coverage_10_90"],
            "test_skill": test["skill_%"], "test_rmse": test["rmse"], "test_cov": test["coverage_10_90"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    grid = GRID[:2] if args.quick else GRID
    if args.quick:
        BASE["epochs"] = 4; BASE["pretrain_epochs"] = 2

    results = []
    t0 = time.time()
    for i, cfg in enumerate(grid, 1):
        print(f"\n===== [{i}/{len(grid)}] config: {cfg['name']} =====")
        try:
            r = run_config(cfg)
            results.append(r)
            print(f"  -> val skill {r['val_skill']:+.1f}% (cov {r['val_cov']:.2f}) | "
                  f"test skill {r['test_skill']:+.1f}% (cov {r['test_cov']:.2f})")
        except Exception as e:
            print(f"  FAILED: {e}")

    results.sort(key=lambda r: -r["val_skill"])       # select on winter/val
    print(f"\n{'='*70}\nLEADERBOARD (sorted by VAL/winter skill) — {time.time()-t0:.0f}s total")
    print(f"{'config':26} {'val_skill':>9} {'val_cov':>7} {'test_skill':>10} {'test_cov':>8}")
    for r in results:
        print(f"{r['name']:26} {r['val_skill']:+8.1f}% {r['val_cov']:6.2f} "
              f"{r['test_skill']:+9.1f}% {r['test_cov']:7.2f}")

    if results:
        best = results[0]
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        out = CKPT_DIR / "best_config.json"
        out.write_text(json.dumps({"selected_on": "val_skill", **best}, indent=2))
        print(f"\nBEST: {best['name']}  val {best['val_skill']:+.1f}%  test {best['test_skill']:+.1f}%")
        print("saved ->", out)
        cur = next((r for r in results if r["name"] == "base"), None)
        if cur:
            print(f"vs base: val {best['val_skill']-cur['val_skill']:+.1f} pts, "
                  f"test {best['test_skill']-cur['test_skill']:+.1f} pts")


if __name__ == "__main__":
    main()
