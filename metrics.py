"""
metrics.py
=================================================================
The ONE scorer every engine reports through. Two rules the whole project
lives by (see PLAN.md "Non-negotiable guardrails"):

  1. Never report a raw RMSE alone — always report **% skill vs persistence**.
     A model that "gets RMSE 74" means nothing until you know persistence is 86.
  2. Never random-split a time series, and never CV across the 19 CAMS cells
     without grouping by `point_id` — there are only 19 independent locations,
     so a random fold leaks a cell's own hours into its test set.

Everything is numpy-in / float-out and de-normalised-space (AQI points), so
Engine 1 (GNN) and the GBDT baseline are scored on the same footing.
=================================================================
"""
from __future__ import annotations

import numpy as np


# --- point estimates -------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def skill_vs_persistence(y_true, y_pred, y_persist) -> float:
    """% RMSE improvement over the naive 'tomorrow = today' baseline.

    Positive = better than persistence. This is THE headline number; a GNN that
    cannot clear 0 % has learned nothing the baseline didn't already know.
    """
    r_model = rmse(y_true, y_pred)
    r_base = rmse(y_true, y_persist)
    if r_base == 0:
        return float("nan")
    return float(100.0 * (r_base - r_model) / r_base)


def scoreboard(y_true, y_pred, y_persist=None, label: str = "") -> dict:
    """The dict every eval prints. Keeps reporting uniform across engines."""
    out = {"label": label, "n": int(len(y_true)),
           "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}
    if y_persist is not None:
        out["rmse_persistence"] = rmse(y_true, y_persist)
        out["skill_%"] = skill_vs_persistence(y_true, y_pred, y_persist)
    return out


def format_scoreboard(s: dict) -> str:
    base = f"[{s['label']}] n={s['n']:>7,}  RMSE={s['rmse']:6.2f}  MAE={s['mae']:6.2f}"
    if "skill_%" in s:
        base += f"  | persistence={s['rmse_persistence']:6.2f}  skill={s['skill_%']:+5.1f}%"
    return base


# --- quantile / uncertainty ------------------------------------------------
def pinball_loss(y_true, y_pred, q: float) -> float:
    """Mean pinball (quantile) loss at quantile q in (0,1)."""
    y_true, y_pred = _clean(y_true, y_pred)
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def crps_from_quantiles(y_true, q_preds, levels) -> float:
    """CRPS estimate from a set of predicted quantiles (improvement 2.8/2.10).

    A proper scoring rule for the WHOLE predictive distribution (lower = better),
    approximated as 2·mean pinball loss across the quantile levels. With just
    3 quantiles it's coarse but valid and comparable across models.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1, 1)
    q = np.asarray(q_preds, dtype=float)
    lv = np.asarray(levels, dtype=float).reshape(1, -1)
    m = np.isfinite(y).ravel() & np.isfinite(q).all(axis=1)
    e = y[m] - q[m]
    pin = np.maximum(lv * e, (lv - 1) * e)
    return float(2.0 * pin.mean())


def pit_calibration(y_true, q_preds, levels) -> dict:
    """Reliability check (PIT): empirical P(y <= q_level) should equal `level`.

    Returns {level: empirical_coverage}. A well-calibrated forecast matches the
    diagonal (e.g. ~10% of truths fall below the p10 prediction).
    """
    y = np.asarray(y_true, dtype=float).reshape(-1, 1)
    q = np.asarray(q_preds, dtype=float)
    m = np.isfinite(y).ravel() & np.isfinite(q).all(axis=1)
    emp = (y[m] <= q[m]).mean(axis=0)
    return {round(float(l), 2): round(float(e), 3) for l, e in zip(levels, emp)}


def interval_coverage(y_true, y_lo, y_hi) -> float:
    """Fraction of truths inside [lo, hi]. For a 0.1/0.9 band, target ~0.80."""
    y_true = np.asarray(y_true, dtype=float)
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_lo) & np.isfinite(y_hi)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((y_true[m] >= y_lo[m]) & (y_true[m] <= y_hi[m])))


# --- chronological / grouped CV -------------------------------------------
def point_id_folds(point_ids: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Yield (train_idx, test_idx) where each test fold holds OUT whole cells.

    Grid-level CV must group by point_id: the 19 cells are the only independent
    units, so a fold that splits a cell's hours across train/test leaks. Use
    this for the GBDT baseline / attribution CV, NOT for the GNN's chronological
    train/val/test (that comes pre-baked in the `split` column).
    """
    point_ids = np.asarray(point_ids)
    cells = np.unique(point_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    for fold in np.array_split(cells, min(n_splits, len(cells))):
        test = np.isin(point_ids, fold)
        yield np.where(~test)[0], np.where(test)[0]


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.normal(200, 60, 5000)
    persist = y + rng.normal(0, 86, 5000)      # baseline ~ RMSE 86
    good = y + rng.normal(0, 60, 5000)         # model  ~ RMSE 60
    print(format_scoreboard(scoreboard(y, good, persist, "self-test")))
    print("pinball q0.9:", round(pinball_loss(y, good, 0.9), 3))
    print("coverage 10/90:", round(interval_coverage(y, good - 80, good + 80), 3))
    pids = rng.integers(7, 34, 5000)
    folds = list(point_id_folds(pids, 5))
    print(f"cv folds: {len(folds)}  test sizes: {[len(t) for _, t in folds]}")
