"""
report_forecast_quality.py — improvements 2.8 + 2.10: forecast-quality report.

No retraining — re-scores the served ensemble with proper probabilistic metrics:
  * CRPS  — a proper score for the whole predictive distribution (lower = better).
  * PIT reliability — empirical P(y <= p10/p50/p90); should match 0.1/0.5/0.9.
  * interval coverage — before vs after the conformal wrapper (2.3).

    python report_forecast_quality.py
"""
from __future__ import annotations

import numpy as np
import torch

from advisor.serving import get_forecast_service
from stgnn_data import load_stgnn, build_supervised_pairs
from train_gnn import group_by_timestep
from metrics import (rmse, skill_vs_persistence, interval_coverage,
                     crps_from_quantiles, pit_calibration)

LEVELS = (0.1, 0.5, 0.9)


@torch.no_grad()
def collect(svc, d, split, horizon, apply_conformal, stride=3):
    delta = svc.conformal.get(horizon, 0.0) if apply_conformal else 0.0
    g = group_by_timestep(build_supervised_pairs(d, split))
    Y, Q, P = [], [], []
    for t in list(g.keys())[::stride]:
        node, y, ypers = g[t]
        q = svc._forward_all_raw(svc.ctx.node_x(t), t, horizon)   # raw (no conformal)
        q = q[node].copy()
        q[:, 0] -= delta; q[:, -1] += delta
        Y.append(y); Q.append(q); P.append(ypers)
    return np.concatenate(Y), np.concatenate(Q), np.concatenate(P)


def main():
    svc = get_forecast_service()
    h = svc.cfg.horizon
    d = load_stgnn(horizon=h)
    print(f"forecast-quality report — h{h} ensemble, test split\n")

    y, q_raw, yp = collect(svc, d, "test", h, apply_conformal=False)
    _, q_cal, _ = collect(svc, d, "test", h, apply_conformal=True)
    p50 = q_raw[:, 1]

    print(f"  point:   RMSE {rmse(y, p50):.1f}   skill vs persistence {skill_vs_persistence(y, p50, yp):+.1f}%")
    print(f"  CRPS:    {crps_from_quantiles(y, q_raw, LEVELS):.2f}  (lower = better; whole-distribution score)")
    print(f"  PIT reliability (empirical P[y<=q], want 0.10 / 0.50 / 0.90):")
    print(f"     raw       -> {pit_calibration(y, q_raw, LEVELS)}")
    print(f"     conformal -> {pit_calibration(y, q_cal, LEVELS)}")
    print(f"  p10-p90 coverage (target 0.80):  raw {interval_coverage(y, q_raw[:,0], q_raw[:,2]):.2f}"
          f"  ->  conformal {interval_coverage(y, q_cal[:,0], q_cal[:,2]):.2f}")


if __name__ == "__main__":
    main()
