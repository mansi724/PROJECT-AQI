"""
calibrate_conformal.py — improvement 2.3: conformalise the forecast intervals.

Conformalized Quantile Regression (CQR, Romano et al. 2019) — a POST-HOC wrapper,
no retraining. The model's raw p10/p90 band under-covers (~0.65 vs the 0.80
target). CQR computes, on a held-out calibration split, one correction δ per
horizon such that the widened band [p10-δ, p90+δ] achieves the target coverage,
with a finite-sample guarantee on exchangeable data.

Runs per available horizon (24/48/72), calibrates on **val** (the winter split),
and reports coverage on **test** before vs after. Saves δ to
`models/checkpoints/conformal.json`; `ForecastService` applies it at serve time.

    python -m src.train.calibrate_conformal
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from advisor.serving import get_forecast_service
from src.data.stgnn_data import load_stgnn, build_supervised_pairs
from src.train.train_gnn import group_by_timestep
from src.metrics import interval_coverage

TARGET = 0.80                      # nominal p10–p90 coverage
CKPT = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "conformal.json"


@torch.no_grad()
def _bands(svc, d, split, horizon, stride=3):
    """Return (y, p10, p90) for a split at a horizon, using the served model(s)."""
    g = group_by_timestep(build_supervised_pairs(d, split))
    keys = list(g.keys())[::stride]
    y, p10, p90 = [], [], []
    for t in keys:
        node, yy, _ = g[t]
        q = svc._forward_all(svc.ctx.node_x(t), t, horizon)      # [N, Q] de-normalised
        y.append(yy); p10.append(q[node, 0]); p90.append(q[node, -1])
    return (np.concatenate(y), np.concatenate(p10), np.concatenate(p90))


def cqr_delta(y, p10, p90, target=TARGET) -> float:
    """CQR conformity score = max(p10 - y, y - p90); δ = its target-quantile."""
    scores = np.maximum(p10 - y, y - p90)
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * target)))
    return float(np.sort(scores)[k - 1])


def main():
    svc = get_forecast_service()
    svc.conformal = {}                 # calibrate on RAW bands (avoid double-applying)
    horizons = svc.available_horizons()
    out = {"target": TARGET, "delta": {}}
    print(f"calibrating horizons {horizons} on val (winter), target coverage {TARGET:.0%}\n")
    for h in horizons:
        d = load_stgnn(horizon=h)                      # pairs with the right t+h label shift
        yv, lo_v, hi_v = _bands(svc, d, "val", h)
        delta = cqr_delta(yv, lo_v, hi_v)
        out["delta"][str(h)] = delta
        # verify on test
        yt, lo_t, hi_t = _bands(svc, d, "test", h)
        cov_before = interval_coverage(yt, lo_t, hi_t)
        cov_after = interval_coverage(yt, lo_t - delta, hi_t + delta)
        width_before = float(np.mean(hi_t - lo_t))
        width_after = float(np.mean((hi_t + delta) - (lo_t - delta)))
        print(f"h{h:>3}: delta={delta:6.1f} | test coverage {cov_before:.2f} -> {cov_after:.2f} "
              f"(target {TARGET:.2f}) | band width {width_before:.0f} -> {width_after:.0f} AQI")

    CKPT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {CKPT}")
    print("ForecastService will apply these deltas to p10/p90 automatically.")


if __name__ == "__main__":
    main()
