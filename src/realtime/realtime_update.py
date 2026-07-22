"""
realtime_update.py  —  the LIVE ingestion path (IMPROVEMENTS §1.7)
=================================================================
A scheduled job that pulls the last few days of CAMS air-quality + weather for
the 19 grid cells, rebuilds *exactly* the features the frozen GNN was trained on,
scales them with the *already-fitted* production scalers, and drops a rolling
window into `data/realtime/`. The advisor/dashboard then serves forecasts from
`now` instead of a fixed historical hour.

Why this is safe (no retrain, no distribution drift)
----------------------------------------------------
* SAME SOURCE. Live pollutants come from the identical Open-Meteo CAMS endpoint
  used for the 3-year history (`air-quality-api.open-meteo.com`); only the query
  switches from `start_date/end_date` to `past_days/forecast_days`. Weather comes
  from the Open-Meteo forecast API (the ERA5 archive lags ~5 days, so it cannot be
  "live"); it is the same variable set and units.
* SAME FEATURES. Every engineered column (bias-corrected PM, CPCB AQI, lags,
  rolls, meteorology, attribution signatures, time flags) is produced by importing
  the *same functions* `build_gnn_dataset.py` used — not a re-implementation.
* SAME SCALING. Normalisation reuses the production `scalers.joblib` (fit on the
  training split only). Nothing is re-fitted, so a live row lands in the exact
  z-space the model expects.

Outputs (all under data/realtime/ — the one folder the historical pipeline may
NOT touch; this is its dedicated writer):
    dynamic_grid.parquet        raw, un-scaled  -> RawDynamics (display panels)
    dynamic_grid_norm.parquet   scaled          -> load_stgnn  (the GNN)
    status.json                 fetched_at, coverage, latest live hour

Activate live serving (so the advisor picks these up):
    set AQI_REALTIME=1        (PowerShell:  $env:AQI_REALTIME = "1")
then (re)start the API.  Run this script from cron/Task Scheduler hourly.

Usage:
    python -m src.realtime.realtime_update                 # default 10 past + 1 forecast day
    python -m src.realtime.realtime_update --past 14 --forecast 2
    python -m src.realtime.realtime_update --dry-run       # fetch + build, don't write
=================================================================
"""
from __future__ import annotations

import argparse
import json
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from src import config as C
from src.data.build_gnn_dataset import (
    PM25_FACTOR, PM10_FACTOR, cpcb_aqi, add_time_features, add_meteo,
    add_attribution_features, add_lags_rolls, add_targets,
)

BASE = Path(__file__).resolve().parents[2]
GNN = BASE / "data" / "gnn"
GNN_PROC = BASE / "data" / "gnn_processed"
OUT = C.REALTIME_DIR                       # data/realtime/  (this script's job)

WARMUP_H = 48                              # first 48 h per cell feed the 48 h lags → drop them

# Weather variables we fetch live (core + boundary layer height, which the
# forecast API supports). Kept identical to the historical weather set.
WEATHER_VARS = C.OPENMETEO_WEATHER_VARS + ["boundary_layer_height"]
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"   # forecast model (archive lags days)


def log(m: str):
    # Windows consoles default to cp1252, which cannot encode some glyphs; stay ASCII-safe.
    print(f"[realtime] {m}".encode("ascii", "replace").decode("ascii"), flush=True)


# ---------------------------------------------------------------------------
# 1. Fetch  — one AQ call + one weather call per grid cell, merged on time
# ---------------------------------------------------------------------------
def _hourly_json(url: str, lat: float, lon: float, hourly: list[str],
                 past_days: int, forecast_days: int) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(hourly),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    last_err = None
    for attempt in range(1, C.MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 429:
                _time.sleep(C.RETRY_BACKOFF * attempt)
                continue
            r.raise_for_status()
            h = r.json().get("hourly", {})
            if not h.get("time"):
                return pd.DataFrame()
            df = pd.DataFrame(h)
            df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception as e:                       # noqa: BLE001
            last_err = e
            _time.sleep(C.RETRY_BACKOFF * attempt)
    log(f"  fetch failed ({lat},{lon}): {last_err}")
    return pd.DataFrame()


def served_cells() -> list[tuple[int, float, float]]:
    """The grid cells the model actually uses — the subset of C.grid_points()
    that contain wards (19 of 36). Taken from nodes_static so live point_ids match
    the historical dataset exactly (same enumeration → same lat/lon per point_id)."""
    served = set(pd.read_parquet(GNN / "nodes_static.parquet",
                                 columns=["point_id"])["point_id"].unique().tolist())
    return [(pid, lat, lon) for pid, lat, lon in C.grid_points() if pid in served]


def fetch_grid(past_days: int, forecast_days: int) -> pd.DataFrame:
    """Raw per-cell hourly frame for the served grid cells (pollutants + weather)."""
    frames = []
    pts = served_cells()
    for i, (pid, lat, lon) in enumerate(pts, 1):
        aq = _hourly_json(C.OPENMETEO_AQ_URL, lat, lon, C.OPENMETEO_AQ_VARS,
                          past_days, forecast_days)
        _time.sleep(C.OPENMETEO_SLEEP)
        wx = _hourly_json(WEATHER_URL, lat, lon, WEATHER_VARS,
                          past_days, forecast_days)
        _time.sleep(C.OPENMETEO_SLEEP)
        if aq.empty or wx.empty:
            log(f"  cell {pid}: incomplete (aq={len(aq)} wx={len(wx)}) — skipped")
            continue
        m = aq.merge(wx, on="time", how="inner")
        m.insert(0, "point_id", pid)
        frames.append(m)
        if i % 5 == 0 or i == len(pts):
            log(f"  fetched {i}/{len(pts)} cells")
    if not frames:
        raise SystemExit("realtime: no cells fetched — check connectivity / Open-Meteo")
    df = pd.concat(frames, ignore_index=True)
    # FIRMS fire is key-gated and near-zero outside the Oct–Nov stubble season;
    # a live keyless pull isn't available, so default to 0 (honest, documented).
    df["fire_count"] = 0.0
    df["fire_frp_sum"] = 0.0
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering — reuse the SAME builders as the historical dataset
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = (df.drop_duplicates(["point_id", "time"])
            .sort_values(["point_id", "time"])
            .reset_index(drop=True))
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month

    # --- CPCB-anchored monthly PM bias correction (identical to build_dynamic) ---
    df["pm2_5_raw"] = df["pm2_5"]
    df["pm10_raw"] = df["pm10"]
    df["pm2_5"] = (df["pm2_5"] * df["month"].map(PM25_FACTOR)).astype("float32")
    df["pm10"] = (df["pm10"] * df["month"].map(PM10_FACTOR)).astype("float32")

    # --- CPCB AQI on the time-ordered 19-cell series (rolling(24) == 24 real h) ---
    df["aqi"], df["dominant_pollutant"] = cpcb_aqi(df, group_col="point_id")

    df = add_time_features(df)
    df = add_meteo(df)
    df = add_attribution_features(df)
    df = add_lags_rolls(df)
    df = add_targets(df)                       # targets NaN at the tail — fine, unused for serving
    df = df.drop(columns=["month"])

    # Drop only the leading warm-up per cell (the 48 h that seed the lags). The
    # TAIL is kept on purpose — those are the recent + short-forecast hours we
    # want to serve live.
    df["_rank"] = df.groupby("point_id").cumcount()
    df = df[df["_rank"] >= WARMUP_H].drop(columns="_rank").reset_index(drop=True)
    df["split"] = "realtime"
    return df


# ---------------------------------------------------------------------------
# 3. Normalise with the ALREADY-FITTED production scalers (never re-fit)
# ---------------------------------------------------------------------------
def _impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    df[cols] = (df.groupby("point_id")[cols]
                  .transform(lambda s: s.interpolate(limit_direction="both")))
    for c in cols:
        if df[c].isna().any():
            df[c] = df.groupby("point_id")[c].transform(lambda s: s.fillna(s.median()))
            df[c] = df[c].fillna(df[c].median())
    return df


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(GNN_PROC / "scalers.joblib")
    scalers, groups = bundle["scalers"], bundle["groups"]
    out = df.copy()

    robust_cols = groups["dyn_robust"]
    standard_cols = groups["dyn_standard"]
    log1p_cols = groups["dyn_log1p"]

    # Continuous columns get interpolated within each cell first (as in preprocess);
    # fire counts (log1p group) are 0-filled, never interpolated.
    out = _impute(out, robust_cols + standard_cols)
    for c in log1p_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    def _apply(obj, cols, log1p=False):
        cols = [c for c in cols if c in out.columns]
        if not cols:
            return
        X = out[cols].to_numpy(dtype="float64")
        if log1p:
            X = np.log1p(np.clip(X, 0, None))
        out[cols] = obj.transform(X).astype("float32")

    _apply(scalers["dyn_robust"], robust_cols)
    _apply(scalers["dyn_standard"], standard_cols)
    _apply(scalers["dyn_log1p_standard"], log1p_cols, log1p=True)
    # DYN_ASIS (sin/cos, binary flags, fire_upwind) are left untouched — as in preprocess.
    return out


# ---------------------------------------------------------------------------
# 4. Align to the production schema so the loaders can concat without surprises
# ---------------------------------------------------------------------------
def _align(df: pd.DataFrame, template_path: Path) -> pd.DataFrame:
    template = pd.read_parquet(template_path, columns=None)
    cols = list(template.columns)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def main():
    ap = argparse.ArgumentParser(description="Live CAMS+weather ingestion for the AQI advisor")
    ap.add_argument("--past", type=int, default=10, help="past days to fetch (>=3 for lag warm-up)")
    ap.add_argument("--forecast", type=int, default=1, help="forecast days to append")
    ap.add_argument("--dry-run", action="store_true", help="fetch + build but do not write")
    args = ap.parse_args()

    if args.past < 3:
        raise SystemExit("--past must be >= 3 so the 48 h lags/rolls have warm-up context")

    t0 = _time.time()
    log(f"fetching {args.past}d past + {args.forecast}d forecast for "
        f"{len(served_cells())} served grid cells ...")
    raw = fetch_grid(args.past, args.forecast)
    feats = build_features(raw)
    norm = normalise(feats)

    latest_obs = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    live = feats[feats["time"] <= latest_obs]
    n_live = int(live["time"].nunique())
    span = (f"{feats['time'].min()} -> {feats['time'].max()}")
    log(f"built {len(feats):,} cell-hours over {feats['point_id'].nunique()} cells | {span}")
    log(f"  {n_live} observed hours <= now, {int(feats['time'].nunique()) - n_live} forecast hours")

    if args.dry_run:
        log("dry-run: nothing written")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    raw_out = OUT / "dynamic_grid.parquet"
    norm_out = OUT / "dynamic_grid_norm.parquet"
    feats_aligned = _align(feats, BASE / "data" / "gnn" / "dynamic_grid.parquet")
    norm_aligned = _align(norm, GNN_PROC / "dynamic_grid_norm.parquet")
    feats_aligned.to_parquet(raw_out, index=False)
    norm_aligned.to_parquet(norm_out, index=False)

    status = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "past_days": args.past, "forecast_days": args.forecast,
        "cells": int(feats["point_id"].nunique()),
        "cell_hours": int(len(feats)),
        "time_min": str(feats["time"].min()),
        "time_max": str(feats["time"].max()),
        "latest_observed_hour": str(live["time"].max()) if len(live) else None,
        "observed_hours": n_live,
        "forecast_hours": int(feats["time"].nunique()) - n_live,
        "source": {"air_quality": C.OPENMETEO_AQ_URL, "weather": WEATHER_URL},
        "note": "fire set to 0 (FIRMS live pull is key-gated; negligible outside Oct–Nov).",
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    log(f"WROTE {raw_out.relative_to(BASE)}  ({feats_aligned.shape})")
    log(f"WROTE {norm_out.relative_to(BASE)}  ({norm_aligned.shape})")
    log(f"latest observed live hour: {status['latest_observed_hour']}")
    log(f"done in {_time.time() - t0:.0f}s. Serve live with  AQI_REALTIME=1  then restart the API.")


if __name__ == "__main__":
    main()
