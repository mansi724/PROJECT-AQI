"""
build_features.py
=================================================================
Turns the merged dataset (data/final/delhi_ward_dataset_clean.csv)
into a MODEL-READY feature table for the three engines:

  1. AQI forecasting      -> lags, rolling stats, cyclical time, forecast target
  2. Source attribution   -> wind-direction (upwind) + stagnation features
  3. Action recommendation-> ward health-risk / priority score

Run:
    python build_features.py                       # default in/out
    python build_features.py --horizon 24          # forecast horizon (hours)
    python build_features.py --sample 200000       # quick test on N rows
    python build_features.py --format csv          # csv instead of parquet

Output:  data/final/model_ready.parquet  (or .csv)
No source columns are modified; only new features are added.
=================================================================
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

BASE = Path(__file__).resolve().parent
IN_DEFAULT = BASE / "data" / "final" / "delhi_ward_dataset.csv"
OUT_DIR = BASE / "data" / "final"

# Columns we build time-series features on
LAG_COLS = ["aqi", "pm2_5", "pm10", "nitrogen_dioxide"]
LAGS = [1, 3, 6, 12, 24, 48]
ROLL_COLS = ["aqi", "pm2_5"]

# Stubble-burning smoke reaches Delhi on NW winds (Punjab/Haryana are NW).
# wind_direction_10m is the direction the wind blows FROM (met convention).
NW_MIN, NW_MAX = 290.0, 350.0
RUSH_HOURS = {8, 9, 10, 18, 19, 20}

CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def log(m):
    print(f"[features] {m}", flush=True)


# ---------------------------------------------------------------------------
def load(in_path, sample):
    if not in_path.exists():
        sys.exit(f"[features] input not found: {in_path}\n"
                 f"           point --input at your cleaned dataset.")
    log(f"loading {in_path.name} ...")
    df = pd.read_csv(in_path, low_memory=False)
    if sample:
        # keep whole wards so lags stay valid, just take the first few wards
        keep = df["ward_id"].drop_duplicates().head(max(1, sample // 26000))
        df = df[df["ward_id"].isin(keep)].copy()
    log(f"rows={len(df):,}  wards={df['ward_id'].nunique()}  cols={df.shape[1]}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values(["ward_id", "time"]).reset_index(drop=True)
    # shrink memory: float64 -> float32
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    return df


# ---------------------------------------------------------------------------
def add_time_features(df):
    log("cyclical time features ...")
    h, m, d = df["hour"], df["month"], df["dayofweek"]
    df["hour_sin"] = np.sin(2 * np.pi * h / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * h / 24).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * m / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * m / 12).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * d / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * d / 7).astype("float32")
    df["is_weekend"] = (d >= 5).astype("int8")
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype("int8")
    # winter = Delhi's high-pollution season (Oct-Feb)
    df["is_winter"] = df["month"].isin([10, 11, 12, 1, 2]).astype("int8")
    return df


def add_lags_rolls(df):
    log("lag + rolling features (per ward) ...")
    g = df.groupby("ward_id", sort=False)
    for col in LAG_COLS:
        for lag in LAGS:
            df[f"{col}_lag_{lag}h"] = g[col].shift(lag).astype("float32")
    for col in ROLL_COLS:
        s = g[col]
        df[f"{col}_roll_mean_6h"] = s.transform(lambda x: x.rolling(6, min_periods=3).mean()).astype("float32")
        df[f"{col}_roll_mean_24h"] = s.transform(lambda x: x.rolling(24, min_periods=12).mean()).astype("float32")
        df[f"{col}_roll_max_24h"] = s.transform(lambda x: x.rolling(24, min_periods=12).max()).astype("float32")
        df[f"{col}_roll_std_24h"] = s.transform(lambda x: x.rolling(24, min_periods=12).std()).astype("float32")
    # short-term trend
    df["aqi_diff_1h"] = (df["aqi"] - df["aqi_lag_1h"]).astype("float32")
    df["aqi_diff_24h"] = (df["aqi"] - df["aqi_lag_24h"]).astype("float32")
    return df.copy()   # de-fragment after many column inserts


def add_meteo_features(df):
    log("meteorology / dispersion features ...")
    ws = df["wind_speed_10m"].astype("float32")
    blh = df["boundary_layer_height"].astype("float32")
    wd = df["wind_direction_10m"].astype("float32")
    df["wind_dir_sin"] = np.sin(np.deg2rad(wd)).astype("float32")
    df["wind_dir_cos"] = np.cos(np.deg2rad(wd)).astype("float32")
    # ventilation high = good dispersion; stagnation high = pollution builds up
    df["ventilation_index"] = (ws * blh).astype("float32")
    df["stagnation_index"] = (1.0 / ((ws + 0.1) * (blh + 1.0))).astype("float32")
    return df


def add_source_features(df):
    log("source-attribution (directional) features ...")
    # 1 when wind carries NW stubble smoke toward Delhi
    df["wind_from_nw"] = ((df["wind_direction_10m"] >= NW_MIN) &
                          (df["wind_direction_10m"] <= NW_MAX)).astype("int8")
    df["fire_upwind"] = (df["fire_frp_sum"] * df["wind_from_nw"]).astype("float32")
    # local traffic pressure (road capacity x rush hour) and industry under stagnant air
    road_feat = "road_capacity_3km" if "road_capacity_3km" in df.columns else "road_km_3km"
    df["traffic_load"] = (df[road_feat] * df["is_rush_hour"]).astype("float32")
    df["industry_stagnation"] = (df["industry_count_5km"] * df["stagnation_index"]).astype("float32")
    # pollution trapped when calm + low mixing height
    df["buildup_pressure"] = (df["stagnation_index"] * df["aqi_lag_1h"]).astype("float32")
    return df


def add_encodings(df):
    log("categorical encodings ...")
    cat = pd.Categorical(df["aqi_category"], categories=CATEGORY_ORDER, ordered=True)
    df["aqi_category_code"] = cat.codes.astype("int8")           # -1 where NaN
    df["dominant_pollutant_code"] = df["dominant_pollutant"].astype("category").cat.codes.astype("int8")
    return df


def add_action_features(df):
    log("action-recommendation / health-risk features ...")
    # normalise receptor exposure 0..1
    v = df["vulnerable_sites_3km"].astype("float32")
    vmax = v.max() if v.max() > 0 else 1.0
    df["vuln_norm"] = (v / vmax).astype("float32")
    # priority = how bad the air is x how many people/receptors are exposed
    df["health_risk_score"] = (df["aqi"] * (0.5 + df["vuln_norm"]) *
                               (1.0 + 0.3 * df["population_density_mean"].rank(pct=True))).astype("float32")
    # simple ordinal risk band from AQI category
    df["health_risk_level"] = df["aqi_category_code"].clip(lower=0).astype("int8")
    return df


def add_target_and_split(df, horizon):
    log(f"forecast target (t+{horizon}h) + persistence baseline + time split ...")
    g = df.groupby("ward_id", sort=False)
    df[f"target_aqi_t{horizon}"] = g["aqi"].shift(-horizon).astype("float32")
    # naive baseline: predict t+h AQI = current AQI
    df[f"persistence_aqi_t{horizon}"] = df["aqi"].astype("float32")
    # time-based split (no leakage): 70% train / 15% val / 15% test by date
    q70 = df["time"].quantile(0.70)
    q85 = df["time"].quantile(0.85)
    split = np.where(df["time"] <= q70, "train",
             np.where(df["time"] <= q85, "val", "test"))
    df["split"] = split
    log(f"  train<= {q70.date()}  |  val<= {q85.date()}  |  test after")
    return df


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(IN_DEFAULT))
    ap.add_argument("--horizon", type=int, default=24, help="forecast horizon in hours")
    ap.add_argument("--sample", type=int, default=0, help="test on ~N rows (whole wards)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = ap.parse_args()

    df = load(Path(args.input), args.sample)

    # optional: merge EDGAR emission-inventory features (from add_emissions.py)
    emis_path = BASE / "data" / "processed" / "ward_emissions.csv"
    if emis_path.exists():
        emis = pd.read_csv(emis_path)
        for c in emis.select_dtypes("float64").columns:
            emis[c] = emis[c].astype("float32")
        df = df.merge(emis, on="ward_id", how="left")
        log(f"merged emission inventory: +{emis.shape[1]-1} cols from {emis_path.name}")
    else:
        log("no ward_emissions.csv found (optional) — run add_emissions.py to add it")

    df = add_time_features(df)
    df = add_lags_rolls(df)
    df = add_meteo_features(df)
    df = add_source_features(df)
    df = add_encodings(df)
    df = add_action_features(df)
    df = add_target_and_split(df, args.horizon)

    new_cols = [c for c in df.columns if c not in (
        "ward_id", "ward_name", "time", "point_id")]
    log(f"final shape: {df.shape[0]:,} rows x {df.shape[1]} cols "
        f"(+{df.shape[1] - 49} engineered)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        out = OUT_DIR / "model_ready.parquet"
        try:
            df.to_parquet(out, index=False, compression="snappy")
        except Exception as e:
            log(f"parquet failed ({e}); install pyarrow  ->  pip install pyarrow. Falling back to CSV.")
            out = OUT_DIR / "model_ready.csv"
            df.to_csv(out, index=False)
    else:
        out = OUT_DIR / "model_ready.csv"
        df.to_csv(out, index=False)

    sz = out.stat().st_size / 1e6
    log(f"SAVED -> {out}  ({sz:.1f} MB)")
    # tiny sanity print
    tcol = f"target_aqi_t{args.horizon}"
    valid = df.dropna(subset=[tcol] + [f"aqi_lag_{l}h" for l in LAGS])
    log(f"rows usable for training (target + all lags present): {len(valid):,}")
    log("Done.")


if __name__ == "__main__":
    main()
