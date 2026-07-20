"""
preprocess_dataset.py
=================================================================
Produce CLEANED + NORMALISED copies of the four data/gnn files, ready for the
STGNN. Implements the agreed missing-value + normalisation strategy.

Outputs -> data/gnn_processed/
    nodes_static_norm.parquet
    dynamic_grid_norm.parquet
    edges_norm.parquet
    labels_station_clean.parquet
    scalers.joblib          # every fitted transformer + column groups (for serving)
    NORMALISATION_MANIFEST.json

Key rules enforced here
-----------------------
* Scalers are FIT ON TRAIN ROWS ONLY, then applied to val/test. Fitting on the
  whole file would leak test statistics into training and inflate scores.
* Targets and ground-truth labels are NEVER normalised and NEVER imputed.
* Missing values are handled per the strategy table:
    - dynamic: drop first 48 h + last 72 h per cell, then interpolate residual
      gaps within each cell (linear), fire -> 0.
    - static / edges: no missing values.
    - labels: aqi_station is complete; Engine-2 label nulls are left in place
      to be DROPPED at engine training time (never imputed).
Run:  python preprocess_dataset.py
=================================================================
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

BASE = Path(__file__).resolve().parent
GNN = BASE / "data" / "gnn"
OUT = BASE / "data" / "gnn_processed"
OUT.mkdir(parents=True, exist_ok=True)

WARMUP_H = 48      # drop first 48 h per cell (lag/roll warm-up)
HORIZON_H = 72     # drop last 72 h per cell (t+72 label unavailable)


def log(m): print(f"[preprocess] {m}", flush=True)


# ======================================================================
# Column -> transform maps
# ======================================================================
# Heavy-tailed pollutant concentrations + everything derived from them
# (lags / rolling / diffs). RobustScaler resists winter smog spikes.
DYN_ROBUST = [
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
    "ozone", "aerosol_optical_depth", "dust", "aqi",
    *[f"aqi_lag_{h}h" for h in (1, 3, 6, 12, 24, 48)],
    *[f"pm2_5_lag_{h}h" for h in (1, 3, 6, 12, 24, 48)],
    *[f"pm10_lag_{h}h" for h in (1, 3, 6, 12, 24, 48)],
    *[f"nitrogen_dioxide_lag_{h}h" for h in (1, 3, 6, 12, 24, 48)],
    "aqi_roll_mean_6h", "aqi_roll_mean_24h", "aqi_roll_max_24h", "aqi_roll_std_24h",
    "pm2_5_roll_mean_6h", "pm2_5_roll_mean_24h", "pm2_5_roll_max_24h", "pm2_5_roll_std_24h",
    "aqi_diff_1h", "aqi_diff_24h",
]
# Bounded / roughly-gaussian continuous variables -> z-score
DYN_STANDARD = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_gusts_10m", "boundary_layer_height",
    "ventilation_index", "stagnation_index",
    "pm25_pm10_ratio", "dust_fraction", "so2_no2_ratio",
]
# Counts with many zeros and a long tail -> log1p then z-score
DYN_LOG1P = ["fire_count", "fire_frp_sum"]
# Already bounded (sin/cos or 0/1) -> leave untouched
DYN_ASIS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "wind_dir_sin", "wind_dir_cos", "is_weekend", "is_rush_hour", "is_winter",
    "is_stubble_season", "is_diwali_window", "wind_from_nw", "fire_upwind",
]
# Interpolated (continuous) columns = everything we scale except log/asis stay too
DYN_INTERPOLATE = DYN_ROBUST + DYN_STANDARD
# Kept raw, never scaled: identifiers, targets, leak copies
DYN_TARGETS = ["target_aqi_t24", "target_aqi_t48", "target_aqi_t72", "persistence_aqi_t24"]
DYN_KEEPRAW = ["point_id", "time", "split", "dominant_pollutant",
               "pm2_5_raw", "pm10_raw", "wind_direction_10m"]  # degrees kept but redundant vs sin/cos

# ---- static ----
STAT_LOG1P = [
    "ward_area_km2", "road_km_3km", "road_capacity_3km", "road_km_in_ward",
    "road_capacity_in_ward", "road_km_per_km2", "industry_count_5km",
    "industry_count_in_ward", "vulnerable_sites_3km", "vulnerable_sites_in_ward",
    "population_sum", "population_density_mean",
    "emis_power_co", "emis_industry_co", "emis_residential_co", "emis_transport_co",
    "emis_power_nox", "emis_industry_nox", "emis_residential_nox", "emis_transport_nox",
    "emis_power_pm25", "emis_industry_pm25", "emis_residential_pm25", "emis_transport_pm25",
    "emis_power_so2", "emis_industry_so2", "emis_residential_so2", "emis_transport_so2",
    "emis_total_pm25", "emis_total_co", "emis_total_so2", "emis_total_nox",
]
STAT_STANDARD = ["elevation_mean", "lu_builtup_fraction", "lu_tree_fraction"]
STAT_KEEPRAW = ["ward_id", "ward_name", "ward_lat", "ward_lon", "point_id",
                "node_idx", "lu_majority_class"]


def fit_apply(df, cols, scaler, train_mask, log1p=False):
    """Fit `scaler` on TRAIN rows of `cols`, transform all rows. Returns fitted scaler."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None, []
    X = df[cols].to_numpy(dtype="float64")
    if log1p:
        X = np.log1p(np.clip(X, 0, None))
    scaler.fit(X[train_mask])
    df[cols] = scaler.transform(X).astype("float32")
    return scaler, cols


# ======================================================================
# 1. dynamic_grid
# ======================================================================
def process_dynamic():
    log("dynamic_grid: loading")
    df = pd.read_parquet(GNN / "dynamic_grid.parquet").sort_values(["point_id", "time"])
    n0 = len(df)

    # ---- schema / duplicate / timestamp checks ----
    dups = df.duplicated(["point_id", "time"]).sum()
    if dups:
        log(f"  removing {dups} duplicate (point_id,time) rows")
        df = df.drop_duplicates(["point_id", "time"])

    # ---- drop warm-up (first 48h) and horizon tail (last 72h) per cell ----
    df["_rank"] = df.groupby("point_id").cumcount()
    df["_n"] = df.groupby("point_id")["_rank"].transform("max")
    keep = (df["_rank"] >= WARMUP_H) & (df["_rank"] <= df["_n"] - HORIZON_H)
    df = df[keep].drop(columns=["_rank", "_n"]).reset_index(drop=True)
    log(f"  rows {n0:,} -> {len(df):,} (dropped first {WARMUP_H}h + last {HORIZON_H}h per cell)")

    # ---- interpolate residual gaps within each cell (linear, both directions) ----
    interp_cols = [c for c in DYN_INTERPOLATE if c in df.columns]
    df[interp_cols] = (df.groupby("point_id")[interp_cols]
                         .transform(lambda s: s.interpolate(limit_direction="both")))
    # fire: NaN means "no detection" -> 0
    for c in DYN_LOG1P:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    # any straggler feature NaN -> cell median then global median
    feat_all = interp_cols + DYN_LOG1P
    for c in feat_all:
        if df[c].isna().any():
            df[c] = df.groupby("point_id")[c].transform(lambda s: s.fillna(s.median()))
            df[c] = df[c].fillna(df[c].median())

    # ---- normalise (FIT ON TRAIN ONLY) ----
    train_mask = (df["split"] == "train").to_numpy()
    log(f"  fitting scalers on {train_mask.sum():,} train rows")
    scalers = {}
    scalers["robust"], _ = fit_apply(df, DYN_ROBUST, RobustScaler(), train_mask)
    scalers["standard"], _ = fit_apply(df, DYN_STANDARD, StandardScaler(), train_mask)
    scalers["log1p_standard"], _ = fit_apply(df, DYN_LOG1P, StandardScaler(), train_mask, log1p=True)

    # ---- verify: zero NaN in feature cols (targets may still be null? no—tail dropped) ----
    feat_check = feat_all + [c for c in DYN_ASIS if c in df.columns]
    assert df[feat_check].isna().sum().sum() == 0, "feature NaNs remain!"
    assert df[[c for c in DYN_TARGETS if c in df.columns]].isna().sum().sum() == 0, "target NaNs remain!"

    df.to_parquet(OUT / "dynamic_grid_norm.parquet", index=False)
    log(f"  wrote dynamic_grid_norm.parquet  {df.shape}")
    return scalers, feat_all


# ======================================================================
# 2. nodes_static
# ======================================================================
def process_static():
    log("nodes_static: loading")
    df = pd.read_parquet(GNN / "nodes_static.parquet").sort_values("node_idx").reset_index(drop=True)
    all_mask = np.ones(len(df), dtype=bool)   # static: fit on all 289 wards
    scalers = {}
    scalers["static_log1p_standard"], _ = fit_apply(df, STAT_LOG1P, StandardScaler(), all_mask, log1p=True)
    scalers["static_standard"], _ = fit_apply(df, STAT_STANDARD, StandardScaler(), all_mask)
    assert df[[c for c in STAT_LOG1P + STAT_STANDARD if c in df.columns]].isna().sum().sum() == 0
    df.to_parquet(OUT / "nodes_static_norm.parquet", index=False)
    log(f"  wrote nodes_static_norm.parquet  {df.shape}")
    return scalers


# ======================================================================
# 3. edges
# ======================================================================
def process_edges():
    log("edges: loading")
    df = pd.read_parquet(GNN / "edges.parquet")
    # bearing -> sin/cos (keep degrees too: the STGNN wind gate needs raw bearing)
    rad = np.deg2rad(df["bearing_deg"].to_numpy())
    df["bearing_sin"] = np.sin(rad).astype("float32")
    df["bearing_cos"] = np.cos(rad).astype("float32")
    sc = StandardScaler()
    df["dist_km_z"] = sc.fit_transform(df[["dist_km"]].to_numpy()).astype("float32")
    df.to_parquet(OUT / "edges_norm.parquet", index=False)
    log(f"  wrote edges_norm.parquet  {df.shape}  (added bearing_sin/cos, dist_km_z; kept bearing_deg raw)")
    return {"edges_dist_standard": sc}


# ======================================================================
# 4. labels_station  (clean only — never normalise ground truth)
# ======================================================================
def process_labels():
    log("labels_station: loading")
    df = pd.read_parquet(GNN / "labels_station.parquet")
    # aqi_station (the target) is 0-null; leave everything raw.
    # Engine-2 label nulls (pm25_pm10_ratio_obs / source_class_obs) are KEPT so
    # they can be DROPPED at engine training time — never imputed.
    n_lab = int(df["source_class_obs"].notna().sum())
    log(f"  aqi_station nulls: {int(df['aqi_station'].isna().sum())} (target must be complete)")
    log(f"  usable Engine-2 label rows: {n_lab:,} / {len(df):,} ({100*n_lab/len(df):.1f}%)")
    df.to_parquet(OUT / "labels_station_clean.parquet", index=False)
    log(f"  wrote labels_station_clean.parquet  {df.shape}")


# ======================================================================
def main():
    dyn_scalers, dyn_feats = process_dynamic()
    stat_scalers = process_static()
    edge_scalers = process_edges()
    process_labels()

    all_scalers = {**{f"dyn_{k}": v for k, v in dyn_scalers.items()},
                   **stat_scalers, **edge_scalers}
    joblib.dump({
        "scalers": all_scalers,
        "groups": {
            "dyn_robust": DYN_ROBUST, "dyn_standard": DYN_STANDARD,
            "dyn_log1p": DYN_LOG1P, "dyn_asis": DYN_ASIS,
            "stat_log1p": STAT_LOG1P, "stat_standard": STAT_STANDARD,
        },
        "warmup_h": WARMUP_H, "horizon_h": HORIZON_H,
    }, OUT / "scalers.joblib")

    manifest = {
        "dynamic_grid_norm.parquet": {
            "RobustScaler": DYN_ROBUST, "StandardScaler": DYN_STANDARD,
            "log1p+StandardScaler": DYN_LOG1P, "left_as_is (bounded/binary)": DYN_ASIS,
            "kept_raw (ids/targets/leak)": DYN_TARGETS + DYN_KEEPRAW,
        },
        "nodes_static_norm.parquet": {
            "log1p+StandardScaler": STAT_LOG1P, "StandardScaler": STAT_STANDARD,
            "kept_raw (ids/categorical)": STAT_KEEPRAW,
        },
        "edges_norm.parquet": {
            "added": ["bearing_sin", "bearing_cos", "dist_km_z"],
            "kept_raw": ["src", "dst", "src_ward", "dst_ward", "bearing_deg", "dist_km"],
        },
        "labels_station_clean.parquet": {
            "not_normalised (ground truth)": ["aqi_station"],
            "label_nulls_kept_drop_at_train": ["pm25_pm10_ratio_obs", "source_class_obs"],
        },
        "rules": {
            "scalers_fit_on": "train split only (dynamic); all wards (static)",
            "targets": "never normalised, never imputed",
            "warmup_dropped_h": WARMUP_H, "horizon_tail_dropped_h": HORIZON_H,
        },
    }
    (OUT / "NORMALISATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    log(f"wrote scalers.joblib + NORMALISATION_MANIFEST.json")
    log(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
