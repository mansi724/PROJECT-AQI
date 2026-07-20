"""
apply_corrections.py
=================================================================
Applies the required DATASET corrections to model_ready.parquet and
regenerates all downstream features consistently.

What it does (in order):
  1. CPCB bias-correction of PM2.5 / PM10 (fixes inverted seasonality).
     - monthly multiplicative factors anchored on CPCB ground truth
       (Feb/Mar/Dec are the only months with usable CPCB PM), extended
       by meteorological season. Capped to a defensible band.
  2. Recomputes the CPCB National AQI from the corrected pollutants
     (grid level), incl. aqi_category + dominant_pollutant.
  3. Land-use spatial DOWNSCALING so wards inside one CAMS cell differ
     (mass-conserving within each grid cell). -> real ward-level aqi.
  4. Adds `season` + `is_stubble_season` flags.
  5. Regenerates every derived feature (lags/rolls/target/health/split)
     by REUSING build_features.py's own functions -> guaranteed consistent.
  6. Drops the sparse cpcb_* columns from the training file
     (they live in data/processed/cpcb_ground_truth.csv for validation).
  7. Drops warm-up rows with no forecast target.

Keeps raw copies: aqi_raw, pm2_5_raw, pm10_raw.
Backs up the original to model_ready_backup.parquet.
Run:  python apply_corrections.py
=================================================================
"""
import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import build_features as bf   # reuse the exact same feature formulas

BASE = Path(__file__).resolve().parent
SRC = BASE / "data" / "final" / "model_ready.parquet"
BACKUP = BASE / "data" / "final" / "model_ready_backup.parquet"
OUT = BASE / "data" / "final" / "model_ready.parquet"

# Base (non-engineered) columns we load; everything else is rebuilt.
BASE_COLS = [
    "ward_id", "ward_name", "time", "point_id", "ward_lat", "ward_lon",
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
    "ozone", "aerosol_optical_depth", "dust", "temperature_2m",
    "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "boundary_layer_height", "fire_count", "fire_frp_sum", "hour", "dayofweek",
    "month", "dist_to_grid_km", "nearest_station_id", "dist_to_station_km",
    "road_km_3km", "road_capacity_3km", "industry_count_5km",
    "vulnerable_sites_3km", "population_sum", "population_density_mean",
    "elevation_mean", "lu_builtup_fraction", "lu_tree_fraction",
    "lu_majority_class",
    "cpcb_co", "cpcb_pm10", "cpcb_so2", "cpcb_no2", "cpcb_pm25", "cpcb_o3",
]

# --- 1. Bias-correction factors (CPCB-anchored, season-extended, capped) -----
# Anchors from CPCB/model median ratios: Feb pm25 1.1/pm10 1.4,
# Mar pm25 0.6/pm10 0.4 (CAMS over-predicts pre-monsoon dust),
# Dec pm25 1.4/pm10 2.6 (CAMS under-predicts winter accumulation; pm10 capped).
PM25_FACTOR = {1: 1.30, 2: 1.10, 3: 0.65, 4: 0.65, 5: 0.65, 6: 0.75,
               7: 1.00, 8: 1.00, 9: 1.00, 10: 1.10, 11: 1.30, 12: 1.35}
PM10_FACTOR = {1: 1.50, 2: 1.40, 3: 0.45, 4: 0.45, 5: 0.45, 6: 0.60,
               7: 0.90, 8: 0.90, 9: 0.90, 10: 0.95, 11: 1.40, 12: 1.70}

# --- 2. CPCB AQI breakpoints as (concentration, sub-index) node pairs --------
# piecewise-linear & monotonic -> np.interp gives the exact CPCB sub-index.
BP = {
    "pm2_5": ([0, 30, 60, 90, 120, 250, 500],   [0, 50, 100, 200, 300, 400, 500]),
    "pm10":  ([0, 50, 100, 250, 350, 430, 600], [0, 50, 100, 200, 300, 400, 500]),
    "no2":   ([0, 40, 80, 180, 280, 400, 600],  [0, 50, 100, 200, 300, 400, 500]),
    "so2":   ([0, 40, 80, 380, 800, 1600, 2000],[0, 50, 100, 200, 300, 400, 500]),
    "co":    ([0, 1, 2, 10, 17, 34, 50],         [0, 50, 100, 200, 300, 400, 500]),  # mg/m3
    "o3":    ([0, 50, 100, 168, 208, 748, 1000], [0, 50, 100, 200, 300, 400, 500]),
}
COLMAP = {"pm2_5": "pm2_5", "pm10": "pm10", "no2": "nitrogen_dioxide",
          "so2": "sulphur_dioxide", "co": "carbon_monoxide", "o3": "ozone"}
POLL_LABEL = {"pm2_5": "pm2_5", "pm10": "pm10", "no2": "nitrogen_dioxide",
              "so2": "sulphur_dioxide", "co": "carbon_monoxide", "o3": "ozone"}


def log(m): print(f"[correct] {m}", flush=True)


def subindex(conc, key):
    xp, fp = BP[key]
    return np.interp(conc, xp, fp)  # clamps <min->0, >max->500


def recompute_grid_aqi(df):
    """CPCB National AQI from (corrected) pollutants, rolled per grid point.

    BUG FIX (2026-07-16): this used to sort by [point_id, time] and then roll 24
    *rows* per point_id group. A cell holds up to 52 wards at the SAME timestamp,
    so the window spanned less than one real hour and every ward got a different
    value based on its ward_id sort position (cell 21 smeared AQI 201->403 where
    one number was correct; distinct values per cell-hour == min(n_wards, 24) in
    19/19 cells). The rolling means MUST be computed on the unique (point_id,
    time) series — every pollutant is identical across wards in a cell anyway —
    and then broadcast back. Also ~15x faster.
    """
    log("recomputing CPCB AQI from corrected pollutants (on unique cell-hours) ...")
    srcs = [s for s in COLMAP.values() if s in df.columns]
    grid = (df[["point_id", "time"] + srcs]
            .drop_duplicates(["point_id", "time"])
            .sort_values(["point_id", "time"])
            .reset_index(drop=True))
    subs = {}
    g = grid.groupby("point_id", sort=False)
    for key, src in COLMAP.items():
        if src not in grid.columns:
            continue
        win, mp = (8, 6) if key in ("co", "o3") else (24, 16)
        roll = g[src].transform(lambda x: x.rolling(win, min_periods=mp).mean())
        if key == "co":
            roll = roll / 1000.0  # ug/m3 -> mg/m3
        subs[key] = subindex(roll.to_numpy(dtype="float64"), key)
    sub = pd.DataFrame(subs, index=grid.index)
    have_pm = sub[[c for c in ("pm2_5", "pm10") if c in sub]].notna().any(axis=1)
    enough = sub.notna().sum(axis=1) >= 3
    valid = have_pm & enough
    grid["aqi"] = np.round(np.where(valid, sub.max(axis=1), np.nan)).astype("float32")
    grid["dominant_pollutant"] = pd.Series(pd.NA, index=grid.index, dtype="object")
    if valid.any():
        grid.loc[valid, "dominant_pollutant"] = sub.loc[valid].idxmax(axis=1).map(POLL_LABEL)

    # broadcast the per-cell series back onto every ward in that cell
    df = df.drop(columns=[c for c in ("aqi", "dominant_pollutant") if c in df.columns])
    df = df.merge(grid[["point_id", "time", "aqi", "dominant_pollutant"]],
                  on=["point_id", "time"], how="left")
    return df


def downscale_to_wards(df):
    """Land-use residual downscaling, mass-conserving within each grid cell."""
    log("land-use downscaling (per-ward, mass-conserving in grid cell) ...")
    ward = (df.groupby("ward_id", sort=False)
              .agg(point_id=("point_id", "first"),
                   rc=("road_capacity_3km", "first"),
                   ind=("industry_count_5km", "first"),
                   bu=("lu_builtup_fraction", "first"),
                   pd_=("population_density_mean", "first")))
    # 0..1 percentile-rank of each land-use driver, averaged -> emission potential
    idx = (ward[["rc", "ind", "bu", "pd_"]].rank(pct=True)).mean(axis=1)
    ward["idx"] = idx
    # center within grid cell so the cell mean is preserved (mass conserving)
    ward["cell_mean"] = ward.groupby("point_id")["idx"].transform("mean")
    ward["factor"] = (1.0 + 0.30 * (ward["idx"] - ward["cell_mean"])).clip(0.80, 1.20)
    fmap = ward["factor"].to_dict()
    f = df["ward_id"].map(fmap).astype("float32").to_numpy()
    df["aqi"] = (df["aqi"] * f).round().astype("float32")
    log(f"  ward factor range {ward['factor'].min():.3f}..{ward['factor'].max():.3f}")
    return df


def add_category(df):
    bins = [-1, 50, 100, 200, 300, 400, 100000]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    df["aqi_category"] = pd.cut(df["aqi"], bins=bins, labels=labels).astype(object)
    return df


def add_season(df):
    m = df["month"]
    season = np.select(
        [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8, 9])],
        ["winter", "pre_monsoon", "monsoon"], default="post_monsoon")
    df["season"] = season
    df["is_stubble_season"] = m.isin([10, 11]).astype("int8")  # paddy-residue burning
    return df


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    if not BACKUP.exists():
        log("backing up original -> model_ready_backup.parquet")
        shutil.copy2(SRC, BACKUP)

    log(f"loading base columns from {SRC.name} ...")
    df = pd.read_parquet(SRC, columns=BASE_COLS)
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in ("ward_name", "nearest_station_id", "lu_majority_class"):
        if c in df:
            df[c] = df[c].astype("category")
    df["time"] = pd.to_datetime(df["time"])
    log(f"rows={len(df):,}  wards={df['ward_id'].nunique()}  cols={df.shape[1]}")

    # keep raw copies for transparency
    df["pm2_5_raw"] = df["pm2_5"]
    df["pm10_raw"] = df["pm10"]

    # 1. bias-correct PM
    log("applying monthly PM bias-correction ...")
    df["pm2_5"] = (df["pm2_5"] * df["month"].map(PM25_FACTOR)).astype("float32")
    df["pm10"] = (df["pm10"] * df["month"].map(PM10_FACTOR)).astype("float32")

    # 2. recompute grid AQI, 3. downscale, category
    df = recompute_grid_aqi(df)
    df["aqi_raw"] = df["aqi"]          # grid-level corrected AQI, before downscale
    df = downscale_to_wards(df)
    df = add_category(df)

    # 4. season flags
    df = add_season(df)

    # 5. regenerate all engineered features with build_features' own code
    df = df.sort_values(["ward_id", "time"]).reset_index(drop=True)
    df = bf.add_time_features(df)
    df = bf.add_lags_rolls(df)
    df = bf.add_meteo_features(df)
    df = bf.add_source_features(df)
    df = bf.add_encodings(df)
    df = bf.add_action_features(df)
    df = bf.add_target_and_split(df, 24)
    gc.collect()

    # 6. drop sparse cpcb_* (validation-only, kept in cpcb_ground_truth.csv)
    cpcb = [c for c in df.columns if c.startswith("cpcb_")]
    df = df.drop(columns=cpcb)
    log(f"dropped {len(cpcb)} cpcb_* columns from training file")

    # 7. drop warm-up rows with no forecast target
    before = len(df)
    df = df[df["target_aqi_t24"].notna()].reset_index(drop=True)
    log(f"dropped {before - len(df):,} rows without a t+24h target")

    log(f"final shape: {df.shape[0]:,} x {df.shape[1]}")
    tmp = OUT.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(OUT)
    log(f"SAVED -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
