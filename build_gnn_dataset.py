"""
build_gnn_dataset.py
=================================================================
Builds the DYNAMIC half of the GNN dataset + the real CPCB label table.

Three problems are fixed here at once:

1. THE ROLLING BUG.  apply_corrections.py sorted by [point_id, time] and then
   rolled 24 *rows* per point_id group. Because a cell holds up to 52 wards at
   the SAME timestamp, the "24-hour average" spanned less than one real hour,
   and each ward got a different value based on its ward_id sort position
   (cell 21 smeared AQI 201->403 where one number was correct).
   Here the AQI is computed on the deduplicated 19-cell series, where
   rolling(24) is genuinely 24 hours.

2. REPETITION / SIZE.  Every dynamic column is constant within (point_id,time)
   — verified, all 19 columns. So the 7,601,856-row flat table holds only
   499,776 unique dynamic records; the rest is a 15x broadcast. We store the
   grid series once and let the loader broadcast at training time.

3. CIRCULAR DOWNSCALING.  The old per-ward factor was a deterministic function
   of four features the model also received, so the model could invert it.
   It is NOT applied here. Per-ward variation must come from real ward static
   features (build_ward_static.py) + the graph + real CPCB labels.

Outputs (data/gnn/):
  dynamic_grid.parquet   19 cells x 26,304 h — pollutants, weather, AQI, lags
  labels_station.parquet real CPCB station AQI, mapped to its ward

Run:  python build_gnn_dataset.py            # dynamic grid
      python build_gnn_dataset.py --labels   # labels only (after CPCB pull)
=================================================================
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

BASE = Path(__file__).resolve().parent
GNN = BASE / "data" / "gnn"
BACKUP = BASE / "data" / "final" / "model_ready_backup.parquet"

DYNAMIC = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
           "ozone", "aerosol_optical_depth", "dust", "temperature_2m",
           "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
           "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
           "boundary_layer_height", "fire_count", "fire_frp_sum"]

# CPCB-anchored monthly bias factors (unchanged from apply_corrections.py)
PM25_FACTOR = {1: 1.30, 2: 1.10, 3: 0.65, 4: 0.65, 5: 0.65, 6: 0.75,
               7: 1.00, 8: 1.00, 9: 1.00, 10: 1.10, 11: 1.30, 12: 1.35}
PM10_FACTOR = {1: 1.50, 2: 1.40, 3: 0.45, 4: 0.45, 5: 0.45, 6: 0.60,
               7: 0.90, 8: 0.90, 9: 0.90, 10: 0.95, 11: 1.40, 12: 1.70}

BP = {
    "pm2_5": ([0, 30, 60, 90, 120, 250, 500],    [0, 50, 100, 200, 300, 400, 500]),
    "pm10":  ([0, 50, 100, 250, 350, 430, 600],  [0, 50, 100, 200, 300, 400, 500]),
    "no2":   ([0, 40, 80, 180, 280, 400, 600],   [0, 50, 100, 200, 300, 400, 500]),
    "so2":   ([0, 40, 80, 380, 800, 1600, 2000], [0, 50, 100, 200, 300, 400, 500]),
    "co":    ([0, 1, 2, 10, 17, 34, 50],         [0, 50, 100, 200, 300, 400, 500]),
    "o3":    ([0, 50, 100, 168, 208, 748, 1000], [0, 50, 100, 200, 300, 400, 500]),
}
COLMAP = {"pm2_5": "pm2_5", "pm10": "pm10", "no2": "nitrogen_dioxide",
          "so2": "sulphur_dioxide", "co": "carbon_monoxide", "o3": "ozone"}
POLL_LABEL = {"pm2_5": "pm2_5", "pm10": "pm10", "no2": "nitrogen_dioxide",
              "so2": "sulphur_dioxide", "co": "carbon_monoxide", "o3": "ozone"}


def log(m): print(f"[gnn] {m}", flush=True)


def subindex(conc, key):
    xp, fp = BP[key]
    return np.interp(conc, xp, fp)


def cpcb_aqi(df, group_col=None):
    """CPCB National AQI. Rolls 24h (8h for CO/O3) over a TIME-ORDERED series.

    `df` must already be one row per (location, hour) — that is the whole point.
    If group_col is None the frame is treated as a single station series.
    """
    subs = {}
    for key, src in COLMAP.items():
        if src not in df.columns:
            continue
        win, mp = (8, 6) if key in ("co", "o3") else (24, 16)
        s = df[src]
        if group_col:
            roll = df.groupby(group_col, sort=False)[src].transform(
                lambda x: x.rolling(win, min_periods=mp).mean())
        else:
            roll = s.rolling(win, min_periods=mp).mean()
        if key == "co":
            roll = roll / 1000.0  # ug/m3 -> mg/m3
        subs[key] = subindex(roll.to_numpy(dtype="float64"), key)
    sub = pd.DataFrame(subs, index=df.index)
    have_pm = sub[[c for c in ("pm2_5", "pm10") if c in sub]].notna().any(axis=1)
    enough = sub.notna().sum(axis=1) >= 3
    valid = have_pm & enough
    aqi = np.where(valid, sub.max(axis=1), np.nan)
    # idxmax on an all-NA row is deprecated -> only ask for rows we keep
    dom = pd.Series(pd.NA, index=df.index, dtype="object")
    if valid.any():
        dom.loc[valid] = sub.loc[valid].idxmax(axis=1).map(POLL_LABEL)
    return np.round(aqi).astype("float32"), dom


def add_time_features(df):
    t = df["time"].dt
    df["hour_sin"] = np.sin(2 * np.pi * t.hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * t.hour / 24).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * t.month / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * t.month / 12).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * t.dayofweek / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * t.dayofweek / 7).astype("float32")
    df["is_weekend"] = (t.dayofweek >= 5).astype("int8")
    df["is_rush_hour"] = t.hour.isin([8, 9, 10, 18, 19, 20]).astype("int8")
    df["is_winter"] = t.month.isin([11, 12, 1, 2]).astype("int8")
    df["is_stubble_season"] = t.month.isin([10, 11]).astype("int8")
    m, d = t.month, t.day
    df["is_diwali_window"] = (((m == 11) & (d <= 15)) | ((m == 10) & (d >= 20))).astype("int8")
    return df


def add_lags_rolls(df):
    """Lags/rolls per GRID CELL on a time-ordered series (correct by construction)."""
    g = df.groupby("point_id", sort=False)
    for col in ["aqi", "pm2_5", "pm10", "nitrogen_dioxide"]:
        s = g[col]
        for L in (1, 3, 6, 12, 24, 48):
            df[f"{col}_lag_{L}h"] = s.shift(L).astype("float32")
    for col in ["aqi", "pm2_5"]:
        s = g[col]
        df[f"{col}_roll_mean_6h"] = s.transform(lambda x: x.rolling(6, min_periods=3).mean()).astype("float32")
        df[f"{col}_roll_mean_24h"] = s.transform(lambda x: x.rolling(24, min_periods=12).mean()).astype("float32")
        df[f"{col}_roll_max_24h"] = s.transform(lambda x: x.rolling(24, min_periods=12).max()).astype("float32")
        df[f"{col}_roll_std_24h"] = s.transform(lambda x: x.rolling(24, min_periods=12).std()).astype("float32")
    df["aqi_diff_1h"] = (df["aqi"] - df["aqi_lag_1h"]).astype("float32")
    df["aqi_diff_24h"] = (df["aqi"] - df["aqi_lag_24h"]).astype("float32")
    return df


def add_meteo(df):
    wd = np.deg2rad(df["wind_direction_10m"])
    df["wind_dir_sin"] = np.sin(wd).astype("float32")
    df["wind_dir_cos"] = np.cos(wd).astype("float32")
    df["ventilation_index"] = (df["wind_speed_10m"] * df["boundary_layer_height"]).astype("float32")
    df["stagnation_index"] = (1.0 / (1.0 + df["wind_speed_10m"] *
                                     df["boundary_layer_height"] / 1000.0)).astype("float32")
    df["wind_from_nw"] = (((df["wind_direction_10m"] >= 270) &
                           (df["wind_direction_10m"] <= 340)).astype("int8"))
    return df


def add_attribution_features(df):
    """Physical source-signature features (Engine 2).

    These are the honest core of attribution. There is no source-apportionment
    ground truth anywhere, and 6 pollutants is not chemical speciation, so PMF /
    receptor modelling is off the table. What IS real and measurable:

      pm25_pm10_ratio  the classic combustion-vs-dust discriminator. In the CPCB
                       station data it flips exactly as Delhi physics predicts —
                       0.62 in December (biomass + vehicles under an inversion)
                       vs 0.36 in April (construction / road / wind-blown dust).
      fire_upwind      FIRMS fire radiative power gated on NW wind = the stubble
                       transport corridor into Delhi.
      dust_fraction    CAMS' own dust product as a share of PM10.

    The station-side twin of pm25_pm10_ratio is a MEASURED label, so the
    dust/combustion split is a supervised, checkable task — not a heuristic.
    """
    pm10 = df["pm10"].clip(lower=1.0)
    df["pm25_pm10_ratio"] = (df["pm2_5"] / pm10).clip(0, 1).astype("float32")
    df["dust_fraction"] = (df["dust"] / pm10).clip(0, 1).astype("float32")
    # stubble: burning upwind only matters when the wind actually comes from it
    df["fire_upwind"] = (df["fire_frp_sum"] * df["wind_from_nw"]).astype("float32")
    # secondary/industrial marker: SO2 is dominated by coal & industry
    df["so2_no2_ratio"] = (df["sulphur_dioxide"] /
                           df["nitrogen_dioxide"].clip(lower=1.0)).clip(0, 10).astype("float32")
    return df


def add_targets(df, horizons=(24, 48, 72)):
    g = df.groupby("point_id", sort=False)["aqi"]
    for h in horizons:
        df[f"target_aqi_t{h}"] = g.shift(-h).astype("float32")
    df["persistence_aqi_t24"] = df["aqi"].astype("float32")  # naive baseline
    return df


def add_split(df):
    """Chronological split — never random (adjacent hours are near-duplicates)."""
    t = df["time"]
    q70, q85 = t.quantile(0.70), t.quantile(0.85)
    df["split"] = np.select([t <= q70, t <= q85], ["train", "val"], default="test")
    return df


def build_dynamic():
    GNN.mkdir(parents=True, exist_ok=True)
    log(f"loading grid columns from {BACKUP.name} ...")
    df = pd.read_parquet(BACKUP, columns=["ward_id", "point_id", "time"] + DYNAMIC)
    df = df[df["ward_id"].notna()]          # drop the null-ward_id junk record
    n0 = len(df)

    # every DYNAMIC column is constant within (point_id, time) -> lossless
    df = (df.drop(columns="ward_id")
            .drop_duplicates(["point_id", "time"])
            .sort_values(["point_id", "time"])
            .reset_index(drop=True))
    log(f"deduped {n0:,} ward-hours -> {len(df):,} grid-hours "
        f"({n0/len(df):.1f}x smaller, zero information lost)")

    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month

    log("bias-correcting PM (CPCB-anchored monthly factors) ...")
    df["pm2_5_raw"] = df["pm2_5"]
    df["pm10_raw"] = df["pm10"]
    df["pm2_5"] = (df["pm2_5"] * df["month"].map(PM25_FACTOR)).astype("float32")
    df["pm10"] = (df["pm10"] * df["month"].map(PM10_FACTOR)).astype("float32")

    log("computing CPCB AQI on the 19-cell series (rolling(24) == 24 real hours) ...")
    df["aqi"], df["dominant_pollutant"] = cpcb_aqi(df, group_col="point_id")

    df = add_time_features(df)
    df = add_meteo(df)
    df = add_attribution_features(df)
    df = add_lags_rolls(df)
    df = add_targets(df)
    df = add_split(df)
    df = df.drop(columns=["month"])

    out = GNN / "dynamic_grid.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    mb = out.stat().st_size / 1e6
    log(f"SAVED {out.name}: {df.shape[0]:,} x {df.shape[1]}  ({mb:.0f} MB)")

    # --- verify the bug is gone -------------------------------------------
    h = df["time"].iloc[len(df) // 2]
    snap = df[df["time"] == h]
    log(f"verification @ {h}: {len(snap)} cells, {snap['aqi'].nunique()} distinct AQI "
        f"(one per cell — no ward smear)")
    log(f"split: {df['split'].value_counts().to_dict()}")
    return df


def build_labels():
    """Real CPCB station AQI -> nearest ward. This is the honest target."""
    import geopandas as gpd
    from shapely.geometry import Point

    gt = C.PROCESSED_DIR / "cpcb_ground_truth.csv"
    if not gt.exists():
        raise SystemExit("cpcb_ground_truth.csv missing — run refresh_ground_truth.py")
    df = pd.read_csv(gt, parse_dates=["time"])
    log(f"ground truth: {len(df):,} station-hours, {df['station_id'].nunique()} stations")
    if "lat" not in df.columns:
        raise SystemExit("cpcb_ground_truth.csv has no lat/lon — re-run refresh_ground_truth.py")

    ren = {"cpcb_pm25": "pm2_5", "cpcb_pm10": "pm10", "cpcb_no2": "nitrogen_dioxide",
           "cpcb_so2": "sulphur_dioxide", "cpcb_co": "carbon_monoxide", "cpcb_o3": "ozone"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df = df.sort_values(["station_id", "time"]).reset_index(drop=True)

    # AQI per station on its own time-ordered series
    log("computing CPCB AQI per station ...")
    df["aqi_station"], df["dominant_pollutant_station"] = cpcb_aqi(df, group_col="station_id")

    # --- MEASURED source signature (Engine 2's real label) -----------------
    # PM2.5/PM10 from actual instruments. Thresholds are the standard
    # combustion/dust split and match what this data shows: winter ~0.60-0.62,
    # pre-monsoon ~0.36. Because it is measured, a model that predicts it can
    # be SCORED — which is what separates this from heuristic attribution.
    ok = df["pm2_5"].gt(0) & df["pm10"].gt(0)
    ratio = (df["pm2_5"] / df["pm10"].clip(lower=1.0)).where(ok)
    df["pm25_pm10_ratio_obs"] = ratio.where(ratio.between(0.05, 1.0)).astype("float32")
    df["source_class_obs"] = pd.cut(
        df["pm25_pm10_ratio_obs"], bins=[0, 0.40, 0.60, 1.01],
        labels=["dust_dominated", "mixed", "combustion_dominated"]).astype(object)

    # map each station to the ward containing it (fallback: nearest centroid)
    nodes = pd.read_parquet(GNN / "nodes_static.parquet")
    wards = gpd.read_file(C.DIRS["wards"] / "delhi_wards.geojson")
    wards = wards[wards["Ward_No"].notna()].copy()
    wards["ward_id"] = wards["Ward_No"].astype(str).str.strip()
    wards = wards[["ward_id", "geometry"]].to_crs("EPSG:32643")

    st = df[["station_id", "lat", "lon"]].drop_duplicates("station_id")
    gst = gpd.GeoDataFrame(st, geometry=[Point(x, y) for x, y in zip(st["lon"], st["lat"])],
                           crs="EPSG:4326").to_crs("EPSG:32643")
    j = gpd.sjoin(gst, wards, how="left", predicate="within")[["station_id", "ward_id"]]
    inside = j["ward_id"].notna().sum()
    # stations outside Delhi's ward layer (NCR: Gurugram, Noida...) -> nearest ward
    miss = j[j["ward_id"].isna()]["station_id"]
    if len(miss):
        near = gpd.sjoin_nearest(gst[gst["station_id"].isin(miss)], wards,
                                 how="left")[["station_id", "ward_id"]]
        j = pd.concat([j[j["ward_id"].notna()], near], ignore_index=True)
    log(f"stations mapped: {inside} inside a ward, {len(miss)} snapped to nearest")

    df = df.merge(j.drop_duplicates("station_id"), on="station_id", how="left")
    df = df.merge(nodes[["ward_id", "node_idx", "point_id"]], on="ward_id", how="left")
    df = df[df["aqi_station"].notna()]

    # --- label-aware split -------------------------------------------------
    # OpenAQ only serves hourly rollups from ~2025-02 for these sensors, so the
    # real labels cover ~17 months, not the full 3-year grid window. Reusing the
    # full-range `split` puts almost no winter in train (5.9% winter, mean AQI
    # 143) but 53% in val (mean AQI 257) — training on clean air and validating
    # on severe smog. `split_lab` re-splits chronologically INSIDE the labelled
    # era so train carries the winter.
    t = df["time"]
    lo, hi = t.min(), t.max()
    span = hi - lo
    q70, q85 = lo + span * 0.70, lo + span * 0.85
    df["split_lab"] = np.select([t <= q70, t <= q85], ["train", "val"], default="test")

    out = GNN / "labels_station.parquet"
    keep = ["station_id", "ward_id", "node_idx", "point_id", "time", "aqi_station",
            "dominant_pollutant_station", "pm2_5", "pm10", "nitrogen_dioxide",
            "pm25_pm10_ratio_obs", "source_class_obs", "lat", "lon", "split_lab"]
    df[[c for c in keep if c in df.columns]].to_parquet(out, index=False)
    log(f"SAVED {out.name}: {len(df):,} labelled station-hours, "
        f"{df['ward_id'].nunique()} distinct wards labelled, "
        f"{df['time'].min()} -> {df['time'].max()}")
    r = df["pm25_pm10_ratio_obs"]
    log(f"measured source signature: {r.notna().sum():,} hours with a real PM2.5/PM10 ratio")
    log(f"  source_class_obs: {df['source_class_obs'].value_counts().to_dict()}")
    w = df["time"].dt.month.isin([11, 12, 1, 2])
    rep = df.assign(winter=w).groupby("split_lab").agg(
        n=("aqi_station", "size"), winter_share=("winter", "mean"),
        mean_aqi=("aqi_station", "mean"))
    log("split_lab (use THIS for supervised training on real labels):")
    for s, r in rep.iterrows():
        log(f"  {s:5s} n={int(r['n']):>7,}  winter={r['winter_share']:.0%}  mean_aqi={r['mean_aqi']:.0f}")
    return df


def main():
    if "--labels" in sys.argv:
        build_labels()
    else:
        build_dynamic()


if __name__ == "__main__":
    main()
