"""
add_training_features.py
=================================================================
Adds the remaining training-helpful features to model_ready.parquet
(run AFTER apply_corrections.py). All additive — no existing column
is changed. Backup already exists as model_ready_backup.parquet.

Adds:
  1. Multi-horizon targets: target_aqi_t48, target_aqi_t72 (PS wants 24-72h)
  2. Festival/holiday flags: is_diwali_window, is_holiday
  3. Ward identity encoding: point_id_code, ward_hist_aqi (TRAIN-only mean, leak-free)
  4. Directional (upwind) source features from OSM locations:
       industry_upwind, road_upwind  (sources in the direction the wind blows FROM)
  5. EDGAR emission inventory: emis_* (merged from ward_emissions.csv)
=================================================================
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FINAL = BASE / "data" / "final" / "model_ready.parquet"
WARD_FEATURES = BASE / "data" / "processed" / "ward_features.csv"
EMIS = BASE / "data" / "processed" / "ward_emissions.csv"
IND_JSON = BASE / "data" / "raw" / "gis" / "industries" / "osm_industries.json"
ROAD_JSON = BASE / "data" / "raw" / "gis" / "roads" / "osm_major_roads.json"

# Diwali firecracker + post-Diwali smog windows (main day +/- 2 days).
DIWALI_WINDOWS = [  # (year, month, day_lo, day_hi)
    (2023, 11, 10, 14),
    (2024, 10, 30, 31), (2024, 11, 1, 3),
    (2025, 10, 18, 22),
]
# Fixed-date national holidays (reduced traffic) + we add Diwali below.
FIXED_HOLIDAYS = {(1, 26), (8, 15), (10, 2)}  # Republic, Independence, Gandhi Jayanti
ROAD_WEIGHT = {"motorway": 3, "trunk": 3, "primary": 2, "secondary": 1}


def log(m): print(f"[train-feat] {m}", flush=True)


def bearing(lat1, lon1, lat2, lon2):
    """Compass bearing (deg, 0=N) from point1 to point2."""
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(np.radians(lat2))
    x = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
         np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_sources():
    """Return (industry_pts[(lat,lon,w)], road_pts[(lat,lon,w)])."""
    ind = json.load(open(IND_JSON, encoding="utf-8"))["elements"]
    ipts = [(e["lat"], e["lon"], 1.0) for e in ind if "lat" in e and "lon" in e]
    roads = json.load(open(ROAD_JSON, encoding="utf-8"))["elements"]
    rpts = []
    for w in roads:
        g = w.get("geometry")
        if not g:
            continue
        wt = ROAD_WEIGHT.get(w.get("tags", {}).get("highway"), 0.5)
        mid = g[len(g) // 2]  # segment midpoint
        rpts.append((mid["lat"], mid["lon"], wt))
    log(f"OSM sources: {len(ipts)} industry nodes, {len(rpts)} road segments")
    return np.array(ipts), np.array(rpts)


def sector_vectors(wards, pts, radius_km):
    """For each ward -> 8-vector of source weight per 45deg compass sector
    (sector = bearing from ward to source, i.e. where the source lies)."""
    plat, plon, pw = pts[:, 0], pts[:, 1], pts[:, 2]
    out = {}
    for _, r in wards.iterrows():
        d = haversine_km(r.ward_lat, r.ward_lon, plat, plon)
        m = d <= radius_km
        if not m.any():
            out[r.ward_id] = np.zeros(8); continue
        b = bearing(r.ward_lat, r.ward_lon, plat[m], plon[m])
        sec = (b // 45).astype(int) % 8
        vec = np.zeros(8)
        np.add.at(vec, sec, pw[m])
        out[r.ward_id] = vec
    return out


def main():
    log(f"loading {FINAL.name} ...")
    df = pd.read_parquet(FINAL)
    log(f"rows={len(df):,} cols={df.shape[1]}")
    df = df.sort_values(["ward_id", "time"]).reset_index(drop=True)
    t = pd.to_datetime(df["time"])
    year, day = t.dt.year.to_numpy(), t.dt.day.to_numpy()
    month = df["month"].to_numpy()

    # 1. multi-horizon targets
    log("multi-horizon targets t48/t72 ...")
    g = df.groupby("ward_id", sort=False)["aqi"]
    df["target_aqi_t48"] = g.shift(-48).astype("float32")
    df["target_aqi_t72"] = g.shift(-72).astype("float32")

    # 2. festival / holiday flags
    log("festival + holiday flags ...")
    diwali = np.zeros(len(df), dtype="int8")
    for (yr, mo, d_lo, d_hi) in DIWALI_WINDOWS:
        diwali |= ((year == yr) & (month == mo) & (day >= d_lo) & (day <= d_hi))
    df["is_diwali_window"] = diwali
    hol = diwali.copy()
    for (mo, d) in FIXED_HOLIDAYS:
        hol |= ((month == mo) & (day == d))
    df["is_holiday"] = hol.astype("int8")

    # 3. ward identity encoding
    log("ward encoding (point_id_code, train-only ward_hist_aqi) ...")
    df["point_id_code"] = df["point_id"].astype("category").cat.codes.astype("int16")
    train_mean = (df[df["split"] == "train"].groupby("ward_id")["aqi"].mean())
    df["ward_hist_aqi"] = df["ward_id"].map(train_mean).astype("float32")

    # 4. directional upwind source features
    log("directional upwind source features (OSM) ...")
    wards = pd.read_csv(WARD_FEATURES)[["ward_id", "ward_lat", "ward_lon"]]
    ipts, rpts = load_sources()
    ind_vec = sector_vectors(wards, ipts, radius_km=8.0)
    road_vec = sector_vectors(wards, rpts, radius_km=4.0)
    # wind_direction_10m = direction wind blows FROM -> its sector is the upwind sector
    wsec = ((df["wind_direction_10m"].to_numpy() % 360) // 45).astype(int) % 8
    wid = df["ward_id"].to_numpy()
    imat = np.array([ind_vec.get(w, np.zeros(8)) for w in wid])
    rmat = np.array([road_vec.get(w, np.zeros(8)) for w in wid])
    rows = np.arange(len(df))
    df["industry_upwind"] = imat[rows, wsec].astype("float32")
    df["road_upwind"] = rmat[rows, wsec].astype("float32")

    # 5. EDGAR emissions
    if EMIS.exists():
        emis = pd.read_csv(EMIS)
        for c in emis.select_dtypes("float64").columns:
            emis[c] = emis[c].astype("float32")
        df = df.merge(emis, on="ward_id", how="left")
        log(f"merged EDGAR emissions: +{emis.shape[1]-1} cols")
    else:
        log("no ward_emissions.csv — skipping EDGAR merge")

    log(f"final shape: {df.shape[0]:,} x {df.shape[1]}")
    tmp = FINAL.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(FINAL)
    log(f"SAVED -> {FINAL} ({FINAL.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
