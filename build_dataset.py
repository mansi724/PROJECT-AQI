"""
Build the final merged dataset for the Delhi ward-level AQI project.

Run AFTER download.py:
    python build_dataset.py

Outputs (in data/processed/):
  final_dataset.csv    hourly rows: grid point x time, with pollution, weather,
                       fire signals and static GIS features  (ML-ready)
  ward_features.csv    one row per ward: static features + assigned grid point
  ward_grid_map.csv    ward_id -> grid point_id mapping

Never touches data/realtime/.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

OUT_DIR = C.PROCESSED_DIR
FINAL_CSV = OUT_DIR / "final_dataset.csv"
WARD_CSV = OUT_DIR / "ward_features.csv"
MAP_CSV = OUT_DIR / "ward_grid_map.csv"


def log(msg):
    print(f"[build] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def read_openmeteo_csv(path: Path) -> pd.DataFrame:
    """Open-Meteo CSVs have 2-3 metadata lines before the real header."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("time")), None)
    if hdr is None:
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines[hdr:])))
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df



def load_grid_series(folder: Path, prefix: str) -> pd.DataFrame:
    """Concatenate per-point per-year Open-Meteo CSVs into one long frame."""
    frames = []
    for path in sorted(folder.glob(f"{prefix}_p*_*.csv")):
        pid = int(path.stem.split("_")[1][1:])
        df = read_openmeteo_csv(path)
        if df.empty:
            continue
        df["point_id"] = pid
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # per point+time, keep last (re-downloads overwrite dupes)
    return out.drop_duplicates(subset=["point_id", "time"], keep="last")


# ---------------------------------------------------------------------------
# 1. Hourly pollution (CAMS) + weather (ERA5)
# ---------------------------------------------------------------------------
def build_hourly():
    aq = load_grid_series(C.DIRS["aqi_cams"], "cams")
    wx = load_grid_series(C.DIRS["weather"], "era5")
    if aq.empty:
        sys.exit("[build] No CAMS AQI CSVs found. Run: python download.py aqi")
    if wx.empty:
        sys.exit("[build] No weather CSVs found. Run: python download.py weather")
    log(f"pollution rows: {len(aq):,} | weather rows: {len(wx):,}")
    df = pd.merge(aq, wx, on=["point_id", "time"], how="inner")
    log(f"merged hourly rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# 1.5 CPCB Ground Truth Observations (Hourly API)
# ---------------------------------------------------------------------------
def build_cpcb_ground_truth() -> pd.DataFrame:
    out_dir = C.DIRS["aqi_openaq"]
    stations_file = out_dir / "stations.json"
    if not stations_file.exists():
        log("cpcb ground truth: stations.json missing - skipping")
        return pd.DataFrame()
    
    frames = []
    for path in out_dir.glob("station_*_*.json"):
        parts = path.stem.split("_")
        st_id = int(parts[1])
        param = parts[2]
        
        try:
            data = json.loads(path.read_text(encoding="utf-8")).get("results", [])
        except Exception:
            continue
        if not data:
            continue
            
        rows = []
        for row in data:
            if "period" in row:
                t = row["period"]["datetimeFrom"]["utc"]
                v = row.get("average", row.get("value"))
            else:
                t = row["date"]["utc"]
                v = row.get("value")
            rows.append({"station_id": st_id, "time": t, param: v})
            
        if rows:
            df = pd.DataFrame(rows)
            # Keep UTC — the CAMS/weather grid is downloaded on-the-hour in UTC.
            # (The old code shifted to IST, so timestamps never matched the grid.)
            t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)

            if getattr(C, "TEMPORAL_RESOLUTION", "hourly") == "daily":
                df["time"] = t.dt.floor("D")
            else:
                # CPCB reports sub-hourly (:15/:30/:45) — snap onto the hourly grid
                df["time"] = t.dt.floor("h")

            # Drop stale / out-of-window records (source contains data back to 2016)
            lo = pd.Timestamp(C.START_DATE)
            hi = pd.Timestamp(C.END_DATE) + pd.Timedelta(days=1)
            df = df[(df["time"] >= lo) & (df["time"] <= hi)]
            df = df.groupby(["station_id", "time"]).mean(numeric_only=True).reset_index()
            frames.append(df)
            
    if not frames:
        return pd.DataFrame()
        
    all_cpcb = pd.concat(frames, ignore_index=True)
    merged = all_cpcb.groupby(["station_id", "time"]).mean(numeric_only=True).reset_index()
    rename_cols = {c: f"cpcb_{c}" for c in merged.columns if c not in ["station_id", "time"]}
    merged = merged.rename(columns=rename_cols)
    out_path = C.PROCESSED_DIR / "cpcb_ground_truth.csv"
    merged.to_csv(out_path, index=False)
    log(f"cpcb ground truth: merged to {out_path.name} with {len(merged)} records")
    return merged


# ---------------------------------------------------------------------------
# 2. Fire features (daily, wide bbox) -> joined on date
# ---------------------------------------------------------------------------
def build_fire_daily() -> pd.DataFrame:
    files = list(C.DIRS["fire"].glob("*.csv"))
    if not files:
        log("fire: no CSVs found — fire features will be 0 (run download.py fire)")
        return pd.DataFrame(columns=["date", "fire_count", "fire_frp_sum"])
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "latitude" not in cols or "acq_date" not in cols:
            continue
        b = C.FIRE_BBOX
        df = df[(df[cols["latitude"]].between(b["min_lat"], b["max_lat"])) &
                (df[cols["longitude"]].between(b["min_lon"], b["max_lon"]))]
        keep = pd.DataFrame({
            "date": pd.to_datetime(df[cols["acq_date"]]).dt.date,
            "frp": pd.to_numeric(df[cols.get("frp", cols["latitude"])], errors="coerce")
                   if "frp" in cols else 0.0,
        })
        frames.append(keep)
    if not frames:
        return pd.DataFrame(columns=["date", "fire_count", "fire_frp_sum"])
    allf = pd.concat(frames, ignore_index=True).drop_duplicates()
    daily = allf.groupby("date").agg(fire_count=("date", "size"),
                                     fire_frp_sum=("frp", "sum")).reset_index()
    log(f"fire: {daily['fire_count'].sum():,.0f} detections across {len(daily)} days")
    return daily


# ---------------------------------------------------------------------------
# 3. Static GIS features per grid point (roads, industry, vulnerability)
# ---------------------------------------------------------------------------
def _osm_points(path: Path):
    """Extract representative (lat, lon) points + geometry from Overpass JSON."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = []
    for el in data.get("elements", []):
        if el["type"] == "node":
            pts.append((el["lat"], el["lon"], None))
        elif "center" in el:
            pts.append((el["center"]["lat"], el["center"]["lon"], None))
        elif "geometry" in el:
            g = el["geometry"]
            pts.append((g[0]["lat"], g[0]["lon"], g))
    return pts


def _road_length_km(geom):
    if not geom:
        return 0.0
    total = 0.0
    for a, b in zip(geom[:-1], geom[1:]):
        total += haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    return total


def build_static_features() -> pd.DataFrame:
    roads_raw = C.DIRS["roads"] / "osm_major_roads.json"
    inds_raw = C.DIRS["industries"] / "osm_industries.json"
    vuln_raw = C.DIRS["vulnerability"] / "osm_vulnerability.json"

    # capacity weights by road class (bigger road = more traffic/emissions)
    ROAD_W = {"motorway": 5, "trunk": 4, "primary": 3, "secondary": 2, "tertiary": 1,
              "motorway_link": 2.5, "trunk_link": 2, "primary_link": 1.5,
              "secondary_link": 1, "tertiary_link": 0.5}
    DEF_LANES = {"motorway": 3, "trunk": 2, "primary": 2, "secondary": 2, "tertiary": 1}

    road_segs = []          # (lat, lon, seg_km, capacity) per way midpoint
    if roads_raw.exists():
        data = json.loads(roads_raw.read_text(encoding="utf-8"))
        for el in data.get("elements", []):
            g = el.get("geometry")
            if not g:
                continue
            tg = el.get("tags", {})
            hw = tg.get("highway", "tertiary")
            w = ROAD_W.get(hw, 1)
            try:
                lanes = float(str(tg.get("lanes", "")).split(";")[0])
            except (ValueError, TypeError):
                lanes = DEF_LANES.get(hw.replace("_link", ""), 1)
            seg_km = _road_length_km(g)
            mid = g[len(g) // 2]
            # capacity = class weight x lane count x length  (traffic-volume proxy)
            road_segs.append((mid["lat"], mid["lon"], seg_km, w * lanes * seg_km))
    inds = _osm_points(inds_raw)
    vuln = _osm_points(vuln_raw)
    log(f"static: {len(road_segs)} road ways, {len(inds)} industry features, "
        f"{len(vuln)} vulnerability sites")

    stations_file = C.DIRS["aqi_openaq"] / "stations.json"
    st_list = []
    if stations_file.exists():
        st_list = json.loads(stations_file.read_text(encoding="utf-8")).get("results", [])

    rows = []
    for pid, lat, lon in C.grid_points():
        near = [(s, cap) for la, lo, s, cap in road_segs if haversine_km(lat, lon, la, lo) <= 3]
        road_km = sum(s for s, _ in near)
        road_capacity = sum(cap for _, cap in near)
        n_ind = sum(1 for la, lo, _ in inds if haversine_km(lat, lon, la, lo) <= 5)
        n_vuln = sum(1 for la, lo, _ in vuln if haversine_km(lat, lon, la, lo) <= 3)
        
        nearest_st_id = pd.NA
        min_st_dist = 9999
        if st_list:
            for st in st_list:
                clat, clon = st["coordinates"]["latitude"], st["coordinates"]["longitude"]
                dst = haversine_km(lat, lon, clat, clon)
                if dst < min_st_dist:
                    min_st_dist = dst
                    nearest_st_id = st["id"]

        rows.append({"point_id": pid, "lat": lat, "lon": lon,
                     "road_km_3km": round(road_km, 2),
                     "road_capacity_3km": round(road_capacity, 1),
                     "industry_count_5km": n_ind,
                     "vulnerable_sites_3km": n_vuln,
                     "nearest_station_id": nearest_st_id,
                     "dist_to_station_km": round(min_st_dist, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Wards -> nearest grid point + ward static features
# ---------------------------------------------------------------------------
def build_wards(static: pd.DataFrame):
    ward_file = C.DIRS["wards"] / "delhi_wards.geojson"
    if not ward_file.exists():
        log("wards: delhi_wards.geojson missing — skipping ward outputs (see README)")
        return
    gj = json.loads(ward_file.read_text(encoding="utf-8"))
    feats = gj.get("features", [])
    if not feats:
        log("wards: geojson has no features — skipping")
        return
        
    stations_file = C.DIRS["aqi_openaq"] / "stations.json"
    st_list = []
    if stations_file.exists():
        st_list = json.loads(stations_file.read_text(encoding="utf-8")).get("results", [])

    def centroid(geom):
        coords = []

        def collect(c):
            if isinstance(c[0], (int, float)):
                coords.append(c)
            else:
                for x in c:
                    collect(x)
        collect(geom["coordinates"])
        arr = np.array(coords, dtype=float)
        return float(arr[:, 1].mean()), float(arr[:, 0].mean())  # lat, lon

    rows = []
    for i, f in enumerate(feats):
        props = f.get("properties", {})
        name = next((props[k] for k in props
                     if str(k).lower() in ("ward_name", "wardname", "name", "ward")), f"ward_{i}")
        wid = next((props[k] for k in props
                    if str(k).lower() in ("ward_no", "wardno", "ward_id", "id", "wardcode")), i)
        try:
            lat, lon = centroid(f["geometry"])
        except Exception:
            continue
            
        nearest_st_id = 0
        min_st_dist = 9999
        if st_list:
            for st in st_list:
                clat, clon = st["coordinates"]["latitude"], st["coordinates"]["longitude"]
                dst = haversine_km(lat, lon, clat, clon)
                if dst < min_st_dist:
                    min_st_dist = dst
                    nearest_st_id = st["id"]
        d = static.assign(dist=[haversine_km(lat, lon, r.lat, r.lon)
                                for r in static.itertuples()])
        nearest = d.loc[d["dist"].idxmin()]
        rows.append({"ward_id": wid, "ward_name": name,
                     "ward_lat": round(lat, 5), "ward_lon": round(lon, 5),
                     "point_id": int(nearest["point_id"]),
                     "dist_to_grid_km": round(float(nearest["dist"]), 2),
                     "nearest_station_id": nearest_st_id,
                     "dist_to_station_km": round(min_st_dist, 2),
                     "road_km_3km": nearest["road_km_3km"],
                     "road_capacity_3km": nearest["road_capacity_3km"],
                     "industry_count_5km": nearest["industry_count_5km"],
                     "vulnerable_sites_3km": nearest["vulnerable_sites_3km"]})
    wards = pd.DataFrame(rows)
    wards.to_csv(WARD_CSV, index=False)
    wards[["ward_id", "ward_name", "point_id"]].to_csv(MAP_CSV, index=False)
    log(f"wards: {len(wards)} wards mapped -> {WARD_CSV.name}, {MAP_CSV.name}")


CPCB_BREAKPOINTS = {
    # PM2.5, 24h avg, µg/m³
    "pm2_5":  [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)],
    # PM10, 24h avg, µg/m³
    "pm10":   [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)],
    # NO2, 24h avg, µg/m³
    "no2":    [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,600,401,500)],
    # SO2, 24h avg, µg/m³
    "so2":    [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2000,401,500)],
    # CO, 8h avg, mg/m³  (divide µg/m³ value by 1000 first)
    "co":     [(0,1.0,0,50),(1.1,2.0,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)],
    # O3, 8h avg, µg/m³
    "o3":     [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)],
}

def _subindex(c, bp):
    if pd.isna(c): return np.nan
    for lo, hi, ilo, ihi in bp:
        if lo <= c <= hi:
            return round((ihi-ilo)/(hi-lo)*(c-lo)+ilo)
    return 500 if c > bp[-1][1] else np.nan

def add_cpcb_aqi(df, colmap):
    df = df.sort_values(['point_id','time'])
    roll = {}
    for key, src in colmap.items():
        if src not in df.columns:
            continue
        win = 8 if key in ('co','o3') else 24
        mp  = 6 if win == 8 else 16
        s = df.groupby('point_id')[src].transform(lambda x: x.rolling(win, min_periods=mp).mean())
        if key == 'co': s = s / 1000.0          # µg/m³ -> mg/m³
        roll[key] = s.map(lambda c: _subindex(c, CPCB_BREAKPOINTS[key]))
    sub = pd.DataFrame(roll, index=df.index)
    have_pm = sub[['pm2_5','pm10']].notna().any(axis=1) if ('pm2_5' in sub.columns or 'pm10' in sub.columns) else False
    enough  = sub.notna().sum(axis=1) >= 3
    df['aqi'] = np.where(have_pm & enough, sub.max(axis=1), np.nan)
    df['dominant_pollutant'] = np.where(df['aqi'].notna(), sub.idxmax(axis=1), None)
    bins=[-1,50,100,200,300,400,10000]
    labels=['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
    df['aqi_category']=pd.cut(df['aqi'],bins=bins,labels=labels)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    assert not str(OUT_DIR.resolve()).startswith(str(C.REALTIME_DIR.resolve()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hourly = build_hourly()
    fire = build_fire_daily()
    static = build_static_features()
    cpcb = build_cpcb_ground_truth()

    hourly["date"] = hourly["time"].dt.date
    if not fire.empty:
        hourly = hourly.merge(fire, on="date", how="left")
        hourly[["fire_count", "fire_frp_sum"]] = hourly[["fire_count", "fire_frp_sum"]].fillna(0)
    else:
        hourly["fire_count"] = 0
        hourly["fire_frp_sum"] = 0.0
    hourly = hourly.drop(columns=["date"]).merge(static, on="point_id", how="left")
    
    if not cpcb.empty:
        # Merge cpcb data using nearest_station_id and time
        hourly = hourly.merge(cpcb, left_on=["nearest_station_id", "time"], right_on=["station_id", "time"], how="left")
        if "station_id" in hourly.columns:
            hourly = hourly.drop(columns=["station_id"])

    # time features for modelling
    hourly["hour"] = hourly["time"].dt.hour
    hourly["dayofweek"] = hourly["time"].dt.dayofweek
    hourly["month"] = hourly["time"].dt.month

    hourly = hourly.sort_values(["point_id", "time"])

    # Interpolate boundary_layer_height (Fix 6)
    if "boundary_layer_height" in hourly.columns:
        hourly["boundary_layer_height"] = hourly.groupby("point_id")["boundary_layer_height"].transform(
            lambda x: x.interpolate(limit_direction="both")
        )

    # Compute CPCB AQI (Fix 1)
    colmap = {
        'pm2_5': 'pm2_5',
        'pm10': 'pm10',
        'no2': 'nitrogen_dioxide',
        'so2': 'sulphur_dioxide',
        'co': 'carbon_monoxide',
        'o3': 'ozone',
    }
    hourly = add_cpcb_aqi(hourly, colmap)

    hourly.to_csv(FINAL_CSV, index=False)
    log(f"FINAL: {FINAL_CSV}  ({len(hourly):,} rows x {len(hourly.columns)} cols, "
        f"{FINAL_CSV.stat().st_size/1e6:.1f} MB)")

    build_wards(static)
    log("Done.")


if __name__ == "__main__":
    main()
