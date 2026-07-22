"""
Historical data download pipeline — Delhi ward-level AQI Forecasting,
Source Attribution & Action Recommendation project.

Usage:
    python -m src.data.download                    # download everything
    python -m src.data.download weather aqi fire   # only selected datasets

Rules enforced by this script:
  * Free sources only. Default sources need NO API key.
  * Skip any file that already exists.
  * NEVER write to data/realtime/.
  * Download only — merging happens in build_dataset.py.
"""

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import requests

from src import config as C

# ---------------------------------------------------------------------------
# Safety guard: forbid any write into data/realtime/
# ---------------------------------------------------------------------------
def _assert_not_realtime(path: Path) -> Path:
    path = Path(path).resolve()
    if C.REALTIME_DIR.resolve() in path.parents or path == C.REALTIME_DIR.resolve():
        raise PermissionError(f"BLOCKED: attempted write into protected folder {C.REALTIME_DIR}")
    return path


def ensure_dirs():
    for d in C.DIRS.values():
        _assert_not_realtime(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def log(msg: str):
    print(f"[download] {msg}", flush=True)


def skip_if_exists(path: Path) -> bool:
    path = _assert_not_realtime(path)
    if path.exists() and path.stat().st_size > 0:
        log(f"SKIP (exists): {path.relative_to(C.BASE_DIR)}")
        return True
    return False


def http_download(url: str, dest: Path, headers=None, params=None, quiet_404=False) -> bool:
    """Stream a URL to dest with retries. Returns True on success."""
    dest = _assert_not_realtime(dest)
    if skip_if_exists(dest):
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    delay = C.RETRY_BACKOFF
    for attempt in range(1, C.MAX_RETRIES + 1):
        try:
            with requests.get(url, headers=headers, params=params,
                              stream=True, timeout=C.REQUEST_TIMEOUT) as r:
                if r.status_code == 404 and quiet_404:
                    return False
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=C.CHUNK_SIZE):
                        f.write(chunk)
            tmp.rename(dest)
            log(f"OK: {dest.relative_to(C.BASE_DIR)}")
            return True
        except Exception as e:
            log(f"attempt {attempt}/{C.MAX_RETRIES} failed for {url}: {e}")
            tmp.unlink(missing_ok=True)
            if attempt < C.MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    return False


def month_range(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def year_spans():
    """Split [START_DATE, END_DATE] into per-calendar-year (start, end) spans."""
    for year in range(C.START_DATE.year, C.END_DATE.year + 1):
        s = max(C.START_DATE, C.START_DATE.replace(year=year, month=1, day=1)
                if year > C.START_DATE.year else C.START_DATE)
        from datetime import date as _d
        s = max(C.START_DATE, _d(year, 1, 1))
        e = min(C.END_DATE, _d(year, 12, 31))
        if s <= e:
            yield year, s, e


# ---------------------------------------------------------------------------
# 1a. AQI — CAMS reanalysis via Open-Meteo (KEYLESS, hourly, per grid point)
# ---------------------------------------------------------------------------
def download_aqi():
    out_dir = C.DIRS["aqi_cams"]
    n_ok = n_fail = 0
    for pid, lat, lon in C.grid_points():
        for year, s, e in year_spans():
            dest = out_dir / f"cams_p{pid:03d}_{year}.csv"
            if skip_if_exists(dest):
                continue
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": ",".join(C.OPENMETEO_AQ_VARS),
                "start_date": s.isoformat(), "end_date": e.isoformat(),
                "format": "csv", "timezone": "UTC",
            }
            url = C.OPENMETEO_AQ_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
            ok = http_download(url, dest)
            n_ok += ok
            n_fail += (not ok)
            time.sleep(C.OPENMETEO_SLEEP)
    log(f"aqi (CAMS): {n_ok} files ok, {n_fail} failed")
    _download_aqi_openaq()


# 1b. AQI — CPCB station ground truth via OpenAQ (optional, needs free key)
def _download_aqi_openaq():
    if not C.OPENAQ_API_KEY:
        log("aqi (OpenAQ stations): OPENAQ_API_KEY not set — skipped (optional)")
        return
    out_dir = C.DIRS["aqi_openaq"]
    headers = {"X-API-Key": C.OPENAQ_API_KEY}
    base = "https://api.openaq.org/v3"
    stations_file = out_dir / "stations.json"
    if not skip_if_exists(stations_file):
        r = requests.get(f"{base}/locations", headers=headers,
                         params={"bbox": ",".join(map(str, C.BBOX_WSEN)), "limit": 1000},
                         timeout=C.REQUEST_TIMEOUT)
        r.raise_for_status()
        _assert_not_realtime(stations_file).write_text(json.dumps(r.json(), indent=2))
        log(f"OK: {stations_file.relative_to(C.BASE_DIR)}")
    stations = json.loads(stations_file.read_text()).get("results", [])
    log(f"aqi (OpenAQ): {len(stations)} stations in bbox")
    
    progress_file = out_dir / "progress.json"
    progress = {
        "Total stations": len(stations),
        "Total pollutants": 0,
        "Total files expected": 0,
        "Files completed": 0,
        "Files remaining": 0,
        "Estimated time remaining": "Calculating...",
        "Number of API retries": 0,
        "Number of failed downloads": 0,
        "_start_time": time.time(),
        "_files_processed_this_run": 0
    }
    if progress_file.exists():
        try:
            old_p = json.loads(progress_file.read_text())
            progress["Files completed"] = old_p.get("Files completed", 0)
            progress["Number of API retries"] = old_p.get("Number of API retries", 0)
            progress["Number of failed downloads"] = old_p.get("Number of failed downloads", 0)
        except Exception:
            pass

    pollutants_found = set()
    total_expected = 0
    for st in stations:
        for sensor in st.get("sensors", []):
            if sensor.get("parameter", {}).get("name", "") in C.AQI_PARAMETERS:
                pollutants_found.add(sensor["parameter"]["name"])
                total_expected += (C.END_DATE.year - C.START_DATE.year + 1)
                
    progress["Total pollutants"] = len(pollutants_found)
    progress["Total files expected"] = total_expected
    progress["Files remaining"] = total_expected - progress["Files completed"]
    progress_file.write_text(json.dumps(progress, indent=4))

    for st_idx, st in enumerate(stations):
        for sensor in st.get("sensors", []):
            param = sensor.get("parameter", {}).get("name", "")
            if param not in C.AQI_PARAMETERS:
                continue
            for year in range(C.START_DATE.year, C.END_DATE.year + 1):
                # OpenAQ v3: hourly aggregates live at /sensors/{id}/hours, and the
                # date-range params are datetime_from/datetime_to (NOT date_from/date_to —
                # the wrong names were silently ignored, so only a default slice came back).
                endpoint = "hours" if getattr(C, "TEMPORAL_RESOLUTION", "hourly") == "hourly" else "days"
                dest = out_dir / f"station_{st['id']}_{param}_{year}_{endpoint}.json"
                if skip_if_exists(dest):
                    # update progress for already checked files if not already in stats to prevent infinite looping time,
                    # but since they were already there, we just skip. (Or we can assume they count against total normally).
                    continue
                try:
                    all_results = []
                    if endpoint == "hours":
                        from datetime import date as _d
                        s_yr = max(C.START_DATE, _d(year, 1, 1))
                        e_yr = min(C.END_DATE, _d(year, 12, 31))
                        for y, m in month_range(s_yr, e_yr):
                            nm, ny = (1, y + 1) if m == 12 else (m + 1, y)
                            d_start = f"{y}-{m:02d}-01T00:00:00Z"
                            d_end = f"{ny}-{nm:02d}-01T00:00:00Z"
                            page = 1
                            while True:
                                page_results = []
                                for attempt in range(1, 6):
                                    r = requests.get(f"{base}/sensors/{sensor['id']}/{endpoint}",
                                                     headers=headers,
                                                     params={"datetime_from": d_start,
                                                             "datetime_to": d_end,
                                                             "limit": 1000, "page": page},
                                                     timeout=C.REQUEST_TIMEOUT)
                                    if r.status_code in [404]:
                                        break
                                    if r.status_code == 429:
                                        log(f"Rate limited... sleeping 10s (attempt {attempt})")
                                        time.sleep(10)
                                        progress["Number of API retries"] += 1
                                        continue
                                    if not r.ok:
                                        # To debug any other issues without failing silently
                                        r.raise_for_status()
                                        
                                    page_results = r.json().get("results", [])
                                    all_results.extend(page_results)
                                    break
                                if not page_results or len(page_results) < 1000:
                                    break
                                page += 1
                                time.sleep(0.3)
                    else:
                        r = requests.get(f"{base}/sensors/{sensor['id']}/{endpoint}",
                                         headers=headers,
                                         params={"datetime_from": f"{year}-01-01T00:00:00Z",
                                                 "datetime_to": f"{year}-12-31T23:59:59Z", "limit": 1000},
                                         timeout=C.REQUEST_TIMEOUT)
                        if r.status_code in [404]:
                            all_results = []
                        else:
                            r.raise_for_status()
                            all_results = r.json().get("results", [])
                        time.sleep(0.3)
                        
                    _assert_not_realtime(dest).write_text(json.dumps({"results": all_results}, indent=2))
                    log(f"OK: {dest.relative_to(C.BASE_DIR)} ({len(all_results)} records)")
                    progress["Files completed"] += 1
                    progress["_files_processed_this_run"] += 1
                except Exception as e:
                    log(f"aqi (OpenAQ): sensor {sensor['id']} {year} failed: {e}")
                    progress["Number of failed downloads"] += 1
                    
                # Update progress tracker
                progress["Files remaining"] = progress["Total files expected"] - progress["Files completed"]
                elapsed = time.time() - progress["_start_time"]
                files_this_run = progress["_files_processed_this_run"]
                if files_this_run > 0:
                    avg_time = elapsed / files_this_run
                    rem_sec = avg_time * progress["Files remaining"]
                    progress["Estimated time remaining"] = f"{round(rem_sec / 60, 1)} minutes"
                    
                # log live update
                if files_this_run % 2 == 0:
                    log(f"PROGRESS UPDATE: {progress['Files completed']}/{progress['Total files expected']} completed. ETA: {progress['Estimated time remaining']}")
                
                # Write to disk cleanly
                out_prog = {k: v for k, v in progress.items() if not k.startswith("_")}
                progress_file.write_text(json.dumps(out_prog, indent=4))


# ---------------------------------------------------------------------------
# 2. Weather — ERA5 via Open-Meteo archive (KEYLESS, hourly, per grid point)
# ---------------------------------------------------------------------------
def _weather_vars():
    """Probe once whether optional vars (boundary_layer_height) are accepted."""
    full = C.OPENMETEO_WEATHER_VARS + C.OPENMETEO_WEATHER_OPTIONAL
    try:
        r = requests.get(C.OPENMETEO_WEATHER_URL, params={
            "latitude": 28.65, "longitude": 77.2,
            "hourly": ",".join(full),
            "start_date": "2024-01-01", "end_date": "2024-01-01",
        }, timeout=C.REQUEST_TIMEOUT)
        if r.status_code == 200:
            return full
    except Exception:
        pass
    log("weather: optional vars not supported by archive API — using core set")
    return C.OPENMETEO_WEATHER_VARS


def download_weather():
    out_dir = C.DIRS["weather"]
    variables = _weather_vars()
    n_ok = n_fail = 0
    for pid, lat, lon in C.grid_points():
        for year, s, e in year_spans():
            dest = out_dir / f"era5_p{pid:03d}_{year}.csv"
            if skip_if_exists(dest):
                continue
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": ",".join(variables),
                "start_date": s.isoformat(), "end_date": e.isoformat(),
                "format": "csv", "timezone": "UTC",
            }
            url = C.OPENMETEO_WEATHER_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
            ok = http_download(url, dest)
            n_ok += ok
            n_fail += (not ok)
            time.sleep(C.OPENMETEO_SLEEP)
    log(f"weather (ERA5/Open-Meteo): {n_ok} files ok, {n_fail} failed")


# 2b. Optional: ERA5 NetCDF grids via Copernicus CDS (needs free account)
def download_era5nc():
    try:
        import cdsapi
    except ImportError:
        log("era5nc: 'cdsapi' not installed — pip install cdsapi (optional)")
        return
    kwargs = {"url": C.CDSAPI_URL, "key": C.CDSAPI_KEY} if C.CDSAPI_KEY else {}
    try:
        client = cdsapi.Client(**kwargs)
    except Exception as e:
        log(f"era5nc: CDS credentials missing — {e} (optional)")
        return
    for year, month in month_range(C.START_DATE, C.END_DATE):
        dest = C.DIRS["weather"] / f"era5_{year}_{month:02d}.nc"
        if skip_if_exists(dest):
            continue
        try:
            client.retrieve("reanalysis-era5-single-levels", {
                "product_type": "reanalysis", "variable": C.ERA5_VARIABLES,
                "year": str(year), "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": list(C.BBOX_NWSE), "format": "netcdf",
            }, str(_assert_not_realtime(dest)))
            log(f"OK: {dest.relative_to(C.BASE_DIR)}")
        except Exception as e:
            log(f"era5nc: {year}-{month:02d} failed: {e}")


# ---------------------------------------------------------------------------
# 3. NASA FIRMS fire — WIDE bbox (Punjab/Haryana stubble burning included)
# ---------------------------------------------------------------------------
def download_fire():
    out_dir = C.DIRS["fire"]
    b = C.FIRE_BBOX
    if C.FIRMS_MAP_KEY:
        area = f"{b['min_lon']},{b['min_lat']},{b['max_lon']},{b['max_lat']}"
        cur = C.START_DATE
        while cur <= C.END_DATE:
            span = min(10, (C.END_DATE - cur).days + 1)
            dest = out_dir / f"firms_{cur.isoformat()}_{span}d.csv"
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                   f"{C.FIRMS_MAP_KEY}/{C.FIRMS_SOURCE}/{area}/{span}/{cur.isoformat()}")
            http_download(url, dest)
            cur += timedelta(days=span)
    else:
        log("fire: FIRMS_MAP_KEY not set — falling back to public per-country yearly CSVs")
        for year in range(C.START_DATE.year, C.END_DATE.year + 1):
            dest = out_dir / f"viirs-snpp_{year}_India.csv"
            url = C.FIRMS_COUNTRY_CSV.format(year=year)
            if not http_download(url, dest, quiet_404=True):
                log(f"fire: {year} country CSV unavailable — get a free key "
                    f"at https://firms.modaps.eosdis.nasa.gov/api/map_key/")


# ---------------------------------------------------------------------------
# 4. Sentinel-5P product catalogs (Copernicus Data Space OData — keyless)
# ---------------------------------------------------------------------------
def download_sentinel5p():
    out_dir = C.DIRS["sentinel5p"]
    w, s, e, n = C.BBOX_WSEN
    footprint = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    odata = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    for product in C.S5P_PRODUCTS:
        for year, month in month_range(C.START_DATE, C.END_DATE):
            dest = out_dir / f"s5p_{product.strip('_')}_{year}_{month:02d}_catalog.json"
            if skip_if_exists(dest):
                continue
            start = f"{year}-{month:02d}-01T00:00:00.000Z"
            nm, ny = (1, year + 1) if month == 12 else (month + 1, year)
            end = f"{ny}-{nm:02d}-01T00:00:00.000Z"
            params = {
                "$filter": (f"Collection/Name eq 'SENTINEL-5P' "
                            f"and contains(Name,'{product}') "
                            f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') "
                            f"and ContentDate/Start ge {start} and ContentDate/Start lt {end}"),
                "$top": 1000, "$select": "Id,Name,ContentDate,ContentLength",
            }
            try:
                r = requests.get(odata, params=params, timeout=C.REQUEST_TIMEOUT)
                r.raise_for_status()
                _assert_not_realtime(dest).write_text(json.dumps(r.json(), indent=2))
                log(f"OK: {dest.relative_to(C.BASE_DIR)}")
            except Exception as e:
                log(f"sentinel5p: {product} {year}-{month:02d} failed: {e}")


# ---------------------------------------------------------------------------
# 5. MODIS AOD (optional — needs free Earthdata token)
# ---------------------------------------------------------------------------
def download_modis():
    if not C.EARTHDATA_TOKEN:
        log("modis: EARTHDATA_TOKEN not set — skipped (optional; CAMS AOD already "
            "included in the aqi dataset)")
        return
    out_dir = C.DIRS["modis"]
    headers = {"Authorization": f"Bearer {C.EARTHDATA_TOKEN}"}
    cmr = "https://cmr.earthdata.nasa.gov/search/granules.json"
    w, s, e, n = C.BBOX_WSEN
    for year, month in month_range(C.START_DATE, C.END_DATE):
        nm, ny = (1, year + 1) if month == 12 else (month + 1, year)
        params = {
            "short_name": C.MODIS_PRODUCT, "version": C.MODIS_VERSION,
            "bounding_box": f"{w},{s},{e},{n}",
            "temporal": f"{year}-{month:02d}-01T00:00:00Z,{ny}-{nm:02d}-01T00:00:00Z",
            "page_size": 2000,
        }
        try:
            r = requests.get(cmr, params=params, timeout=C.REQUEST_TIMEOUT)
            r.raise_for_status()
            entries = r.json()["feed"]["entry"]
        except Exception as ex:
            log(f"modis: CMR search {year}-{month:02d} failed: {ex}")
            continue
        for entry in entries:
            links = [l["href"] for l in entry.get("links", [])
                     if l["href"].endswith(".hdf") and l["href"].startswith("https")]
            if links:
                http_download(links[0], out_dir / Path(links[0]).name, headers=headers)


# ---------------------------------------------------------------------------
# 6-7. OpenStreetMap via Overpass (keyless)
# ---------------------------------------------------------------------------
def _overpass(query: str, dest: Path):
    if skip_if_exists(dest):
        return
    try:
        r = requests.post(C.OVERPASS_URL, data={"data": query},
                      headers={"User-Agent": "aqi-project/1.0"},
                      timeout=C.REQUEST_TIMEOUT * 3)
        r.raise_for_status()
        _assert_not_realtime(dest).write_text(r.text, encoding="utf-8")
        log(f"OK: {dest.relative_to(C.BASE_DIR)}")
    except Exception as e:
        log(f"overpass: {dest.name} failed: {e}")


def _bbox_str():
    b = C.BBOX
    return f"{b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']}"


def download_roads():
    q = f"""[out:json][timeout:300];
(way["highway"~"motorway|trunk|primary|secondary|tertiary"]({_bbox_str()}););
out geom;"""
    _overpass(q, C.DIRS["roads"] / "osm_major_roads.json")


def download_industries():
    q = f"""[out:json][timeout:300];
(
  node["man_made"="works"]({_bbox_str()});
  way["man_made"="works"]({_bbox_str()});
  way["landuse"="industrial"]({_bbox_str()});
  node["power"="plant"]({_bbox_str()});
  way["power"="plant"]({_bbox_str()});
  node["amenity"="fuel"]({_bbox_str()});
);
out geom;"""
    _overpass(q, C.DIRS["industries"] / "osm_industries.json")


# ---------------------------------------------------------------------------
# 8. Vulnerability layer — hospitals, schools, elder care (Overpass, keyless)
# ---------------------------------------------------------------------------
def download_vulnerability():
    q = f"""[out:json][timeout:300];
(
  node["amenity"~"hospital|clinic|school|college|kindergarten|nursing_home"]({_bbox_str()});
  way["amenity"~"hospital|clinic|school|college|kindergarten|nursing_home"]({_bbox_str()});
);
out center;"""
    _overpass(q, C.DIRS["vulnerability"] / "osm_vulnerability.json")


# ---------------------------------------------------------------------------
# 9. Delhi ward boundaries (250 wards) — tried from candidate free mirrors
# ---------------------------------------------------------------------------
def download_wards():
    dest = C.DIRS["wards"] / "delhi_wards.geojson"
    if skip_if_exists(dest):
        return
    for url in C.WARD_GEOJSON_URLS:
        if http_download(url, dest, quiet_404=True):
            # sanity check: must be valid GeoJSON with features
            try:
                gj = json.loads(dest.read_text(encoding="utf-8"))
                if gj.get("features"):
                    log(f"wards: {len(gj['features'])} ward polygons")
                    return
            except Exception:
                pass
            dest.unlink(missing_ok=True)
    log("wards: no mirror worked — download manually (see README 'Ward boundaries') "
        "and save as data/raw/gis/wards/delhi_wards.geojson")


# ---------------------------------------------------------------------------
# 10. ESA WorldCover, 11. SRTM DEM, 12. WorldPop (all keyless)
# ---------------------------------------------------------------------------
def download_landuse():
    tile = "N27E075"  # 3x3 deg tile containing Delhi
    ver = "v200" if C.WORLDCOVER_YEAR >= 2021 else "v100"
    url = (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
           f"{ver}/{C.WORLDCOVER_YEAR}/map/"
           f"ESA_WorldCover_10m_{C.WORLDCOVER_YEAR}_{ver}_{tile}_Map.tif")
    http_download(url, C.DIRS["landuse"] / f"worldcover_{C.WORLDCOVER_YEAR}_{tile}.tif")


def download_dem():
    for tile in C.SRTM_TILES:
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile[:3]}/{tile}.hgt.gz"
        http_download(url, C.DIRS["dem"] / f"{tile}.hgt.gz")


def download_population():
    dest = C.DIRS["population"] / Path(C.WORLDPOP_URL).name
    http_download(C.WORLDPOP_URL, dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DOWNLOADERS = {
    "wards": download_wards,
    "aqi": download_aqi,
    "weather": download_weather,
    "fire": download_fire,
    "roads": download_roads,
    "industries": download_industries,
    "vulnerability": download_vulnerability,
    "landuse": download_landuse,
    "dem": download_dem,
    "population": download_population,
    "sentinel5p": download_sentinel5p,
    "modis": download_modis,
    "era5nc": download_era5nc,
}


def main():
    ensure_dirs()
    selected = sys.argv[1:] or list(DOWNLOADERS)
    unknown = [s for s in selected if s not in DOWNLOADERS]
    if unknown:
        sys.exit(f"Unknown dataset(s): {unknown}. Choose from: {list(DOWNLOADERS)}")

    log(f"Region: {C.REGION_NAME} | Range: {C.START_DATE} -> {C.END_DATE}")
    log(f"Grid: {sum(1 for _ in C.grid_points())} points at {C.GRID_RES} deg")
    for name in selected:
        log(f"=== {name} ===")
        try:
            DOWNLOADERS[name]()
        except Exception as e:
            log(f"{name}: FAILED — {e}")
    log("Done. Re-run anytime; existing files are skipped.")
    log("Next: python -m src.data.build_gnn_dataset  →  data/processed/final_dataset.csv")


if __name__ == "__main__":
    main()
