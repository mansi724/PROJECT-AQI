"""
refresh_ground_truth.py
=================================================================
Re-pulls CPCB station ground truth from OpenAQ v3 using the CORRECT
endpoint (/sensors/{id}/hours) and params (datetime_from/datetime_to).

Design notes (learned the hard way):
  * CHECKPOINTED — every sensor's result is written to disk the moment it
    lands (data/processed/cpcb_cache/sensor_<id>.parquet). A crash or a
    killed shell loses at most one sensor, and re-running skips whatever is
    already cached. The previous version accumulated everything in RAM and
    wrote once at the end, so a 4-hour run died with nothing to show.
  * THREADED — 4 workers. Sequential was ~17h for the full station list.
  * SCOPED — only PM25/PM10/NO2 by default. PM dominates the CPCB AQI in
    Delhi; pulling all 6 params triples runtime for little label value.

Run:  python refresh_ground_truth.py            # pull + assemble
      python refresh_ground_truth.py --assemble # assemble cache only
=================================================================
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

import config as C

BASE = Path(__file__).resolve().parent
OUT_DIR = C.DIRS["aqi_openaq"]
STATIONS = OUT_DIR / "stations.json"
PROC = C.PROCESSED_DIR
CACHE = PROC / "cpcb_cache"
BASEURL = "https://api.openaq.org/v3"
WIN_LO = pd.Timestamp(C.START_DATE)
WIN_HI = pd.Timestamp(C.END_DATE) + pd.Timedelta(days=1)

# Params that actually drive the CPCB AQI. PM2.5/PM10 are the binding
# sub-indices in Delhi almost year-round; NO2 matters near traffic.
PARAMS = {"pm25", "pm10", "no2"}
# Only bother with stations that still reported recently.
ALIVE_SINCE = pd.Timestamp("2025-01-01")
WORKERS = 4

_print_lock = threading.Lock()


def log(m):
    with _print_lock:
        print(f"[refresh] {m}", flush=True)


def _pull_range(sid, headers, d_from, d_to):
    """Paginate one date range (kept < ~10k rows to avoid OpenAQ's deep-page 500)."""
    out = []
    page = 1
    while page <= 12:
        r = None
        for attempt in range(5):
            try:
                r = requests.get(f"{BASEURL}/sensors/{sid}/hours", headers=headers,
                                 params={"datetime_from": d_from, "datetime_to": d_to,
                                         "limit": 1000, "page": page}, timeout=60)
            except requests.RequestException:
                time.sleep(3)
                continue
            if r.status_code == 429:
                time.sleep(8)
                continue
            if r.status_code in (404, 410):
                return out
            if r.status_code >= 500:
                time.sleep(3)
                continue
            break
        if r is None or r.status_code != 200:
            return out
        res = r.json().get("results", [])
        out.extend(res)
        if len(res) < 1000:
            break
        page += 1
        time.sleep(0.25)
    return out


def pull_sensor_hours(sid, headers):
    """Pull the whole window in yearly chunks (each < 10k hours)."""
    out = []
    for yr in range(WIN_LO.year, WIN_HI.year + 1):
        lo = max(WIN_LO, pd.Timestamp(yr, 1, 1))
        hi = min(WIN_HI, pd.Timestamp(yr + 1, 1, 1))
        if lo >= hi:
            continue
        out.extend(_pull_range(sid, headers,
                               lo.strftime("%Y-%m-%dT00:00:00Z"),
                               hi.strftime("%Y-%m-%dT00:00:00Z")))
    return out


def sensor_job(task, headers):
    """Pull one sensor and checkpoint it to disk immediately."""
    sid, station_id, param, lat, lon = task
    dest = CACHE / f"sensor_{sid}.parquet"
    if dest.exists():
        return sid, param, -1  # already cached

    recs = pull_sensor_hours(sid, headers)
    rows = []
    for r in recs:
        t = r.get("period", {}).get("datetimeFrom", {}).get("utc")
        v = r.get("value")
        if t is None or v is None:
            continue
        rows.append((station_id, t, param, v, lat, lon))

    df = pd.DataFrame(rows, columns=["station_id", "time", "param", "value", "lat", "lon"])
    # write even when empty -> records "this sensor is done, it has nothing"
    df.to_parquet(dest, index=False)
    return sid, param, len(rows)


def build_tasks(stations):
    tasks = []
    for st in stations:
        last = (st.get("datetimeLast") or {}).get("utc")
        if not last or pd.Timestamp(last[:19]) < ALIVE_SINCE:
            continue
        coords = st.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is None or lon is None:
            continue
        for sensor in st.get("sensors", []):
            param = sensor.get("parameter", {}).get("name", "")
            if param not in PARAMS:
                continue
            # NOTE: stations often expose several sensors for the same param
            # (a retired one and a live one) and the metadata gives no way to
            # tell them apart. Pull them all; assemble() averages duplicate
            # station-hours, and dead sensors simply return 0 rows.
            tasks.append((sensor["id"], st["id"], param, lat, lon))
    return tasks


def assemble():
    """Fold every cached sensor file into cpcb_ground_truth.csv."""
    files = sorted(CACHE.glob("sensor_*.parquet"))
    if not files:
        raise SystemExit("no cache files — run the pull first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["value"].notna()]
    log(f"assembling {len(files)} cached sensors -> {len(df):,} raw rows")

    df["time"] = (pd.to_datetime(df["time"], utc=True, errors="coerce")
                    .dt.tz_localize(None).dt.floor("h"))
    df = df[(df["time"] >= WIN_LO) & (df["time"] <= WIN_HI)]
    # drop physically impossible readings (OpenAQ carries -999 style nulls)
    df = df[(df["value"] > 0) & (df["value"] < 2000)]

    coords = (df.groupby("station_id")[["lat", "lon"]].first().reset_index())
    wide = (df.pivot_table(index=["station_id", "time"], columns="param",
                           values="value", aggfunc="mean").reset_index())
    wide.columns.name = None
    wide = wide.rename(columns={p: f"cpcb_{p}" for p in PARAMS if p in wide.columns})
    wide = wide.merge(coords, on="station_id", how="left")

    out_path = PROC / "cpcb_ground_truth.csv"
    if out_path.exists():
        out_path.replace(PROC / "cpcb_ground_truth_OLD.csv")
    wide.to_csv(out_path, index=False)
    log(f"SAVED -> {out_path.name}: {len(wide):,} station-hours, "
        f"{wide['time'].min()} -> {wide['time'].max()}, "
        f"{wide['station_id'].nunique()} stations")
    for c in [c for c in wide.columns if c.startswith("cpcb_")]:
        log(f"  {c}: {wide[c].notna().sum():,} non-null")
    return wide


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if "--assemble" in sys.argv:
        assemble()
        return

    key = (C.OPENAQ_API_KEY or "").strip()
    if not key:
        raise SystemExit("OPENAQ_API_KEY missing in .env")
    headers = {"X-API-Key": key}
    stations = json.loads(STATIONS.read_text(encoding="utf-8")).get("results", [])
    tasks = build_tasks(stations)
    done = {f.stem.replace("sensor_", "") for f in CACHE.glob("sensor_*.parquet")}
    todo = [t for t in tasks if str(t[0]) not in done]
    log(f"{len(tasks)} sensors in scope ({sorted(PARAMS)}), "
        f"{len(done)} already cached, {len(todo)} to pull, {WORKERS} workers")
    log(f"window {WIN_LO.date()} -> {WIN_HI.date()}")

    n_done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(sensor_job, t, headers): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            n_done += 1
            try:
                sid, param, n = fut.result()
                log(f"[{n_done}/{len(todo)}] station {t[1]} {param}: {n:,} hours")
            except Exception as e:
                log(f"[{n_done}/{len(todo)}] station {t[1]} {t[2]}: FAILED {e}")

    log("pull complete — assembling")
    assemble()


if __name__ == "__main__":
    main()
