"""
build_emissions.py
=================================================================
Reads the EDGAR v8.1 (FT2022) sector emission grids downloaded into
data/raw/gis/emissions/ and maps them to each Delhi ward, producing
data/processed/ward_emissions.csv.

EDGAR txt format: 3 header lines (date, metadata, 'lat;lon;emission'),
then 'lat;lon;emission' rows in Tonnes / 0.1deg cell / year.
Filenames: v8.1_FT2022_AP_{POLL}_2022_{SECTOR}.txt
=================================================================
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
EMIS_DIR = BASE / "data" / "raw" / "gis" / "emissions"
PROC = BASE / "data" / "processed"
WARD_FEATURES = PROC / "ward_features.csv"
BBOX = {"min_lat": 28.40, "max_lat": 28.90, "min_lon": 76.84, "max_lon": 77.35}

SECTOR_MAP = {"ENE": "power", "IND": "industry", "TRO": "transport", "RCO": "residential"}
POLL_MAP = {"PM2.5": "pm25", "NOX": "nox", "SO2": "so2", "CO": "co"}


def log(m): print(f"[emissions] {m}", flush=True)


def parse_tokens(fname):
    up = fname.upper()
    poll = next((v for k, v in POLL_MAP.items() if k in up), None)
    sector = None
    for code in SECTOR_MAP:
        if re.search(rf"(^|[_\-.]){code}([_\-.]|$)", up):
            sector = SECTOR_MAP[code]
            break
    return poll, sector


def read_grid(path):
    lo_lat, hi_lat = BBOX["min_lat"] - 0.3, BBOX["max_lat"] + 0.3
    lo_lon, hi_lon = BBOX["min_lon"] - 0.3, BBOX["max_lon"] + 0.3
    # skip the 3 text header lines; data is lat;lon;emission
    df = pd.read_csv(path, sep=";", skiprows=3, header=None,
                     names=["lat", "lon", "emi"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df[(df.lat >= lo_lat) & (df.lat <= hi_lat) &
              (df.lon >= lo_lon) & (df.lon <= hi_lon)]


def nearest_emi(grid, lat, lon):
    if grid.empty:
        return 0.0
    d = (grid["lat"].values - lat) ** 2 + (grid["lon"].values - lon) ** 2
    return float(grid["emi"].values[d.argmin()])


def main():
    wards = pd.read_csv(WARD_FEATURES)[["ward_id", "ward_lat", "ward_lon"]]
    files = sorted(glob.glob(str(EMIS_DIR / "*.txt")))
    if not files:
        raise SystemExit(f"no EDGAR txt in {EMIS_DIR}")
    log(f"{len(files)} emission grids; {len(wards)} wards")

    cols = {}
    for f in files:
        poll, sector = parse_tokens(Path(f).name)
        if not poll or not sector:
            log(f"  skip {Path(f).name} (unrecognized)"); continue
        grid = read_grid(f)
        key = f"emis_{sector}_{poll}"
        vals = wards.apply(lambda r: nearest_emi(grid, r.ward_lat, r.ward_lon), axis=1)
        cols[key] = cols.get(key, 0) + vals.fillna(0)
        log(f"  OK {Path(f).name} -> {key}  (Delhi cells={len(grid)})")

    out = wards[["ward_id"]].copy()
    for k, v in cols.items():
        out[k] = v.values
    for poll in set(k.rsplit("_", 1)[-1] for k in cols):
        pc = [k for k in cols if k.endswith("_" + poll)]
        out[f"emis_total_{poll}"] = out[pc].sum(axis=1)

    PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROC / "ward_emissions.csv", index=False)
    log(f"SAVED -> ward_emissions.csv  ({out.shape[0]} wards x {out.shape[1]-1} cols)")
    log("cols: " + ", ".join(c for c in out.columns if c != "ward_id"))


if __name__ == "__main__":
    main()
