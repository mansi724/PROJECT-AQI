"""
add_emissions.py
=================================================================
Adds an EMISSION INVENTORY layer (EDGAR) to the Delhi ward dataset,
for source attribution: how much each ward emits, by sector.

WHY: source attribution needs to know *where the emissions come from*.
EDGAR gives free, gridded (0.1 deg) anthropogenic emissions per SECTOR
(power, industry, road transport, residential) and per pollutant.

-----------------------------------------------------------------
STEP 1 — download EDGAR grids (once, on a machine with internet)
-----------------------------------------------------------------
Go to:  https://edgar.jrc.ec.europa.eu/dataset_ap81   (EDGAR v8.1 Air Pollutants)
For each pollutant you want (PM2.5, NOx, SO2, CO), download the
SECTOR-specific yearly grid for the latest year, in **TXT** or **NetCDF**.
Sectors that matter most for Delhi:
    ENE  = power industry
    IND  = combustion + processes for manufacturing (industry)
    TRO_noRES / TRO = road transport
    RCO  = residential / commercial / other combustion
Save all downloaded files into:
    data/raw/gis/emissions/
Filenames should contain the pollutant and sector code, e.g.
    v8.1_FT2022_PM2.5_2022_IND.txt   or   ..._NOx_2022_TRO.nc
(The script matches on those tokens in the filename — it is tolerant.)

-----------------------------------------------------------------
STEP 2 — run
-----------------------------------------------------------------
    python add_emissions.py

Output:  data/processed/ward_emissions.csv   (one row per ward_id)
build_features.py will automatically merge it if present.

TXT files need no extra libraries. NetCDF (.nc) needs xarray+netCDF4:
    pip install xarray netcdf4
=================================================================
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import config as C
    EMIS_DIR = C.DIRS.get("emissions", C.RAW_DIR / "gis" / "emissions")
    PROC_DIR = C.PROCESSED_DIR
    BBOX = C.BBOX
except Exception:
    BASE = Path(__file__).resolve().parent
    EMIS_DIR = BASE / "data" / "raw" / "gis" / "emissions"
    PROC_DIR = BASE / "data" / "processed"
    BBOX = {"min_lat": 28.40, "max_lat": 28.90, "min_lon": 76.84, "max_lon": 77.35}

WARD_FEATURES = PROC_DIR / "ward_features.csv"

# EDGAR sector code -> friendly source name
SECTOR_MAP = {
    "ENE": "power",
    "IND": "industry",
    "TRO": "transport", "TRO_noRES": "transport", "TRO_RES": "transport",
    "RCO": "residential",
    "REF_TRF": "power", "PRO": "industry", "TNR": "transport",
}
# pollutant tokens we recognise in filenames -> clean name
POLL_MAP = {
    "PM2.5": "pm25", "PM25": "pm25", "PM2_5": "pm25",
    "NOX": "nox", "NOx": "nox", "NO2": "nox",
    "SO2": "so2", "CO": "co", "PM10": "pm10",
}


def log(m):
    print(f"[emissions] {m}", flush=True)


def parse_tokens(fname):
    """Guess (pollutant, sector) from an EDGAR filename."""
    up = fname.upper()
    poll = None
    for tok, name in POLL_MAP.items():
        if tok.upper() in up:
            poll = name
            break
    sector = None
    # match longest sector codes first
    for code in sorted(SECTOR_MAP, key=len, reverse=True):
        if re.search(rf"(^|[_\-.]){code}([_\-.]|$)", up):
            sector = SECTOR_MAP[code]
            break
    return poll, sector


def read_grid(path):
    """
    Return a DataFrame with columns lat, lon, emi  (clipped to Delhi bbox).
    Supports EDGAR .txt (flat lat;lon;emission) and .nc (NetCDF).
    """
    p = Path(path)
    lo_lat, hi_lat = BBOX["min_lat"] - 0.2, BBOX["max_lat"] + 0.2
    lo_lon, hi_lon = BBOX["min_lon"] - 0.2, BBOX["max_lon"] + 0.2

    if p.suffix.lower() == ".nc":
        try:
            import xarray as xr
        except ImportError:
            log(f"  skip {p.name}: NetCDF needs `pip install xarray netcdf4`")
            return None
        ds = xr.open_dataset(p)
        var = [v for v in ds.data_vars][0]
        da = ds[var]
        latn = "lat" if "lat" in da.dims else [d for d in da.dims if "lat" in d.lower()][0]
        lonn = "lon" if "lon" in da.dims else [d for d in da.dims if "lon" in d.lower()][0]
        da = da.squeeze()
        sub = da.where((da[latn] >= lo_lat) & (da[latn] <= hi_lat) &
                       (da[lonn] >= lo_lon) & (da[lonn] <= hi_lon), drop=True)
        df = sub.to_dataframe(name="emi").reset_index()
        df = df.rename(columns={latn: "lat", lonn: "lon"})
        return df[["lat", "lon", "emi"]].dropna()

    # TXT / CSV: sniff separator, find the 3 numeric columns
    for sep in [";", ",", r"\s+"]:
        try:
            df = pd.read_csv(p, sep=sep, engine="python", comment="#",
                             header=None, nrows=5)
        except Exception:
            continue
        if df.shape[1] >= 3 and df.apply(lambda c: pd.to_numeric(c, errors="coerce").notna().all()).sum() >= 3:
            full = pd.read_csv(p, sep=sep, engine="python", comment="#", header=None)
            full = full.iloc[:, :3]
            full.columns = ["lat", "lon", "emi"]
            full = full.apply(pd.to_numeric, errors="coerce").dropna()
            full = full[(full.lat >= lo_lat) & (full.lat <= hi_lat) &
                        (full.lon >= lo_lon) & (full.lon <= hi_lon)]
            return full
    log(f"  skip {p.name}: could not parse as txt/csv")
    return None


def nearest_emi(grid, lat, lon):
    """Emission of the EDGAR cell nearest a ward centroid."""
    d = (grid["lat"].values - lat) ** 2 + (grid["lon"].values - lon) ** 2
    return float(grid["emi"].values[d.argmin()]) if len(d) else np.nan


def main():
    if not WARD_FEATURES.exists():
        sys.exit(f"[emissions] {WARD_FEATURES} not found — run build_dataset.py first.")
    wards = pd.read_csv(WARD_FEATURES)[["ward_id", "ward_lat", "ward_lon"]]

    files = sorted(glob.glob(str(EMIS_DIR / "*")))
    files = [f for f in files if Path(f).suffix.lower() in (".txt", ".nc", ".csv")]
    if not files:
        sys.exit(f"[emissions] no EDGAR files in {EMIS_DIR}\n"
                 f"            download them first (see the header of this script).")

    log(f"found {len(files)} emission files")
    # accumulate per (sector, pollutant) a value per ward
    cols = {}
    for f in files:
        poll, sector = parse_tokens(Path(f).name)
        if not poll or not sector:
            log(f"  skip {Path(f).name}: can't identify pollutant/sector from name")
            continue
        grid = read_grid(f)
        if grid is None or grid.empty:
            log(f"  skip {Path(f).name}: no grid data in Delhi bbox")
            continue
        key = f"emis_{sector}_{poll}"
        vals = wards.apply(lambda r: nearest_emi(grid, r["ward_lat"], r["ward_lon"]), axis=1)
        cols[key] = cols.get(key, 0) + vals.fillna(0)
        log(f"  OK {Path(f).name} -> {key}")

    if not cols:
        sys.exit("[emissions] no usable files (check filenames contain pollutant + sector).")

    out = wards[["ward_id"]].copy()
    for k, v in cols.items():
        out[k] = v.values
    # totals per pollutant across sectors
    for poll in set(k.split("_")[-1] for k in cols):
        pc = [k for k in cols if k.endswith("_" + poll)]
        out[f"emis_total_{poll}"] = out[pc].sum(axis=1)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "ward_emissions.csv"
    out.to_csv(out_path, index=False)
    log(f"SAVED -> {out_path}  ({out.shape[0]} wards x {out.shape[1]-1} emission cols)")
    log("Columns: " + ", ".join(c for c in out.columns if c != "ward_id"))
    log("Next: re-run  python build_features.py  (it auto-merges this file).")


if __name__ == "__main__":
    main()
