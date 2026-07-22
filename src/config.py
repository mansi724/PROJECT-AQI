"""
Central configuration for the AQI Forecasting historical data pipeline.
Delhi ward-level system: forecasting + source attribution + action recommendations.

Edit values here — never inside download.py / build_dataset.py.
All default sources are FREE and NEED NO API KEY.
Optional keyed sources (OpenAQ stations, CDS NetCDF, FIRMS API, MODIS) add extra depth.
"""

from datetime import date, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REALTIME_DIR = DATA_DIR / "realtime"   # READ-ONLY. The pipeline must NEVER write here.

DIRS = {
    "aqi":           RAW_DIR / "aqi",
    "aqi_cams":      RAW_DIR / "aqi" / "cams",
    "aqi_openaq":    RAW_DIR / "aqi" / "openaq",
    "weather":       RAW_DIR / "weather",
    "fire":          RAW_DIR / "fire",
    "sentinel5p":    RAW_DIR / "satellite" / "sentinel5p",
    "modis":         RAW_DIR / "satellite" / "modis",
    "roads":         RAW_DIR / "gis" / "roads",
    "industries":    RAW_DIR / "gis" / "industries",
    "landuse":       RAW_DIR / "gis" / "landuse",
    "dem":           RAW_DIR / "gis" / "dem",
    "population":    RAW_DIR / "gis" / "population",
    "wards":         RAW_DIR / "gis" / "wards",
    "vulnerability": RAW_DIR / "gis" / "vulnerability",
}

# ---------------------------------------------------------------------------
# Region of interest: Delhi (NCT) — ward-level system
# ---------------------------------------------------------------------------
REGION_NAME = "Delhi NCT (ward-level)"
CITY = "Delhi"
COUNTRY = "IN"

# City bounding box (WGS84) — NCT of Delhi + immediate NCR fringe
BBOX = {
    "min_lat": 28.40,
    "max_lat": 28.90,
    "min_lon": 76.84,
    "max_lon": 77.35,
}
BBOX_WSEN = (BBOX["min_lon"], BBOX["min_lat"], BBOX["max_lon"], BBOX["max_lat"])
BBOX_NWSE = (BBOX["max_lat"], BBOX["min_lon"], BBOX["min_lat"], BBOX["max_lon"])

# WIDE bbox for FIRE data only — covers Punjab/Haryana/W-UP stubble burning,
# the dominant external source of Delhi's winter AQI.
FIRE_BBOX = {
    "min_lat": 27.0,
    "max_lat": 32.0,
    "min_lon": 73.0,
    "max_lon": 79.0,
}

# ---------------------------------------------------------------------------
# Forecasting grid — points where hourly weather + pollution are downloaded.
# ~0.1 deg (~11 km, matches CAMS resolution). Wards are mapped to the
# nearest grid point by build_dataset.py.
# ---------------------------------------------------------------------------
GRID_RES = 0.10

def grid_points():
    """Yield (point_id, lat, lon) covering the city bbox."""
    pid = 0
    lat = BBOX["min_lat"]
    while lat <= BBOX["max_lat"] + 1e-9:
        lon = BBOX["min_lon"]
        while lon <= BBOX["max_lon"] + 1e-9:
            yield pid, round(lat, 4), round(lon, 4)
            pid += 1
            lon += GRID_RES
        lat += GRID_RES

# ---------------------------------------------------------------------------
# Historical date range: last 3 years
# ---------------------------------------------------------------------------
END_DATE = date.today() - timedelta(days=2)   # small lag: archives update ~2 days behind
START_DATE = END_DATE - timedelta(days=3 * 365)

# ---------------------------------------------------------------------------
# KEYLESS default sources (Open-Meteo — free, no registration)
# ---------------------------------------------------------------------------
OPENMETEO_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ERA5-based hourly weather variables
OPENMETEO_WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
]
# Requested additionally; auto-dropped if the API rejects it
OPENMETEO_WEATHER_OPTIONAL = ["boundary_layer_height"]

# CAMS hourly air-quality variables (includes AOD — satellite-equivalent signal)
OPENMETEO_AQ_VARS = [
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide",
    "carbon_monoxide", "ozone", "aerosol_optical_depth", "dust",
]

# ---------------------------------------------------------------------------
# Optional API keys (all free-tier) — extra datasets, not required for the CSV
# ---------------------------------------------------------------------------
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")      # CPCB station ground truth
CDSAPI_URL = os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
CDSAPI_KEY = os.getenv("CDSAPI_KEY", "")              # ERA5 NetCDF grids
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")        # FIRMS area API (else country-CSV fallback)
EARTHDATA_TOKEN = os.getenv("EARTHDATA_TOKEN", "")    # MODIS AOD granules

# ---------------------------------------------------------------------------
# Dataset-specific settings
# ---------------------------------------------------------------------------
TEMPORAL_RESOLUTION = "hourly"  # Options: "hourly", "daily"

AQI_PARAMETERS = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]

ERA5_VARIABLES = [
    "2m_temperature", "2m_dewpoint_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "surface_pressure", "total_precipitation", "boundary_layer_height",
]

FIRMS_SOURCE = "VIIRS_SNPP_SP"
# Keyless fallback: NASA FIRMS public per-country yearly CSVs
FIRMS_COUNTRY_CSV = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/"
    "{year}/viirs-snpp_{year}_India.csv"
)

S5P_PRODUCTS = ["L2__NO2___", "L2__SO2___", "L2__CO____", "L2__AER_AI"]
MODIS_PRODUCT = "MCD19A2"
MODIS_VERSION = "061"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Delhi ward boundaries (2022 delimitation, 250 wards) — candidate free sources,
# tried in order. If all fail, download manually (see README) and save as
# data/raw/gis/wards/delhi_wards.geojson
WARD_GEOJSON_URLS = [
    "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Delhi/Delhi_Wards.geojson",
    "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Delhi/delhi_wards.geojson",
    "https://raw.githubusercontent.com/opencitydata/delhi-wards/master/delhi_wards.geojson",
]

WORLDCOVER_YEAR = 2021
SRTM_TILES = ["N28E076", "N28E077"]

WORLDPOP_YEAR = 2020
WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/"
    f"{WORLDPOP_YEAR}/IND/ind_ppp_{WORLDPOP_YEAR}_1km_Aggregated_UNadj.tif"
)

# ---------------------------------------------------------------------------
# Download behaviour
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_BACKOFF = 5
CHUNK_SIZE = 1024 * 1024
OPENMETEO_SLEEP = 0.6        # seconds between Open-Meteo calls (free-tier friendly)

# ---------------------------------------------------------------------------
# Attributes / Schema mapping
# ---------------------------------------------------------------------------
UNITS = {
    "pm2_5": "μg/m³",
    "pm10": "μg/m³",
    "nitrogen_dioxide": "μg/m³",
    "sulphur_dioxide": "μg/m³",
    "carbon_monoxide": "μg/m³",
    "ozone": "μg/m³",
    "aerosol_optical_depth": "",
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "dew_point_2m": "°C",
    "precipitation": "mm",
    "surface_pressure": "hPa",
    "wind_speed_10m": "km/h",
    "wind_direction_10m": "°",
    "boundary_layer_height": "m",
}
