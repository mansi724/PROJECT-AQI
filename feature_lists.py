"""
feature_lists.py
=================================================================
Single source of truth for WHICH columns feed WHICH engine.
Import these lists in training code so nobody accidentally trains on
a leaking column (the target, the persistence baseline, raw pre-correction
copies, identifiers, or the string label columns).

Usage:
    from feature_lists import forecast_features, EXCLUDE_ALWAYS
    X = df[forecast_features]
    y = df["target_aqi_t24"]
=================================================================
"""

# --- NEVER feed these as inputs to any model -------------------------------
# identifiers, string labels, the answer, and pre-correction raw copies.
EXCLUDE_ALWAYS = [
    "ward_id", "ward_name", "time", "point_id", "nearest_station_id",
    "lu_majority_class", "dominant_pollutant", "aqi_category", "season", "split",
    # leakage: targets + naive baseline + raw pre-correction values
    "target_aqi_t24", "target_aqi_t48", "target_aqi_t72",
    "persistence_aqi_t24", "aqi_raw", "pm2_5_raw", "pm10_raw",
]

TARGETS = {24: "target_aqi_t24", 48: "target_aqi_t48", 72: "target_aqi_t72"}
# persistence baseline for ALL horizons = current aqi (naive "tomorrow=today")
PERSISTENCE = "persistence_aqi_t24"

# --- ENGINE 1: AQI forecasting ---------------------------------------------
# Everything predictive & leak-free. `aqi` at time t IS a valid input when
# predicting t+h. Use *_code instead of the string label columns.
forecast_features = [
    # current pollutants + weather
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
    "ozone", "aerosol_optical_depth", "dust",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "boundary_layer_height", "fire_count", "fire_frp_sum",
    "aqi",  # current AQI — valid feature for a future-horizon target
    # cyclical time + calendar
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_rush_hour", "is_winter", "is_stubble_season",
    "is_diwali_window", "is_holiday",
    # history (most important for forecasting)
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h", "aqi_lag_48h",
    "pm2_5_lag_1h", "pm2_5_lag_3h", "pm2_5_lag_6h", "pm2_5_lag_12h", "pm2_5_lag_24h", "pm2_5_lag_48h",
    "pm10_lag_1h", "pm10_lag_3h", "pm10_lag_6h", "pm10_lag_12h", "pm10_lag_24h", "pm10_lag_48h",
    "nitrogen_dioxide_lag_1h", "nitrogen_dioxide_lag_3h", "nitrogen_dioxide_lag_6h",
    "nitrogen_dioxide_lag_12h", "nitrogen_dioxide_lag_24h", "nitrogen_dioxide_lag_48h",
    "aqi_roll_mean_6h", "aqi_roll_mean_24h", "aqi_roll_max_24h", "aqi_roll_std_24h",
    "pm2_5_roll_mean_6h", "pm2_5_roll_mean_24h", "pm2_5_roll_max_24h", "pm2_5_roll_std_24h",
    "aqi_diff_1h", "aqi_diff_24h",
    # dispersion / wind
    "wind_dir_sin", "wind_dir_cos", "ventilation_index", "stagnation_index",
    # static ward context + identity encoding
    "road_km_3km", "road_capacity_3km", "industry_count_5km", "vulnerable_sites_3km",
    "population_sum", "population_density_mean", "elevation_mean",
    "lu_builtup_fraction", "lu_tree_fraction",
    "point_id_code", "ward_hist_aqi",
]

# --- ENGINE 2: source attribution ------------------------------------------
# directional/local source signals + real EDGAR per-sector magnitudes.
EMISSION_COLS = [
    "emis_power_pm25", "emis_industry_pm25", "emis_residential_pm25", "emis_transport_pm25",
    "emis_power_nox", "emis_industry_nox", "emis_residential_nox", "emis_transport_nox",
    "emis_power_so2", "emis_industry_so2", "emis_residential_so2", "emis_transport_so2",
    "emis_power_co", "emis_industry_co", "emis_residential_co", "emis_transport_co",
    "emis_total_pm25", "emis_total_nox", "emis_total_so2", "emis_total_co",
]
attribution_features = [
    "dominant_pollutant_code",
    "wind_from_nw", "fire_upwind", "wind_direction_10m", "wind_speed_10m",
    "traffic_load", "industry_stagnation", "buildup_pressure",
    "industry_upwind", "road_upwind",
    "road_capacity_3km", "industry_count_5km",
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "ozone",
] + EMISSION_COLS

# --- ENGINE 3: action / health recommendation ------------------------------
action_features = [
    "aqi", "aqi_category_code", "vulnerable_sites_3km",
    "population_sum", "population_density_mean",
    "vuln_norm", "health_risk_score", "health_risk_level",
    # plus the dominant source from Engine 2 at inference time
]

# validation-only (kept in data/processed/cpcb_ground_truth.csv, NOT here)
GROUND_TRUTH_NOTE = "CPCB station readings live in cpcb_ground_truth.csv — score, don't train."

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    df = pd.read_parquet(Path(__file__).resolve().parent / "data/final/model_ready.parquet")
    cols = set(df.columns)
    for name, lst in [("forecast", forecast_features),
                      ("attribution", attribution_features),
                      ("action", action_features)]:
        missing = [c for c in lst if c not in cols]
        print(f"{name}: {len(lst)} features, missing={missing}")
    leak = [c for c in forecast_features if c in EXCLUDE_ALWAYS]
    print(f"forecast/leak overlap (should be []): {leak}")
