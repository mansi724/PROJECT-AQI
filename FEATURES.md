# Feature manifest — which columns feed which engine

`model_ready.parquet` now has **137 columns**. Not all are model inputs. Use
[`feature_lists.py`](feature_lists.py) in training code — it is the single source
of truth and is validated against the actual file (`python feature_lists.py`).

## ⛔ Never train on these (leakage / identifiers)
`ward_id`, `ward_name`, `time`, `point_id`, `nearest_station_id`, `lu_majority_class`,
`dominant_pollutant`, `aqi_category`, `season`, `split` (identifiers / string labels — use the `*_code` columns),
and the answer / pre-correction copies:
`target_aqi_t24/48/72`, `persistence_aqi_t24`, `aqi_raw`, `pm2_5_raw`, `pm10_raw`.

> `aqi` at time *t* **is** a valid input when predicting *t+h* — only the shifted `target_*` leaks.

## Engine 1 — AQI forecasting (81 features)
- **Targets:** `target_aqi_t24` / `t48` / `t72` (24–72h). Baseline to beat = `persistence_aqi_t24` (RMSE 105.4 / MAE 72.4 on test).
- **Inputs:** current pollutants + weather, `aqi`, cyclical time, calendar flags
  (`is_weekend/rush_hour/winter/stubble_season/diwali_window/holiday`), the lag & rolling
  history (most important), dispersion (`ventilation_index`, `stagnation_index`), static
  ward context, and identity encoding (`point_id_code`, `ward_hist_aqi` — train-only mean, leak-free).
- **Split:** train on `split=='train'`, tune on `'val'`, report on `'test'`.
  For cross-validation, **group by `point_id`** — the pollutant signal is only ~19 grid cells.

## Engine 2 — Source attribution (38 features)
- Directional/local source signals: `wind_from_nw`, `fire_upwind`, `industry_upwind`,
  `road_upwind` (sources in the direction wind blows FROM), `traffic_load`,
  `industry_stagnation`, `buildup_pressure`.
- **Real magnitudes:** EDGAR v8.1 (2022) per-sector emissions —
  `emis_{power,industry,residential,transport}_{pm25,nox,so2,co}` + `emis_total_*`.
- Validate the attribution story against the EDGAR sector split.

## Engine 3 — Action / health recommendation (8 features)
`aqi`, `aqi_category_code`, `vulnerable_sites_3km`, `population_*`, `vuln_norm`,
`health_risk_score`, `health_risk_level` — plus the dominant source from Engine 2 at inference.

## Validation only (NOT in model_ready.parquet)
CPCB station readings live in `data/processed/cpcb_ground_truth.csv`. Score against them; never train on them.
