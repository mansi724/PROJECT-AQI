# Preprocessing — cleaned & normalised dataset

`preprocess_dataset.py` turns the four raw files in `data/gnn/` into model-ready copies in
**`data/gnn_processed/`**, applying the agreed missing-value + normalisation strategy. The raw files
are left untouched.

```
python preprocess_dataset.py
```

## Output files

| File | Shape | What changed |
|---|---|---|
| `dynamic_grid_norm.parquet` | 497,496 × 84 | warm-up + horizon rows dropped, gaps filled, features scaled |
| `nodes_static_norm.parquet` | 289 × 42 | static features scaled |
| `edges_norm.parquet` | 1,670 × 9 | added `bearing_sin/cos`, `dist_km_z` (kept `bearing_deg` raw) |
| `labels_station_clean.parquet` | 488,694 × 15 | validated only — ground truth never altered |
| `scalers.joblib` | — | every fitted scaler + column groups (apply the same transform at serving time) |
| `NORMALISATION_MANIFEST.json` | — | machine-readable column → transform map |

## The two rules that make this leak-free

1. **Scalers are fit on the TRAIN split only** (dynamic: 348,935 train rows; static: all 289 wards),
   then applied to val/test. Verified: `temperature_2m` train mean 0.000 / std 1.000, while val mean
   is −0.470 — the shift is real signal, not leaked. Fitting on the whole file would have quietly
   inflated every score.
2. **Targets and ground truth are never normalised and never imputed.** `target_aqi_t24` stays raw
   (27–500), `aqi_station` stays raw (14–500).

## Missing-value handling (as applied)

- **Dynamic grid:** dropped the **first 48 h** (lag/rolling warm-up) and the **last 72 h**
  (t+72 label unavailable) of each of the 19 cells → 2,280 rows removed. Residual gaps
  linear-interpolated **within each cell**; `fire_count`/`fire_frp_sum` NaN → 0 (no detection).
  Result: **0 NaN** anywhere in the file.
- **Static / edges:** already complete — nothing to fill.
- **Labels:** `aqi_station` is 0-null. The Engine-2 label columns
  (`pm25_pm10_ratio_obs`, `source_class_obs`) keep their 2.8 % nulls **on purpose** — drop those rows
  when training Engine 2 (97.2 % = 474,847 rows remain). Never impute a label.

## Normalisation scheme

| Group | Transform | Columns |
|---|---|---|
| Pollutants + their lags/rolls/diffs | **RobustScaler** | `aqi`, `pm2_5`, `pm10`, `nitrogen_dioxide`, `sulphur_dioxide`, `carbon_monoxide`, `ozone`, `aerosol_optical_depth`, `dust`, all `*_lag_*` / `*_roll_*` / `*_diff_*` |
| Weather + dispersion + ratios | **StandardScaler** | `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `surface_pressure`, `wind_speed_10m`, `wind_gusts_10m`, `boundary_layer_height`, `ventilation_index`, `stagnation_index`, `pm25_pm10_ratio`, `dust_fraction`, `so2_no2_ratio` |
| Fire | **log1p → StandardScaler** | `fire_count`, `fire_frp_sum` |
| Static counts / emissions | **log1p → StandardScaler** | population, roads, industry, vulnerable-sites, area, all `emis_*` |
| Static continuous | **StandardScaler** | `elevation_mean`, `lu_builtup_fraction`, `lu_tree_fraction` |
| Cyclical / binary | **left as-is** | `*_sin`, `*_cos`, `is_*`, `wind_from_nw`, `fire_upwind` |
| Wind direction | **sin/cos** | already stored as `wind_dir_sin` / `wind_dir_cos`; edges gained `bearing_sin/cos` |

RobustScaler centres on the median, so RobustScaler columns show a small non-zero mean (e.g. `pm2_5`
train mean +0.35) — that's expected and correct for right-skewed smog data.

## How the STGNN should consume this

- Read from **`data/gnn_processed/`** and **skip re-normalisation** (it's already applied). The
  reference `gnn_data.py` normalises on the fly against the raw files, so pick one path — don't
  double-scale.
- **Keep `bearing_deg` raw** for the wind gate: the model computes
  `cos(bearing_deg − wind_direction)`; feeding the scaled version would break the gate. (`edges_norm`
  keeps both.)
- Leak columns (`pm2_5_raw`, `pm10_raw`, `target_*`, `persistence_aqi_t24`, `dominant_pollutant`)
  are retained raw for reference but must stay out of the feature matrix (`feature_lists.EXCLUDE_ALWAYS`).
- Reuse `scalers.joblib` at inference/serving so live inputs get the identical transform.

## Notes vs. the original strategy table

The strategy table listed some layers this dataset doesn't carry as hourly streams — handled as
follows so nothing is silently faked:
- **Satellite** revisit / forward-fill → the satellite signal here is already the hourly CAMS
  `aerosol_optical_depth` + `dust` (no separate revisit gaps to fill).
- **Construction / industry activity** forward-fill → no hourly construction or industrial-activity
  layer exists; industry is the **annual** EDGAR `emis_*` in `nodes_static` (static, log1p-scaled).
- **Wind direction interpolation** → done in sin/cos space (`wind_dir_sin/cos`), never on degrees, as
  specified.
