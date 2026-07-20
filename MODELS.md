# Modeling phase — Graph Transformer forecast + explainability + attribution

This is the **modeling** layer built on top of the finished dataset
(`data/gnn_processed/`). Nothing in the data pipeline was changed. Everything
here reads the already-cleaned, already-scaled processed files and reports every
number through one scorer (`metrics.py`) as **skill vs persistence**.

> The dataset layers (preprocessing, feature engineering, ward dataset, dynamic
> graph) were already complete before this phase. See `PREPROCESSING.md` /
> `GNN_DATASET` history. This phase implements: **Graph Transformer → AQI
> forecast → SHAP + GNNExplainer → source attribution.** The RAG/LLM/counterfactual
> recommendation engine is explicitly *future work* and is not built.

## Files added this phase

| File | Role |
|---|---|
| `metrics.py` | The one scorer: RMSE/MAE + **% skill vs persistence**, pinball, coverage, point_id-grouped CV |
| `stgnn_data.py` | Model-facing loader for **`data/gnn_processed/`** (already-scaled). Reuses column/leak lists from `gnn_data.py` — one source of truth. Keeps `X_dyn` at `[T,19,F]` and gathers to 289 wards on the fly |
| `models/gnn_forecast.py` | **`WardGraphTransformer`** — GRU temporal encoder (over 19 cells) + wind-aware `TransformerConv` spatial attention + monotone quantile heads (p10/p50/p90) |
| `train_gnn.py` | Semi-supervised training (loss only at 49 labelled wards), optional **dense CAMS-grid pretraining**, batched block-diagonal graph forward, eval vs persistence |
| `explain_gnn.py` | **SHAP** (gradient feature attribution) + **GNNExplainer** (edge/feature masks) on the snapshot model |
| `models/attribution.py` | **Engine 2**: `SourceAttributor` — learned ratio + class heads plus rule-based directional signals → ranked qualitative source profile |
| `train_attribution.py` | Trains/validates the two LightGBM attribution heads |

Checkpoints land in `models/checkpoints/`; explainability CSVs in `data/explain/`.

## Results (all on the honest station target, chronological splits)

Baseline = station persistence ("tomorrow's AQI = today's"), **val RMSE 77.42 /
test 68.68**. Winter skill is read from **val** (test is summer-only), and the
headline is always *skill vs persistence*, never raw RMSE.

### Engine 1 — AQI forecast (horizon 24 h)

| Model | val skill | test skill | test RMSE | p10–p90 coverage |
|---|---|---|---|---|
| **Graph Transformer (temporal)** | ~ +7 % | **+20.8 %** | 54.4 | 0.59 |
| Graph Transformer (**no-history ablation**) | ~ +6 % | **+18.2 %** | 56.2 | 0.63 |

The **no-history ablation still beats persistence by +18 %** — the model is
learning pollution *physics and spatial transport*, not just autocorrelation.
That ablation model is also the explainable one (raw interpretable features).

### Engine 2 — source attribution

| Head | Metric | Result | Baseline |
|---|---|---|---|
| pm2.5/pm10 ratio (dust↔combustion) | MAE (test) | **0.127** | 0.177 (predict-mean) |
| dust / mixed / combustion class | macro-F1 (**val**) | **0.528** | 0.224 (majority) |

Output is a **ranked qualitative profile** (dust / traffic / biomass-burning /
industrial / secondary) with a confidence — never a mass-% apportionment (no
source labels exist; 6 pollutants ≠ speciation).

### Explainability

* **SHAP** (global + per ward-hour): top drivers are relative humidity,
  seasonality, stagnation, boundary-layer height, PM2.5 history, AOD/dust —
  physically sensible.
* **GNNExplainer**: for a target ward it ranks the neighbouring wards whose
  messages mattered. Example: ward 239's forecast is dominated by **Anand Vihar**
  (importance 0.68), a known Delhi hotspot — the graph learned a real transport link.

## Guardrails honoured

1. Never feed leak columns (targets, `*_raw`, persistence, ids) — enforced via
   `gnn_data.LEAK` reused by the loader.
2. Chronological splits only (`split` / `split_lab`); grid CV groups by `point_id`.
3. Report winter skill from **val**, summer from **test**; always vs persistence.
4. Engine 2 is qualitative only.
5. Target & persistence stay **raw AQI**; only inputs are scaled (by preprocessing).

## How to run

```bash
PY=/c/Users/mansi/AppData/Local/Programs/Python/Python310/python.exe   # env with torch+PyG

$PY stgnn_data.py                                   # loader sanity check
$PY train_gnn.py --horizon 24 --pretrain-epochs 12 --epochs 45          # temporal forecaster
$PY train_gnn.py --horizon 24 --pretrain-epochs 12 --epochs 45 --no-temporal   # ablation / explainable
$PY train_attribution.py                            # Engine 2 heads
$PY explain_gnn.py --ward 239 --time-index 24477    # SHAP + GNNExplainer for one ward
$PY explain_gnn.py --global-shap 300                # global feature importance
```

## Environment note (important)

The real working interpreter is **global Python 3.10**
(`C:/Users/mansi/AppData/Local/Programs/Python/Python310/python.exe`) — it has
`torch 2.13.0+cu126` (CUDA), scikit-learn, geopandas, networkx, plus the
`torch-geometric`, `shap`, `lightgbm` added this phase. The project `venv/` is
empty and is **not** used. `PLAN.md` referenced reference files
(`example_train_gnn.py`, `metrics.py`, `GNN_DATASET.md`) that never existed on
disk; the modeling layer here is the real implementation of that plan.

## Not built (future phase, by design)

Prediction store / FastAPI / web map / RAG+LLM advisor / counterfactual
simulation / recommendation ranking. The forecast + attribution outputs are
structured so those can be layered on without changing this pipeline.
