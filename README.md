# Urban Air Quality Intelligence — Delhi

Hyperlocal AQI for **289 Delhi wards**, hourly. One dataset feeds three engines:
**forecasting**, **source attribution**, and **action recommendation (RAG + LLM)**.

**→ The dataset you train on is [`data/gnn/`](data/gnn/). Load it with [`gnn_data.py`](gnn_data.py). Nothing else.**

**Build docs:**
- **[`MODELS.md`](MODELS.md)** — the modeling layer: Graph Transformer forecaster (+22.3% test skill, ensemble), SHAP + GNNExplainer, source attribution. Model-facing loader: [`stgnn_data.py`](stgnn_data.py).
- **[`ADVISOR.md`](ADVISOR.md)** — everything after attribution: Context Builder → Hybrid Retrieval + Knowledge Graph → Cross-Encoder Rerank → LLM Reasoning → Policy Validation → Counterfactual Simulation → Action Ranking → FastAPI + Dashboard (the [`advisor/`](advisor/) package).
- **[`IMPROVEMENTS.md`](IMPROVEMENTS.md)** — prioritized roadmap of everything that can be improved (data, model, RAG, LLM, UI, ops) + the manual-setup checklist (keys, docs, downloads).

---

## 1. What this dataset actually is

Start with the honest picture, because it drives every design decision here.

Free hyperlocal pollutant data does not exist. The pollutant and weather data comes from
**CAMS, an atmospheric model with only 19 grid cells over Delhi**. So at any hour, the raw
inputs have ~4 distinct PM2.5 values for the whole city — not 289. Rain has exactly **1**.

You cannot fake your way out of that. What you *can* do is combine:

| Layer | Resolution | Real? |
|---|---|---|
| Pollutants + weather (CAMS) | 19 cells | Simulated |
| Roads, industry, population, land use | **289 wards** | Real |
| CPCB station measurements | 57 stations → 49 wards | **Real, measured** |
| Ward adjacency graph | 289 nodes, 1,670 edges | Real geometry |

The job of the model is to **learn how ward context bends the coarse regional signal**, using
real station measurements as the answer key. That's a published approach (land-use regression
/ data fusion), and it's why this is a graph problem: 49 wards have ground truth, and the
graph carries it to the other 240.

An earlier version of this dataset invented per-ward variation with a formula. It was circular —
the formula's inputs were also model inputs, so the model just re-derived it. That is deleted.
Real variation now comes from real ward features + the graph + real labels.

---

## 2. The four files

| File | Answers | Shape | Key |
|---|---|---|---|
| `nodes_static.parquet` | **Who** are the wards? | 289 × 42 | `node_idx`, `point_id` |
| `dynamic_grid.parquet` | **What's the air doing** each hour? | 499,776 × 80 | `point_id`, `time` |
| `edges.parquet` | **Who's next to whom?** | 1,670 × 6 | `src`, `dst` |
| `labels_station.parquet` | **What's true**, where we can check | 488,694 × 12 | `node_idx`, `time` |

**47 MB on disk, 138 MB in RAM.** The old flat table was 776 MB / 5.3 GB.

Why so much smaller? Because the old file stored 7,601,856 ward-hour rows when the dynamic data
only has **499,776 unique cell-hours** — every ward in a cell had a byte-identical copy. We
verified all 19 dynamic columns are constant within a `(point_id, time)`, so storing the grid
once and joining at load time loses **nothing**. The loader does:

```python
X_dyn[t][cell_of_node]     # [289, F] — a gather, effectively free
```

Never build `[T, 289, F]` yourself. That's 1.8 GB of duplicated floats — the exact mistake the
old file made.

---

## 3. Quick start

```python
from gnn_data import load_gnn

d = load_gnn(horizon=24)

x = d.node_features(t=1000)        # [289, 84]  static ++ dynamic, joined for you
y = d.targets(t=1000)              # [289]      CAMS target (weak — see §5)
seq = d.window(1000, lookback=24)  # [24, 289, 84] for a sequence model

d.edge_index                       # (2, 1670) ready for PyTorch Geometric
d.labels                           # the real CPCB answers
```

A complete, working training run is in **[`example_train_gnn.py`](example_train_gnn.py)** —
plain torch, no extra install:

```bash
python example_train_gnn.py
```

It predicts real station AQI 24 h ahead, computes loss only at the 49 labelled wards, and
outputs **289 distinct ward predictions**. That's the deliverable in one file: read it first.

```
labels: 438,441 ward-hours over 49 wards (17% of nodes)
  epoch 1  val RMSE  70.22
TEST  RMSE 53.99  MAE 37.62
prediction for all 289 wards: min 118 max 217 (289 distinct)
```

---

## 4. How the three engines work

### Engine 1 — Forecasting
`d.node_features(t)` → GNN → AQI at t+24/48/72 for every ward.
Pretrain on the dense CAMS target, fine-tune on real labels.
**Beat persistence: RMSE 86.03.** A plain GBDT gets 73.98 — that's the bar a GNN must clear.

### Engine 2 — Source attribution
`ATTRIBUTION_DYN` + `ATTRIBUTION_STATIC` → dust-vs-combustion signature per ward.

The weakest engine, but **validated against instruments**: `pm25_pm10_ratio_obs` is a *measured*
source fingerprint at 58 stations (474,847 hours). It flips exactly as physics predicts —
**0.62 in December** (combustion) vs **0.36 in April** (dust). So predicting it is a **scoreable**
task, not a heuristic: ratio regression MAE **0.124** vs 0.177 baseline; source-class acc
**0.601 vs 0.507** majority on `val`.

⚠️ **Report this engine from `val`, not `test`** — the test window is summer and therefore 67.5%
dust, so "always say dust" wins there. Wind-gated edges (`cos(bearing − wind_direction)`) carry
the transport story.

**Never claim** *"industry = 34% of PM2.5"*. No source labels exist and 6 pollutants isn't
chemical speciation, so PMF/receptor modelling is impossible. `emis_*` is annual — zero temporal
signal. Construction is absent entirely.

### Engine 3 — Action recommendation (RAG + LLM)
The dataset supplies **retrieval context**, not the LLM:
- *Who's at risk*: `population_sum`, `population_density_mean`, `vulnerable_sites_3km` (real
  per-ward now — 154 distinct values, was 17)
- *What's coming*: predicted AQI + `dominant_pollutant` from Engines 1–2
- *Where*: `ward_name`, `ward_lat/lon` to ground the retrieved text

You supply the corpus (CPCB/GRAP action tiers, health advisories). Retrieve on
(AQI band × dominant pollutant × vulnerability); let the LLM compose the advice.
**Keep the LLM out of the arithmetic** — it should cite, not compute.

---

## 5. Two rules that will save your results

**1. Use the right split.**

| Column | Where | For |
|---|---|---|
| `split` | `dynamic_grid` | CAMS pretraining only |
| `split_lab` | `labels_station` | **All real-label training/eval** |

Real labels only start **Feb 2025** (OpenAQ serves no earlier hourly data for these sensors),
so they cover ~17 months, not 3 years. Using the full-range `split` puts **5.9 % winter in
train but 53 % in val** — training on clean air, validating on smog. `split_lab` fixes this
(train 32 % winter).

⚠️ The labelled era has **one winter**, so `split_lab` test is summer-only (**0 % winter**).
**You cannot claim winter skill from the test set** — report it from `val`. Delhi is
winter-dominated (Nov mean AQI 368 vs Jul 77), so this is not a technicality.

**2. Never split randomly, never trust the row count.**
Adjacent hours are near-duplicates and wards in a cell share inputs exactly — a random split
leaks and the score is fiction. For CV on grid-derived features, **group by `point_id`**: you
have **19 independent locations**, not 289, and not 7.6 M.

---

## 6. What changed (and what to throw away)

| Fixed | Was |
|---|---|
| **AQI target** | `rolling(24)` spanned 24 *rows*, not hours → one cell's AQI smeared **201→403** across its wards by sort order. Now verified correct (310 vs independent 309.6). |
| **Ward features** | Roads/industry copied from nearest grid point → `road_km_3km` had **19** distinct values. Recomputed on ward geometry → **289**. |
| **Ground truth** | 6,453 rows → **488,694** (89×). The pull is now checkpointed and resumable. |
| **Size** | 5.3 GB / 93 % duplicated → 138 MB, nothing lost. |
| **Junk ward** | A null-named geojson feature injected 26,304 null rows. Dropped. |

**Do not use `data/final/model_ready.parquet` — it is still corrupt.** The code that made it is
fixed, but the file was never regenerated. `model_ready_backup.parquet` must stay (everything
rebuilds from it) but is not for training. `feature_lists.py` / `FEATURES.md` describe the *old*
table — `gnn_data.py` is current.

---

## 7. Rebuilding

```bash
python build_ward_static.py             # nodes + edges          ~1 min
python build_gnn_dataset.py             # dynamic grid           ~2 min
python refresh_ground_truth.py          # CPCB pull (resumable)  ~75 min
python build_gnn_dataset.py --labels    # station labels
python gnn_data.py                      # smoke-test
python example_train_gnn.py             # end-to-end demo
```

`refresh_ground_truth.py` checkpoints every sensor — re-running skips cached ones, and
`--assemble` rebuilds the CSV from cache without re-pulling.

<details>
<summary><b>Raw download pipeline (only if starting from scratch)</b></summary>

```bash
pip install requests cdsapi
python download.py              # everything
python download.py aqi weather  # selected
```

Free API keys (env vars, or edit `config.py`):

| Variable | From | For |
|---|---|---|
| `OPENAQ_API_KEY` | explore.openaq.org/register | CPCB stations |
| `CDSAPI_KEY` | cds.climate.copernicus.eu | ERA5 weather |
| `FIRMS_MAP_KEY` | firms.modaps.eosdis.nasa.gov/api/map_key/ | Fire |
| `EARTHDATA_TOKEN` | urs.earthdata.nasa.gov | MODIS AOD |

Roads, industries, land use, DEM, population need no key. Missing keys skip that dataset only.
Re-running is safe — existing files are skipped. `data/realtime/` is never written to.
Sources/licenses: `datasets.json`. Config (bbox, dates, retries): `config.py`.
</details>

---

## 8. Limits to state out loud

Judges reward honesty here far more than a hidden weakness.

1. **Pollutants/weather are 19 CAMS cells**, broadcast to wards. Real per-ward signal comes from
   static features + graph + labels.
2. **~19 effective spatial samples** for grid-derived features.
3. **Only 49/289 wards (17 %) have a real label.** The rest are inferred — validated on 49,
   asserted on 240.
4. **Emissions are annual** (EDGAR v8.1 FT2022) — no hourly industrial activity.
5. **No construction, real traffic counts, or satellite-imagery encoders.** The old
   `traffic_load` was `road_capacity × is_rush_hour` (20 distinct values) — not carried over.
6. **Bias correction is still 12 hand-tuned monthly scalars.** With 488k real labels, the model
   should now learn this itself — **the highest-value next step.**
7. **No uncertainty yet.** Quantile heads (0.1/0.5/0.9) are cheap and matter more for a health
   product than a marginal RMSE gain.

---

## 9. Docs

| Doc | Read it for |
|---|---|
| **This file** | Orientation — start here |
| [GNN_DATASET.md](GNN_DATASET.md) | Full spec: schemas, splits, engines, measured baselines |
| [DATASET_ISSUES.md](DATASET_ISSUES.md) | Every known problem + fix, with evidence |
| [example_train_gnn.py](example_train_gnn.py) | Working code — the fastest way to understand it |
| [METHODOLOGY.md](METHODOLOGY.md), [CORRECTIONS.md](CORRECTIONS.md) | Older notes — **predate the fixes; treat as history** |
