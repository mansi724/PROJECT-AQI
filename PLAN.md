# Urban Air Quality Intelligence — Build Plan (post-dataset)

End-to-end roadmap for turning the finished `data/gnn/` dataset into a working product:
**3 engines (forecast → attribution → advice) + a web app**. This is the plan only — nothing here
has been run yet.

---

## 0. Where we are

The **dataset phase is complete**. The trainable deliverable is `data/gnn/` (138 MB in RAM):

| File | Shape | What it is |
|---|---|---|
| `nodes_static.parquet` | 289 × 42 | Per-ward static features (real ward geometry) |
| `dynamic_grid.parquet` | 499,776 × 80 | Pollutants + weather + AQI + lags, per CAMS cell-hour (19 × 26,304) |
| `edges.parquet` | 1,670 × 6 | Wind-aware ward graph (queen contiguity + dist/bearing) |
| `labels_station.parquet` | 488,694 × 12 | **Real CPCB station AQI** → 49 wards (the honest target) |

- **`gnn_data.py`** — the only loader training code should import.
- **`feature_lists.py`** — column source-of-truth (which columns feed which engine + leak list).
- **`example_train_gnn.py`** — a **working reference GNN** (`WardGNN`, plain torch, wind-gated
  message passing, semi-supervised). We build Engine 1 by promoting this, not rewriting it.
- **Deprecated — do not touch:** `model_ready.parquet` (corrupted target), `train_model.py`
  (RandomForest on the old flat CSV).

---

## Non-negotiable guardrails

These come from `GNN_DATASET.md` / `feature_lists.py`. Every engine must respect them.

1. **Never feed leak columns** — `feature_lists.EXCLUDE_ALWAYS` / `gnn_data.LEAK` (targets, `*_raw`,
   `*_obs`, persistence, identifiers, string labels). The loader already filters these.
2. **Splits:** supervised train/eval on **`split_lab`** (labelled era). `split` (full 3-year) is
   **CAMS pretraining only**. Never random-split. Grid-level CV groups by `point_id` (19 real
   locations, not 289).
3. **Report winter skill from `val`, never `test`** — the test window is summer-only (0 % winter).
4. **Always report skill relative to persistence** (RMSE **86.03**), never raw RMSE. Bar to beat:
   GBDT all-features **73.98**. The GNN must beat that by exploiting the graph.
5. **Engine 2 = qualitative attribution only** — no quantitative apportionment ("industry = 34 %"),
   no PMF/CMB, no hourly industrial activity (EDGAR is annual). Validate against
   `pm25_pm10_ratio_obs` / `source_class_obs`, report from **val**.
6. **Engine 3 = the LLM cites, does not compute.** All numbers come from Engines 1/2.

---

## Phase 0 — Environment + shared harness

- **Freeze `requirements.txt`** (none exists today). Already in `venv`: torch, sklearn,
  pandas/pyarrow, geopandas. **Add:** `lightgbm`, `fastapi`, `uvicorn`, `sentence-transformers`,
  `chromadb` (or `faiss-cpu`). **Do NOT add torch-geometric** — the reference uses plain torch, and
  its `torch-scatter`/`torch-sparse` wheels are a painful Windows install.
- **`metrics.py`** — one scorer used by every result: RMSE/MAE **+ % skill vs persistence**, plus a
  `point_id`-grouped CV splitter. Nothing gets reported except through this.

**New files:** `requirements.txt`, `metrics.py`

---

## Phase 1 — Engine 1: Spatio-temporal GNN forecaster

Build on `example_train_gnn.py`.

### Keep from the reference (already correct)
- **Static join:** `x = cat[X_static, X_dyn[t][cell_of_node]]` — cheap gather, never materialise
  `[T, 289, F]`.
- **Wind gate:** rebuild wind angle per cell from `wind_dir_sin/cos`, align to edge `bearing`, gate
  messages by `(cos(bearing − wind_dir)+1)/2` so they flow **downwind**.
- **Semi-supervised loss** at the 49 labelled wards only; the other 240 predict through the graph.
- **Normalise on train hours only** (mu/sd from `split_lab=='train'` timesteps).
- **Horizon shift** `t_in = t_idx − horizon` so the label is 24 h ahead of the inputs.

### Upgrade to production (`AQIGraphNet`)
1. **Temporal encoder** — per-node GRU over a lookback window (`d.window(t, L)`, L≈24) before message
   passing. Keep a `--no-temporal` flag → snapshot mode for the **no-history ablation** (must still
   beat persistence — this is the "learns physics, not autocorrelation" evidence).
2. **Deeper spatial stack** — 2–3 wind-gated rounds; optional multi-head edge attention weighted by
   the wind gate.
3. **Quantile heads** — three outputs (0.1 / 0.5 / 0.9) trained with **pinball loss** (replaces the
   reference's `smooth_l1_loss`). Cheap, and Engine 3 needs the uncertainty band.
4. **Horizons** — parameterise `horizon ∈ {24, 48, 72}`.

### Training (`train_gnn.py`)
- **Optional pretrain** on `y_grid` (dense weak CAMS target, full-range `split`) → warm start, then
  **fine-tune on `labels_station`** (`split_lab`).
- Adam (lr 3e-3 → cosine decay), grad clip, **full epochs over all train hours** (drop the
  reference's 400-hour/epoch sampling cap for the real run), early-stop on **val pinball + station
  RMSE**. Save best checkpoint + scaler stats + feature list + config to
  `models/checkpoints/gnn_forecast_h24.pt`.

### Evaluation
Station-ward RMSE/MAE on **val (winter)** and **test (summer)** through `metrics.py`: skill vs
persistence (86.03), beat GBDT (73.98). No-history ablation still beats persistence. Quantile
calibration: 0.1–0.9 band coverage ≈ 80 %. Confirm all-289-ward inference shows real hyperlocal
spread.

**New files:** `models/gnn_forecast.py`, `train_gnn.py`, `models/checkpoints/`

---

## Phase 2 — Engine 2: Source attribution

Use **`ATTRIBUTION_DYN` + `ATTRIBUTION_STATIC` from `gnn_data.py`** (the GNN-dataset column names).
Note: `feature_lists.attribution_features` targets the *deprecated flat table* — some of its columns
(`traffic_load`, `industry_upwind`, …) don't exist in `data/gnn/`; prefer the `gnn_data.py` lists.

Two validated LightGBM heads on `split_lab`:
- **Dust-vs-combustion regressor** → `pm25_pm10_ratio` (label `pm25_pm10_ratio_obs`). Target MAE
  ≈ 0.124 vs predict-mean 0.177.
- **Source-class classifier** → dust / mixed / combustion (`source_class_obs`). Report macro-F1 from
  **val** (beats majority ~+9 pts; test is degenerate summer-dust).

Plus **rule-based directional signals**: `fire_upwind` (stubble transport), wind-gated regional
transport via `cos(bearing − wind_dir)`, NO₂ + `road_capacity_3km` (traffic), EDGAR `emis_*` shares
as **prior context only**. Output per ward-hour: a **ranked qualitative source profile with
confidence** — never a percentage apportionment.

**New files:** `models/attribution.py`, `train_attribution.py`

---

## Phase 3 — Prediction store (glue between models and app)

Run Engine 1 + Engine 2 inference over the recent/eval window for all 289 wards →
`data/serving/predictions.parquet` (ward_id, time, horizon, aqi_p10/p50/p90, dominant_pollutant,
source_profile). The API serves from this store for a fast, deterministic demo (live inference is an
optional endpoint). Also emit `data/serving/wards.geojson` (polygons + centroids) for the map.

**New files:** `build_predictions.py`

---

## Phase 4 — Engine 3: RAG + LLM health/action advisor

- **`rag/llm_client.py`** — thin `LLMClient` interface `generate(system, prompt) -> str` with
  adapters for Anthropic / Ollama / OpenAI selected by env var (provider chosen at implementation
  time). Embeddings: `sentence-transformers` (local, free).
- **`rag/corpus/`** — curated markdown: CPCB **GRAP** action tiers, AQI health-advisory bands,
  pollutant-specific guidance (PM2.5, O₃, NO₂), vulnerable-group advice. Chunk + embed →
  `chromadb`/`faiss` (`rag/index.py`).
- **`rag/advisor.py`** — inputs: ward identity (`ward_name`, `ward_lat/lon`), predicted AQI band +
  quantiles (Engine 1), dominant pollutant/source (Engine 2), vulnerability
  (`population_density_mean`, `vulnerable_sites_3km`). Retrieve top-k on
  **(AQI band × dominant pollutant × vulnerability)**, build a grounded prompt, LLM composes per-ward
  guidance. **Numbers are injected, not generated.**

**New files:** `rag/llm_client.py`, `rag/index.py`, `rag/advisor.py`, `rag/corpus/`

---

## Phase 5 — Serving: FastAPI + web map

**API** (`api/main.py`) — reads the prediction store; loads the GNN lazily for live endpoints:

| Endpoint | Returns |
|---|---|
| `GET /wards` | `wards.geojson` |
| `GET /map?horizon=24` | all 289 wards' `aqi_p50` for choropleth coloring |
| `GET /forecast?ward_id=&horizon=` | p10/p50/p90 band + recent history |
| `GET /attribution?ward_id=` | ranked source profile + signals |
| `POST /advice` | RAG advice for a ward (Engine 3) |

**Frontend** (`web/`) — a **Leaflet choropleth of Delhi's 289 wards** colored by predicted AQI (CPCB
bands). Click a ward → side panel: (a) forecast chart + uncertainty band, (b) source-attribution
bars, (c) LLM health advice. Single-page, dependency-light (vanilla JS + Leaflet, or a small React
app).

**New files:** `api/main.py`, `web/`

---

## Phase 6 — Demo + write-up

README / `run_demo.md`: how to launch API + frontend, headline metrics, and the **honest-limitations**
list (19 CAMS cells, annual emissions, 49/289 wards labelled, summer-only test). Keep the no-history
ablation and persistence-relative framing front and center for Technical-score judges.

---

## File map

| Purpose | File | New/Existing |
|---|---|---|
| Data loader (import only) | `gnn_data.py` | existing — reuse as-is |
| Column source-of-truth | `feature_lists.py` | existing |
| Reference GNN to build on | `example_train_gnn.py` | existing — promote |
| Metrics harness | `metrics.py` | new |
| GNN model | `models/gnn_forecast.py` | new |
| GNN trainer/eval | `train_gnn.py` | new |
| Attribution | `models/attribution.py`, `train_attribution.py` | new |
| Prediction store | `build_predictions.py` | new |
| RAG advisor | `rag/llm_client.py`, `rag/index.py`, `rag/advisor.py`, `rag/corpus/` | new |
| API | `api/main.py` | new |
| Frontend | `web/` | new |
| Deprecated (ignore) | `train_model.py`, `model_ready.parquet` | — |

---

## Build order

`metrics.py` → Engine 1 (model → train → eval) → Engine 2 → `build_predictions.py` → Engine 3 RAG →
API → frontend → demo/write-up. Engines 1 and 2 share `gnn_data.py` + `metrics.py`, so the harness
comes first.

## How to verify each phase

1. `python gnn_data.py` — loader shapes/splits/labels print correctly.
2. `python example_train_gnn.py` — reference runs (sanity before refactor).
3. `python train_gnn.py` — converges; **val RMSE beats 73.98**; no-history ablation beats 86.03;
   band coverage ≈ 80 %.
4. `python train_attribution.py` — ratio MAE < 0.177; class macro-F1 (val) > majority.
5. `python build_predictions.py` — writes `predictions.parquet` + `wards.geojson`.
6. `uvicorn api.main:app` — `/map`, `/forecast`, `/attribution`, `/advice` respond; a known
   station-ward's forecast matches the eval number; `/advice` cites corpus text with no invented
   figures.
7. Open `web/` — map renders; ward click populates forecast band + sources + advice.
