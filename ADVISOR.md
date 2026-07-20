# Delhi AQI Advisor — Post-Attribution Intelligence Pipeline

Everything **after Source Attribution** in the platform: it turns the frozen
forecasting stack's outputs into **policy-grounded, validated, ranked action
recommendations** with counterfactual "what-if" simulation, served through a
FastAPI backend and an interactive dashboard.

```
AQI Forecast → SHAP + GNNExplainer → Source Attribution        (frozen pipeline)
        ↓
Context Builder → Hybrid Retrieval (+ Knowledge Graph) → Cross-Encoder Rerank
        ↓
LLM Reasoning → Policy Validation → Counterfactual Simulation → Action Ranking → Dashboard
```

The forecasting stack (Graph Transformer, SHAP, GNNExplainer, LightGBM
attribution) is **reused, never retrained**. The single seam is
`advisor/serving.py` (`ForecastService`).

> Interpreter: this project runs on the **global Python 3.10**
> (`C:/Users/mansi/AppData/Local/Programs/Python/Python310/python.exe`), which
> has the CUDA Torch build. The empty project `venv/` is not used. Substitute
> `python` below with that interpreter (or activate an env with `requirements.txt`).

---

## Module map (`advisor/`)

| Part | Module | What it does |
|---|---|---|
| — | `config.py` | Central, env-overridable config (paths, models, weights, intervention maps). No hardcoded paths. |
| — | `feature_space.py` | Raw↔scaled transforms (reuses `scalers.joblib`) + raw display values. |
| — | `serving.py` | **Reuse layer**: predict / SHAP / GNNExplainer / attribution / counterfactual. |
| 1 | `context_builder.py` | Forecast outputs → structured `WardContext` JSON. |
| 2/3 | `kb/schema.py`, `kb/ingest.py`, `data/kb/corpus/` | Metadata schema + curated corpus + multi-format ingestion (PDF/DOCX/HTML/MD, OCR fallback, chunk, dedup, citations). |
| 4 | `embeddings/embedder.py`, `embeddings/vector_store.py` | Sentence-Transformers embeddings + ChromaDB (incremental indexing, doc updates). |
| 7 | `kg/knowledge_graph.py` | NetworkX graph (sources, pollutants, stages, interventions, policies) + traversal. |
| 5 | `retrieval/bm25.py`, `retrieval/hybrid.py` | Hybrid retrieval = semantic + BM25 + metadata + knowledge-graph. |
| 6 | `retrieval/reranker.py` | Cross-encoder reranker (ms-marco / bge). |
| 8 | `llm/client.py`, `llm/reasoner.py` | Provider-agnostic **cloud** LLM (Groq/Anthropic/Gemini) + grounded, structured-JSON reasoning. Offline mock fallback. |
| 9 | `validation/policy_validator.py` | Rule validation of every LLM action (stage/city/season/source/authority/conflicts). |
| 10 | `simulation/counterfactual.py` | What-if simulation by editing features + re-running the frozen GNN. |
| 11 | `ranking/action_ranker.py` | Multi-objective ranking (improvement, confidence, policy strength, feasibility, cost, time). |
| — | `pipeline.py` | End-to-end orchestrator (`advise(ward_id)`). |
| 12 | `api/main.py`, `dashboard/index.html` | FastAPI backend + single-page Leaflet/Plotly dashboard. |

Every module runs standalone (`python -m advisor.<module>`) with a self-test.

---

## Setup

```bash
pip install -r requirements.txt
```

### Configure the LLM (cloud only — no Ollama/local)
Add to `.env` (default provider is **Groq**, model **gemma2-9b-it**):
```
GROQ_API_KEY=your_groq_key
# optional overrides:
# ADVISOR_LLM_PROVIDER=groq        # groq | anthropic | gemini | mock
# ADVISOR_LLM_MODEL=gemma2-9b-it
# ANTHROPIC_API_KEY=...            # if provider=anthropic
# GEMINI_API_KEY=...               # if provider=gemini
```
Without a key the pipeline still runs end-to-end using a **deterministic offline
reasoner** (clearly flagged `provider: mock`) so nothing is blocked.

---

## Build the knowledge base (Parts 2–4, 7)

```bash
# 1. Ingest documents -> chunks (curated corpus + anything you drop in data/kb/raw_docs/)
python -m advisor.kb.ingest

# 2. Embed + build the ChromaDB vector index (incremental; --reset to rebuild)
python -m advisor.embeddings.vector_store --reset

# 3. Build the knowledge graph
python -m advisor.kg.knowledge_graph
```

### Adding your own policy documents
Drop PDFs / DOCX / HTML / TXT into `data/kb/raw_docs/`. Optionally add a sidecar
`<file>.meta.json` to set metadata explicitly, e.g.:
```json
{"title":"CAQM GRAP Revised Schedule","authority":"CAQM","document_type":"grap",
 "aqi_stage":"Stage III","source_category":"action_plan","effective_date":"2022-08-05",
 "pollutant":["PM2.5","PM10"],"tags":["construction","diesel"]}
```
Then re-run steps 1–3. Indexing is incremental; `update_document(doc_id, chunks)`
replaces a single document's chunks.

---

## Run retrieval / reasoning / simulation standalone

```bash
python -m advisor.context_builder                 # Part 1 structured JSON
python -m advisor.retrieval.hybrid                # Part 5 hybrid retrieval
python -m advisor.retrieval.reranker              # Part 6 reranking
python -m advisor.llm.reasoner                     # Part 8 grounded reasoning
python -m advisor.validation.policy_validator     # Part 9 validation
python -m advisor.simulation.counterfactual       # Part 10 what-if
python -m advisor.ranking.action_ranker           # Part 11 ranking
python -m advisor.pipeline                         # full end-to-end for one ward
```

---

## Run the backend + dashboard (Part 12)

```bash
python -m uvicorn advisor.api.main:app --host 127.0.0.1 --port 8000
# open http://localhost:8000
```

### API endpoints
| Endpoint | Returns |
|---|---|
| `GET /health` | status + configured LLM provider |
| `GET /wards` | all 289 wards (id, name, lat/lon) |
| `GET /map?time_index=` | every ward's predicted AQI (one graph forward) for the choropleth |
| `GET /layers?time_index=` | per-ward multi-metric snapshot (AQI, current, forecast-diff, PM2.5/PM10/NO₂, traffic, industry, wind) for all map layers |
| `GET /layers/source?time_index=` | per-ward dominant source class (Source Attribution layer) |
| `POST /simulate` | Scenario Builder — 7 policy sliders (0–100%) → composed counterfactual, before/after AQI + source split + recommendation |
| `GET /forecast?ward_id=&time_index=` | p10/p50/p90 + 48h history |
| `GET /context?ward_id=&explain=` | the structured `WardContext` (Part 1) |
| `GET /advise?ward_id=&explain=` | **full pipeline**: context + policies + reasoning + validation + counterfactuals + ranked actions |
| `POST /counterfactual` | `{ward_id, actions[], intensity}` or `{feature_multipliers}` → before/after AQI |

The dashboard shows the Delhi ward map (colored by predicted AQI), and per ward:
forecast + uncertainty band, source-attribution bars, ranked recommended actions
with counterfactual before/after AQI and policy citations, retrieved policies, and
an on-demand SHAP + GNNExplainer panel.

---

## Complete end-to-end demo

```bash
pip install -r requirements.txt
python -m advisor.kb.ingest
python -m advisor.embeddings.vector_store --reset
python -m advisor.kg.knowledge_graph
python -m uvicorn advisor.api.main:app --port 8000
# open http://localhost:8000 and click a ward
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `provider: mock` in `/advise` | No LLM key. Add `GROQ_API_KEY` to `.env`. The pipeline still works without it. |
| `[embedder] ... unavailable ... hashing fallback` | First run needs internet to download the MiniLM model; re-run once online, or set `ADVISOR_EMBED_MODEL` to a locally-cached model. |
| Reranker `passthrough` | Cross-encoder model not downloaded; same as above. |
| `FileNotFoundError: gnn_forecast_h24_notemporal.pt` | Train the explainable snapshot model first: `python train_gnn.py --horizon 24 --no-temporal --pretrain-epochs 12 --epochs 45`. |
| OCR skipped for a scanned PDF | Install system `tesseract` + `poppler` and `pip install pdf2image`. Text PDFs need nothing extra. |
| Chroma "collection exists" / stale chunks | `python -m advisor.embeddings.vector_store --reset` to rebuild. |
| Windows console `UnicodeEncodeError` | Cosmetic (arrows/em-dashes); set `PYTHONUTF8=1` or ignore — files/JSON are unaffected. |

---

## What is NOT included (by design)

The recommendation **content** depends on a live LLM key for full quality; the
offline mock is intentionally conservative (only KG-mapped interventions). The
curated corpus is a factual starting set — add the official GRAP/CAQM/CPCB PDFs to
`data/kb/raw_docs/` for authoritative citations. No forecasting model was modified.
