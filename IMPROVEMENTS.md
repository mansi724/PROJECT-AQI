# Improvement Roadmap — Delhi AQI Intelligence Platform

A single backlog of **everything that can be made better**, across data, the
forecasting model, explainability, attribution, RAG, the LLM, validation,
counterfactuals, the dashboard, the backend, and ops — plus the **manual steps
only you can do** (API keys, downloads, official documents).

Nothing here is required for the platform to run today; it already works
end-to-end. This is the map for taking it from "working hackathon build" to
"production decision-support system."

### How to read this
- **Priority** — `P0` do first (biggest impact / unblocks other work) · `P1` high value · `P2` nice-to-have / polish.
- **Effort** — `S` hours · `M` a day or two · `L` multi-day.
- **🔑 Manual** — needs something only you can provide (a key, a paid tier, an official document, a decision).
- **File** — where the change goes, so you can jump straight in.

---

## 0. The single highest-leverage fix (do this first)

| # | Item | Pri | Effort | Status |
|---|---|---|---|---|
| 0.1 | ~~Recover the full CPCB ground-truth history~~ | — | — | ✅ **ALREADY DONE** (verified 2026-07-20) |

> **UPDATE — 0.1 is complete; do NOT re-download.** The CPCB re-pull was already
> executed (via `refresh_ground_truth.py`): `cpcb_ground_truth.csv` now holds
> **577,175 station-hours across 64 stations (2023→2026)** — a ~160× increase over
> the old sparse ~3,588 hours — and `labels_station` was rebuilt from it (**49 wards,
> 488,694 label-hours, 2025-02 → 2026-07**). The sparsity *bug* is fixed and there is
> nothing left to recover from OpenAQ (data only exists densely from 2025; median
> ~11,000 h/station). **The "49 wards" is the physical CPCB monitor count, not a bug**
> — re-downloading cannot raise it. To get more labelled wards, add other sensor
> networks (see 1.2). This entry replaces the old, now-stale `cpcb-download-bug` note.

### Data reality check — which limits are actually fixable

The download/sparsity **bug is fixed**. The remaining items are structural limits of
*free* data — not bugs. Keep this table for planning; none of them block the model,
serving, RAG, LLM, or UI work below.

| Limitation | Fixable? | How | Effort / cost |
|---|---|---|---|
| **~49 wards labelled** | ✅ Yes | Add **other sensor networks** — PurpleAir (free API), AQI.in, DPCC/IMD stations, low-cost sensors. Each network = more labelled wards. *Most tractable data upgrade.* | Medium — real data-acquisition; mostly free, some scraping/permission. |
| **Summer-only test window** | ✅ Yes | Evaluation choice, not a data flaw. Use season-stratified CV, or wait as more winters accumulate. | Low — code/design. |
| **19 CAMS grid cells** (pollutant/weather has only 19 distinct locations broadcast to 289 wards) | 🟡 Partial | Add **satellite** (Sentinel-5P NO₂, MODIS/VIIRS aerosol — free, ~1–7 km) for real sub-cell texture, or a finer reanalysis. GNN + land-use already partly compensate. **Biggest structural limit.** | High — a sub-project; satellite has cloud gaps. |
| **Annual EDGAR emissions** (static yearly, not hourly) | 🟡 Hard | Hourly inventories are rare/paid; best you can do is proxy activity (traffic/industrial indices), already approximated directionally. | High / often not worth it. |
| Corrupted `model_ready.parquet` target | N/A | Deprecated file, **not used** by the GNN or advisor — no effect. Delete/retire (see 1.5). | Trivial. |

**Takeaway:** the data is at the realistic ceiling of free sources. The one clearly
worth-it data upgrade is **more monitors (1.2)**; everything else is either partial,
expensive, or best handled as an honest limitations slide in the demo.

---

## 1. Data & ground truth

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 1.1 | CPCB re-pull ✅ done (0.1). Still worth: **re-anchor the bias-correction** on the fuller 577k-hour record (the PM factors were set on the old sparse data) | P1 | M | `apply_corrections.py` |
| 1.2 | Add **more real monitors** (IMD, DPCC low-cost sensors, PurpleAir) to raise labelled-ward coverage above 49 | P1 | L | 🔑 new source adapters in `download.py` |
| 1.3 | **Native hyperlocal pollutants** — the pollutant grid is only **19 CAMS cells** broadcast to wards. If you can get satellite TROPOMI/NO₂ or a finer reanalysis, wards stop sharing identical dynamics | P1 | L | 🔑 `build_dataset.py` |
| 1.4 | Populate **EDGAR hourly/sectoral** emissions if a finer product exists (currently annual, static) | P2 | M | `build_emissions.py` |
| 1.5 | Regenerate `model_ready.parquet` or fully retire it — it still carries the old corrupted target (`aqi-rolling-window-bug`); keep only `data/gnn/` + `data/gnn_processed/` | P1 | S | `apply_corrections.py` |
| 1.6 | Add a **data-freshness / drift monitor** (schema + range checks) run before every training/serving cycle | P2 | M | new `data_checks.py` |
| 1.7 | **Real-time ingestion path** — a scheduled job pulling the last 24 h of CAMS + weather so the dashboard shows *live* forecasts, not a fixed historical window | P1 | L | 🔑 new `realtime_update.py` + cron |

---

## 2. Forecasting model (Graph Transformer)

**Now serving:** the **4-model ensemble** (test skill **+23.6%** / val **+10.4%**, coverage 0.65)
is persisted (`gnn_forecast_h24_notemporal_ens{0..3}.pt`) and `ForecastService` averages
it for every forecast, map layer, and counterfactual. Explainability still uses the single
interpretable model.

> ### ⚠️ Execution order — do this section **recipe-first, ensemble-last**
> Most items below **change the model and therefore require retraining** (🔁). If you
> train the ensemble, then later change the architecture/hyperparameters, the ensemble
> weights are thrown out and must be retrained. So follow this order to avoid wasted runs:
>
> 1. **Settle the recipe on a SINGLE model** — 2.4 (hyperparameter search) → 2.5 (encoder) →
>    2.3 (loss/dropout part). Cheap single runs (~2–4 min each).
> 2. **2.2** — train h48 / h72 with the winning recipe.
> 3. **2.1** — retrain the ensemble of the winning recipe as the **final** serving model.
>    *(The save + serving-average plumbing is already built — only the weights refresh,
>    one command: `python train_ensemble.py --k 4 --no-temporal --pretrain-epochs 10 --epochs 40 --aux-grid-weight 0`.)*
> 4. **Post-hoc, no retraining** — 2.3 conformal wrapper, 2.10 CRPS reporting.
>
> Legend: 🔁 = needs retraining · 🩹 = post-hoc, no retraining.

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 2.1 | ✅ **DONE — ensemble persisted & serving** (avg of 4 snapshot members; test +23.6% / val +10.4%). Re-run the one command after any recipe change to refresh. | P0 | — | 🔁 `train_ensemble.py`, `advisor/serving.py` |
| 2.4 | ✅ **DONE (2026-07-20)** — 12-config sweep (`sweep_gnn.py`). Finding: base recipe already well-tuned; no dramatic winner. **Adopted `dropout 0.3`** (0.2→0.3) — beats base on val skill, test skill, and calibration (coverage 0.63→0.74). Now the default in `train_gnn.py`/`train_ensemble.py`; `best_config.json` records it. *(Single-seed → margins are noisy; the calibration gain is the reliable part.)* | P1 | — | `sweep_gnn.py`, `train_gnn.py` |
| 2.5 | 🔁 **Better temporal encoder** — replace/augment the GRU with a small Temporal-Transformer or TCN; add attention over the lookback window | P1 | M | `models/gnn_forecast.py` |
| 2.3 | 🔁/🩹 **Fix winter calibration** — coverage 0.65 vs target 0.80. Training part: heavier dropout, quantile-reg. **Post-hoc part: conformal-prediction wrapper on residuals (no retrain).** | P1 | M | `train_gnn.py`, `models/gnn_forecast.py` |
| 2.2 | 🔁 **Train the 48h & 72h horizons** (with the settled recipe) — only h24 exists, so the dashboard's 48/72h tabs currently reuse the 24h number. `train_gnn.py --horizon 48/72` already works | P1 | S | `train_gnn.py` |
| 2.7 | 🔁 **Physics-informed features/loss** — advection term from wind + boundary-layer height as a soft constraint | P2 | L | `models/gnn_forecast.py` |
| 2.9 | 🔁 **Online / incremental fine-tuning** as new station data arrives (warm-start from the last checkpoint) | P2 | M | `train_gnn.py` |
| 2.8 | 🩹 **Probabilistic upgrade / CRPS** — you already have quantiles; report the full predictive distribution | P2 | M | `metrics.py` |
| 2.10 | 🩹 **Report CRPS + PIT histograms** (no retrain — re-scores the existing model), for forecast-quality credibility | P2 | S | `metrics.py` |

---

## 3. Explainability (SHAP + GNNExplainer)

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 3.1 | **Cache / precompute explanations** — SHAP+GNNExplainer take ~5 s per ward. Precompute for all wards nightly into `data/serving/` so the dashboard is instant | P1 | M | `explain_gnn.py`, new `build_predictions.py` |
| 3.2 | **Global SHAP summary** in the UI (beeswarm / mean-|SHAP| across wards), not only per-ward | P2 | S | `explain_gnn.py` (`shap_global` exists) |
| 3.3 | **Temporal attribution** — which past hours drove the forecast (attribute over the GRU lookback) | P2 | M | `explain_gnn.py` |
| 3.4 | **Counterfactual explanations** — "AQI would drop below 300 if NO₂ fell 20%" auto-generated from the simulator | P2 | M | ties `explain_gnn.py` + `simulation/` |
| 3.5 | **Stability check** — run GNNExplainer with several seeds and report edge-importance variance (trust signal) | P2 | S | `explain_gnn.py` |

---

## 4. Source attribution

Current: ratio head test MAE 0.127 (vs 0.177), class head val macro-F1 0.528 (vs 0.224).

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 4.1 | **Add species that enable real apportionment** (VOC/black-carbon/K⁺ tracers) if obtainable → move from qualitative to defensible quantitative shares | P1 | L | 🔑 data-dependent, `models/attribution.py` |
| 4.2 | **Calibrate class probabilities** (isotonic/Platt) so the donut's confidence is trustworthy | P2 | S | `train_attribution.py` |
| 4.3 | **Wind-back-trajectory attribution** — use HYSPLIT/back-trajectories to attribute regional transport more rigorously than the current directional proxy | P2 | L | 🔑 new module |
| 4.4 | **Per-source uncertainty bands** in the attribution output | P2 | S | `models/attribution.py` |
| 4.5 | **Validate against a published Delhi source-apportionment study** and report agreement | P2 | M | write-up |

---

## 5. RAG — knowledge base, ingestion, embeddings, retrieval

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 5.1 | **Add the real official documents** — the corpus is curated *summaries*. Drop the actual **GRAP/CAQM/CPCB/DPCC/NCAP/WHO PDFs** into `data/kb/raw_docs/` (+ sidecar `.meta.json`) and re-ingest for authoritative citations | P0 | M | 🔑 `data/kb/raw_docs/`, `advisor/kb/ingest.py` |
| 5.2 | **Upgrade the embedding model** — swap MiniLM (384-d) for `BAAI/bge-large-en-v1.5` or `bge-m3` (multilingual, Hindi policy text) | P1 | S | `ADVISOR_EMBED_MODEL` in `.env` / `config.py` |
| 5.3 | **Enable the real cross-encoder** `BAAI/bge-reranker-v2-m3` (config already supports it) | P1 | S | `ADVISOR_RERANKER` |
| 5.4 | **Hindi / bilingual corpus** — many Delhi notifications are in Hindi; add multilingual embeddings + translation on ingest | P1 | M | 🔑 `advisor/kb/ingest.py` |
| 5.5 | **Chunk-quality upgrades** — semantic/late chunking, table extraction from PDFs, keep section hierarchy | P2 | M | `advisor/kb/ingest.py` |
| 5.6 | **Query expansion / HyDE** before retrieval for better recall on policy jargon | P2 | M | `advisor/retrieval/hybrid.py` |
| 5.7 | **Retrieval evaluation set** — a small labelled (query → relevant chunk) set to tune the semantic/BM25/graph/metadata weights instead of guessing | P1 | M | new `eval/retrieval_eval.py` |
| 5.8 | **Auto-refresh the corpus** — scheduled scrape of CAQM/CPCB order pages → ingest new orders | P2 | L | 🔑 new scraper |
| 5.9 | **Citation deep-linking** — store page/section so the UI can link to the exact spot in the source PDF | P2 | M | `advisor/kb/schema.py` |

---

## 6. Knowledge graph

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 6.1 | **Auto-extract entities/relations** from ingested docs with an LLM (currently the ontology is curated) → richer, self-updating graph | P1 | M | `advisor/kg/knowledge_graph.py` |
| 6.2 | **Weight `policies_for` by stage match** — a Stage-III query currently can surface Stage-I docs highly; tighten the traversal scoring | P2 | S | `advisor/kg/knowledge_graph.py` |
| 6.3 | **Graph visualisation page** — expose the full KG as an interactive ECharts graph in the dashboard | P2 | M | dashboard |
| 6.4 | Optional **Neo4j** backend if the graph grows large (NetworkX is fine for now) | P2 | L | `advisor/kg/` |

---

## 7. LLM reasoning

Currently runs on a **deterministic mock** until you add a key.

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 7.1 | **Add a cloud LLM key** — `GROQ_API_KEY` in `.env` (Gemma `gemma2-9b-it`) turns on real reasoning. Provider-agnostic (Groq/Anthropic/Gemini) already built | P0 | S | 🔑 `.env`, `advisor/llm/client.py` |
| 7.2 | **Structured outputs / function-calling** — enforce the JSON schema at the API level (Groq/Anthropic tool schema) to eliminate parse fallbacks | P1 | S | `advisor/llm/reasoner.py` |
| 7.3 | **Grounding guardrails** — post-check every cited string actually exists in the retrieved chunks; reject the answer if not (anti-hallucination) | P1 | M | `advisor/llm/reasoner.py` |
| 7.4 | **Self-consistency / critique pass** — a second LLM call critiques the interventions before ranking | P2 | M | `advisor/llm/reasoner.py` |
| 7.5 | **Streaming responses** to the dashboard for perceived speed | P2 | M | `advisor/api/main.py` |
| 7.6 | **Prompt-injection defence** on ingested docs (treat corpus text as untrusted) | P1 | S | `advisor/llm/reasoner.py` |
| 7.7 | **Bilingual advisories** — generate the recommendation in English + Hindi for public-facing use | P2 | S | `advisor/llm/reasoner.py` |
| 7.8 | **Cost/latency controls** — cache LLM answers per (ward, hour, stage); they rarely change | P1 | S | `advisor/llm/` |

---

## 8. Policy validation

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 8.1 | **Externalise rules to config/YAML** so non-developers can edit applicability without touching code | P2 | S | `advisor/validation/policy_validator.py` |
| 8.2 | **Legal-authority check** — verify the citing authority actually has jurisdiction for the action (CAQM vs DPCC vs MCD) | P2 | M | validator |
| 8.3 | **Temporal validity** — reject actions from superseded orders using `effective_date` | P2 | S | validator |
| 8.4 | **Explain rejections in the UI** — show *why* an action was filtered (currently only counted) | P2 | S | dashboard + validator |

---

## 9. Counterfactual simulation

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 9.1 | **Calibrate the feature-multiplier map** against real intervention studies (odd-even, construction bans) instead of hand-set factors | P1 | M | `config.INTERVENTION_FEATURE_MAP` / `SCENARIO_SLIDERS` |
| 9.2 | **Spatial spillover** — apply the intervention to neighbouring wards too (message passing already propagates it; make it explicit/optional) | P2 | M | `advisor/serving.py::counterfactual` |
| 9.3 | **Recompute true post-intervention attribution** (currently approximated by shrinking shares) by re-running the attribution heads on modified features | P1 | M | `advisor/api/main.py::simulate` |
| 9.4 | **Uncertainty on the counterfactual** — show p10–p90 of the *simulated* AQI, not just p50 | P2 | S | `advisor/simulation/counterfactual.py` |
| 9.5 | **Cost-effectiveness curve** — AQI drop per unit cost across intensities (optimise the budget) | P2 | M | `advisor/ranking/` |

---

## 10. Action ranking

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 10.1 | **Real cost/feasibility/time data** — replace the placeholder catalogue values with sourced estimates | P1 | M | 🔑 `config.ACTION_CATALOGUE` |
| 10.2 | **User-tunable objective weights** — a slider set in the UI so a policymaker can reweight cost vs speed vs impact | P2 | S | dashboard + `ranking` |
| 10.3 | **Pareto-front view** — surface non-dominated action sets, not a single ranking | P2 | M | `advisor/ranking/` |
| 10.4 | **Portfolio optimisation** — best *combination* under a budget, not per-action ranking | P2 | L | new module |

---

## 11. UI / Dashboard

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 11.1 | **Wind rose + radar + heatmap** charts (ECharts is already loaded; requested but not yet built) — wind rose per ward, pollutant radar, ward×hour AQI heatmap | P1 | M | `advisor/dashboard/index.html` |
| 11.2 | **Animate AQI over time** — play button that steps the time slider and animates the choropleth | P1 | S | dashboard |
| 11.3 | **Precomputed prediction store** so ward clicks are instant (see 3.1) — currently `/advise` runs the full pipeline live (~0.5 s, +5 s if explain) | P1 | M | new `build_predictions.py` + `/advise` reads store |
| 11.4 | **Loading skeletons** instead of spinners for panels | P2 | S | dashboard |
| 11.5 | **Mobile/tablet responsive** pass — the KPI grid + two-column main need breakpoints below 900 px | P1 | M | dashboard CSS |
| 11.6 | **Accessibility** — ARIA labels, keyboard nav, focus states, colourblind toggle (palette already CVD-safe) | P1 | M | dashboard |
| 11.7 | **Heatmap map-mode** (Leaflet.heat) as an alternative to the choropleth | P2 | S | dashboard |
| 11.8 | **Export a real PDF report** (per ward: forecast + sources + actions + citations) via a server-side renderer, beyond `window.print()` | P2 | M | `advisor/api/` |
| 11.9 | **48/72h tabs become real** once 2.2 is done | P1 | S | dashboard + model |
| 11.10 | **If you truly want React/shadcn** — a separate `web/` Vite+TS+shadcn app consuming the same API (a rewrite, so keep the current one as fallback) | P2 | L | new `web/` |
| 11.11 | **Alerts feed** — the bell badge counts severe wards; make it a real dropdown list with jump-to-ward | P2 | S | dashboard |

---

## 12. Backend / API / serving

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 12.1 | **Precompute + cache** all per-ward forecasts/attribution/explanations to a store; serve from it (fast, deterministic demo) | P0 | M | new `build_predictions.py`, `data/serving/` |
| 12.2 | **Async + concurrency** — the pipeline is sync; make retrieval/LLM calls async so the server handles parallel users | P1 | M | `advisor/api/main.py` |
| 12.3 | **Response caching layer** keyed by (ward, time, stage) | P1 | S | `advisor/api/` |
| 12.4 | **Input validation + rate limiting + CORS** for a public deployment | P1 | S | `advisor/api/main.py` |
| 12.5 | **/healthz with model+index+LLM readiness** for orchestration | P2 | S | `advisor/api/` |
| 12.6 | **Serving-time scaler reuse** for live data (`scalers.joblib`) — already wired via `feature_space.py`; add a `serving_transform` entry point for fresh inputs | P1 | S | `advisor/feature_space.py` |

---

## 13. Testing, quality, MLOps

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 13.1 | **Unit + integration tests** (pytest) for each `advisor` module and the pipeline; each module has a `__main__` self-test to lift from | P1 | M | new `tests/` |
| 13.2 | **Golden-output tests** for the pipeline on a fixed ward/hour to catch regressions | P1 | S | `tests/` |
| 13.3 | **Experiment tracking** (MLflow/W&B) for training runs instead of `*.log` files | P2 | M | `train_gnn.py` |
| 13.4 | **Model registry + versioning** for checkpoints (so serving pins a version) | P2 | M | `models/checkpoints/` |
| 13.5 | **CI** (GitHub Actions): lint + tests + a smoke `advise()` on every push | P1 | M | `.github/` |
| 13.6 | **Dockerise** (API + models + Chroma) for one-command deploy | P1 | M | new `Dockerfile`, `docker-compose.yml` |
| 13.7 | **`git init`** — the project is not a git repo; version-control it before anything else | P0 | S | 🔑 repo root |
| 13.8 | **Pin the environment** — the working interpreter is global py3.10 with CUDA torch; move it into a real venv from `requirements.txt` so it's reproducible | P1 | S | 🔑 `requirements.txt` |
| 13.9 | **Structured logging + error tracking** (replace prints; add Sentry-style capture) | P2 | M | across `advisor/` |

---

## 14. Security & governance (for a government-grade deployment)

| # | Item | Pri | Effort | Note |
|---|---|---|---|---|
| 14.1 | **Auth** (SSO/role-based) — policymakers vs public | P1 | M | 🔑 |
| 14.2 | **Audit log** of every recommendation shown (what, when, on which model version) | P1 | M | traceability |
| 14.3 | **Secrets management** — keys out of `.env` into a vault for production | P1 | S | 🔑 |
| 14.4 | **Model card + data sheet** — document limitations (19 grid cells, 49 labelled wards, annual emissions, summer-only test) publicly | P1 | S | write-up |
| 14.5 | **Prompt-injection & content safety** on LLM outputs before display | P1 | M | `advisor/llm/` |

---

## Manual setup checklist (things only YOU can do) 🔑

Work top-down; the first three unlock the most.

- [ ] **`git init`** and commit the current state (13.7).
- [x] ~~OpenAQ API key → re-pull CPCB ground truth (0.1)~~ — ✅ already done (577k station-hours). *No action needed.*
- [ ] **`GROQ_API_KEY`** (or Anthropic/Gemini) → `.env` — turns the LLM from mock to real (7.1).
- [ ] **Official policy PDFs** into `data/kb/raw_docs/` (+ `.meta.json`) → re-ingest (5.1).
- [ ] Create a **clean venv** from `requirements.txt` and stop relying on the global interpreter (13.8).
- [ ] **Persist the ensemble** checkpoints and point serving at them (2.1).
- [ ] **Train h48 + h72** models so the timeline tabs are real (2.2).
- [ ] Decide on **real cost/feasibility numbers** for the action catalogue (10.1).
- [ ] (If deploying) provision **auth, HTTPS, secrets, and a server/GPU** (§14).

---

## Suggested order (impact-first)

1. **Unblock**: `git init`, `GROQ_API_KEY`, real policy PDFs. *(CPCB re-pull already done — skip.)*
2. **Model** *(recipe-first, ensemble-last)*: 2.1 ✅ done · then settle recipe on a single model (2.4 → 2.5 → 2.3) → train h48/h72 (2.2) → **retrain the ensemble last** → conformal + CRPS post-hoc (2.3/2.10).
3. **Serving**: precompute prediction store → instant dashboard.
4. **RAG/LLM**: stronger embeddings + reranker, grounding guardrails, structured outputs.
5. **UI**: wind rose/radar/heatmap, animate-over-time, responsive, precomputed explanations.
6. **Hardening**: tests, CI, Docker, auth, model card.

> Reality check to keep front-and-centre (and cite in any demo): the honest
> limitations are **19 CAMS grid cells**, **49/289 wards labelled**, **annual
> emissions**, and a **summer-only test window**. Every improvement above is
> ultimately about loosening one of those four constraints — fix the data first,
> and everything downstream gets easier.
