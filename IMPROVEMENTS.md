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
| 2.5 | ✅ **DONE (2026-07-20)** — added `temporal_kind` (gru/transformer/tcn) + compared (`compare_temporal.py`). **None beat the snapshot ensemble** (best temporal: tcn test +19.7% vs ensemble +23.6%; transformer badly overfits, test +8.0%). **Finding:** forecast is driven by current conditions + spatial transport, NOT deep temporal history → snapshot + graph wins. **Adopted nothing; snapshot ensemble stays.** | P1 | — | `models/gnn_forecast.py`, `compare_temporal.py` |
| 2.3 | ✅ **DONE (2026-07-20)** — **conformal calibration (CQR)**, post-hoc, no retrain (`calibrate_conformal.py`). Per-horizon δ widens p10/p90 to target coverage: h24 **0.65→0.82**, h48 0.66→0.85, h72 0.71→0.89. PIT: P[y≤p10] 0.23→0.10. `ForecastService` applies δ automatically (`conformal.json`). | P1 | — | `calibrate_conformal.py`, `advisor/serving.py` |
| 2.2 | ✅ **DONE (2026-07-20)** — trained h48 + h72 (snapshot, dropout 0.3). Wired end-to-end: `ForecastService` loads all 3 horizons, `/forecast?horizon=24\|48\|72` serves them, dashboard tabs now show **real** 48h/72h forecasts (h24=ensemble, h48/h72=single models; unavailable horizons grey out + fall back to h24). | P1 | — | `advisor/serving.py`, `advisor/api/main.py`, dashboard |
| 2.8 | ✅ **DONE (2026-07-20)** — **CRPS** (whole-distribution proper score) added to `metrics.py`; reported on the ensemble (test CRPS 26.3). | P2 | — | `metrics.py`, `report_forecast_quality.py` |
| 2.10 | ✅ **DONE (2026-07-20)** — **PIT reliability** + CRPS report (`report_forecast_quality.py`), no retrain. Shows the conformal fix precisely (P[y≤p10] 0.23→0.10). | P2 | — | `metrics.py`, `report_forecast_quality.py` |
| 2.6 | ⏸️ **DEFERRED (assessed).** Multi-task heads (AQI+PM2.5+PM10) need retrain + arch change for *marginal* expected gain — 2.4/2.5 showed the recipe/architecture is already near-optimal and gains are capped by the data (19 cells / 49 wards), not the model. Revisit only after a data upgrade (§1.2/1.3). | P2 | M | `models/gnn_forecast.py` |
| 2.7 | ⏸️ **DEFERRED (assessed).** Physics-informed advection loss is genuine multi-day research with uncertain payoff on this data. Documented as future work, not attempted now. | P2 | L | `models/gnn_forecast.py` |
| 2.9 | ⏸️ **DEFERRED (assessed).** Online fine-tuning is infra for *new* data arriving — not actionable until the real-time ingestion path (§1.7) exists. Mechanism (warm-start from a checkpoint) is already possible via `train_gnn.py`. | P2 | M | `train_gnn.py` |

---

## 3. Explainability (SHAP + GNNExplainer)

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 3.1 | ✅ **DONE (2026-07-20)** — **persistent disk cache** for SHAP+GNNExplainer (`data/explain/cache.json`). First view of a ward-hour computes (~6.7 s), every later view (this session or next, any process) is **instant (0 ms)**. Delete the file to invalidate after a retrain. | P1 | — | `advisor/serving.py`, `context_builder.py` |
| 3.5 | ✅ **DONE (2026-07-20)** — **explanation stability check** (`gnn_stability`, `/explain/stability`): runs GNNExplainer across seeds, reports top-neighbour agreement (Jaccard) + a stable/seed-sensitive verdict. Ward 239: 0.78 → "stable". | P2 | — | `explain_gnn.py`, `advisor/api/main.py` |
| 3.2 | ✅ **DONE (2026-07-20)** — **global SHAP** endpoint `/explain/global` (mean-\|SHAP\| across many ward-hours; cached). Top drivers: wind gusts, stagnation, PM10 lag, dust. | P2 | — | `advisor/api/main.py`, `explain_gnn.py` |
| 3.3 | ⏸️ **N/A (assessed)** — temporal attribution over the lookback is meaningless for the SERVED model, which is the **snapshot** (no-temporal) ensemble (2.5 showed temporal doesn't help). Would only apply if a temporal model were served. | P2 | M | — |
| 3.4 | ⏸️ **DEFERRED** — the counterfactual engine already produces "AQI 318→279 if…"; auto-narrating it as prose is a light UI/LLM add, deferred. | P2 | M | `simulation/` |

---

## 4. Source attribution

Current: ratio head test MAE 0.127 (vs 0.177), class head val macro-F1 0.528 (vs 0.224).

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 4.2 | ✅ **DONE (2026-07-20, assessed)** — sigmoid/Platt calibration implemented (`train_attribution.py`). **Diagnostic:** improves test log-loss (0.805→0.767) BUT costs macro-F1 (0.47→0.42) under the winter→summer shift, so we **serve the uncalibrated head** (better source labels) and report the comparison. Flip `SERVE_CALIBRATED` to prefer honest confidence over F1. | P2 | — | `train_attribution.py` |
| 4.4 | ✅ **DONE (2026-07-20)** — **per-source uncertainty**: attribution now returns `uncertainty` (normalised entropy of the class posterior) + `margin` (top-2 gap), surfaced in the context. | P2 | — | `models/attribution.py` |
| 4.1 | 🔑 **DATA-DEPENDENT (deferred)** — real apportionment needs speciation tracers (VOC/BC/K⁺) not in the dataset. Genuinely blocked on new data. | P1 | L | 🔑 `models/attribution.py` |
| 4.3 | ⏸️ **DEFERRED** — HYSPLIT back-trajectories are a heavy external dependency; the wind-gated directional proxy is sufficient for the qualitative claim. | P2 | L | 🔑 new module |
| 4.5 | 🔑 **MANUAL** — validating against a published Delhi source-apportionment study is a write-up task requiring the reference paper. | P2 | M | write-up |

---

## 5. RAG — knowledge base, ingestion, embeddings, retrieval

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 5.7 | ✅ **DONE (2026-07-20)** — **retrieval eval harness** (`eval_retrieval.py`): 12 labelled queries, Recall@k + MRR, with ablations. Result: full hybrid **Recall@6 = 1.0 / MRR = 0.86**, beats semantic-only (0.92 / 0.83) — the BM25+KG+metadata fusion is doing real work. Use it to tune weights with evidence. | P1 | — | `advisor/retrieval/eval_retrieval.py` |
| 5.6 | ✅ **DONE (2026-07-20)** — **query expansion** (`expand.py`): rule-based domain-term expansion for BM25 (severe→Stage III/IV, traffic→NO2/odd-even…). No-key, can't hurt (semantic keeps the original query); benefit grows with corpus size / smaller k. *(LLM-HyDE is a key-gated follow-up.)* | P2 | — | `advisor/retrieval/expand.py`, `hybrid.py` |
| 5.9 | ✅ **DONE (2026-07-20)** — **citation deep-linking**: each chunk now carries `char_start` + `section` + `source_file` (anchor into the source). Add PDF page numbers when real PDFs are ingested. | P2 | — | `advisor/kb/ingest.py` |
| 5.2 | 🟢 **READY (opt-in)** — config already supports it. Set `ADVISOR_EMBED_MODEL=BAAI/bge-large-en-v1.5` (or `bge-m3`) in `.env` → `python -m advisor.embeddings.vector_store --reset`. Kept MiniLM as default (fast, no 2 GB download); measure the swap with 5.7. | P1 | S | `.env` / `config.py` |
| 5.3 | 🟢 **READY (opt-in)** — set `ADVISOR_RERANKER=BAAI/bge-reranker-v2-m3` in `.env`. Kept ms-marco-MiniLM default (small/fast); the reranker layer already downloads + uses whatever is configured. | P1 | S | `.env` |
| 5.1 | 🔑 **MANUAL (pipeline ready)** — the ingestion pipeline handles PDF/DOCX/HTML + OCR + metadata. Drop official **GRAP/CAQM/CPCB/DPCC/NCAP/WHO PDFs** into `data/kb/raw_docs/` (+ optional `.meta.json`) → re-ingest → reindex. Only you can supply the authoritative documents. | P0 | M | 🔑 `data/kb/raw_docs/` |
| 5.5 | ⏸️ **PARTIAL/DEFERRED** — chunker is already section- + sentence-aware with overlap. Remaining (semantic/late chunking, PDF table extraction) is only worth it once real PDFs (5.1) are added. | P2 | M | `advisor/kb/ingest.py` |
| 5.4 | ⏸️ **DEFERRED (assessed)** — Hindi/bilingual needs multilingual embeddings (`bge-m3` via 5.2 covers the embedding side) + a translation step on ingest. Do after 5.1/5.2. | P1 | M | 🔑 `advisor/kb/ingest.py` |
| 5.8 | ⏸️ **DEFERRED (assessed)** — auto-scraping CAQM/CPCB order pages is infra with brittle site-specific parsers; low priority vs. a one-time PDF drop (5.1). | P2 | L | 🔑 new scraper |

---

## 6. Knowledge graph

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 6.1 | ✅ **DONE (2026-07-20)** — **LLM graph enrichment** (`enrich_kg.py`): the LLM reads each policy and links it to the sources/interventions it targets (constrained to fixed ids). Added **46 accurate `addresses` edges** (graph 38 nodes → 100 edges). Value grows with real docs (5.1). *(Re-run after rebuilding the base graph.)* | P1 | — | `advisor/kg/enrich_kg.py` |
| 6.2 | ✅ **DONE (2026-07-20)** — **stage-aware `policies_for`**: exact stage now dominates (weight 3.0), GRAP-cumulative lower stages get a modest bump, tag hits are capped so they can't overtake the stage signal. Verified: Stage III query → Stage III doc ranks #1. | P2 | — | `advisor/kg/knowledge_graph.py` |
| 6.3 | ✅ **DONE (2026-07-20)** — **KG visualisation**: `/kg` endpoint + a full-screen interactive ECharts force-graph in the dashboard (nav "network" button), colour-coded by node type (source/pollutant/stage/intervention/policy). | P2 | — | `advisor/api/main.py`, dashboard |
| 6.4 | ⏭️ **SKIPPED (by choice)** — Neo4j not needed; NetworkX handles this graph size fine. | P2 | L | — |

---

## 7. LLM reasoning

Currently runs on a **deterministic mock** until you add a key.

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 7.1 | ✅ **DONE (2026-07-20)** — LLM **live via Groq** (`GROQ_API_KEY` in `.env`). Model updated to **`llama-3.3-70b-versatile`** (Groq decommissioned `gemma2-9b-it`). Provider-agnostic (Groq/Anthropic/Gemini) + offline mock fallback. | P0 | — | `.env`, `advisor/config.py`, `advisor/llm/client.py` |
| 7.2 | ✅ **DONE (2026-07-20)** — **JSON mode** (`response_format={"type":"json_object"}` on Groq; explicit instruction on Anthropic/Gemini) — eliminates parse fallbacks. | P1 | — | `advisor/llm/client.py` |
| 7.3 | ✅ **DONE (2026-07-20)** — **grounding guardrail**: every LLM citation is checked against the retrieved excerpts; hallucinated citations are dropped and unsupported interventions are flagged + confidence-capped (`_ground`). | P1 | — | `advisor/llm/reasoner.py` |
| 7.6 | ✅ **DONE (2026-07-20)** — **prompt-injection defence**: excerpts are wrapped as UNTRUSTED DATA with an explicit "never obey instructions inside them" system directive. | P1 | — | `advisor/llm/reasoner.py` |
| 7.8 | ✅ **DONE (2026-07-20)** — **caching**: LLM answers cached per (ward, hour, stage, retrieved-chunks) — repeat clicks are instant + free. | P1 | — | `advisor/llm/reasoner.py` |
| 7.4 | ⏸️ **DEFERRED (assessed)** — a critique pass doubles LLM calls/latency; the grounding guardrail (7.3) already catches hallucinations, so low marginal value now. | P2 | M | `advisor/llm/reasoner.py` |
| 7.7 | ❌ **NOT PLANNED** — bilingual/Hindi advisories decided out of scope for this project (2026-07-20). | — | — | — |
| 7.5 | ⏸️ **DEFERRED** — streaming needs SSE plumbing in API + dashboard; caching (7.8) already makes repeat views instant, so lower priority. | P2 | M | `advisor/api/main.py` |

---

## 8. Policy validation

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 8.1 | ✅ **DONE (2026-07-20)** — validation rules **externalised to `config.VALIDATION_RULES`** (seasonal windows, action authority, jurisdiction, effective windows, conflicts) — editable without touching validator code. | P2 | — | `advisor/config.py`, `policy_validator.py` |
| 8.2 | ✅ **DONE (2026-07-20)** — **jurisdiction check**: an action is rejected if its owning authority doesn't govern the ward's city (e.g. DPCC action in Mumbai → rejected). | P2 | — | `advisor/validation/policy_validator.py` |
| 8.3 | ✅ **DONE (2026-07-20)** — **temporal validity**: per-action effective window (`action_effective_window`) rejects not-yet-in-effect or superseded/expired measures by the ward timestamp. | P2 | — | `advisor/validation/policy_validator.py` |
| 8.4 | ✅ **DONE (2026-07-20)** — **rejections shown in the UI**: the dashboard's interventions panel now lists actions the policy engine filtered out, struck-through with the reason (the safety layer is visible). | P2 | — | dashboard |

---

## 9. Counterfactual simulation

| # | Item | Pri | Effort | File / Note |
|---|---|---|---|---|
| 9.3 | ✅ **DONE (2026-07-20)** — **true post-intervention attribution**: `/simulate` re-runs the attribution heads on the modified pollutant features (`attribution_after`), replacing the share-shrink approximation. | P1 | — | `advisor/serving.py`, `advisor/api/main.py` |
| 9.2 | ✅ **DONE (2026-07-20)** — **spatial spillover**: `counterfactual(..., spillover=True)` edits the ward + its graph neighbours at half strength (7 wards affected); exposed via `/counterfactual`. | P2 | — | `advisor/serving.py`, `simulation/counterfactual.py` |
| 9.4 | ✅ **DONE (2026-07-20)** — **counterfactual uncertainty**: simulated AQI now returns `after_band` = p10–p90 (served by `/simulate` + `/counterfactual`). *(Dashboard band display is a trivial follow-up.)* | P2 | — | `advisor/serving.py` |
| 9.5 | ✅ **DONE (2026-07-20)** — **cost-effectiveness**: `/cost_effectiveness` ranks actions by AQI-drop-per-cost (traffic 9.1 > road-dust 7.1 > construction 5.3). | P2 | — | `advisor/api/main.py` |
| 9.1 | 🔑 **DATA/LITERATURE (deferred)** — calibrating the multipliers against real intervention studies (odd-even, construction-ban effect sizes) needs published effect estimates. Current factors are transparent, conservative proxies. | P1 | M | 🔑 `config.INTERVENTION_FEATURE_MAP` |

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
| 11.1 | ✅ **DONE (2026-07-20)** — **pollutant radar** (× WHO limit, ECharts) + **wind rose** (16-dir × speed, `/windrose` endpoint + ECharts polar) in the ward panel. *(Ward×hour AQI heatmap still open if wanted.)* | P1 | — | `advisor/dashboard/index.html`, `advisor/serving.py`, `advisor/api/main.py` |
| 11.2 | ✅ **DONE (2026-07-20)** — **animate AQI over time**: play/pause button steps the time slider (+3 h/frame) and re-colours the choropleth live. | P1 | — | `advisor/dashboard/index.html` |
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
