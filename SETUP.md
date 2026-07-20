# Team Setup — Delhi AQI Intelligence Platform

Onboarding guide for continuing this project. Read this first, then see the
**[roadmap in `IMPROVEMENTS.md`](IMPROVEMENTS.md)** for what to work on next.

> **What this project is:** a ward-level Delhi AQI **forecasting → explainability →
> source attribution → policy-grounded action advisor**, with a FastAPI backend and
> a decision-support dashboard. Deep-dive docs: **[`MODELS.md`](MODELS.md)** (the ML
> models) and **[`ADVISOR.md`](ADVISOR.md)** (the RAG/LLM/counterfactual pipeline).

---

## 1. Prerequisites
- **Python 3.10** (important — the project is pinned to 3.10).
- **git**
- *(Optional)* an **NVIDIA GPU + CUDA** — only needed to *train* models fast. The
  app and inference run fine on CPU.

## 2. Clone & install
```bash
git clone <your-repo-url>
cd "real data"

# make a clean virtual environment (do NOT commit it)
python -m venv .venv
# activate it:
#   Windows PowerShell:  .venv\Scripts\Activate.ps1
#   macOS/Linux:         source .venv/bin/activate

pip install -r requirements.txt
```
> **GPU note:** if you have an NVIDIA GPU, install the CUDA build of torch instead:
> `pip install torch --index-url https://download.pytorch.org/whl/cu126`

## 3. Configure keys
```bash
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
```
Then open `.env` and add a **`GROQ_API_KEY`** (free at https://console.groq.com) to
enable real LLM recommendations. Without it, the app still runs on a deterministic
**offline mock** — so this is optional to get started.

## 4. What data you get (and what you don't)
**Included in the repo (ready to use):**
- `data/gnn/` + `data/gnn_processed/` — the trainable dataset (already built).
- `models/checkpoints/*.pt` — the trained forecasting models (incl. the 4-member ensemble).
- `data/kb/corpus/` — the policy knowledge-base documents.
- `data/raw/gis/wards/delhi_wards.geojson` — ward boundaries for the map.

**NOT in the repo (too big / rebuildable / secret):**
- `data/raw/` (4 GB source data) and `data/final/` (3.6 GB deprecated) — **not needed**
  to run or to do model work. Ask the team lead for a Drive link only if you plan to
  rebuild the dataset from scratch.
- `data/kb/chroma/` (vector index) — **you rebuild this locally**, see step 5.
- `.env` — everyone keeps their own.

## 5. Build the knowledge base index (one-time, ~1 min)
The vector index and knowledge graph are git-ignored (they're generated), so build
them once after cloning:
```bash
python -m advisor.kb.ingest                       # docs -> chunks
python -m advisor.embeddings.vector_store --reset # build the ChromaDB index
python -m advisor.kg.knowledge_graph              # build the knowledge graph
```
(First run downloads the embedding + reranker models, ~200 MB — needs internet once.)

## 6. Run it
```bash
# quick sanity checks (each module self-tests):
python stgnn_data.py                 # data loader
python -m advisor.pipeline           # full advise() for one ward

# launch the backend + dashboard:
python -m uvicorn advisor.api.main:app --port 8000
# open http://localhost:8000
```

## 7. Where to continue — the roadmap
**[`IMPROVEMENTS.md`](IMPROVEMENTS.md)** is the prioritized backlog (data / model /
RAG / LLM / UI / ops), with a "recipe-first, ensemble-last" order for the model work
and a manual-setup checklist. Current state:
- ✅ Data recovery (CPCB) — done; ~49 wards is the physical monitor ceiling.
- ✅ **2.1** — the forecast now serves a **4-model ensemble** (test skill +23.6%).
- 🔄 **2.4** (hyperparameter sweep) — in progress; see `sweep_gnn.py`.

## 8. Repo conventions
- **Never commit `.env`** or anything in `data/final/`, `data/raw/` (except the kept
  geojson), or `venv/` — the `.gitignore` handles this.
- The interpreter is **Python 3.10**; keep it that way.
- Each `advisor/*` module and each `*.py` script has a `__main__` self-test — run it
  to verify your change before committing.
- Retraining a model? See the "recipe-first, ensemble-last" note in `IMPROVEMENTS.md`
  §2 so you don't waste runs.

## 9. Common issues
| Symptom | Fix |
|---|---|
| `provider: mock` in recommendations | Add `GROQ_API_KEY` to `.env`. |
| Retrieval returns nothing | You skipped step 5 — build the Chroma index + KG. |
| `ModuleNotFoundError` | Wrong env — activate `.venv` and `pip install -r requirements.txt`. |
| Torch has no CUDA | Install the cu126 wheel (step 2 note); or run on CPU. |
| Map is blank | `data/raw/gis/wards/delhi_wards.geojson` missing — re-pull it from the repo. |
