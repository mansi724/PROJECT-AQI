"""
advisor/config.py — single, config-driven source of truth for the advisor stack.

No module below hardcodes a path, model name, or magic number — they all read
from `CONFIG` (or accept an injected `AdvisorConfig`, for dependency injection in
tests). Environment variables (loaded from `.env`) override the defaults, so the
same code runs locally and in a deployment without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # dotenv optional
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


# --------------------------------------------------------------------------
# AQI bands (CPCB India) — shared by context, validation, dashboard colouring
# --------------------------------------------------------------------------
AQI_BANDS = [
    (0, 50, "Good", "#4caf50"),
    (51, 100, "Satisfactory", "#8bc34a"),
    (101, 200, "Moderate", "#ffeb3b"),
    (201, 300, "Poor", "#ff9800"),
    (301, 400, "Very Poor", "#f44336"),
    (401, 500, "Severe", "#7e0023"),
]

# GRAP (Graded Response Action Plan) stages keyed to the AQI band.
GRAP_STAGES = {
    "Stage I": (201, 300, "Poor"),
    "Stage II": (301, 400, "Very Poor"),
    "Stage III": (401, 450, "Severe"),
    "Stage IV": (451, 500, "Severe+"),
}


# CPCB health advisory per band (grounded in the CPCB AQI corpus doc).
HEALTH_ADVICE = {
    "Good": "Air quality is good — minimal impact. Enjoy outdoor activities.",
    "Satisfactory": "Minor breathing discomfort possible for very sensitive people.",
    "Moderate": "Sensitive groups (asthma, heart disease, children, elderly) may feel "
                "discomfort. Reduce prolonged or heavy outdoor exertion.",
    "Poor": "Breathing discomfort on prolonged exposure. Sensitive groups should limit "
            "outdoor activity; others reduce heavy exertion.",
    "Very Poor": "Respiratory illness on prolonged exposure. Everyone should avoid outdoor "
                 "exertion; sensitive groups stay indoors. Use N95 masks and air purifiers.",
    "Severe": "Serious health impact for all, even with light activity. Stay indoors, run "
              "purifiers, wear N95 outdoors, and avoid all outdoor exertion.",
    "Severe+": "Air-quality emergency. Remain indoors; strictly avoid outdoor exposure. "
               "Sensitive groups are at severe risk.",
}

# WHO 2021 Global AQG 24-hour reference levels (µg/m³) for the 'vs WHO' bars.
WHO_LIMITS = {"pm2_5": 15, "pm10": 45, "nitrogen_dioxide": 25, "sulphur_dioxide": 40,
              "ozone": 100, "carbon_monoxide": 4000}

# Fixed, colourblind-safe categorical colours for source attribution
# (validated via the dataviz palette validator, dark-mode surface).
SOURCE_COLORS = {"traffic": "#3987e5", "dust": "#c98500", "biomass_burning": "#199e70",
                 "industrial": "#d55181", "industry": "#d55181", "secondary": "#008300"}


def health_advice(band: str) -> str:
    return HEALTH_ADVICE.get(band, "")


# ---- Scenario Builder (dashboard) — 7 policy levers -----------------------
# Each slider is an intervention strength 0–100%. The value per feature is the
# multiplier at 100% (strongest); the effective multiplier scales with the
# slider and composes across levers (product per feature) before re-running the
# frozen GNN. Reuses the same raw-feature counterfactual mechanism as Part 10.
SCENARIO_SLIDERS = {
    "traffic_reduction":        {"nitrogen_dioxide": 0.55, "carbon_monoxide": 0.60, "pm2_5": 0.88},
    "construction_activity":    {"pm10": 0.62, "dust": 0.64, "aerosol_optical_depth": 0.85, "pm2_5": 0.92},
    "industrial_emissions":     {"sulphur_dioxide": 0.60, "so2_no2_ratio": 0.80, "pm2_5": 0.90, "pm10": 0.93},
    "road_dust":                {"pm10": 0.75, "dust": 0.72},
    "public_transport":         {"nitrogen_dioxide": 0.85, "carbon_monoxide": 0.88, "pm2_5": 0.95},
    "heavy_vehicle_restriction":{"nitrogen_dioxide": 0.70, "pm2_5": 0.90, "pm10": 0.92},
    "green_cover":              {"pm2_5": 0.93, "pm10": 0.92, "dust": 0.90},
}
SLIDER_META = {
    "traffic_reduction":        {"label": "Traffic Reduction",         "icon": "car",        "source": "traffic"},
    "construction_activity":    {"label": "Construction Control",       "icon": "hard-hat",   "source": "dust"},
    "industrial_emissions":     {"label": "Industrial Emissions",       "icon": "factory",    "source": "industrial"},
    "road_dust":                {"label": "Road Dust Suppression",      "icon": "wind",       "source": "dust"},
    "public_transport":         {"label": "Public Transport Usage",     "icon": "bus",        "source": "traffic"},
    "heavy_vehicle_restriction":{"label": "Heavy Vehicle Restriction",  "icon": "truck",      "source": "traffic"},
    "green_cover":              {"label": "Green Cover Expansion",      "icon": "trees",      "source": "dust"},
}

# Map map-layers to their per-ward metric key (served by /layers).
MAP_LAYERS = {
    "aqi": {"label": "Predicted AQI", "kind": "aqi"},
    "forecast_diff": {"label": "Forecast Change (24h)", "kind": "diverging"},
    "current_aqi": {"label": "Current AQI", "kind": "aqi"},
    "pm2_5": {"label": "PM2.5", "kind": "seq"},
    "pm10": {"label": "PM10", "kind": "seq"},
    "nitrogen_dioxide": {"label": "NO₂ (traffic)", "kind": "seq"},
    "traffic": {"label": "Traffic Load", "kind": "seq"},
    "industry": {"label": "Industry Density", "kind": "seq"},
    "construction": {"label": "Construction / Dust", "kind": "seq"},
    "wind_speed": {"label": "Wind Speed", "kind": "seq"},
    "source": {"label": "Dominant Source", "kind": "categorical"},
}


def aqi_band(aqi: float) -> dict:
    for lo, hi, name, color in AQI_BANDS:
        if lo <= aqi <= hi:
            return {"band": name, "range": [lo, hi], "color": color}
    return {"band": "Severe+", "range": [500, 1000], "color": "#4a0011"}


def grap_stage(aqi: float) -> str | None:
    for stage, (lo, hi, _) in GRAP_STAGES.items():
        if lo <= aqi <= hi:
            return stage
    return None if aqi <= 200 else "Stage IV"


# --------------------------------------------------------------------------
# Counterfactual intervention model (Part 10) — WHICH raw features an action
# moves, and by how much (multiplicative, applied in raw feature space before
# re-scaling & re-running the frozen GNN). Config-driven so new actions need no
# code change. Values are conservative, transparent proxies — not fitted.
# --------------------------------------------------------------------------
INTERVENTION_FEATURE_MAP: dict[str, dict[str, float]] = {
    # action_type -> {raw_feature: multiplier when action fully applied}
    "traffic_restriction": {"nitrogen_dioxide": 0.75, "carbon_monoxide": 0.80,
                            "pm2_5": 0.92, "pm10": 0.95},
    "odd_even": {"nitrogen_dioxide": 0.82, "carbon_monoxide": 0.85, "pm2_5": 0.94},
    "construction_halt": {"pm10": 0.70, "dust": 0.72, "aerosol_optical_depth": 0.85,
                          "pm2_5": 0.90},
    "road_dust_suppression": {"pm10": 0.82, "dust": 0.80},
    "industrial_restriction": {"sulphur_dioxide": 0.70, "so2_no2_ratio": 0.80,
                               "pm2_5": 0.90, "pm10": 0.93},
    "brick_kiln_control": {"sulphur_dioxide": 0.78, "pm2_5": 0.92},
    "biomass_ban": {"fire_count": 0.5, "fire_frp_sum": 0.5, "pm2_5": 0.93},
    "public_transport_boost": {"nitrogen_dioxide": 0.88, "carbon_monoxide": 0.90},
}

# Which source label each action addresses (used by ranking + validation).
ACTION_TARGET_SOURCE = {
    "traffic_restriction": "traffic", "odd_even": "traffic",
    "public_transport_boost": "traffic",
    "construction_halt": "dust", "road_dust_suppression": "dust",
    "industrial_restriction": "industrial", "brick_kiln_control": "industrial",
    "biomass_ban": "biomass_burning",
}

# GRAP stages under which each action is applicable (validation + KG + ranking).
INTERVENTION_STAGES = {
    "road_dust_suppression": ["Stage I", "Stage II", "Stage III", "Stage IV"],
    "public_transport_boost": ["Stage II", "Stage III", "Stage IV"],
    "traffic_restriction": ["Stage III", "Stage IV"],
    "industrial_restriction": ["Stage III", "Stage IV"],
    "brick_kiln_control": ["Stage III", "Stage IV"],
    "construction_halt": ["Stage III", "Stage IV"],
    "odd_even": ["Stage IV"],
    "biomass_ban": ["Stage III", "Stage IV"],
}

# Human-readable action catalogue (feasibility/cost/time drive Part 11 ranking).
ACTION_CATALOGUE = {
    "road_dust_suppression": {"label": "Mechanised sweeping + water sprinkling",
                              "feasibility": 0.9, "cost": 0.3, "time_to_effect_h": 6},
    "public_transport_boost": {"label": "Augment bus/Metro frequency",
                               "feasibility": 0.7, "cost": 0.5, "time_to_effect_h": 12},
    "traffic_restriction": {"label": "Restrict polluting vehicles (BS-III/IV)",
                            "feasibility": 0.6, "cost": 0.4, "time_to_effect_h": 12},
    "industrial_restriction": {"label": "Curb non-compliant industry",
                               "feasibility": 0.6, "cost": 0.6, "time_to_effect_h": 24},
    "brick_kiln_control": {"label": "Close brick kilns / hot-mix plants",
                           "feasibility": 0.6, "cost": 0.5, "time_to_effect_h": 24},
    "construction_halt": {"label": "Halt construction & demolition",
                          "feasibility": 0.5, "cost": 0.8, "time_to_effect_h": 12},
    "odd_even": {"label": "Odd-even vehicle rationing",
                 "feasibility": 0.4, "cost": 0.6, "time_to_effect_h": 24},
    "biomass_ban": {"label": "Enforce biomass/stubble & firecracker ban",
                    "feasibility": 0.5, "cost": 0.4, "time_to_effect_h": 24},
}


@dataclass
class AdvisorConfig:
    # ---- reuse of the frozen forecasting stack ----
    horizon: int = 24
    forecast_ckpt_tag: str = "notemporal"       # explainable snapshot model
    attribution_tag: str = "attribution"
    device: str = _env("ADVISOR_DEVICE", "cpu")  # CPU is plenty for single-ward inference

    # ---- knowledge base / documents ----
    kb_dir: Path = DATA_DIR / "kb"
    kb_corpus_dir: Path = DATA_DIR / "kb" / "corpus"
    kb_raw_docs_dir: Path = DATA_DIR / "kb" / "raw_docs"   # user drops PDFs/DOCX/HTML here
    chunk_size: int = 900
    chunk_overlap: int = 150

    # ---- embeddings + vector store (Part 4) ----
    embedding_model: str = _env("ADVISOR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_dir: Path = DATA_DIR / "kb" / "chroma"
    chroma_collection: str = "aqi_policies"

    # ---- retrieval (Part 5) ----
    retrieval_top_k: int = 20          # candidates before rerank
    final_top_k: int = 6               # after rerank -> LLM
    semantic_weight: float = 0.5
    bm25_weight: float = 0.3
    graph_weight: float = 0.2

    # ---- reranker (Part 6) ----
    reranker_model: str = _env("ADVISOR_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ---- knowledge graph (Part 7) ----
    kg_path: Path = DATA_DIR / "kb" / "knowledge_graph.gpickle"

    # ---- LLM (Part 8) — CLOUD ONLY (no local/Ollama) ----
    llm_provider: str = _env("ADVISOR_LLM_PROVIDER", "groq")   # groq | anthropic | gemini | mock
    llm_model: str = _env("ADVISOR_LLM_MODEL", "gemma2-9b-it")
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1500
    groq_api_key: str = _env("GROQ_API_KEY", "")
    anthropic_api_key: str = _env("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = _env("GEMINI_API_KEY", "")

    # ---- serving outputs ----
    serving_dir: Path = DATA_DIR / "serving"

    # ---- action ranking (Part 11) — multi-objective weights (sum ~1) ----
    ranking_weights: dict = field(default_factory=lambda: {
        "aqi_improvement": 0.35, "confidence": 0.15, "policy_strength": 0.20,
        "feasibility": 0.15, "cost": 0.10, "time_to_effect": 0.05,
    })

    bands: list = field(default_factory=lambda: AQI_BANDS)

    def ensure_dirs(self):
        for p in (self.kb_dir, self.kb_corpus_dir, self.kb_raw_docs_dir,
                  self.chroma_dir, self.serving_dir):
            p.mkdir(parents=True, exist_ok=True)


CONFIG = AdvisorConfig()


if __name__ == "__main__":
    CONFIG.ensure_dirs()
    print("BASE_DIR:", BASE_DIR)
    print("LLM provider/model:", CONFIG.llm_provider, "/", CONFIG.llm_model,
          "| key set:", bool(CONFIG.groq_api_key))
    print("embedding model:", CONFIG.embedding_model)
    print("aqi 312 ->", aqi_band(312), "| GRAP", grap_stage(312))
    print("interventions:", list(INTERVENTION_FEATURE_MAP))
