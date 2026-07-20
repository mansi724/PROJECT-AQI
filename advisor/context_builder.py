"""
advisor/context_builder.py — PART 1: Context Builder.

Turns the forecasting stack's raw outputs (prediction + SHAP + GNNExplainer +
source attribution + meteorology + ward metadata) into ONE structured, JSON-safe
`WardContext`. This object is the contract every downstream module (retrieval,
LLM reasoning, counterfactual, ranking, dashboard) consumes — so nothing below
has to know how the models work, only how to read this schema.

    ctx = ContextBuilder().build(ward_id="52")
    ctx.to_json()   # -> the structured JSON in the spec

All heavy model calls go through `advisor.serving.ForecastService`; the builder
itself is pure composition, so it is cheap and deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

from advisor.config import CONFIG, AdvisorConfig
from advisor.serving import ForecastService, get_forecast_service


@dataclass
class WardContext:
    ward_id: str
    ward_name: str
    lat: float
    lon: float
    timestamp: str
    forecast_horizon: str
    predicted_aqi: float
    aqi_low: float
    aqi_high: float
    aqi_band: str
    grap_stage: str | None
    dominant_sources: dict            # {"traffic":0.41,"industry":0.28,...}
    source_confidence: float
    meteorology: dict
    pollutants: dict
    ward_metadata: dict
    explanations: dict = field(default_factory=dict)
    edgar_prior_pm25: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# canonical source label -> dashboard/LLM display name
_SOURCE_DISPLAY = {
    "dust": "dust", "traffic": "traffic", "biomass_burning": "biomass_burning",
    "industrial": "industry", "secondary": "secondary",
}


class ContextBuilder:
    def __init__(self, service: ForecastService | None = None,
                 config: AdvisorConfig = CONFIG):
        self.cfg = config
        self.svc = service or get_forecast_service()

    def build(self, ward_id: str | None = None, node_idx: int | None = None,
              time=None, time_index=None, include_explanations: bool = True,
              max_sources: int = 4) -> WardContext:
        if node_idx is None:
            if ward_id is None:
                raise ValueError("provide ward_id or node_idx")
            node_idx = self.svc.ward_to_node(ward_id)
        t = self.svc.resolve_time(time=time, time_index=time_index)

        fc = self.svc.forecast(node_idx, t)
        meta = self.svc.node_meta(node_idx)
        prof = self.svc.attribution(node_idx, t)

        # dominant sources: keep the meaningful (score>0) ones, top-N, renormalised
        ranked = [(r["source"], r["score"]) for r in prof["ranked_sources"] if r["score"] > 0]
        ranked = ranked[:max_sources]
        tot = sum(s for _, s in ranked) or 1.0
        dominant = {_SOURCE_DISPLAY.get(k, k): round(v / tot, 3) for k, v in ranked}

        explanations = {}
        if include_explanations:
            explanations = {
                **self.svc.explanations(node_idx, t),          # 3.1 cached SHAP + GNNExplainer
                "dominant_class": prof.get("dominant_class"),
                "class_probs": {k: round(v, 3) for k, v in prof.get("class_probs", {}).items()},
                "predicted_pm25_pm10_ratio": round(prof.get("pred_pm25_pm10_ratio", float("nan")), 3),
                "source_uncertainty": prof.get("uncertainty"),  # 4.4
            }

        return WardContext(
            ward_id=meta["ward_id"], ward_name=meta["ward_name"],
            lat=meta["lat"], lon=meta["lon"],
            timestamp=str(fc.time), forecast_horizon=f"{fc.horizon_h}h",
            predicted_aqi=round(fc.p50, 1), aqi_low=round(fc.p10, 1), aqi_high=round(fc.p90, 1),
            aqi_band=fc.as_dict()["band"], grap_stage=fc.as_dict()["grap_stage"],
            dominant_sources=dominant, source_confidence=prof.get("confidence", 0.5),
            meteorology=self.svc.meteorology(node_idx, t),
            pollutants=self.svc.raw_pollutants(node_idx, t),
            ward_metadata=meta,
            explanations=explanations,
            edgar_prior_pm25=prof.get("edgar_prior_pm25", {}),
        )


if __name__ == "__main__":
    cb = ContextBuilder()
    # fast path first (no SHAP/GNNExplainer) to show the core contract
    core = cb.build(ward_id="239", time_index=24477, include_explanations=False)
    print("=== core context (no explanations) ===")
    print(core.to_json())
    print("\n=== with explanations (SHAP + GNNExplainer) ===")
    full = cb.build(ward_id="239", time_index=24477, include_explanations=True)
    print(json.dumps({"explanations": full.explanations}, indent=2, default=str))
