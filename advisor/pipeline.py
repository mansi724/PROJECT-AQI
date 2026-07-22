"""
advisor/pipeline.py — end-to-end orchestration of the post-attribution stack.

Chains every module in the target architecture into one call:

  Context Builder -> Hybrid Retrieval (+KG) -> Cross-Encoder Rerank ->
  LLM Reasoning -> Policy Validation -> Counterfactual Simulation -> Action Ranking

`AdvisorPipeline.advise(ward_id)` returns ONE structured object the API/dashboard
render directly. Dependencies are injected (all default to the cached singletons),
so the whole flow is testable with the offline mock LLM and swappable in prod.
"""
from __future__ import annotations

from advisor.config import CONFIG
from advisor.context_builder import ContextBuilder
from advisor.retrieval.hybrid import get_hybrid_retriever, RetrievalContext
from advisor.retrieval.reranker import get_reranker
from advisor.llm.reasoner import Reasoner
from advisor.validation.policy_validator import PolicyValidator
from advisor.simulation.counterfactual import CounterfactualSimulator
from advisor.ranking.action_ranker import ActionRanker


class AdvisorPipeline:
    def __init__(self, config=CONFIG, context_builder=None, retriever=None,
                 reranker=None, reasoner=None, validator=None, simulator=None, ranker=None):
        self.cfg = config
        self.context_builder = context_builder or ContextBuilder()
        self.retriever = retriever or get_hybrid_retriever()
        self.reranker = reranker or get_reranker()
        self.reasoner = reasoner or Reasoner()
        self.validator = validator or PolicyValidator()
        self.simulator = simulator or CounterfactualSimulator(self.context_builder.svc)
        self.ranker = ranker or ActionRanker(config)

    def advise(self, ward_id: str | None = None, node_idx: int | None = None,
               time=None, time_index=None, include_explanations: bool = True,
               ranking_weights: dict | None = None) -> dict:
        # 1. Context
        wc = self.context_builder.build(ward_id=ward_id, node_idx=node_idx, time=time,
                                        time_index=time_index,
                                        include_explanations=include_explanations)
        ctx = wc.to_dict()
        node = self.context_builder.svc.ward_to_node(wc.ward_id)
        t = self.context_builder.svc.resolve_time(time=time, time_index=time_index)

        # 2. Retrieval + 3. rerank
        stage = wc.grap_stage or "Stage I"
        rquery = (f"AQI {wc.predicted_aqi} ({wc.aqi_band}), GRAP {stage}. Dominant sources: "
                  f"{', '.join(wc.dominant_sources)}. What control actions are prescribed?")
        rctx = RetrievalContext(query=rquery, aqi_stage=stage,
                                sources=list(wc.dominant_sources), pollutants=["PM2.5", "PM10"])
        candidates = self.retriever.retrieve(rctx, top_k=self.cfg.retrieval_top_k)
        top = self.reranker.rerank(rquery, candidates, top_k=self.cfg.final_top_k)

        # 4. LLM reasoning
        reasoning = self.reasoner.reason(ctx, top)

        # 5. Validation
        validation = self.validator.validate(reasoning.get("interventions", []), ctx)
        valid_actions = validation["valid_actions"]

        # 6. Counterfactual per valid action + combined
        counterfactuals, cf_map = [], {}
        for a in valid_actions:
            r = self.simulator.simulate_action(node, t, a["action"]).as_dict()
            r["action"] = a["action"]
            counterfactuals.append(r)
            cf_map[a["action"]] = r
        combined = None
        if valid_actions:
            combined = self.simulator.simulate_actions(
                node, t, [a["action"] for a in valid_actions]).as_dict()

        # 7. Ranking
        ranked = self.ranker.rank(valid_actions, cf_map, ctx, weights=ranking_weights)

        return {
            "context": ctx,
            "retrieved_policies": [c.as_dict() for c in top],
            "reasoning": reasoning,
            "validation": {"n_valid": validation["n_valid"],
                           "n_rejected": validation["n_rejected"],
                           "rejected": validation["rejected_actions"]},
            "counterfactuals": counterfactuals,
            "combined_counterfactual": combined,
            "ranked_actions": ranked,
            "provider": reasoning.get("_provider"),
        }


_PIPELINE = None


def get_pipeline() -> AdvisorPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = AdvisorPipeline()
    return _PIPELINE


if __name__ == "__main__":
    import json
    out = get_pipeline().advise(ward_id="239", time_index=24477, include_explanations=False)
    print("provider:", out["provider"])
    print("AQI:", out["context"]["predicted_aqi"], out["context"]["aqi_band"],
          "| stage:", out["context"]["grap_stage"])
    print("dominant sources:", out["context"]["dominant_sources"])
    print("retrieved policies:", [p["title"][:40] for p in out["retrieved_policies"]])
    print("validation:", out["validation"]["n_valid"], "valid,",
          out["validation"]["n_rejected"], "rejected")
    print("\nRANKED ACTIONS:")
    for r in out["ranked_actions"]:
        print(f"  #{r['rank']} {r['action']:22} score={r['score']:.3f} "
              f"AQI {r['aqi_before']}->{r['aqi_after']} | {r['title']}")
    if out["combined_counterfactual"]:
        c = out["combined_counterfactual"]
        print(f"\ncombined all actions: AQI {c['aqi_before']} -> {c['aqi_after']} "
              f"({c['improvement_pct']}% better)")
