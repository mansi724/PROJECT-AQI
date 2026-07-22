"""
advisor/ranking/action_ranker.py — PART 11: multi-objective action ranking.

Combines six objectives into one transparent score per intervention:

  * aqi_improvement  — the counterfactual AQI drop (Part 10), the primary signal.
  * confidence       — the LLM's grounded confidence (Part 8).
  * policy_strength  — validation result + citation support (Part 9 / retrieval).
  * feasibility      — how practical the action is (catalogue).
  * cost             — implementation cost (inverted: cheaper is better).
  * time_to_effect   — how fast it works (inverted: faster is better).

Each objective is normalised to [0,1] (higher = better), then combined with the
configured weights. Output is a ranked list with the full breakdown, so a
decision-maker can see *why* an action ranks where it does.
"""
from __future__ import annotations

from advisor.config import CONFIG, ACTION_CATALOGUE, ACTION_TARGET_SOURCE


def _norm(v, lo, hi):
    if hi - lo < 1e-9:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


class ActionRanker:
    def __init__(self, config=CONFIG):
        self.cfg = config
        self.w = config.ranking_weights

    def rank(self, actions: list, counterfactuals: dict, ward_context: dict,
             weights: dict | None = None) -> list[dict]:
        """actions: validated LLM interventions (each a dict with 'action', 'confidence',
        optional '_validation', 'citations'). counterfactuals: {action_id: CounterfactualResult-like dict}.
        `weights` (10.2) overrides the config objective weights so a policymaker can
        reweight impact vs cost vs speed live."""
        w = {**self.w, **(weights or {})}
        if not actions:
            return []
        # improvement range for normalisation (across the candidate set)
        imps = [max(counterfactuals.get(a["action"], {}).get("improvement_pct", 0.0), 0.0)
                for a in actions]
        imax = max(imps) if imps else 1.0

        dominant = set((ward_context.get("dominant_sources") or {}).keys())
        dominant |= {"industrial" if s == "industry" else s for s in list(dominant)}

        ranked = []
        for a in actions:
            aid = a["action"]
            cat = ACTION_CATALOGUE.get(aid, {})
            cf = counterfactuals.get(aid, {})

            imp = max(cf.get("improvement_pct", 0.0), 0.0)
            o_improve = _norm(imp, 0.0, imax if imax > 0 else 1.0)
            o_conf = float(a.get("confidence", 0.6))
            o_policy = self._policy_strength(a, aid, dominant)
            o_feas = float(cat.get("feasibility", 0.5))
            o_cost = 1.0 - float(cat.get("cost", 0.5))                 # cheaper better
            o_time = 1.0 - _norm(cat.get("time_to_effect_h", 24), 0, 48)  # faster better

            objectives = {"aqi_improvement": o_improve, "confidence": o_conf,
                          "policy_strength": o_policy, "feasibility": o_feas,
                          "cost": o_cost, "time_to_effect": o_time}
            score = sum(w[k] * objectives[k] for k in w)
            ranked.append({
                "action": aid,
                "title": a.get("title", cat.get("label", aid)),
                "target_source": a.get("target_source", ACTION_TARGET_SOURCE.get(aid, "")),
                "score": round(score, 4),
                "aqi_before": cf.get("aqi_before"), "aqi_after": cf.get("aqi_after"),
                "aqi_delta": cf.get("delta"), "improvement_pct": cf.get("improvement_pct"),
                "confidence": o_conf, "citations": a.get("citations", []),
                "objectives": {k: round(v, 3) for k, v in objectives.items()},
                "rationale": a.get("rationale", ""),
            })
        ranked.sort(key=lambda d: -d["score"])
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
        self._mark_pareto(ranked)          # 10.3
        return ranked

    @staticmethod
    def _mark_pareto(ranked: list) -> None:
        """10.3 — flag Pareto-optimal actions: those not dominated on ALL objectives
        by another. A dominated action is beaten (>=) everywhere and strictly worse
        somewhere; the survivors are the defensible trade-off frontier."""
        objs = [r["objectives"] for r in ranked]
        keys = list(objs[0].keys()) if objs else []
        for i, ri in enumerate(ranked):
            dominated = False
            for j, rj in enumerate(ranked):
                if i == j:
                    continue
                ge_all = all(objs[j][k] >= objs[i][k] for k in keys)
                gt_any = any(objs[j][k] > objs[i][k] for k in keys)
                if ge_all and gt_any:
                    dominated = True
                    break
            ri["pareto_optimal"] = not dominated

    def _policy_strength(self, action: dict, aid: str, dominant: set) -> float:
        s = 1.0
        val = action.get("_validation", {})
        s -= 0.1 * len(val.get("flags", []))
        if action.get("citations"):
            s = min(1.0, s + 0.1)
        if ACTION_TARGET_SOURCE.get(aid) in dominant:      # addresses a real dominant source
            s = min(1.0, s + 0.1)
        return max(0.0, s)


if __name__ == "__main__":
    from advisor.simulation.counterfactual import CounterfactualSimulator
    sim = CounterfactualSimulator()
    node = sim.svc.ward_to_node("239")
    t = sim.svc.resolve_time(time_index=24477)
    acts = ["road_dust_suppression", "construction_halt", "traffic_restriction"]
    cf = {r["action"]: r for r in sim.rank_actions_by_effect(node, t, acts)}
    actions = [{"action": a, "confidence": 0.75, "citations": ["GRAP"], "_validation": {"flags": []}}
               for a in acts]
    ctx = {"dominant_sources": {"dust": 0.6, "traffic": 0.2}}
    for r in ActionRanker().rank(actions, cf, ctx):
        print(f"#{r['rank']} {r['action']:22} score={r['score']:.3f} "
              f"imp%={r['improvement_pct']} obj={r['objectives']}")
