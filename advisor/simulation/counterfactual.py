"""
advisor/simulation/counterfactual.py — PART 10: counterfactual "what-if" engine.

Does NOT retrain anything. It reuses the already-trained Graph Transformer: for a
candidate action it edits the relevant RAW input feature(s), re-scales them with
the frozen preprocessing scalers, and re-runs inference — reading off the new AQI.

    traffic_restriction  ->  NO2 x0.75, CO x0.80, PM2.5 x0.92  ->  re-run GNN
    AQI 318  ->  279

Supports:
  * single action, at a given intensity in [0,1] (0 = no change, 1 = full effect);
  * multiple simultaneous interventions (feature multipliers compose);
  * arbitrary raw-feature overrides (for the example's explicit "traffic index").
"""
from __future__ import annotations

from dataclasses import dataclass

from advisor.config import CONFIG, INTERVENTION_FEATURE_MAP, ACTION_CATALOGUE
from advisor.serving import ForecastService, get_forecast_service


@dataclass
class CounterfactualResult:
    aqi_before: float
    aqi_after: float
    delta: float
    improvement_pct: float
    applied_features: dict

    def as_dict(self):
        return {"aqi_before": self.aqi_before, "aqi_after": self.aqi_after,
                "delta": self.delta, "improvement_pct": round(self.improvement_pct, 1),
                "applied_features": self.applied_features}


def _scaled_multipliers(action_ids, intensity: float = 1.0) -> dict:
    """Compose per-feature multipliers across actions, scaled by intensity.
    effective = 1 - intensity*(1 - base_multiplier); multipliers compose by product."""
    merged: dict[str, float] = {}
    for aid in action_ids:
        for feat, mult in INTERVENTION_FEATURE_MAP.get(aid, {}).items():
            eff = 1.0 - intensity * (1.0 - mult)
            merged[feat] = merged.get(feat, 1.0) * eff
    return merged


class CounterfactualSimulator:
    def __init__(self, service: ForecastService | None = None, config=CONFIG):
        self.cfg = config
        self.svc = service or get_forecast_service()

    def simulate_action(self, node_idx: int, t: int, action_id: str,
                        intensity: float = 1.0) -> CounterfactualResult:
        mults = _scaled_multipliers([action_id], intensity)
        return self._run(node_idx, t, mults)

    def simulate_actions(self, node_idx: int, t: int, action_ids: list,
                         intensity: float = 1.0) -> CounterfactualResult:
        """Multiple simultaneous interventions (composed)."""
        mults = _scaled_multipliers(action_ids, intensity)
        return self._run(node_idx, t, mults)

    def simulate_raw(self, node_idx: int, t: int, feature_multipliers: dict) -> CounterfactualResult:
        """Explicit raw-feature what-if, e.g. {'nitrogen_dioxide': 0.62/0.83}."""
        return self._run(node_idx, t, feature_multipliers)

    def _run(self, node_idx: int, t: int, mults: dict) -> CounterfactualResult:
        cf = self.svc.counterfactual(node_idx, t, mults)
        before, after = cf["aqi_before"], cf["aqi_after"]
        imp = 100.0 * (before - after) / before if before else 0.0
        return CounterfactualResult(aqi_before=before, aqi_after=after,
                                    delta=cf["delta"], improvement_pct=imp,
                                    applied_features=cf["applied_features"])

    def rank_actions_by_effect(self, node_idx: int, t: int, action_ids: list,
                               intensity: float = 1.0) -> list[dict]:
        """Per-action AQI improvement — the effectiveness input to Part 11 ranking."""
        out = []
        for aid in action_ids:
            r = self.simulate_action(node_idx, t, aid, intensity)
            out.append({"action": aid, "label": ACTION_CATALOGUE.get(aid, {}).get("label", aid),
                        **r.as_dict()})
        out.sort(key=lambda d: d["delta"])   # most negative delta (biggest drop) first
        return out


if __name__ == "__main__":
    sim = CounterfactualSimulator()
    node = sim.svc.ward_to_node("239")
    t = sim.svc.resolve_time(time_index=24477)

    print("single action (construction_halt):",
          sim.simulate_action(node, t, "construction_halt").as_dict())
    print("\nexplicit raw what-if (NO2 0.83->0.62):",
          sim.simulate_raw(node, t, {"nitrogen_dioxide": 0.62 / 0.83}).as_dict())
    print("\ncombined [construction_halt, road_dust_suppression]:",
          sim.simulate_actions(node, t, ["construction_halt", "road_dust_suppression"]).as_dict())
    print("\nper-action effect ranking:")
    for r in sim.rank_actions_by_effect(node, t, ["construction_halt", "road_dust_suppression",
                                                  "traffic_restriction", "industrial_restriction"]):
        print(f"  {r['action']:22} {r['aqi_before']:.0f} -> {r['aqi_after']:.0f} "
              f"(d={r['delta']:+.1f}, {r['improvement_pct']:.1f}%)")
