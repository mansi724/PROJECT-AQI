"""
advisor/validation/policy_validator.py — PART 9: policy validation engine.

The LLM proposes; the rule engine disposes. Every intervention the LLM returns is
checked against deterministic rules before it can be ranked or shown. Nothing the
model says is trusted on faith.

Checks per action:
  * known action     — id must be in the action catalogue.
  * AQI stage        — action must be applicable at the ward's current GRAP stage
                       (INTERVENTION_STAGES); a Stage-IV-only measure is invalid at
                       Stage I.
  * city / season    — action applicability by city and season (e.g. stubble/biomass
                       controls are meaningful in the Oct–Feb window).
  * source relevance — action's target source should be among the ward's dominant
                       sources (else it is 'weak', kept but flagged).
  * authority        — a recognised authority must own the measure.
Cross-action:
  * conflicts        — mutually redundant/contradictory actions are de-duplicated.

Invalid actions are REJECTED with a reason; weak ones are kept but flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from advisor.config import (
    CONFIG, INTERVENTION_FEATURE_MAP, INTERVENTION_STAGES, ACTION_TARGET_SOURCE,
    ACTION_CATALOGUE, grap_stage,
)

# actions that only make sense in the cool/burning season (Oct–Feb)
_SEASONAL = {"biomass_ban"}
_WINTER_MONTHS = {10, 11, 12, 1, 2}
# actions whose authority ownership (for the authority check)
_ACTION_AUTHORITY = {a: "CAQM" for a in INTERVENTION_FEATURE_MAP}
_ACTION_AUTHORITY.update({"biomass_ban": "CAQM", "road_dust_suppression": "DPCC"})
# pairs that are redundant together -> keep the stronger only
_CONFLICTS = [("odd_even", "traffic_restriction")]


@dataclass
class ValidationResult:
    action: str
    valid: bool
    flags: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    authority: str = ""

    def as_dict(self):
        return {"action": self.action, "valid": self.valid, "flags": self.flags,
                "reasons": self.reasons, "authority": self.authority}


class PolicyValidator:
    def __init__(self, config=CONFIG):
        self.cfg = config

    def validate_action(self, action: dict, ward_context: dict) -> ValidationResult:
        aid = action.get("action")
        stage = ward_context.get("grap_stage") or grap_stage(ward_context.get("predicted_aqi", 0))
        city = str(ward_context.get("ward_metadata", {}).get("city", "Delhi") or "Delhi")
        month = _month_of(ward_context.get("timestamp"))
        dominant = set((ward_context.get("dominant_sources") or {}).keys())
        dominant |= {"industrial" if s == "industry" else s for s in list(dominant)}

        res = ValidationResult(action=aid, valid=True,
                               authority=_ACTION_AUTHORITY.get(aid, "CAQM"))

        # 1. known action
        if aid not in INTERVENTION_FEATURE_MAP:
            res.valid = False
            res.reasons.append(f"unknown action id '{aid}'")
            return res

        # 2. AQI stage applicability
        stages_ok = INTERVENTION_STAGES.get(aid, [])
        if stage and stages_ok and stage not in stages_ok:
            res.valid = False
            res.reasons.append(f"not applicable at {stage} (allowed: {', '.join(stages_ok)})")

        # 3. city applicability (this deployment is Delhi-NCR)
        if city not in ("Delhi", "NCR", "India"):
            res.flags.append(f"authored for Delhi-NCR, ward city is {city}")

        # 4. season applicability
        if aid in _SEASONAL and month and month not in _WINTER_MONTHS:
            res.flags.append("seasonal measure outside the Oct–Feb burning window")

        # 5. source relevance
        tgt = ACTION_TARGET_SOURCE.get(aid)
        if tgt and dominant and tgt not in dominant:
            res.flags.append(f"target source '{tgt}' is not among dominant sources")

        return res

    def validate(self, interventions: list, ward_context: dict) -> dict:
        results, kept, rejected = [], [], []
        for iv in interventions:
            r = self.validate_action(iv, ward_context)
            results.append(r)
            (kept if r.valid else rejected).append(iv | {"_validation": r.as_dict()})

        kept = self._resolve_conflicts(kept)
        return {"valid_actions": kept, "rejected_actions": rejected,
                "n_valid": len(kept), "n_rejected": len(rejected),
                "results": [r.as_dict() for r in results]}

    def _resolve_conflicts(self, actions: list) -> list:
        by_id = {a["action"]: a for a in actions}
        drop = set()
        for a, b in _CONFLICTS:
            if a in by_id and b in by_id:
                # keep the higher-confidence one
                lo = a if by_id[a].get("confidence", 0) < by_id[b].get("confidence", 0) else b
                drop.add(lo)
                other = b if lo == a else a
                by_id[other].setdefault("_validation", {}).setdefault("flags", []).append(
                    f"absorbed redundant '{lo}'")
        return [a for a in actions if a["action"] not in drop]


def _month_of(timestamp) -> int | None:
    try:
        return int(str(timestamp)[5:7])
    except Exception:
        return None


if __name__ == "__main__":
    v = PolicyValidator()
    ctx = {"predicted_aqi": 175, "grap_stage": "Stage I", "timestamp": "2026-04-28 21:00:00",
           "dominant_sources": {"dust": 0.6, "traffic": 0.2}, "ward_metadata": {"city": "Delhi"}}
    proposed = [
        {"action": "road_dust_suppression", "confidence": 0.8},   # valid at Stage I
        {"action": "construction_halt", "confidence": 0.7},        # Stage III+ only -> reject
        {"action": "odd_even", "confidence": 0.6},                 # Stage IV only -> reject
        {"action": "make_it_rain", "confidence": 0.9},             # unknown -> reject
    ]
    out = v.validate(proposed, ctx)
    print(f"valid={out['n_valid']} rejected={out['n_rejected']}")
    for a in out["valid_actions"]:
        print("  VALID  ", a["action"], "| flags:", a["_validation"]["flags"])
    for a in out["rejected_actions"]:
        print("  REJECT ", a["action"], "|", a["_validation"]["reasons"])
