"""
advisor/llm/reasoner.py — PART 8: grounded LLM reasoning.

Builds a strictly-grounded prompt and returns structured JSON. The model is
instructed to (1) analyse the AQI prediction, (2) analyse source attribution,
(3) analyse the retrieved policies, (4) propose interventions from a FIXED action
vocabulary, (5) justify each, (6) cite the supporting documents, (7) assign a
confidence — and to use ONLY the supplied context (no outside facts).

The chosen `action` ids come from the shared catalogue, so the counterfactual
engine (Part 10) and ranker (Part 11) can consume them directly.
"""
from __future__ import annotations

import json

from advisor.config import CONFIG, INTERVENTION_FEATURE_MAP, ACTION_CATALOGUE
from advisor.llm.client import get_llm_client, LLMClient

ALLOWED_ACTIONS = list(INTERVENTION_FEATURE_MAP.keys())

SYSTEM_PROMPT = (
    "You are an air-quality policy advisor for Delhi wards. You must ground every "
    "statement ONLY in the provided ward context and the retrieved policy excerpts. "
    "Do NOT invent facts, numbers, policies, or citations. If the evidence is "
    "insufficient, say so and lower the confidence. All quantitative values come "
    "from the context you are given — never compute or guess new ones. "
    "Return STRICT JSON only, no prose outside the JSON."
)

_OUTPUT_SCHEMA = """
Return JSON with exactly these keys:
{
  "aqi_assessment": "<1-2 sentences on the predicted AQI, band and GRAP stage>",
  "source_analysis": "<1-2 sentences on the dominant sources and why>",
  "policy_basis": ["<citation strings of the policies you relied on>"],
  "interventions": [
    {
      "action": "<one of the ALLOWED_ACTIONS ids>",
      "title": "<short human title>",
      "target_source": "<the source this addresses>",
      "rationale": "<why, grounded in the retrieved policy + attribution>",
      "citations": ["<citation strings from the excerpts>"],
      "confidence": <0..1>
    }
  ]
}
"""


class Reasoner:
    def __init__(self, client: LLMClient | None = None, config=CONFIG):
        self.cfg = config
        self.client = client or get_llm_client()

    def _user_prompt(self, context: dict, chunks: list) -> str:
        cites = [{"citation": getattr(c, "citation", ""),
                  "title": (c.metadata if hasattr(c, "metadata") else {}).get("title", ""),
                  "text": getattr(c, "text", "")[:600]} for c in chunks]
        payload = {"context": context, "citations": cites}
        excerpts = "\n\n".join(
            f"[{i+1}] {c['citation']}\n{c['text']}" for i, c in enumerate(cites))
        catalogue = {a: ACTION_CATALOGUE.get(a, {}).get("label", a) for a in ALLOWED_ACTIONS}
        return (
            f"ALLOWED_ACTIONS (choose only from these ids): {json.dumps(catalogue)}\n\n"
            f"WARD CONTEXT and evidence are between the markers below; the mock "
            f"reasoner also reads them, so keep them intact.\n"
            f"<<<CONTEXT_JSON>>>{json.dumps(payload)}<<<END_CONTEXT_JSON>>>\n\n"
            f"RETRIEVED POLICY EXCERPTS:\n{excerpts}\n\n"
            f"TASK: Follow the 7 steps and {_OUTPUT_SCHEMA}"
        )

    def reason(self, context: dict, chunks: list) -> dict:
        prompt = self._user_prompt(context, chunks)
        raw = self.client.generate(SYSTEM_PROMPT, prompt)
        data = _extract_json(raw)
        return self._sanitise(data, context)

    def _sanitise(self, data: dict, context: dict) -> dict:
        if not isinstance(data, dict):
            data = {}
        data.setdefault("aqi_assessment", "")
        data.setdefault("source_analysis", "")
        data.setdefault("policy_basis", [])
        acts = data.get("interventions") or []
        clean = []
        for a in acts:
            if not isinstance(a, dict):
                continue
            action = a.get("action")
            if action not in INTERVENTION_FEATURE_MAP:      # drop invalid/hallucinated ids
                continue
            a.setdefault("title", ACTION_CATALOGUE.get(action, {}).get("label", action))
            a["confidence"] = float(a.get("confidence", 0.6))
            a.setdefault("citations", [])
            clean.append(a)
        data["interventions"] = clean
        data["_provider"] = self.client.provider
        return data


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except Exception:
            return {}
    return {}


if __name__ == "__main__":
    from advisor.context_builder import ContextBuilder
    from advisor.retrieval.hybrid import get_hybrid_retriever, RetrievalContext
    from advisor.retrieval.reranker import get_reranker

    wc = ContextBuilder().build(ward_id="239", time_index=24477, include_explanations=False)
    rctx = RetrievalContext(
        query=f"AQI {wc.predicted_aqi} {wc.aqi_band}; sources {list(wc.dominant_sources)}; actions?",
        aqi_stage=wc.grap_stage or "Stage I", sources=list(wc.dominant_sources),
        pollutants=["PM2.5", "PM10"])
    cands = get_hybrid_retriever().retrieve(rctx, top_k=10)
    top = get_reranker().rerank(rctx.query, cands, top_k=5)

    out = Reasoner().reason(wc.to_dict(), top)
    print("provider:", out.get("_provider"))
    print("aqi_assessment:", out["aqi_assessment"])
    print("interventions:")
    for iv in out["interventions"]:
        print(f"  - {iv['action']:22} conf={iv.get('confidence')} | {iv.get('title')}")
