"""
advisor/kg/enrich_kg.py — improvement 6.1: LLM-based knowledge-graph enrichment.

The base graph links policies to sources/interventions by keyword *tags*
(`_link_policies`). This step upgrades that: it asks the LLM to read each policy
document and classify — from the FIXED vocabularies — which pollution sources it
targets and which interventions it prescribes. The resulting `addresses` edges
are more accurate than tag-matching, and the value grows as real official
documents are added (5.1).

Constrained + grounded by construction: the LLM may only pick from the known
source / intervention ids, so it can't invent graph nodes. No-op (with a clear
message) if no LLM key is configured.

    python -m advisor.kg.enrich_kg
"""
from __future__ import annotations

import json

from advisor.config import INTERVENTION_FEATURE_MAP
from advisor.kb.ingest import Ingestor
from advisor.kg.knowledge_graph import KnowledgeGraph, get_knowledge_graph, SOURCE_POLLUTANTS
from advisor.llm.client import get_llm_client

SOURCES = list(SOURCE_POLLUTANTS.keys())
ACTIONS = list(INTERVENTION_FEATURE_MAP.keys())

SYSTEM = ("You classify Indian air-quality policy documents. Use ONLY the provided "
          "id lists. Do not invent ids. Return strict JSON.")


def _classify(client, title: str, text: str) -> dict:
    prompt = (
        f"SOURCE ids: {SOURCES}\n"
        f"INTERVENTION ids: {ACTIONS}\n\n"
        f"Document title: {title}\n"
        f"Excerpt:\n{text[:1800]}\n\n"
        f"Which pollution SOURCES does this document target, and which INTERVENTIONS "
        f"does it prescribe? Choose only from the id lists above. "
        f'Return JSON: {{"sources": [...], "interventions": [...]}}')
    raw = client.generate(SYSTEM, prompt, json_mode=True)
    try:
        d = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return {"sources": [], "interventions": []}
    return {"sources": [s for s in d.get("sources", []) if s in SOURCES],
            "interventions": [a for a in d.get("interventions", []) if a in ACTIONS]}


def enrich(save: bool = True) -> dict:
    client = get_llm_client()
    if client.is_mock:
        print("[enrich_kg] no LLM key -> skipping (set GROQ_API_KEY to enable 6.1).")
        return {"skipped": True}

    kg = get_knowledge_graph()
    chunks = Ingestor.load_chunks()
    # group chunk text per doc
    by_doc: dict[str, dict] = {}
    for c in chunks:
        d = by_doc.setdefault(c.doc_id, {"title": c.metadata.get("title", ""), "text": ""})
        if len(d["text"]) < 1800:
            d["text"] += " " + c.text

    added = 0
    for doc_id, info in by_doc.items():
        if doc_id not in kg.g:
            continue
        res = _classify(client, info["title"], info["text"])
        for s in res["sources"]:
            kg._add(s, "source"); kg._rel(doc_id, s, "addresses"); added += 1
        for a in res["interventions"]:
            if a in kg.g:
                kg._rel(doc_id, a, "addresses"); added += 1
        print(f"  {info['title'][:44]:44} -> sources {res['sources']} | actions {len(res['interventions'])}")

    if save:
        kg.save()
    print(f"\n[enrich_kg] added {added} LLM-derived 'addresses' edges | {kg.stats()}")
    return {"added_edges": added, **kg.stats()}


if __name__ == "__main__":
    enrich()
