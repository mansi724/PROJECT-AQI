"""
advisor/retrieval/eval_retrieval.py — PART 5.7: retrieval evaluation harness.

Turns "is retrieval good?" into a number. A small labelled set maps realistic
queries to the document(s) that SHOULD be retrieved (by `document_type` +
optional `aqi_stage`). We score the hybrid retriever with:

  * Recall@k — did a relevant chunk make the top-k?
  * MRR      — how high did the first relevant chunk rank?

It also runs an **ablation**: expansion on vs off (5.6), and semantic-only vs
full hybrid — so weight/config changes can be judged by evidence, not vibes.

    python -m advisor.retrieval.eval_retrieval
"""
from __future__ import annotations

from dataclasses import replace

from advisor.config import CONFIG
from advisor.retrieval.hybrid import HybridRetriever, RetrievalContext

# labelled set: query -> relevance rule (doc_type, optional aqi_stage substring)
GOLD = [
    ("What actions are required when AQI is severe?", "grap", "Stage III"),
    ("emergency truck ban and construction halt", "grap", "Stage IV"),
    ("restrictions on BS-III diesel four wheelers", "grap", "Stage III"),
    ("mechanised sweeping and water sprinkling for dust", "grap", None),
    ("ban on firecrackers during Diwali", "dpcc_notification", None),
    ("stubble burning management in Punjab and Haryana", "caqm_order", None),
    ("WHO safe annual limit for PM2.5", "who_guideline", None),
    ("national clean air programme reduction target", "ncap", None),
    ("AQI category health advisory for sensitive groups", "cpcb_guideline", None),
    ("odd-even vehicle rationing scheme", "grap", "Stage IV"),
    ("close brick kilns and stone crushers", "grap", "Stage III"),
    ("augment bus and metro public transport", "grap", "Stage II"),
]


def _relevant(meta, doc_type, stage) -> bool:
    if meta.get("document_type") != doc_type:
        return False
    if stage and stage not in str(meta.get("aqi_stage", "")):
        return False
    return True


def evaluate(retriever, k=6, expansion=True, weights=None) -> dict:
    cfg = retriever.cfg
    if weights or (expansion is not None):
        cfg = replace(cfg, query_expansion=expansion, **(weights or {}))
        retriever = HybridRetriever(config=cfg, vector_store=retriever.vs,
                                    bm25=retriever.bm25, kg=retriever.kg)
    hits, rr = 0, 0.0
    for q, dt, stage in GOLD:
        res = retriever.retrieve(RetrievalContext(query=q), top_k=k)
        ranks = [i for i, r in enumerate(res) if _relevant(r.metadata, dt, stage)]
        if ranks:
            hits += 1
            rr += 1.0 / (ranks[0] + 1)
    n = len(GOLD)
    return {"recall@k": round(hits / n, 3), "MRR": round(rr / n, 3), "n": n, "k": k}


def main():
    from advisor.retrieval.hybrid import get_hybrid_retriever
    r = get_hybrid_retriever()
    print(f"Retrieval eval on {len(GOLD)} labelled queries (k={CONFIG.final_top_k})\n")
    print("  full hybrid + expansion :", evaluate(r, expansion=True))
    print("  full hybrid, NO expansion:", evaluate(r, expansion=False))
    print("  semantic-only (bm25=0,kg=0):",
          evaluate(r, expansion=False, weights={"semantic_weight": 1.0, "bm25_weight": 0.0,
                                                "graph_weight": 0.0}))


if __name__ == "__main__":
    main()
