"""
advisor/retrieval/hybrid.py — PART 5: hybrid retrieval.

NOT a plain vector search. Four signals are fused:

  1. Semantic   — Chroma cosine similarity (dense meaning).
  2. BM25       — lexical overlap (exact policy terms).
  3. Metadata   — boost chunks whose aqi_stage / pollutant match the live context.
  4. Knowledge graph — boost chunks from policy docs the KG links to the context's
     {stage, sources} (relationship evidence, not just text).

Each signal is min-max normalised, then combined with the configured weights.
Returns a merged, de-duplicated Top-K ready for the cross-encoder reranker (Part 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from advisor.config import CONFIG
from advisor.embeddings.vector_store import get_vector_store
from advisor.retrieval.bm25 import get_bm25
from advisor.kg.knowledge_graph import get_knowledge_graph


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    citation: str
    score: float
    components: dict = field(default_factory=dict)

    def as_dict(self):
        return {"chunk_id": self.chunk_id, "text": self.text, "citation": self.citation,
                "title": self.metadata.get("title"), "authority": self.metadata.get("authority"),
                "aqi_stage": self.metadata.get("aqi_stage"), "score": round(self.score, 4),
                "components": {k: round(v, 4) for k, v in self.components.items()}}


def _minmax(d: dict) -> dict:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


@dataclass
class RetrievalContext:
    """The bits of the ward context that steer retrieval."""
    query: str
    aqi_stage: str | None = None
    sources: list = field(default_factory=list)
    pollutants: list = field(default_factory=list)


class HybridRetriever:
    def __init__(self, config=CONFIG, vector_store=None, bm25=None, kg=None):
        self.cfg = config
        self.vs = vector_store or get_vector_store()
        self.bm25 = bm25 or get_bm25()
        self.kg = kg or get_knowledge_graph()

    def retrieve(self, ctx: RetrievalContext, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.cfg.retrieval_top_k
        pool: dict[str, dict] = {}     # chunk_id -> {text, metadata, citation}

        def remember(r):
            pool.setdefault(r["chunk_id"], {"text": r["text"], "metadata": r["metadata"],
                                            "citation": r.get("citation", r["metadata"].get("citation", ""))})

        # 1. semantic
        sem_raw = {}
        for r in self.vs.query(ctx.query, top_k=top_k):
            remember(r); sem_raw[r["chunk_id"]] = r["score"]
        # 2. bm25
        bm_raw = {}
        for r in self.bm25.query(ctx.query, top_k=top_k):
            remember(r); bm_raw[r["chunk_id"]] = r["score"]
        # 4. knowledge-graph policy boost -> doc_ids
        kg_docs = {p["doc_id"]: p["weight"] for p in
                   self.kg.policies_for(stage=ctx.aqi_stage, sources=ctx.sources,
                                        pollutants=ctx.pollutants)}

        sem_n, bm_n = _minmax(sem_raw), _minmax(bm_raw)
        w = self.cfg
        results = []
        for cid, info in pool.items():
            meta = info["metadata"]
            s_sem, s_bm = sem_n.get(cid, 0.0), bm_n.get(cid, 0.0)
            # graph score from the chunk's doc
            doc_id = meta.get("doc_id", "")
            g = kg_docs.get(doc_id, 0.0)
            g_norm = min(g / 3.0, 1.0)
            # metadata match boost
            meta_bonus = 0.0
            st = meta.get("aqi_stage")
            if ctx.aqi_stage and st in (ctx.aqi_stage, "all"):
                meta_bonus += 0.5
            polls = str(meta.get("pollutant", ""))
            if any(p in polls for p in ctx.pollutants):
                meta_bonus += 0.3
            score = (w.semantic_weight * s_sem + w.bm25_weight * s_bm +
                     w.graph_weight * g_norm + 0.15 * meta_bonus)
            results.append(RetrievedChunk(
                chunk_id=cid, text=info["text"], metadata=meta, citation=info["citation"],
                score=score, components={"semantic": s_sem, "bm25": s_bm,
                                         "graph": g_norm, "metadata": meta_bonus}))
        results.sort(key=lambda r: -r.score)
        return results[:top_k]


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever()


if __name__ == "__main__":
    r = get_hybrid_retriever()
    ctx = RetrievalContext(
        query="AQI is very poor and traffic and dust are the dominant sources; what should be done?",
        aqi_stage="Stage II", sources=["traffic", "dust"], pollutants=["PM2.5", "PM10"])
    print(f"context: stage={ctx.aqi_stage} sources={ctx.sources}\n")
    for rc in r.retrieve(ctx, top_k=6):
        c = rc.components
        print(f"[{rc.score:.3f}] sem={c['semantic']:.2f} bm={c['bm25']:.2f} "
              f"kg={c['graph']:.2f} md={c['metadata']:.1f} | "
              f"{rc.metadata['authority']:5} {rc.metadata.get('aqi_stage'):8} | {rc.metadata['title'][:42]}")
