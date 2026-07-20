"""
advisor/retrieval/reranker.py — PART 6: cross-encoder reranker.

Hybrid retrieval optimises recall (cast a wide net). The cross-encoder optimises
precision: it scores each (query, chunk) PAIR jointly — far more accurate than the
bi-encoder cosine used for first-stage retrieval — so only the genuinely most
relevant chunks reach the LLM.

Default: `cross-encoder/ms-marco-MiniLM-L-6-v2` (small, strong). Swap for
`BAAI/bge-reranker-base` via `ADVISOR_RERANKER` if you prefer. If the model can't
load (offline), it degrades to an identity passthrough that preserves the hybrid
order, flagged by `.is_active`.
"""
from __future__ import annotations

from functools import lru_cache

from advisor.config import CONFIG


class Reranker:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or CONFIG.reranker_model
        self._model = None
        self.is_active = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"[reranker] '{self.model_name}' unavailable ({e}); "
                  f"passthrough (hybrid order kept).")
            self.is_active = False

    def rerank(self, query: str, chunks: list, top_k: int = None) -> list:
        top_k = top_k or CONFIG.final_top_k
        if not chunks:
            return []
        if self._model is None:
            out = chunks[:top_k]
            for c in out:
                _set(c, "rerank_score", None)
            return out
        pairs = [[query, _text(c)] for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(chunks, scores), key=lambda cs: -float(cs[1]))
        out = []
        for c, s in ranked[:top_k]:
            _set(c, "rerank_score", float(s))
            out.append(c)
        return out


def _text(c):
    return c.text if hasattr(c, "text") else c.get("text", "")


def _set(c, k, v):
    if hasattr(c, "components"):
        c.components[k] = v if v is not None else 0.0
    elif isinstance(c, dict):
        c[k] = v


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


if __name__ == "__main__":
    from advisor.retrieval.hybrid import get_hybrid_retriever, RetrievalContext
    ctx = RetrievalContext(
        query="Very poor air quality driven by traffic; which restrictions apply and to whom?",
        aqi_stage="Stage II", sources=["traffic"], pollutants=["PM2.5", "NO2"])
    cands = get_hybrid_retriever().retrieve(ctx, top_k=12)
    print(f"hybrid candidates: {len(cands)}")
    rr = get_reranker()
    print("reranker active:", rr.is_active)
    for c in rr.rerank(ctx.query, cands, top_k=5):
        rs = c.components.get("rerank_score")
        print(f"  rerank={rs if rs is None else round(rs,3)} | "
              f"{c.metadata['authority']:5} {c.metadata.get('aqi_stage'):8} | {c.metadata['title'][:48]}")
