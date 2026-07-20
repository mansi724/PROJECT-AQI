"""
advisor/retrieval/bm25.py — lexical BM25 index over the chunk corpus.

The keyword half of hybrid retrieval: catches exact terms an embedding may blur
(e.g. "BS-III", "odd-even", "PM2.5", "Stage IV") that matter a lot for policy.
"""
from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from advisor.kb.ingest import Ingestor

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


class BM25Index:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self.chunks}
        corpus = [tokenize(c.text + " " + c.metadata.get("title", "")) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def query(self, text: str, top_k: int = 20) -> list[dict]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(text))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [{"chunk_id": self.chunks[i].chunk_id,
                 "text": self.chunks[i].text,
                 "metadata": self.chunks[i].metadata,
                 "citation": self.chunks[i].citation,
                 "score": float(scores[i])} for i in order if scores[i] > 0]


@lru_cache(maxsize=1)
def get_bm25() -> BM25Index:
    return BM25Index(Ingestor.load_chunks())


if __name__ == "__main__":
    idx = get_bm25()
    for q in ["BS-III diesel four wheeler restriction", "firecracker ban Diwali"]:
        print(f"\nquery: {q!r}")
        for r in idx.query(q, top_k=3):
            print(f"  [{r['score']:.2f}] {r['metadata']['title'][:55]} | {r['text'][:70]}...")
