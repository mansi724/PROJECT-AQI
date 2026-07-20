"""
advisor/embeddings/vector_store.py — PART 4: ChromaDB vector store.

Persistent, cosine-space Chroma collection with:
  * incremental indexing  — only new/changed chunks are embedded (chunk_id carries
    a content hash, so an edited chunk gets a new id and is re-added).
  * document updates       — `update_document(doc_id, chunks)` deletes the old
    version's chunks then upserts the new ones.
  * metadata-filtered search for the hybrid retriever (Part 5).

Chroma metadata must be scalar, so list fields (pollutant, tags) are flattened to
comma-joined strings; retrieval filters on the flattened forms.
"""
from __future__ import annotations

from functools import lru_cache

import chromadb

from advisor.config import CONFIG
from advisor.kb.schema import Chunk
from advisor.embeddings.embedder import get_embedder


def _flatten_meta(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        if isinstance(v, (list, tuple)):
            out[k] = ", ".join(str(x) for x in v)
        elif v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class VectorStore:
    def __init__(self, config=CONFIG, embedder=None):
        self.cfg = config
        config.ensure_dirs()
        self.embedder = embedder or get_embedder()
        self.client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=config.chroma_collection, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.collection.count()

    def existing_ids(self) -> set:
        got = self.collection.get(include=[])
        return set(got.get("ids", []))

    def index_chunks(self, chunks: list[Chunk], incremental: bool = True) -> dict:
        if incremental:
            have = self.existing_ids()
            todo = [c for c in chunks if c.chunk_id not in have]
        else:
            todo = list(chunks)
        if not todo:
            return {"added": 0, "total": self.count()}
        embs = self.embedder.encode([c.text for c in todo]).tolist()
        self.collection.upsert(
            ids=[c.chunk_id for c in todo],
            embeddings=embs,
            documents=[c.text for c in todo],
            metadatas=[_flatten_meta({**c.metadata, "citation": c.citation,
                                      "doc_id": c.doc_id}) for c in todo],
        )
        return {"added": len(todo), "total": self.count()}

    def update_document(self, doc_id: str, chunks: list[Chunk]) -> dict:
        self.collection.delete(where={"doc_id": doc_id})
        doc_chunks = [c for c in chunks if c.doc_id == doc_id]
        return self.index_chunks(doc_chunks, incremental=False)

    def query(self, text: str, top_k: int = None, where: dict | None = None) -> list[dict]:
        top_k = top_k or self.cfg.retrieval_top_k
        q = self.embedder.encode([text]).tolist()
        res = self.collection.query(query_embeddings=q, n_results=top_k, where=where)
        out = []
        ids = res["ids"][0]
        for i, cid in enumerate(ids):
            out.append({
                "chunk_id": cid,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": 1.0 - float(res["distances"][0][i]),   # cosine sim
            })
        return out

    def reset(self):
        self.client.delete_collection(self.cfg.chroma_collection)
        self.collection = self.client.get_or_create_collection(
            name=self.cfg.chroma_collection, metadata={"hnsw:space": "cosine"})


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()


def build_index(reset: bool = False) -> dict:
    """Ingest corpus -> chunks -> (re)index. The Part-4 entry point."""
    from advisor.kb.ingest import Ingestor
    ing = Ingestor()
    chunks = ing.ingest()
    ing.write(chunks)
    vs = get_vector_store()
    if reset:
        vs.reset()
    stats = vs.index_chunks(chunks, incremental=not reset)
    return {"chunks": len(chunks), **stats}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--query", default="What actions are required when AQI is severe (Stage III)?")
    args = ap.parse_args()

    print("build:", build_index(reset=args.reset))
    vs = get_vector_store()
    print(f"\nquery: {args.query!r}")
    for r in vs.query(args.query, top_k=4):
        print(f"  [{r['score']:.3f}] {r['metadata']['authority']:6} "
              f"{r['metadata'].get('aqi_stage'):9} | {r['metadata']['title'][:50]}")
        print(f"          {r['text'][:110]}...")
    print("\nmetadata-filtered (aqi_stage='Stage IV'):")
    for r in vs.query("emergency truck ban", top_k=3, where={"aqi_stage": "Stage IV"}):
        print(f"  [{r['score']:.3f}] {r['metadata']['title'][:60]}")
