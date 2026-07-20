"""
advisor/embeddings/embedder.py — PART 4: embedding generation.

A thin, cached wrapper over Sentence-Transformers so the whole stack shares ONE
loaded model. If the model cannot be loaded (e.g. fully offline first run with no
cache), it falls back to a deterministic hashing embedder so downstream code and
tests still run — clearly flagged via `.is_semantic`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from advisor.config import CONFIG


class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or CONFIG.embedding_model
        self._model = None
        self.is_semantic = True
        self._dim = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"[embedder] '{self.model_name}' unavailable ({e}); "
                  f"using deterministic hashing fallback.")
            self.is_semantic = False
            self._dim = 384

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self._model is not None:
            vecs = self._model.encode(list(texts), normalize_embeddings=normalize,
                                      show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        return self._hash_encode(texts, normalize)

    def _hash_encode(self, texts, normalize: bool) -> np.ndarray:
        """Fallback: hashed bag-of-tokens projected to `dim`. Not semantic, but
        stable and non-zero so retrieval plumbing is testable offline."""
        out = np.zeros((len(texts), self._dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                out[i, hash(tok) % self._dim] += 1.0
        if normalize:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.clip(n, 1e-9, None)
        return out


@lru_cache(maxsize=2)
def get_embedder(model_name: str = None) -> Embedder:
    return Embedder(model_name)


if __name__ == "__main__":
    emb = get_embedder()
    v = emb.encode(["GRAP Stage III bans construction",
                    "traffic restrictions reduce NO2"])
    print("model:", emb.model_name, "| semantic:", emb.is_semantic, "| dim:", emb.dim)
    print("shape:", v.shape, "| row norm:", round(float(np.linalg.norm(v[0])), 3))
    # cosine similarity sanity
    a = emb.encode("construction dust ban during severe air quality")[0]
    b = emb.encode("halt building activity when AQI is severe")[0]
    c = emb.encode("odd even traffic scheme")[0]
    print("sim(construction, construction-para):", round(float(a @ b), 3),
          "| sim(construction, traffic):", round(float(a @ c), 3))
