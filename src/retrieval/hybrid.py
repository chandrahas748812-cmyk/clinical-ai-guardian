"""Hybrid retrieval: dense + BM25 sparse + cross-encoder rerank.

Pipeline: query -> [dense search, BM25 search] -> reciprocal rank fusion ->
cross-encoder rerank -> top-k chunks with citations.

All heavy dependencies are optional:
- sentence-transformers -> deterministic hash embeddings (mock) when missing
- cross-encoders -> lexical-overlap reranker (mock) when missing
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:  # optional real embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False


# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------

def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic mock embedding: hashed token buckets, L2-normalized.

    Not semantically meaningful — but deterministic, so retrieval, fusion and
    tests are reproducible offline. Real deployments swap in
    sentence-transformers via `DenseEmbedder`.
    """
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h >> 16) % dim] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class DenseEmbedder:
    """Dense embeddings with automatic mock fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.dim = dim
        self.backend = "mock"
        self._model = None
        if _ST_AVAILABLE:
            try:
                self._model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
                self.dim = self._model.get_sentence_embedding_dimension()
            except Exception:
                self._model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            return [list(map(float, v)) for v in self._model.encode(texts)]
        return [_hash_embed(t, self.dim) for t in texts]


def cosine(a: List[float], b: List[float]) -> float:
    denom = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
    return sum(x * y for x, y in zip(a, b)) / denom


# ----------------------------------------------------------------------------
# BM25 (from-scratch, dependency-free)
# ----------------------------------------------------------------------------

class BM25:
    """Classic BM25 over a static corpus of chunk texts."""

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self._tokens = [self._tok(d) for d in docs]
        self._avgdl = sum(len(t) for t in self._tokens) / max(len(self._tokens), 1)
        df: Dict[str, int] = {}
        for toks in self._tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def scores(self, query: str) -> List[float]:
        q_terms = self._tok(query)
        out = []
        for toks in self._tokens:
            dl = len(toks) or 1
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += idf * num / den
            out.append(s)
        return out


# ----------------------------------------------------------------------------
# Reranker (cross-encoder with mock fallback)
# ----------------------------------------------------------------------------

class Reranker:
    """Cross-encoder rerank with deterministic lexical fallback."""

    def __init__(self):
        self.backend = "mock"
        self._model = None
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.backend = "cross-encoder"
        except Exception:
            self._model = None

    def rerank(self, query: str, docs: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        if self._model is not None:  # pragma: no cover
            scores = self._model.predict([(query, d) for d in docs])
            ranked = sorted(enumerate(docs), key=lambda p: scores[p[0]], reverse=True)
            return [(i, float(scores[i])) for i, _ in ranked[:top_k]]
        scored = []
        for i, d in enumerate(docs):
            q = set(re.findall(r"[a-z0-9]+", query.lower()))
            dw = set(re.findall(r"[a-z0-9]+", d.lower()))
            stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "was", "for"}
            overlap = len((q - stop) & (dw - stop))
            scored.append((i, overlap / max(len(q - stop), 1)))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_k]


# ----------------------------------------------------------------------------
# Chunk
# ----------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    start: int
    end: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str  # dense | bm25 | fused | reranked


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    """RRF fusion of multiple ranked index lists -> [(doc_idx, fused_score)]."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda p: p[1], reverse=True)


class HybridRetriever:
    """Dense + BM25 -> RRF -> cross-encoder rerank."""

    def __init__(
        self,
        chunks: List[Chunk],
        embedder: Optional[DenseEmbedder] = None,
        reranker: Optional[Reranker] = None,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.chunks = chunks
        self.embedder = embedder or DenseEmbedder()
        self.reranker = reranker or Reranker()
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._doc_embs = self.embedder.embed([c.text for c in chunks])
        self._bm25 = BM25([c.text for c in chunks])

    def _dense_rank(self, query: str, top_n: int) -> List[int]:
        q = self.embedder.embed([query])[0]
        scored = sorted(
            range(len(self.chunks)),
            key=lambda i: cosine(q, self._doc_embs[i]),
            reverse=True,
        )
        return scored[:top_n]

    def _bm25_rank(self, query: str, top_n: int) -> List[int]:
        scores = self._bm25.scores(query)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

    def retrieve(self, query: str, top_k: int = 5, prefetch: int = 20) -> List[ScoredChunk]:
        dense_rank = self._dense_rank(query, prefetch)
        bm25_rank = self._bm25_rank(query, prefetch)
        fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
        cand_idx = [i for i, _ in fused[: prefetch * 2 // 2 + prefetch // 2]][:prefetch]
        if not cand_idx:
            cand_idx = [i for i, _ in fused[:top_k]]
        reranked = self.reranker.rerank(query, [self.chunks[i].text for i in cand_idx], top_k=top_k)
        return [
            ScoredChunk(chunk=self.chunks[cand_idx[i]], score=float(s), source="reranked")
            for i, s in reranked
        ]
