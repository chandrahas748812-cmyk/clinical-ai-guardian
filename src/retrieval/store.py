"""In-memory document store: chunking + indexing + hybrid retrieval.

Production deployments swap this for a managed vector DB (pgvector /
OpenSearch / Pinecone); the interface is deliberately narrow so the swap is
one class.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .chunking import chunk_document
from .hybrid import Chunk, DenseEmbedder, HybridRetriever, Reranker, ScoredChunk


class DocumentStore:
    """Chunk, index, and search clinical documents."""

    def __init__(
        self,
        chunk_strategy: str = "semantic",
        chunk_kwargs: Optional[dict] = None,
        embedder: Optional[DenseEmbedder] = None,
    ):
        self.chunk_strategy = chunk_strategy
        self.chunk_kwargs = chunk_kwargs or {}
        self._embedder = embedder
        self.chunks: List[Chunk] = []
        self._docs: Dict[str, str] = {}
        self._retriever: Optional[HybridRetriever] = None

    def add_document(self, doc_id: str, text: str, meta: Optional[dict] = None) -> int:
        """Chunk and stage a document. Returns number of chunks created."""
        self._docs[doc_id] = text
        chunks = chunk_document(text, doc_id, self.chunk_strategy, **self.chunk_kwargs)
        for c in chunks:
            if meta:
                c.meta.update(meta)
        self.chunks.extend(chunks)
        self._retriever = None  # invalidate index
        return len(chunks)

    def build_index(self) -> None:
        """Build dense + BM25 indexes over all staged chunks."""
        self._retriever = HybridRetriever(
            self.chunks,
            embedder=self._embedder or DenseEmbedder(),
            reranker=Reranker(),
        )

    def search(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        if self._retriever is None:
            self.build_index()
        assert self._retriever is not None
        return self._retriever.retrieve(query, top_k=top_k)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        for c in self.chunks:
            if c.id == chunk_id:
                return c
        return None

    def stats(self) -> dict:
        return {
            "documents": len(self._docs),
            "chunks": len(self.chunks),
            "strategy": self.chunk_strategy,
        }
