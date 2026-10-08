"""Hybrid retrieval package: chunking, dense+BM25+rerank, store, metrics."""

from .chunking import chunk_document, chunk_fixed, chunk_hierarchical, chunk_semantic
from .hybrid import BM25, Chunk, DenseEmbedder, HybridRetriever, Reranker, ScoredChunk
from .metrics import evaluate_run, hit_rate, mrr, precision_at_k, recall_at_k
from .store import DocumentStore

__all__ = [
    "BM25", "Chunk", "DenseEmbedder", "DocumentStore", "HybridRetriever",
    "Reranker", "ScoredChunk",
    "chunk_document", "chunk_fixed", "chunk_semantic", "chunk_hierarchical",
    "evaluate_run", "hit_rate", "mrr", "precision_at_k", "recall_at_k",
]
