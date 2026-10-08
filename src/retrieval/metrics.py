"""Retrieval quality metrics: precision@k, recall@k, MRR, hit rate.

Conventions: `retrieved` is an ordered list of chunk ids; `relevant` is the
set of chunk ids judged relevant for the query.
"""

from __future__ import annotations

from typing import Dict, List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for cid in top if cid in relevant) / len(top)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for cid in top if cid in relevant) / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved: List[str], relevant: Set[str], k: int) -> float:
    return 1.0 if any(cid in relevant for cid in retrieved[:k]) else 0.0


def evaluate_run(
    runs: List[Dict[str, object]], k_values: List[int] = (1, 3, 5)
) -> Dict[str, float]:
    """Aggregate metrics over eval runs.

    Each run: {"retrieved": [chunk_ids...], "relevant": {chunk_ids...}}.
    """
    agg: Dict[str, float] = {}
    n = max(len(runs), 1)
    for k in k_values:
        agg[f"precision@{k}"] = sum(
            precision_at_k(r["retrieved"], r["relevant"], k) for r in runs  # type: ignore
        ) / n
        agg[f"recall@{k}"] = sum(
            recall_at_k(r["retrieved"], r["relevant"], k) for r in runs  # type: ignore
        ) / n
        agg[f"hit_rate@{k}"] = sum(
            hit_rate(r["retrieved"], r["relevant"], k) for r in runs  # type: ignore
        ) / n
    agg["mrr"] = sum(mrr(r["retrieved"], r["relevant"]) for r in runs) / n  # type: ignore
    return agg
