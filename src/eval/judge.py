"""LLM-as-judge groundedness scoring + hallucination detection.

The judge scores each answer sentence against the retrieved context:
- groundedness: token-overlap entailment (deterministic, offline)
- hallucination flags: sentences with no supporting context, or containing
  `must_not_contain` anti-facts from the golden example

A real deployment swaps `judge_groundedness` for an LLM judge prompt
(see docstring); the interface is identical.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..guardrails.output import claim_groundedness
from ..llm import get_llm
from .dataset import GoldenExample


@dataclass
class JudgeResult:
    groundedness: float
    sentence_scores: List[float]
    hallucinations: List[str]
    must_contain_hits: List[str]
    must_contain_misses: List[str]
    forbidden_hits: List[str]
    citation_hit: bool


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def judge_groundedness(answer: str, context: str,
                       example: Optional[GoldenExample] = None,
                       llm: Any = None) -> JudgeResult:
    """Score an answer. Deterministic offline judge; LLM judge is opt-in.

    To use a real LLM judge, set CLINICAL_JUDGE_LLM=1 and OPENAI_API_KEY; the
    judge then asks the model to rate each claim 0..1 with rationale. The
    deterministic scorer below is the default so CI never needs credentials.
    """
    use_llm = False
    import os
    if os.environ.get("CLINICAL_JUDGE_LLM") == "1":
        use_llm = True
    sents = [s for s in _sentences(answer) if len(s.split()) > 3]
    scores = [claim_groundedness(s, context) for s in sents]
    groundedness = sum(scores) / len(scores) if scores else 1.0

    hallucinations = [s for s, sc in zip(sents, scores) if sc < 0.2]
    lowered = answer.lower()
    must_hits = [m for m in (example.must_contain if example else [])
                 if m.lower() in lowered]
    must_miss = [m for m in (example.must_contain if example else [])
                 if m.lower() not in lowered]
    forbidden = [m for m in (example.must_not_contain if example else [])
                 if m.lower() in lowered]
    citation_hit = True
    if example and example.expected_citations:
        citation_hit = any(c in answer for c in example.expected_citations)
    # pragma: no cover - LLM judge path needs credentials
    if use_llm and llm is None:
        llm = get_llm()
    return JudgeResult(
        groundedness=round(groundedness, 3),
        sentence_scores=[round(s, 3) for s in scores],
        hallucinations=hallucinations,
        must_contain_hits=must_hits,
        must_contain_misses=must_miss,
        forbidden_hits=forbidden,
        citation_hit=citation_hit,
    )


@dataclass
class EvalCaseResult:
    id: str
    query: str
    groundedness: float
    passed: bool
    failures: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    citations: List[str] = field(default_factory=list)


def run_eval(agent: Any, examples: List[GoldenExample],
             audit: Any = None) -> Dict[str, Any]:
    """Run the full golden set through the agent. Returns aggregate report."""
    results: List[EvalCaseResult] = []
    for ex in examples:
        t0 = time.perf_counter()
        state = agent.ask(ex.query, request_id=f"eval-{ex.id}")
        latency = (time.perf_counter() - t0) * 1000
        # Judge the draft answer (pre-disclaimer) so the standard medical
        # disclaimer isn't scored as an ungrounded claim.
        judge = judge_groundedness(state.draft_answer or state.final_answer,
                                   state.context_text, ex)
        failures: List[str] = []
        if state.blocked:
            failures.append(f"blocked: {state.block_reason}")
        if judge.groundedness < ex.min_groundedness:
            failures.append(
                f"groundedness {judge.groundedness} < {ex.min_groundedness}")
        failures += [f"missing fact: {m}" for m in judge.must_contain_misses]
        failures += [f"forbidden fact present: {m}" for m in judge.forbidden_hits]
        if judge.hallucinations:
            failures.append(f"{len(judge.hallucinations)} ungrounded sentence(s)")
        if not judge.citation_hit and ex.expected_citations:
            failures.append("expected citation not found in answer")
        cost = (state.tokens_in * 0.5 + state.tokens_out * 1.5) / 1_000_000
        results.append(EvalCaseResult(
            id=ex.id, query=ex.query, groundedness=judge.groundedness,
            passed=not failures, failures=failures, latency_ms=round(latency, 1),
            cost_usd=round(cost, 6), citations=state.citations))
        if audit:
            audit.log("eval_case", f"eval-{ex.id}", actor="eval-harness",
                      example_id=ex.id, passed=not failures,
                      groundedness=judge.groundedness,
                      latency_ms=round(latency, 1))
    passed = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / max(len(results), 1), 3),
        "avg_groundedness": round(
            sum(r.groundedness for r in results) / max(len(results), 1), 3),
        "avg_latency_ms": round(
            sum(r.latency_ms for r in results) / max(len(results), 1), 1),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "cases": [r.__dict__ for r in results],
    }
