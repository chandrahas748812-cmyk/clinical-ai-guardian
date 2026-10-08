"""Clinical Q&A agent graph: triage -> retrieve -> synthesize -> verify -> respond.

Runs on LangGraph's StateGraph when the package is installed; otherwise
falls back to an equivalent pure-Python sequential runner. Both paths
execute the same node functions, so behavior is identical and the fallback
keeps `example.py` and tests runnable offline.

Node overview:
- triage:    input guardrail + intent classification + PII redaction of query
- retrieve:  hybrid search over the document store
- synthesize: LLM answer/summary/extraction with citations
- verify:    output guardrail (groundedness, PII leak, citations); decides HITL
- respond:   human-in-the-loop gate, disclaimer, final assembly
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional

from ..guardrails.input import InputGuardrail
from ..guardrails.output import OutputGuardrail
from ..privacy.redactor import PIIRedactor, RedactionMap
from ..retrieval.hybrid import Chunk, ScoredChunk
from ..retrieval.store import DocumentStore
from . import tools
from .state import AgentState


# ----------------------------------------------------------------------------
# Node functions (framework-agnostic)
# ----------------------------------------------------------------------------

def node_triage(state: AgentState, deps: Dict[str, Any]) -> AgentState:
    t0 = time.perf_counter()
    guard: InputGuardrail = deps["input_guardrail"]
    verdict = guard.check(state.query)
    if not verdict.allowed:
        state.blocked = True
        state.block_reason = verdict.reason
        state.intent = "rejected"
        state.timings_ms["triage"] = (time.perf_counter() - t0) * 1000
        return state
    q = state.query.lower()
    if any(w in q for w in ("summariz", "summary", "tldr", "overview")):
        state.intent = "summarize"
    elif any(w in q for w in ("extract", "list all", "what medications", "allergies")):
        state.intent = "extract"
    else:
        state.intent = "qa"
    redactor: PIIRedactor = deps["redactor"]
    result = redactor.redact(state.query)
    state.redacted_query = result.redacted_text
    state.redaction_seal = result.redaction_map.seal_hash()
    # stash map for downstream PII-leak scanning (in-memory only, never logged)
    deps["_redaction_map"] = result.redaction_map
    state.timings_ms["triage"] = (time.perf_counter() - t0) * 1000
    return state


def node_retrieve(state: AgentState, deps: Dict[str, Any]) -> AgentState:
    t0 = time.perf_counter()
    store: DocumentStore = deps["store"]
    chunks = tools.tool_retrieve(state.redacted_query or state.query, store,
                                 top_k=deps.get("top_k", 5))
    # Privacy: redact PII from retrieved chunks BEFORE they reach the LLM.
    # The raw documents stay in the secure store; only redacted text flows
    # downstream. The redaction map is kept in-memory for authorized
    # re-identification and PII-leak scanning, never logged.
    redactor: PIIRedactor = deps["redactor"]
    redacted_chunks = []
    for sc in chunks:
        result = redactor.redact(sc.chunk.text)
        redacted_chunk = ScoredChunk(
            chunk=Chunk(
                id=sc.chunk.id, text=result.redacted_text, doc_id=sc.chunk.doc_id,
                start=sc.chunk.start, end=sc.chunk.end, meta=sc.chunk.meta),
            score=sc.score, source=sc.source)
        redacted_chunks.append(redacted_chunk)
        # Merge maps so output guardrail can scan for leaks of doc PII too.
        existing = deps.get("_redaction_map")
        if existing is not None:
            existing.mapping.update(result.redaction_map.mapping)
    state.chunks = redacted_chunks
    state.context_text = tools.build_context(redacted_chunks)
    state.citations = [c.chunk.id for c in redacted_chunks]
    state.timings_ms["retrieve"] = (time.perf_counter() - t0) * 1000
    return state


def node_synthesize(state: AgentState, deps: Dict[str, Any]) -> AgentState:
    t0 = time.perf_counter()
    llm = deps.get("llm")
    if state.intent == "summarize":
        out = tools.tool_summarize(state.context_text, llm)
    elif state.intent == "extract":
        out = tools.tool_extract(state.context_text, llm)
    else:
        out = tools.tool_answer(state.redacted_query or state.query,
                                state.context_text, llm)
    state.draft_answer = out["text"]
    state.tokens_in += out["tokens_in"]
    state.tokens_out += out["tokens_out"]
    state.timings_ms["synthesize"] = (time.perf_counter() - t0) * 1000
    return state


def node_verify(state: AgentState, deps: Dict[str, Any]) -> AgentState:
    t0 = time.perf_counter()
    redaction_map: Optional[RedactionMap] = deps.get("_redaction_map")
    guard = OutputGuardrail(
        min_groundedness=deps.get("min_groundedness", 0.35),
        redaction_map=redaction_map,
    )
    verdict = guard.check(state.draft_answer, state.context_text)
    state.groundedness = verdict.groundedness
    if not verdict.allowed:
        # Low groundedness or PII leak -> escalate to human, don't auto-respond.
        state.needs_hitl = True
        state.hitl_reason = verdict.reason
    state.timings_ms["verify"] = (time.perf_counter() - t0) * 1000
    return state


def node_respond(state: AgentState, deps: Dict[str, Any]) -> AgentState:
    t0 = time.perf_counter()
    if state.needs_hitl:
        reviewer: Callable[[AgentState], str] = deps.get(
            "hitl_reviewer", _default_reviewer)
        decision = reviewer(state)
        state.hitl_decision = decision
        if decision != "approve":
            state.final_answer = (
                "This answer was flagged for clinical review and was not "
                f"approved for release ({state.hitl_reason}). A clinician will "
                "follow up. [escalated]"
            )
            state.timings_ms["respond"] = (time.perf_counter() - t0) * 1000
            return state
    # Restore original PII only for approved, in-trust-boundary consumers.
    # The API layer decides whether the caller is entitled to re-identified text.
    state.final_answer = state.draft_answer
    if state.final_answer and "disclaimer" not in state.final_answer.lower():
        from ..guardrails.output import DISCLAIMER
        state.final_answer = state.final_answer.rstrip() + DISCLAIMER
    state.timings_ms["respond"] = (time.perf_counter() - t0) * 1000
    return state


def _default_reviewer(state: AgentState) -> str:
    """Non-interactive default: reject escalations (safe default for demos)."""
    return "reject"


# ----------------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------------

_NODES = [node_triage, node_retrieve, node_synthesize, node_verify, node_respond]


def run_sequential(state: AgentState, deps: Dict[str, Any]) -> AgentState:
    """Pure-Python fallback runner: executes nodes in order, honoring blocks."""
    for node in _NODES:
        state = node(state, deps)
        if state.blocked:
            break
    return state


def build_langgraph_app(deps: Dict[str, Any]):  # pragma: no cover - optional dep
    """Build a LangGraph StateGraph over the same node functions."""
    from langgraph.graph import StateGraph, END  # type: ignore

    workflow = StateGraph(AgentState)

    def _wrap(name, fn):
        def inner(s: AgentState) -> AgentState:
            return fn(s, deps)
        inner.__name__ = name
        return inner

    for node in _NODES:
        workflow.add_node(node.__name__, _wrap(node.__name__, node))
    workflow.set_entry_point("node_triage")
    workflow.add_edge("node_triage", "node_retrieve")
    workflow.add_edge("node_retrieve", "node_synthesize")
    workflow.add_edge("node_synthesize", "node_verify")
    workflow.add_edge("node_verify", "node_respond")
    workflow.add_edge("node_respond", END)

    def _after_triage(s: AgentState) -> str:
        return "node_retrieve" if not s.blocked else END
    workflow.add_conditional_edges("node_triage", _after_triage)
    return workflow.compile()


class ClinicalAgent:
    """Facade: runs the clinical Q&A pipeline with LangGraph when available."""

    def __init__(self, deps: Dict[str, Any]):
        self.deps = deps
        self._app = None
        try:
            self._app = build_langgraph_app(deps)
            self.engine = "langgraph"
        except Exception:
            self.engine = "sequential"

    def ask(self, query: str, request_id: str = "") -> AgentState:
        state = AgentState(query=query, request_id=request_id)
        if self._app is not None:  # pragma: no cover - optional dep
            out = self._app.invoke(state)
            return out if isinstance(out, AgentState) else AgentState(**out)
        return run_sequential(state, self.deps)
