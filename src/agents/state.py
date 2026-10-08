"""Agent state for the clinical Q&A pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..retrieval.hybrid import ScoredChunk


@dataclass
class AgentState:
    """State threaded through triage -> retrieve -> synthesize -> verify -> respond."""

    query: str
    request_id: str = ""
    # triage
    intent: str = "unknown"          # qa | summarize | extract | rejected
    triage_reason: str = ""
    # privacy
    redacted_query: str = ""
    redaction_seal: str = ""
    # retrieval
    chunks: List[ScoredChunk] = field(default_factory=list)
    context_text: str = ""
    # synthesis
    draft_answer: str = ""
    citations: List[str] = field(default_factory=list)
    # verification
    groundedness: float = 0.0
    needs_hitl: bool = False
    hitl_reason: str = ""
    hitl_decision: Optional[str] = None  # approve | reject
    # response
    final_answer: str = ""
    blocked: bool = False
    block_reason: str = ""
    # telemetry
    timings_ms: Dict[str, float] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
