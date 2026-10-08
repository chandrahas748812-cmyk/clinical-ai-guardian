"""Agent tools: retrieval, summarization, structured extraction.

Tools are plain functions taking (state, deps) so they are trivially unit
testable and can be wrapped as LangChain/LangGraph tools in production.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..llm import get_llm
from ..retrieval.hybrid import ScoredChunk


def build_context(chunks: List[ScoredChunk], max_chars: int = 4000) -> str:
    """Assemble retrieved chunks into a cited context block."""
    parts = []
    total = 0
    for sc in chunks:
        tag = f"[{sc.chunk.id}]"
        piece = f"{tag} {sc.chunk.text}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def tool_retrieve(query: str, retriever: Any, top_k: int = 5) -> List[ScoredChunk]:
    return retriever.search(query, top_k=top_k)


def tool_answer(question: str, context: str, llm: Any = None) -> Dict[str, Any]:
    llm = llm or get_llm()
    prompt = (
        "You answer clinical questions using ONLY the provided context. "
        "Cite every factual claim with [doc]. If the context lacks the answer, "
        "say so explicitly.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )
    resp = llm.chat([{"role": "user", "content": prompt}])
    return {"text": resp.text, "tokens_in": resp.input_tokens,
            "tokens_out": resp.output_tokens, "latency_ms": resp.latency_ms}


def tool_summarize(context: str, llm: Any = None) -> Dict[str, Any]:
    llm = llm or get_llm()
    prompt = (
        "Summarize the following clinical document context in 3-5 sentences. "
        "Cite claims with [doc].\n\n"
        f"CONTEXT:\n{context}\n\nSUMMARY:"
    )
    resp = llm.chat([{"role": "user", "content": prompt}])
    return {"text": resp.text, "tokens_in": resp.input_tokens,
            "tokens_out": resp.output_tokens, "latency_ms": resp.latency_ms}


def tool_extract(context: str, llm: Any = None) -> Dict[str, Any]:
    llm = llm or get_llm()
    prompt = (
        "Extract structured clinical fields (diagnoses, medications, dosages, "
        "allergies, vitals) from the context as 'Field: value' lines. "
        "Cite with [doc].\n\n"
        f"CONTEXT:\n{context}\n\nEXTRACT:"
    )
    resp = llm.chat([{"role": "user", "content": prompt}])
    return {"text": resp.text, "tokens_in": resp.input_tokens,
            "tokens_out": resp.output_tokens, "latency_ms": resp.latency_ms}
