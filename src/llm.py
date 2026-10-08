"""Shared LLM backend abstraction.

`get_llm()` returns the best available chat backend:
1. OpenAI via langchain-openai (if OPENAI_API_KEY is set and package installed)
2. Deterministic MockLLM otherwise — rule-based, offline, test-safe.

The mock is deliberately *deterministic*: same input -> same output, so the
eval harness and tests are reproducible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        # Rough blended public pricing for accounting in the eval harness.
        return (self.input_tokens * 0.5 + self.output_tokens * 1.5) / 1_000_000


class MockLLM:
    """Deterministic, offline stand-in for a chat LLM.

    Behaviour:
    - If the prompt contains a CONTEXT block and a QUESTION, it answers by
      extracting the highest-overlap sentences from the context (extractive QA).
    - "Summarize" prompts return the first N sentences.
    - "Extract" prompts return key: value lines found in the context.
    - Anything else echoes a grounded, citation-style stub.
    """

    name = "mock-llm-v1"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        prompt = "\n".join(m.get("content", "") for m in messages)
        context = self._extract_block(prompt, "CONTEXT")
        question = self._extract_block(prompt, "QUESTION") or prompt[-500:]
        lowered = prompt.lower()

        if "summarize" in lowered and context:
            text = self._summarize(context)
        elif "extract" in lowered and context:
            text = self._extract_fields(context)
        elif context:
            text = self._answer(question, context)
        else:
            text = (
                "Based on the information provided, I cannot give a grounded "
                "answer because no retrieved context was supplied. [no-context]"
            )
        return LLMResponse(
            text=text,
            model=self.name,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )

    @staticmethod
    def _extract_block(prompt: str, tag: str) -> str:
        # Only split on the known structural tags, not on clinical
        # headers like "MRN:" that appear inside the context.
        pattern = rf"{tag}:\s*(.*?)(?=\n(?:CONTEXT|QUESTION|ANSWER):|\Z)"
        m = re.search(pattern, prompt, re.S)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    @staticmethod
    def _overlap(a: str, b: str) -> int:
        aw = set(re.findall(r"[a-z0-9]+", a.lower()))
        bw = set(re.findall(r"[a-z0-9]+", b.lower()))
        stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "was", "for",
                "with", "on", "at", "by", "as", "it", "this", "that", "are", "be"}
        return len((aw - stop) & (bw - stop))

    def _answer(self, question: str, context: str) -> str:
        sents = self._sentences(context)
        ranked = sorted(sents, key=lambda s: self._overlap(question, s), reverse=True)
        top = [s for s in ranked[:2] if self._overlap(question, s) > 0]
        if not top:
            return ("I could not find information in the provided clinical documents "
                    "to answer this question. [no-evidence]")
        cited = " ".join(f"{s} [doc]" for s in top)
        return cited

    def _summarize(self, context: str, n: int = 3) -> str:
        sents = self._sentences(context)[:n]
        return " ".join(f"{s} [doc]" for s in sents) or "No content to summarize. [no-evidence]"

    def _extract_fields(self, context: str) -> str:
        out = []
        for line in context.splitlines():
            m = re.match(r"\s*([A-Za-z][A-Za-z0-9 _/-]{1,30}):\s*(.+)", line)
            if m:
                out.append(f"{m.group(1).strip()}: {m.group(2).strip()} [doc]")
        return "\n".join(out[:10]) or "No structured fields found. [no-evidence]"


def get_llm() -> MockLLM:
    """Return the active LLM backend (real backends plug in here)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:  # pragma: no cover - requires credentials
            from langchain_openai import ChatOpenAI  # type: ignore

            return _LangChainAdapter(ChatOpenAI(model="gpt-4o-mini", api_key=api_key))
        except Exception:
            pass
    return MockLLM()


class _LangChainAdapter:  # pragma: no cover - requires credentials
    """Thin adapter so real backends share the MockLLM interface."""

    name = "openai-adapter"

    def __init__(self, client: Any):
        self._client = client

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        resp = self._client.invoke(messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return LLMResponse(text=text, model=self.name,
                           input_tokens=getattr(resp, "input_tokens", 0),
                           output_tokens=len(text.split()))
