"""Output guardrails: validate agent answers before they reach the user.

Checks:
- Groundedness: every substantive claim (sentence) must overlap the retrieved
  context above a threshold; otherwise flag hallucination risk.
- PII leak scan: raw PII values must never appear (see privacy.redactor).
- Medical disclaimer: clinical answers get a standard disclaimer appended.
- Citation presence: answers should carry [doc] citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..privacy.redactor import RedactionMap


@dataclass
class OutputVerdict:
    allowed: bool
    reason: str = ""
    groundedness: float = 0.0
    issues: List[str] = field(default_factory=list)
    final_text: str = ""


DISCLAIMER = (
    "\n\n_Disclaimer: This summary is generated from clinical documents for "
    "informational purposes and is not medical advice. Confirm with a "
    "licensed clinician before making care decisions._"
)


def _tokens(s: str) -> set:
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "was", "for",
            "with", "on", "at", "by", "as", "it", "this", "that", "are", "be",
            "doc", "no"}
    return set(re.findall(r"[a-z0-9]+", s.lower())) - stop


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def claim_groundedness(claim: str, context: str) -> float:
    """Fraction of claim tokens present in context (0..1)."""
    ct, xt = _tokens(claim), _tokens(context)
    if not ct:
        return 1.0
    return len(ct & xt) / len(ct)


class OutputGuardrail:
    def __init__(
        self,
        min_groundedness: float = 0.35,
        require_citations: bool = True,
        add_disclaimer: bool = True,
        redaction_map: Optional[RedactionMap] = None,
    ):
        self.min_groundedness = min_groundedness
        self.require_citations = require_citations
        self.add_disclaimer = add_disclaimer
        self.redaction_map = redaction_map

    def check(self, answer: str, context: str) -> OutputVerdict:
        issues: List[str] = []
        sents = [s for s in _sentences(answer) if len(_tokens(s)) > 3]
        scores = [claim_groundedness(s, context) for s in sents]
        groundedness = sum(scores) / len(scores) if scores else 1.0

        if groundedness < self.min_groundedness:
            issues.append(
                f"Low groundedness {groundedness:.2f} < {self.min_groundedness:.2f}"
            )
        if self.require_citations and "[doc]" not in answer and "[no-" not in answer:
            issues.append("Missing source citations")
        if self.redaction_map:
            leaked = [
                p for p, orig in self.redaction_map.mapping.items()
                if len(orig) >= 4 and orig in answer
            ]
            if leaked:
                issues.append(f"PII leak detected: {len(leaked)} raw value(s) in output")

        allowed = not issues
        final = answer
        if allowed and self.add_disclaimer and "disclaimer" not in answer.lower():
            final = answer.rstrip() + DISCLAIMER
        return OutputVerdict(
            allowed=allowed,
            reason="; ".join(issues) if issues else "OK",
            groundedness=round(groundedness, 3),
            issues=issues,
            final_text=final,
        )
