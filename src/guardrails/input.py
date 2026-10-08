"""Input guardrails: validate user queries before they reach the pipeline.

Blocks:
- Prompt-injection patterns (ignore instructions, system prompts, jailbreaks)
- Oversized inputs
- Off-topic queries (non-clinical) when strict topic mode is on
- Queries that look like PII harvesting ("list all SSNs...")

All checks are pure functions -> easy to test and to wire into FastAPI
dependencies or the agent graph's triage node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GuardrailVerdict:
    allowed: bool
    reason: str = ""
    triggered_rule: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


_INJECTION_PATTERNS: List[tuple] = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "prompt-injection"),
    (r"disregard\s+(all\s+)?(previous|prior)\s+instructions", "prompt-injection"),
    (r"system\s*prompt", "prompt-injection"),
    (r"you\s+are\s+now\s+", "role-hijack"),
    (r"jailbreak|DAN\s+mode|do\s+anything\s+now", "jailbreak"),
    (r"reveal\s+(your\s+)?(system|initial)\s+(prompt|instructions)", "exfiltration"),
    (r"print\s+(all\s+)?(patient|medical)\s+records", "pii-harvesting"),
    (r"list\s+all\s+(ssn|social security|patient)s?\b", "pii-harvesting"),
    (r"dump\s+(the\s+)?(database|context|documents)", "exfiltration"),
    (r"<\s*script|javascript\s*:", "xss"),
]

_CLINICAL_KEYWORDS = {
    "patient", "diagnosis", "treatment", "medication", "dosage", "symptom",
    "clinical", "medical", "doctor", "nurse", "hospital", "therapy", "lab",
    "test", "result", "allergy", "vital", "blood", "prescription", "surgery",
    "discharge", "admission", "chronic", "acute", "prognosis", "screening",
    "vaccine", "dose", "mg", "diagnosed", "condition", "disease", "syndrome",
}


@dataclass
class InputPolicy:
    max_chars: int = 2000
    min_chars: int = 3
    strict_topic: bool = True
    block_injection: bool = True


class InputGuardrail:
    def __init__(self, policy: Optional[InputPolicy] = None):
        self.policy = policy or InputPolicy()

    def check(self, query: str) -> GuardrailVerdict:
        q = query or ""
        if len(q.strip()) < self.policy.min_chars:
            return GuardrailVerdict(False, "Query too short.", "min-length")
        if len(q) > self.policy.max_chars:
            return GuardrailVerdict(
                False,
                f"Query exceeds {self.policy.max_chars} characters.",
                "max-length",
                {"length": len(q)},
            )
        if self.policy.block_injection:
            lowered = q.lower()
            for pattern, rule in _INJECTION_PATTERNS:
                if re.search(pattern, lowered):
                    return GuardrailVerdict(
                        False, f"Blocked: suspected {rule}.", rule,
                        {"pattern": pattern},
                    )
        if self.policy.strict_topic:
            words = set(re.findall(r"[a-z]+", q.lower()))
            if not (words & _CLINICAL_KEYWORDS):
                return GuardrailVerdict(
                    False,
                    "Query does not appear clinical; strict topic mode is on.",
                    "off-topic",
                )
        return GuardrailVerdict(True, "OK")
