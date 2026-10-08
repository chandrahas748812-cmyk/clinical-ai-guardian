"""PII redaction with reversible token mapping for audit.

Redaction replaces each detection with a typed placeholder like
`[PERSON_1]`. The mapping (placeholder -> original text) is stored in a
`RedactionMap` that can be sealed (hashed) for the audit log; reversal
requires the map object itself, which in production lives in a separate
secure store — never in the audit log.

This mirrors the Safe Harbor approach: redacted text is safe to pass to
LLMs and logs, while the map enables authorized re-identification workflows.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .detector import PIIDetection, PIIDetector


@dataclass
class RedactionMap:
    """Placeholder -> original PII. Reversible only with this object."""

    mapping: Dict[str, str] = field(default_factory=dict)
    salt: str = field(default_factory=lambda: secrets.token_hex(8))

    def seal_hash(self) -> str:
        """Hash of the mapping for the audit log (proves *what* was redacted
        without storing the PII itself)."""
        canonical = "|".join(f"{k}={v}" for k, v in sorted(self.mapping.items()))
        return hashlib.sha256(f"{self.salt}:{canonical}".encode()).hexdigest()

    def restore(self, text: str) -> str:
        for placeholder, original in self.mapping.items():
            text = text.replace(placeholder, original)
        return text


@dataclass
class RedactionResult:
    redacted_text: str
    detections: List[PIIDetection]
    redaction_map: RedactionMap
    entities_redacted: Dict[str, int]


class PIIRedactor:
    """Detect + redact, returning text safe for downstream LLM calls."""

    def __init__(self, detector: Optional[PIIDetector] = None):
        self.detector = detector or PIIDetector()

    def redact(self, text: str) -> RedactionResult:
        detections = self.detector.detect(text)
        # Replace from the end so offsets stay valid.
        detections_sorted = sorted(detections, key=lambda d: d.start, reverse=True)
        redacted = text
        mapping: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        entities: Dict[str, int] = {}
        for d in detections_sorted:
            counters[d.entity_type] = counters.get(d.entity_type, 0) + 1
            placeholder = f"[{d.entity_type}_{counters[d.entity_type]}]"
            # De-duplicate: same original text + type reuses the placeholder.
            existing = next((p for p, o in mapping.items() if o == d.text), None)
            if existing is None:
                mapping[placeholder] = d.text
                existing = placeholder
            redacted = redacted[: d.start] + existing + redacted[d.end:]
            entities[d.entity_type] = entities.get(d.entity_type, 0) + 1
        return RedactionResult(
            redacted_text=redacted,
            detections=detections,
            redaction_map=RedactionMap(mapping=mapping),
            entities_redacted=entities,
        )

    def scan_for_leak(self, text: str, redaction_map: RedactionMap) -> List[str]:
        """Check whether any original PII value appears in `text`.

        Used by output guardrails: the LLM must never echo raw PII.
        Returns the list of leaked entity types.
        """
        leaked = []
        for placeholder, original in redaction_map.mapping.items():
            if len(original) >= 4 and original in text:
                entity = placeholder.strip("[]").rsplit("_", 1)[0]
                if entity not in leaked:
                    leaked.append(entity)
        return leaked
