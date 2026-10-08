"""PII detection for clinical text (Presidio-style, dependency-free).

Detectors combine:
- Regex patterns: SSN, phone, email, MRN, dates, ZIP, credit card
- Heuristic NER: person names via title patterns (Dr./Mr./Ms./Mrs. + Name)
  and "Patient: <Name>" style headers common in clinical notes

Each detection carries an entity type, character offsets, and a confidence
score. Thresholds are configurable per entity type (see configs/privacy.yaml).

This is a *reference* detector, not a certified de-identifier: real HIPAA
de-identification pipelines use trained NER (Presidio/spaCy) plus human
review. See README "production gaps".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PIIDetection:
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float
    detector: str = "regex"

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type, "start": self.start, "end": self.end,
            "text": self.text, "confidence": round(self.confidence, 3),
            "detector": self.detector,
        }


# (entity_type, pattern, confidence, description)
_PATTERNS: List[Tuple[str, str, float, str]] = [
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b", 0.99, "US Social Security number"),
    ("SSN", r"\b\d{3}\s\d{2}\s\d{4}\b", 0.90, "SSN space-separated"),
    ("PHONE", r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", 0.95, "phone"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.99, "email"),
    ("MRN", r"\bMRN[:\s#]*([A-Z0-9-]{4,12})\b", 0.97, "medical record number"),
    ("MRN", r"\b(?:Medical Record|Chart)(?: Number| No\.?)[:\s#]*([A-Z0-9-]{4,12})\b", 0.90, "MRN verbose"),
    ("DATE", r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b", 0.95, "MM/DD/YYYY"),
    ("DATE", r"\b(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])\b", 0.95, "YYYY-MM-DD"),
    ("DATE", r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b", 0.92, "Month D, YYYY"),
    ("ZIP", r"\b\d{5}(?:-\d{4})?\b", 0.60, "ZIP code (low confidence alone)"),
    ("CREDIT_CARD", r"\b(?:\d[ -]*?){13,16}\b", 0.70, "card-like number"),
]

# Name heuristics: titles and clinical header patterns.
_NAME_PATTERNS: List[Tuple[str, str, float]] = [
    (r"\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", "PERSON", 0.90),
    (r"\bPatient(?: Name)?\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", "PERSON", 0.95),
    (r"\bDOB\s*:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)\b", "PERSON", 0.80),
    # Standalone name pair mid-sentence (conservative: 3+ letter parts,
    # not sentence-initial, filtered against stop words below).
    (r"(?<=[a-z]\s)([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b", "PERSON", 0.65),
]

# Common non-name capitalized words to avoid false positives.
_STOP_WORDS = {
    "Patient", "Doctor", "Hospital", "Clinic", "Department", "Emergency",
    "Intensive", "Care", "Unit", "Room", "Bed", "Floor", "Building",
}


class PIIDetector:
    """Configurable PII detector with per-entity confidence thresholds."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.thresholds: Dict[str, float] = cfg.get("thresholds", {})
        self.enabled: List[str] = cfg.get("enabled_entities") or [
            "SSN", "PHONE", "EMAIL", "MRN", "DATE", "PERSON", "ZIP",
        ]
        self.default_threshold: float = cfg.get("default_threshold", 0.5)

    def _passes(self, entity_type: str, confidence: float) -> bool:
        thr = self.thresholds.get(entity_type, self.default_threshold)
        return confidence >= thr

    def detect(self, text: str) -> List[PIIDetection]:
        out: List[PIIDetection] = []
        for entity, pattern, conf, _desc in _PATTERNS:
            if entity not in self.enabled:
                continue
            for m in re.finditer(pattern, text):
                if not self._passes(entity, conf):
                    continue
                out.append(PIIDetection(entity, m.start(), m.end(), m.group(0),
                                       conf, detector="regex"))
        for pattern, entity, conf in _NAME_PATTERNS:
            if entity not in self.enabled or not self._passes(entity, conf):
                continue
            for m in re.finditer(pattern, text):
                name = m.group(1)
                if any(w in _STOP_WORDS for w in name.split()):
                    continue
                s = m.start(1)
                out.append(PIIDetection(entity, s, s + len(name), name, conf,
                                       detector="heuristic-ner"))
        # de-duplicate overlapping spans, keep highest confidence
        out.sort(key=lambda d: (d.start, -(d.end - d.start), -d.confidence))
        merged: List[PIIDetection] = []
        for d in out:
            if merged and d.start < merged[-1].end:
                if d.confidence > merged[-1].confidence:
                    merged[-1] = d
                continue
            merged.append(d)
        return merged

    def detect_types(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detect(text):
            counts[d.entity_type] = counts.get(d.entity_type, 0) + 1
        return counts
