"""Golden dataset format for clinical Q&A evaluation.

JSONL, one object per line:
{
  "id": "q001",
  "query": "What dosage of metformin was prescribed?",
  "expected_citations": ["note-1#c2"],
  "must_contain": ["500mg", "twice daily"],      # key facts the answer must include
  "must_not_contain": ["1000mg"],                # anti-hallucination checks
  "min_groundedness": 0.4
}

`must_contain` / `must_not_contain` give deterministic, LLM-free assertions;
`expected_citations` checks retrieval; `min_groundedness` gates the judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class GoldenExample:
    id: str
    query: str
    expected_citations: List[str] = field(default_factory=list)
    must_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    min_groundedness: float = 0.35


def load_golden(path: str) -> List[GoldenExample]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            examples.append(GoldenExample(
                id=d["id"], query=d["query"],
                expected_citations=d.get("expected_citations", []),
                must_contain=d.get("must_contain", []),
                must_not_contain=d.get("must_not_contain", []),
                min_groundedness=d.get("min_groundedness", 0.35),
            ))
    return examples


def save_golden(examples: List[GoldenExample], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps({
                "id": e.id, "query": e.query,
                "expected_citations": e.expected_citations,
                "must_contain": e.must_contain,
                "must_not_contain": e.must_not_contain,
                "min_groundedness": e.min_groundedness,
            }) + "\n")
