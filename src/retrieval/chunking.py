"""Configurable chunking strategies for clinical documents.

Strategies:
- fixed:      sliding window of N chars with overlap (fast, predictable)
- semantic:   split on section headers / sentence boundaries, merge small pieces
- hierarchical: parent (section) -> child (sentence windows); children carry
              parent context for better retrieval grounding

All chunkers emit `Chunk` objects with stable IDs: {doc_id}#c{index}.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .hybrid import Chunk


def _sentences(text: str) -> List[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]
    return parts


def chunk_fixed(text: str, doc_id: str, size: int = 800, overlap: int = 120) -> List[Chunk]:
    """Sliding character window with overlap."""
    chunks: List[Chunk] = []
    step = max(size - overlap, 1)
    idx = 0
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(id=f"{doc_id}#c{idx}", text=piece,
                               doc_id=doc_id, start=start, end=end,
                               meta={"strategy": "fixed"}))
            idx += 1
        if end >= len(text):
            break
    return chunks


_SECTION_RE = re.compile(r"^([A-Z][A-Za-z0-9 ,/\-()]{2,60}):?\s*$", re.M)


def chunk_semantic(
    text: str,
    doc_id: str,
    max_chars: int = 1200,
    min_chars: int = 200,
) -> List[Chunk]:
    """Split on clinical section headers; merge tiny sections; split huge ones."""
    sections: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if _SECTION_RE.match(line.strip()) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    sections = [s for s in sections if s]

    chunks: List[Chunk] = []
    idx = 0
    buf = ""
    for sec in sections:
        if len(sec) > max_chars:  # split long sections on sentences
            if buf:
                chunks.append(_mk(doc_id, idx, buf, "semantic")); idx += 1; buf = ""
            for s in _sentences(sec):
                if len(buf) + len(s) > max_chars and len(buf) >= min_chars:
                    chunks.append(_mk(doc_id, idx, buf, "semantic")); idx += 1; buf = ""
                buf = (buf + " " + s).strip()
        elif len(buf) + len(sec) <= max_chars:
            buf = (buf + "\n" + sec).strip()
        else:
            chunks.append(_mk(doc_id, idx, buf, "semantic")); idx += 1
            buf = sec
    if buf.strip():
        chunks.append(_mk(doc_id, idx, buf, "semantic"))
    # fix offsets
    offset = 0
    for c in chunks:
        c.start = text.find(c.text[:40], offset)
        c.start = c.start if c.start >= 0 else offset
        c.end = c.start + len(c.text)
        offset = c.end
    return chunks


def _mk(doc_id: str, idx: int, text: str, strategy: str) -> Chunk:
    return Chunk(id=f"{doc_id}#c{idx}", text=text.strip(), doc_id=doc_id,
                 start=0, end=len(text), meta={"strategy": strategy})


def chunk_hierarchical(
    text: str,
    doc_id: str,
    parent_max: int = 1500,
    child_size: int = 400,
    child_overlap: int = 60,
) -> List[Chunk]:
    """Parent section chunks + child sentence-window chunks.

    Child chunks carry `parent_id` in meta so retrieval can expand context
    (parent retrieval) while citations stay precise (child level).
    """
    parents = chunk_semantic(text, doc_id, max_chars=parent_max)
    out: List[Chunk] = []
    for p in parents:
        p.meta["level"] = "parent"
        out.append(p)
        kids = chunk_fixed(p.text, f"{p.id}:ch", size=child_size, overlap=child_overlap)
        for k in kids:
            k.meta.update({"strategy": "hierarchical", "level": "child",
                           "parent_id": p.id, "parent_text": p.text[:500]})
            out.append(k)
    # re-id children sequentially
    ci = 0
    for c in out:
        if c.meta.get("level") == "child":
            c.id = f"{doc_id}#hc{ci}"
            ci += 1
    return out


CHUNKERS = {
    "fixed": chunk_fixed,
    "semantic": chunk_semantic,
    "hierarchical": chunk_hierarchical,
}


def chunk_document(text: str, doc_id: str, strategy: str = "semantic",
                  **kwargs) -> List[Chunk]:
    """Dispatch to a named chunking strategy."""
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    return CHUNKERS[strategy](text, doc_id, **kwargs)
