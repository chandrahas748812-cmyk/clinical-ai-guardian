"""Tests: chunking strategies."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.chunking import chunk_document, chunk_fixed, chunk_semantic, chunk_hierarchical


TEXT = (
    "CHIEF COMPLAINT:\nType 2 diabetes follow-up.\n\n"
    "MEDICATIONS:\n- Metformin 500mg twice daily.\n- Lisinopril 10mg daily.\n\n"
    "ALLERGIES:\n- Penicillin (rash).\n"
)


def test_fixed_chunking_overlap():
    chunks = chunk_fixed("abcdefghij" * 100, "d1", size=100, overlap=20)
    assert len(chunks) > 5
    # overlap means consecutive chunks share text
    assert chunks[0].text[-20:] in chunks[1].text
    assert all(c.id.startswith("d1#c") for c in chunks)


def test_fixed_chunking_ids_stable():
    a = chunk_fixed(TEXT, "doc", size=50, overlap=10)
    b = chunk_fixed(TEXT, "doc", size=50, overlap=10)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_semantic_chunking_sections():
    chunks = chunk_semantic(TEXT, "d2", max_chars=100)
    assert len(chunks) >= 2
    assert all(c.meta["strategy"] == "semantic" for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "Metformin" in joined and "Penicillin" in joined


def test_hierarchical_parent_child_links():
    chunks = chunk_hierarchical(TEXT, "d3", parent_max=500, child_size=80, child_overlap=10)
    parents = [c for c in chunks if c.meta.get("level") == "parent"]
    children = [c for c in chunks if c.meta.get("level") == "child"]
    assert parents, "expected parent chunks"
    assert children, "expected child chunks"
    parent_ids = {p.id for p in parents}
    assert all(c.meta["parent_id"] in parent_ids for c in children)


def test_chunk_document_dispatch():
    for strategy in ("fixed", "semantic", "hierarchical"):
        chunks = chunk_document(TEXT, "dx", strategy=strategy)
        assert len(chunks) > 0


def test_chunk_document_unknown_strategy():
    try:
        chunk_document(TEXT, "dx", strategy="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
