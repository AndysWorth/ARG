"""Knowledge graph tests.

Covers every test point from Section 6:
  * CRUD round-trips and stats reflect counts
  * Traversal at depth 1, 2, and across cycles
  * Reverse-link lookup
  * list_all_documents() / get_graph_json() shape + counts
  * most_linked_docs() ranking
  * orphaned_docs()
  * docs_by_chunk_count() ordering, no truncation
  * remove_document() cascades through chunks + edges
  * Persistence across a close + reopen
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from arg.crawler.extractors import Document
from arg.graph import Chunk, KnowledgeGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _doc(path: Path, *, title: str | None = None, file_type: str = "html") -> Document:
    """Build a Document whose ``path`` becomes its doc_id once resolved."""
    return Document(
        path=path,
        content="",
        metadata={"title": title or path.stem, "file_type": file_type},
    )


@pytest.fixture
def kg_path(tmp_path: Path) -> Path:
    return tmp_path / "kuzu_db"


@pytest.fixture
def kg(kg_path: Path):
    graph = KnowledgeGraph(kg_path)
    yield graph
    graph.close()


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Basic insert + stats
# ---------------------------------------------------------------------------


def test_add_document_idempotent(kg, docs_dir):
    p = docs_dir / "a.html"
    p.touch()
    kg.add_document(_doc(p, title="A"))
    kg.add_document(_doc(p, title="A renamed"))
    docs = kg.list_all_documents()
    assert len(docs) == 1
    assert docs[0]["title"] == "A renamed"


def test_stats_reflects_inserts(kg, docs_dir):
    a, b = docs_dir / "a.html", docs_dir / "b.html"
    a.touch()
    b.touch()
    kg.add_document(_doc(a))
    kg.add_document(_doc(b))
    kg.add_chunk(Chunk("c1", "text1", 3), str(a.resolve()), 0)
    kg.add_chunk(Chunk("c2", "text2", 5), str(a.resolve()), 1)
    kg.add_link(str(a.resolve()), str(b.resolve()), "anchor")

    s = kg.stats()
    assert s == {
        "documents": 2,
        "chunks": 2,
        "link_edges": 1,
        "contains_edges": 2,
    }


def test_add_chunk_recomputes_chunk_count(kg, docs_dir):
    p = docs_dir / "a.html"
    p.touch()
    kg.add_document(_doc(p))
    doc_id = str(p.resolve())
    kg.add_chunk(Chunk("c1", "t", 1), doc_id, 0)
    kg.add_chunk(Chunk("c2", "t", 1), doc_id, 1)
    kg.add_chunk(Chunk("c3", "t", 1), doc_id, 2)
    assert kg.get_doc_metadata(doc_id)["chunk_count"] == 3


def test_add_chunk_idempotent_does_not_double_count(kg, docs_dir):
    p = docs_dir / "a.html"
    p.touch()
    kg.add_document(_doc(p))
    doc_id = str(p.resolve())
    kg.add_chunk(Chunk("c1", "t", 1), doc_id, 0)
    kg.add_chunk(Chunk("c1", "updated", 2), doc_id, 0)  # same chunk_id
    assert kg.get_doc_metadata(doc_id)["chunk_count"] == 1


def test_add_chunk_rejects_negative_position(kg, docs_dir):
    p = docs_dir / "a.html"
    p.touch()
    kg.add_document(_doc(p))
    with pytest.raises(ValueError, match="position must be >= 0"):
        kg.add_chunk(Chunk("c1", "t", 1), str(p.resolve()), -1)


def test_get_chunks_for_doc_ordered_by_position(kg, docs_dir):
    p = docs_dir / "a.html"
    p.touch()
    kg.add_document(_doc(p))
    doc_id = str(p.resolve())
    # Insert in scrambled order; expect ordered output.
    kg.add_chunk(Chunk("c2", "t", 1), doc_id, 2)
    kg.add_chunk(Chunk("c0", "t", 1), doc_id, 0)
    kg.add_chunk(Chunk("c1", "t", 1), doc_id, 1)
    assert kg.get_chunks_for_doc(doc_id) == ["c0", "c1", "c2"]


def test_get_doc_metadata_returns_empty_dict_when_missing(kg):
    assert kg.get_doc_metadata("/nonexistent/path.html") == {}


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def _build_chain(kg, docs_dir, names: list[str]) -> list[str]:
    """Build a chain of docs linked in order; return their doc_ids."""
    ids = []
    for name in names:
        p = docs_dir / name
        p.touch()
        kg.add_document(_doc(p))
        ids.append(str(p.resolve()))
    for src, tgt in pairwise(ids):
        kg.add_link(src, tgt, "")
    return ids


def test_get_linked_docs_depth_one(kg, docs_dir):
    a, b, _c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    assert kg.get_linked_docs(a, depth=1) == [b]


def test_get_linked_docs_depth_two(kg, docs_dir):
    a, b, c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    assert sorted(kg.get_linked_docs(a, depth=2)) == sorted([b, c])


def test_get_linked_docs_handles_cycles(kg, docs_dir):
    """A->B->C->A: large depth must not infinite-loop or include source."""
    ids = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    a, b, c = ids
    # Close the cycle.
    kg.add_link(c, a, "")
    result = kg.get_linked_docs(a, depth=5)
    assert a not in result  # source excluded even when reachable via cycle
    assert set(result) == {b, c}


def test_get_linked_docs_zero_depth_returns_empty(kg, docs_dir):
    a, _b, _c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    assert kg.get_linked_docs(a, depth=0) == []


def test_get_linked_docs_rejects_huge_depth(kg, docs_dir):
    a, *_ = _build_chain(kg, docs_dir, ["a.html", "b.html"])
    with pytest.raises(ValueError, match="depth must be <= "):
        kg.get_linked_docs(a, depth=1000)


# ---------------------------------------------------------------------------
# Reverse links
# ---------------------------------------------------------------------------


def test_get_reverse_links(kg, docs_dir):
    a, b, c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    # Add an extra inbound edge: c -> b
    kg.add_link(c, b, "see also")

    reverse = kg.get_reverse_links(b)
    # Both a->b and c->b show up. Anchor texts differ.
    anchors = sorted((r["doc_id"], r["anchor_text"]) for r in reverse)
    assert anchors == sorted([(a, ""), (c, "see also")])


def test_get_reverse_links_empty_when_no_inbound(kg, docs_dir):
    a, _b, _c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    assert kg.get_reverse_links(a) == []  # a is the source-only end of the chain


# ---------------------------------------------------------------------------
# list_all + graph_json
# ---------------------------------------------------------------------------


def test_list_all_documents(kg, docs_dir):
    a, b, c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    docs = kg.list_all_documents()
    assert [d["doc_id"] for d in docs] == sorted([a, b, c])
    for d in docs:
        assert "title" in d
        assert "file_type" in d
        assert "chunk_count" in d


def test_get_graph_json_shape_and_counts(kg, docs_dir):
    _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    graph = kg.get_graph_json()
    assert set(graph.keys()) == {"nodes", "edges"}
    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2  # a->b, b->c
    # Every edge endpoint must correspond to a known node.
    node_ids = {n["id"] for n in graph["nodes"]}
    for edge in graph["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids
        assert "anchor_text" in edge


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def test_most_linked_docs_ranks_by_inbound(kg, docs_dir):
    # Hub-and-spoke: a, b, c all link to "hub"; nobody links to a/b/c.
    paths = [docs_dir / n for n in ["hub.html", "a.html", "b.html", "c.html"]]
    for p in paths:
        p.touch()
        kg.add_document(_doc(p))
    hub_id = str(paths[0].resolve())
    for p in paths[1:]:
        kg.add_link(str(p.resolve()), hub_id, "")

    ranked = kg.most_linked_docs(top_n=4)
    assert ranked[0]["doc_id"] == hub_id
    assert ranked[0]["inbound"] == 3
    # Remaining docs have inbound == 0.
    for entry in ranked[1:]:
        assert entry["inbound"] == 0


def test_most_linked_docs_limits_top_n(kg, docs_dir):
    for name in ("a.html", "b.html", "c.html", "d.html"):
        p = docs_dir / name
        p.touch()
        kg.add_document(_doc(p))
    ranked = kg.most_linked_docs(top_n=2)
    assert len(ranked) == 2


def test_most_linked_docs_rejects_non_positive(kg):
    assert kg.most_linked_docs(top_n=0) == []
    assert kg.most_linked_docs(top_n=-3) == []


def test_orphaned_docs_returns_only_zero_inbound(kg, docs_dir):
    a, _b, _c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    orphans = kg.orphaned_docs()
    assert orphans == [a]  # b and c each have an inbound edge


def test_docs_by_chunk_count_no_truncation(kg, docs_dir):
    counts = {"a.html": 1, "b.html": 5, "c.html": 3, "d.html": 0, "e.html": 7}
    for name, count in counts.items():
        p = docs_dir / name
        p.touch()
        kg.add_document(_doc(p))
        doc_id = str(p.resolve())
        for i in range(count):
            kg.add_chunk(Chunk(f"{name}::chunk::{i}", "t", 1), doc_id, i)

    by_count_desc = kg.docs_by_chunk_count(descending=True)
    assert len(by_count_desc) == 5  # no truncation
    chunk_counts = [d["chunk_count"] for d in by_count_desc]
    assert chunk_counts == sorted(chunk_counts, reverse=True)

    by_count_asc = kg.docs_by_chunk_count(descending=False)
    chunk_counts_asc = [d["chunk_count"] for d in by_count_asc]
    assert chunk_counts_asc == sorted(chunk_counts_asc)


# ---------------------------------------------------------------------------
# Remove cascade
# ---------------------------------------------------------------------------


def test_remove_document_deletes_node_chunks_and_edges(kg, docs_dir):
    a, b, c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    # Give B some chunks.
    for i in range(3):
        kg.add_chunk(Chunk(f"b::chunk::{i}", "t", 1), b, i)
    # Give A→B an extra LINKS_TO edge (multi-edge allowed).
    kg.add_link(a, b, "extra")

    kg.remove_document(b)

    # B is gone.
    assert kg.get_doc_metadata(b) == {}
    # B's chunks are gone.
    assert kg.get_chunks_for_doc(b) == []
    # A→B edges are gone.
    assert kg.get_reverse_links(b) == []
    # The chain's downstream edge B→C is also gone.
    assert kg.get_linked_docs(a, depth=2) == []
    # Other docs unaffected.
    assert kg.get_doc_metadata(a) != {}
    assert kg.get_doc_metadata(c) != {}


def test_remove_document_chunks_collection_empty_after(kg, docs_dir):
    p = docs_dir / "a.html"
    p.touch()
    kg.add_document(_doc(p))
    doc_id = str(p.resolve())
    for i in range(2):
        kg.add_chunk(Chunk(f"a::chunk::{i}", "t", 1), doc_id, i)
    assert kg.stats()["chunks"] == 2

    kg.remove_document(doc_id)
    assert kg.stats()["chunks"] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persistence_across_reopen(tmp_path, docs_dir):
    db_path = tmp_path / "kuzu_db"

    kg = KnowledgeGraph(db_path)
    a, b, c = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    kg.add_chunk(Chunk("a::chunk::0", "hello", 1), a, 0)
    expected_stats = kg.stats()
    kg.close()

    # Re-open the same directory in a fresh process-like state.
    kg2 = KnowledgeGraph(db_path)
    try:
        assert kg2.stats() == expected_stats
        assert kg2.get_chunks_for_doc(a) == ["a::chunk::0"]
        assert sorted(kg2.get_linked_docs(a, depth=2)) == sorted([b, c])
    finally:
        kg2.close()


def test_pdf_file_type_stored(kg, docs_dir):
    p = docs_dir / "manual.pdf"
    p.touch()
    kg.add_document(_doc(p, title="Manual", file_type="pdf"))
    meta = kg.get_doc_metadata(str(p.resolve()))
    assert meta["file_type"] == "pdf"
    assert meta["title"] == "Manual"


def test_context_manager_closes_cleanly(tmp_path):
    db_path = tmp_path / "kuzu_db"
    with KnowledgeGraph(db_path) as kg:
        assert kg.stats() == {
            "documents": 0,
            "chunks": 0,
            "link_edges": 0,
            "contains_edges": 0,
        }
    # Re-open after context-manager close.
    with KnowledgeGraph(db_path) as kg2:
        assert kg2.stats()["documents"] == 0


# ---------------------------------------------------------------------------
# Feature 0003 — graph correctness
# ---------------------------------------------------------------------------


def test_add_link_idempotent(kg, docs_dir):
    """Same (src, tgt, anchor) twice → get_linked_docs returns target once."""
    a, b, _ = _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html"])
    # a->b was already created by _build_chain; add again with same anchor.
    kg.add_link(a, b, "")
    result = kg.get_linked_docs(a, depth=1)
    assert result.count(b) == 1


def test_add_link_distinct_anchors(kg, docs_dir):
    """Two calls with different anchors → single edge; target returned once."""
    a, b = _build_chain(kg, docs_dir, ["a.html", "b.html"])[:2]
    kg.add_link(a, b, "first anchor")
    kg.add_link(a, b, "second anchor")
    assert kg.stats()["link_edges"] == 1
    result = kg.get_linked_docs(a, depth=1)
    assert result.count(b) == 1


def test_list_all_documents_offset_without_limit(kg, docs_dir):
    """offset is applied even when limit=0 (no limit)."""
    _build_chain(kg, docs_dir, ["a.html", "b.html", "c.html", "d.html"])
    all_docs = kg.list_all_documents()
    assert len(all_docs) == 4
    paged = kg.list_all_documents(limit=0, offset=2)
    assert len(paged) == 2
    assert paged == all_docs[2:]


def test_list_documents_by_chunk_count_pagination(kg, docs_dir):
    """5 docs with varying chunk counts → correct 3-item page in descending order."""
    names = ["a.html", "b.html", "c.html", "d.html", "e.html"]
    chunk_counts = [1, 5, 3, 0, 7]
    for name, count in zip(names, chunk_counts, strict=True):
        p = docs_dir / name
        p.touch()
        kg.add_document(_doc(p))
        doc_id = str(p.resolve())
        for i in range(count):
            kg.add_chunk(Chunk(f"{name}::chunk::{i}", "t", 1), doc_id, i)

    # First page: top 3 by chunk_count desc → counts 7, 5, 3.
    page1 = kg.list_documents_by_chunk_count(limit=3, offset=0)
    assert len(page1) == 3
    counts_page1 = [d["chunk_count"] for d in page1]
    assert counts_page1 == sorted(counts_page1, reverse=True)
    assert counts_page1 == [7, 5, 3]

    # Second page: remaining 2 docs → counts 1, 0.
    page2 = kg.list_documents_by_chunk_count(limit=3, offset=3)
    assert len(page2) == 2
    assert [d["chunk_count"] for d in page2] == [1, 0]
