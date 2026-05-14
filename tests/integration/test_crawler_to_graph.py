"""Integration: crawler → knowledge graph.

These tests don't need Ollama (crawl + graph writes are purely local), so
they aren't gated on the ``require_ollama`` fixture.
"""

from __future__ import annotations

import pytest

from arg.config import ARGConfig
from arg.crawler.crawler import crawl
from arg.graph import KnowledgeGraph

pytestmark = pytest.mark.integration


def _ingest(config: ARGConfig, kg: KnowledgeGraph) -> None:
    """Walk crawler output into the graph; record link edges in pass 2."""
    documents = list(crawl(config.docs_root, config))
    known_paths: set[str] = set()
    for doc in documents:
        kg.add_document(doc)
        known_paths.add(str(doc.path.resolve()))
    for doc in documents:
        src = str(doc.path.resolve())
        for target in doc.metadata.get("links_to", []) or []:
            if target in known_paths and target != src:
                kg.add_link(src, target, "")


def test_all_fixture_documents_appear_as_nodes(base_config, tmp_db):
    kg = KnowledgeGraph(base_config.kuzu_path("default"))
    try:
        _ingest(base_config, kg)
        listed = kg.list_all_documents()
        titles = {d["title"] for d in listed}
        # Titles after extraction strip ' | ' / ' — ' suffix tails. page_a's
        # "Kraken API — Authentication | Kraken API Docs" collapses to
        # "Kraken API" because the em-dash separator wins over the pipe.
        for expected in (
            "ARG Test Documentation",
            "Kraken API",  # page_a — em-dash separator extracts the first segment
            "Rate Limits",  # page_b — pipe-suffix stripping
            "Error Codes",  # page_c — pipe-suffix stripping
        ):
            assert expected in titles, f"missing title: {expected!r}; got {titles!r}"
    finally:
        kg.close()


def test_known_links_appear_as_edges(base_config):
    kg = KnowledgeGraph(base_config.kuzu_path("default"))
    try:
        _ingest(base_config, kg)
        index_id = str((base_config.docs_root / "index.html").resolve())
        page_a_id = str((base_config.docs_root / "page_a.html").resolve())
        page_b_id = str((base_config.docs_root / "page_b.html").resolve())
        page_c_id = str((base_config.docs_root / "subdir" / "page_c.html").resolve())

        index_links = kg.get_linked_docs(index_id, depth=1)
        assert page_a_id in index_links
        assert page_b_id in index_links
        assert page_c_id in index_links
    finally:
        kg.close()


def test_get_linked_docs_depth_one_from_index(base_config):
    kg = KnowledgeGraph(base_config.kuzu_path("default"))
    try:
        _ingest(base_config, kg)
        index_id = str((base_config.docs_root / "index.html").resolve())
        depth1 = set(kg.get_linked_docs(index_id, depth=1))
        # index -> page_a, page_b, page_c
        assert len(depth1) == 3

    finally:
        kg.close()


def test_circular_link_dedup(base_config):
    """page_b links back to index — depth-N traversal must terminate."""
    kg = KnowledgeGraph(base_config.kuzu_path("default"))
    try:
        _ingest(base_config, kg)
        index_id = str((base_config.docs_root / "index.html").resolve())
        # depth=5 over a 4-node graph must not blow up.
        result = kg.get_linked_docs(index_id, depth=5)
        # Source must not appear in its own forward set.
        assert index_id not in result
    finally:
        kg.close()
