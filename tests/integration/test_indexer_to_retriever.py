"""Integration: indexer → retriever, end-to-end on corpus_a."""

from __future__ import annotations

import pytest

from arg.crawler.crawler import crawl
from arg.graph import KnowledgeGraph
from arg.indexer import Indexer
from arg.retriever import HybridRetriever

pytestmark = pytest.mark.integration


def _build(base_config, ollama_embedder):
    """Returns (kg, indexer, retriever) with corpus_a indexed."""
    kg = KnowledgeGraph(base_config.kuzu_path("default"))
    indexer = Indexer(
        config=base_config,
        knowledge_graph=kg,
        embedder=ollama_embedder,
    )
    indexer.index(list(crawl(base_config.docs_root, base_config)))
    retriever = HybridRetriever(
        config=base_config,
        knowledge_graph=kg,
        embedder=ollama_embedder,
        chroma_documents_collection=indexer._docs_coll,
        chroma_chunks_collection=indexer._chunks_coll,
        bm25_index_path=base_config.bm25_index_path("default"),
        cluster_cache_path=base_config.cluster_cache_path("default"),
    )
    return kg, indexer, retriever


def _doc_ids_in(results, name_substring: str) -> bool:
    return any(name_substring in (r.node.metadata.get("doc_id") or "") for r in results)


def test_dense_stage_returns_chunks_for_known_query(base_config, ollama_embedder):
    kg, _, retriever = _build(base_config, ollama_embedder)
    try:
        results = retriever.retrieve("How do I authenticate?", enrich=False)
        assert results, "dense retrieval should return at least one chunk"
        # The auth content lives in page_a.html — it must surface.
        assert _doc_ids_in(results, "page_a.html")
    finally:
        kg.close()


def test_bm25_finds_exact_technical_term(base_config, ollama_embedder):
    """``X-Rate-Limit-Retry-After`` is a literal token in page_b.html."""
    kg, _, retriever = _build(base_config, ollama_embedder)
    try:
        results = retriever.retrieve("X-Rate-Limit-Retry-After header", enrich=False)
        assert results
        assert _doc_ids_in(results, "page_b.html")
    finally:
        kg.close()


def test_rrf_deduplicates_across_stages(base_config, ollama_embedder):
    """RRF must collapse chunks that appear in multiple stages by chunk_id."""
    kg, _, retriever = _build(base_config, ollama_embedder)
    try:
        results = retriever.retrieve("rate limit tiers", enrich=False)
        chunk_ids = [r.node.id_ for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))
        # All scores must be positive after fusion.
        for r in results:
            assert r.score is not None and r.score > 0
    finally:
        kg.close()


def test_graph_expansion_pulls_chunks_from_linked_docs(base_config, ollama_embedder):
    """Index has outgoing links to page_a / page_b / page_c. A query hitting
    index.html should surface chunks from its linked neighbours too."""
    kg, _, retriever = _build(base_config, ollama_embedder)
    try:
        # Use an index-overview-style query so dense matches index.html;
        # graph expansion then must reach into page_a / page_b / page_c.
        results = retriever.retrieve("What does this documentation cover?", enrich=False)
        doc_ids = {r.node.metadata.get("doc_id") for r in results}
        # At least one chunk should come from a neighbour of index.html.
        neighbour_hits = sum(
            1 for d in doc_ids if any(name in (d or "") for name in ("page_a", "page_b", "page_c"))
        )
        assert neighbour_hits >= 1
    finally:
        kg.close()


def test_lost_in_middle_reordering_places_top_at_position_zero(base_config, ollama_embedder):
    kg, _, retriever = _build(base_config, ollama_embedder)
    try:
        results = retriever.retrieve("authentication with API key", enrich=False)
        assert len(results) >= 2
        # Among returned chunks, the highest score must be at position 0.
        scores = [r.score or 0.0 for r in results]
        assert scores[0] == max(scores)
    finally:
        kg.close()


def test_filter_has_table_returns_only_table_chunks(base_config, ollama_embedder):
    kg, _, retriever = _build(base_config, ollama_embedder)
    try:
        results = retriever.retrieve(
            "rate limit per tier",
            enrich=False,
            filters={"has_table": True},
        )
        assert results, "table filter should still surface the rate-limit table chunk"
        for r in results:
            assert r.node.metadata.get("has_table") is True
    finally:
        kg.close()
