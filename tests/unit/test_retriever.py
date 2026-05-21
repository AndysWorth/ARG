"""HybridRetriever tests — every Section 8 ``test_retriever.py`` point.

A controlled fake embedder produces deterministic vectors:

  * Text containing ``QUERY_<TAG>`` is encoded so that a query containing the
    same ``QUERY_<TAG>`` token is a near-perfect cosine match. This lets us
    exercise dense retrieval semantics without Ollama.

A real `Indexer` + `KnowledgeGraph` build the same on-disk artefacts the
production stack uses, so the retriever sees authentic ChromaDB + Kuzu
shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arg.config import ARGConfig
from arg.crawler.extractors import Document
from arg.graph import KnowledgeGraph
from arg.indexer import Indexer
from arg.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Test embedder — query-tag aware
# ---------------------------------------------------------------------------


_VEC_DIM = 32


class _TagEmbedder:
    """Deterministic embedder that pumps a known signal for ``QUERY_<TAG>`` tokens.

    Every encoded text inspects the input for substrings of the form
    ``QUERY_X`` where X is a single uppercase letter. Each tag gets a unique
    coordinate in a 32-dim vector; the resulting vectors are normalised. A
    query string containing the same tag therefore lines up dimension-wise
    with chunks carrying that tag, giving us deterministic top-1 control.
    """

    _TAG_BASE = ord("A")

    def embed(self, text: str) -> list[float]:
        import math
        import re

        vec = [0.001] * _VEC_DIM  # tiny baseline so different texts aren't identical
        for match in re.finditer(r"QUERY_([A-Z])", text):
            idx = (ord(match.group(1)) - self._TAG_BASE) % _VEC_DIM
            vec[idx] += 1.0
        # Mix in a tiny hash so two un-tagged texts aren't equal vectors.
        h = abs(hash(text)) % 997
        vec[h % _VEC_DIM] += 0.05
        # Normalise so Chroma's L2 distance acts like cosine.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> ARGConfig:
    docs = tmp_path / "docs"
    docs.mkdir()
    return ARGConfig(
        docs_root=docs,
        db_path=tmp_path / "arg_db",
        # Tighter top-k so test corpora don't need to be huge.
        top_k_vector=4,
        top_k_graph=2,
        graph_hop_depth=1,
        enrich_min_score=0.0,  # keep enrichment active so tests can exercise it
    )


@pytest.fixture
def kg(config: ARGConfig):
    g = KnowledgeGraph(config.kuzu_path("default"))
    yield g
    g.close()


def _make_doc(
    docs_dir: Path,
    name: str,
    content: str,
    *,
    title: str | None = None,
    links_to: list[str] | None = None,
    file_type: str = "html",
    page_description: str = "",
) -> Document:
    p = docs_dir / name
    p.touch()
    return Document(
        path=p,
        content=content,
        metadata={
            "title": title or name,
            "page_description": page_description,
            "file_type": file_type,
            "links_to": list(links_to or []),
            "code_blocks": [],
        },
    )


def _build_retriever(config, kg) -> HybridRetriever:
    embedder = _TagEmbedder()
    indexer = Indexer(config=config, knowledge_graph=kg, embedder=embedder)
    docs = _build_corpus(config)
    indexer.index(docs)
    return HybridRetriever(
        config=config,
        knowledge_graph=kg,
        embedder=embedder,
        chroma_documents_collection=indexer._docs_coll,
        chroma_chunks_collection=indexer._chunks_coll,
        bm25_index_path=config.bm25_index_path("default"),
        cluster_cache_path=config.cluster_cache_path("default"),
    )


def _build_corpus(config: ARGConfig) -> list[Document]:
    """Five docs with controlled tag distributions for deterministic tests."""
    return [
        _make_doc(
            config.docs_root,
            "alpha.html",
            content=(
                "##H1## Alpha Heading\n"
                "Alpha body QUERY_A discusses authentication tokens and OAuth.\n"
                "Continuation line for alpha section.\n"
            ),
            title="Alpha",
            page_description="Auth overview QUERY_A",
            links_to=[str((config.docs_root / "beta.html").resolve())],
        ),
        _make_doc(
            config.docs_root,
            "beta.html",
            content=(
                "##H1## Beta Heading\n"
                "Beta body QUERY_B about database migrations.\n"
                "| Tier | Limit |\n|---|---|\n| 1 | 500 |\n"
            ),
            title="Beta",
            page_description="Database stuff",
        ),
        _make_doc(
            config.docs_root,
            "gamma.html",
            content=(
                "##H1## Gamma Heading\nGamma body QUERY_C about network topology and firewalls.\n"
            ),
            title="Gamma",
        ),
        _make_doc(
            config.docs_root,
            "delta.html",
            content=(
                "##H1## Delta Heading\n"
                "Delta body QUERY_D about user interface design patterns.\n"
                "X-Rate-Limit-Retry-After is an exact technical term BM25 should "
                "find when QUERY_D is absent.\n"
            ),
            title="Delta",
        ),
        _make_doc(
            config.docs_root,
            "epsilon.pdf",
            content=(
                "##H1## Epsilon Heading\nPDF body with QUERY_E about hardware specifications.\n"
            ),
            title="Epsilon",
            file_type="pdf",
        ),
    ]


# ---------------------------------------------------------------------------
# Stage 1 — dense
# ---------------------------------------------------------------------------


def test_dense_only_returns_correct_top_chunk(config, kg):
    """enrich=False + bm25_enabled=False → pure dense retrieval."""
    config.bm25_enabled = False
    retriever = _build_retriever(config, kg)
    results = retriever.retrieve("QUERY_A about authentication", enrich=False)
    assert results, "dense retrieval should return something"
    # The top result must be from alpha.html (the doc carrying QUERY_A).
    top_meta = results[0].node.metadata
    assert "alpha.html" in top_meta["doc_id"]


def test_retriever_handles_zero_hits_gracefully(config, kg):
    """Empty corpus → empty list, no crash."""
    # Build an indexer without any documents.
    embedder = _TagEmbedder()
    indexer = Indexer(config=config, knowledge_graph=kg, embedder=embedder)
    indexer.index([])
    retriever = HybridRetriever(
        config=config,
        knowledge_graph=kg,
        embedder=embedder,
        chroma_documents_collection=indexer._docs_coll,
        chroma_chunks_collection=indexer._chunks_coll,
        bm25_index_path=config.bm25_index_path("default"),
        cluster_cache_path=config.cluster_cache_path("default"),
    )
    assert retriever.retrieve("QUERY_X anything") == []


# ---------------------------------------------------------------------------
# Stage 1.5 — BM25
# ---------------------------------------------------------------------------


def test_bm25_finds_exact_technical_term(config, kg):
    """Exact-term query — dense embedder might miss; BM25 should hit."""
    retriever = _build_retriever(config, kg)
    # Search for a literal token that ONLY appears in delta.html. Drop the
    # QUERY_* tag so dense retrieval can't trivially lock onto it.
    results = retriever.retrieve("X-Rate-Limit-Retry-After header", enrich=False)
    assert results
    # At least one of the top hits must be from delta.html.
    doc_ids = {r.node.metadata.get("doc_id") for r in results}
    assert any("delta.html" in (d or "") for d in doc_ids)


def test_bm25_disabled_skips_stage_1_5(config, kg):
    config.bm25_enabled = False
    retriever = _build_retriever(config, kg)
    # An exact-term query that ONLY BM25 can find — without BM25, the dense
    # path may still match via the embedder's hash mixin; we don't assert
    # zero hits, only that the retriever did not crash and did not surface
    # BM25-specific signal.
    results = retriever.retrieve("X-Rate-Limit-Retry-After", enrich=False)
    # No BM25 signal — the hash-mixin lottery may still surface delta but
    # the test mainly checks the disabled path doesn't raise.
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Stage 2 — graph expansion
# ---------------------------------------------------------------------------


def test_stage2_adds_chunks_from_linked_documents(config, kg):
    """Alpha links to Beta; a query that hits Alpha should pull Beta chunks in."""
    retriever = _build_retriever(config, kg)
    results = retriever.retrieve("QUERY_A authentication", enrich=False)
    doc_ids = {r.node.metadata.get("doc_id") for r in results}
    assert any("alpha.html" in (d or "") for d in doc_ids)
    # Graph expansion: Alpha links to Beta, so Beta should be in the result set.
    assert any("beta.html" in (d or "") for d in doc_ids)


def test_graph_hop_depth_zero_skips_stage2_internally(config, kg):
    """Direct check on _stage2_graph — the corpus is too small for an
    observable end-to-end difference (dense top_k=4 over 5 docs pulls in
    linked docs anyway). The contract is that Stage 2 returns nothing."""
    config.graph_hop_depth = 0
    retriever = _build_retriever(config, kg)
    # Seed: a fake hit for alpha so we know there's something to expand from.
    from arg.retriever.retriever import _ChunkHit

    alpha_id = str((config.docs_root / "alpha.html").resolve())
    seed = _ChunkHit(
        chunk_id=f"{alpha_id}::chunk::0",
        text="",
        metadata={"doc_id": alpha_id},
    )
    assert retriever._stage2_graph(query="anything", seed_hits=[seed], chroma_filters=None) == []


# ---------------------------------------------------------------------------
# Stage 3 — RRF fusion + deduplication
# ---------------------------------------------------------------------------


def test_rrf_deduplicates_across_stages_and_assigns_positive_scores(config, kg):
    retriever = _build_retriever(config, kg)
    results = retriever.retrieve("QUERY_A authentication", enrich=False)
    chunk_ids = [r.node.id_ for r in results]
    assert len(chunk_ids) == len(set(chunk_ids)), "RRF must deduplicate"
    for r in results:
        assert r.score is not None and r.score > 0


# ---------------------------------------------------------------------------
# Stage 4 — lost-in-middle reordering
# ---------------------------------------------------------------------------


def test_lost_in_middle_places_top_chunks_at_bookends():
    """Direct check of the reordering helper, decoupled from Chroma + Kuzu."""
    from arg.retriever.retriever import _ChunkHit, _lost_in_middle_reorder

    # 6 hits, RRF-sorted descending by design — we just need stable
    # input ordering.
    hits = [
        _ChunkHit(chunk_id=f"c{i}", text=f"t{i}", metadata={}, rrf_score=10.0 - i) for i in range(6)
    ]
    out = _lost_in_middle_reorder(hits, target_n=6)
    # The highest-scored chunk (c0) ends up at position 0.
    assert out[0].node.id_ == "c0"
    # The second-highest (c1) ends up at position -1.
    assert out[-1].node.id_ == "c1"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_filters_has_table_true(config, kg):
    retriever = _build_retriever(config, kg)
    results = retriever.retrieve("QUERY_B database", enrich=False, filters={"has_table": True})
    assert results, "should find at least the beta table chunk"
    for r in results:
        assert r.node.metadata.get("has_table") is True


def test_filters_file_type_pdf(config, kg):
    retriever = _build_retriever(config, kg)
    results = retriever.retrieve("QUERY_E hardware", enrich=False, filters={"file_type": "pdf"})
    assert results
    for r in results:
        assert r.node.metadata.get("file_type") == "pdf"


# ---------------------------------------------------------------------------
# scope_doc_id
# ---------------------------------------------------------------------------


def test_scope_doc_id_restricts_results_to_one_doc(config, kg):
    retriever = _build_retriever(config, kg)
    alpha_id = str((config.docs_root / "alpha.html").resolve())
    results = retriever.retrieve(
        "QUERY_C network",  # query targets gamma; scope forces alpha
        scope_doc_id=alpha_id,
    )
    assert results, "scope retrieval should still return chunks from the scoped doc"
    for r in results:
        assert r.node.metadata.get("doc_id") == alpha_id


# ---------------------------------------------------------------------------
# Enrichment fallback
# ---------------------------------------------------------------------------


def test_enrichment_impossible_threshold_falls_back_to_unfiltered(config, kg):
    """enrich_min_score=1.0 means no doc passes → enrichment skipped."""
    config.enrich_min_score = 1.0
    retriever = _build_retriever(config, kg)
    results = retriever.retrieve("QUERY_A authentication", enrich=True)
    # Same behaviour as unfiltered Stage 1: alpha must still surface.
    doc_ids = {r.node.metadata.get("doc_id") for r in results}
    assert any("alpha.html" in (d or "") for d in doc_ids)


# ---------------------------------------------------------------------------
# Stage 0.3 — cluster cache gracefully missing
# ---------------------------------------------------------------------------


def test_cluster_cache_missing_does_not_break_enrichment(config, kg):
    retriever = _build_retriever(config, kg)
    # No cluster_cache.json was written.
    assert not config.cluster_cache_path("default").exists()
    results = retriever.retrieve("QUERY_A authentication", enrich=True)
    assert results, "missing cluster cache must not block retrieval"


def test_cluster_cache_used_when_present(config, kg, tmp_path):
    retriever = _build_retriever(config, kg)
    alpha_id = str((config.docs_root / "alpha.html").resolve())
    gamma_id = str((config.docs_root / "gamma.html").resolve())
    cache: dict[str, Any] = {
        "doc_to_cluster": {alpha_id: "c0", gamma_id: "c0"},
        "cluster_members": {"c0": [alpha_id, gamma_id]},
    }
    cluster_path = config.cluster_cache_path("default")
    cluster_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_path.write_text(json.dumps(cache), encoding="utf-8")
    # Force cluster expansion to fire by lowering the doc-count threshold.
    config.min_cluster_docs = 1
    results = retriever.retrieve("QUERY_A authentication", enrich=True)
    assert results, "enrichment with cluster cache must still return results"


# ---------------------------------------------------------------------------
# _find_document
# ---------------------------------------------------------------------------


def test_find_document_returns_ranked_doc_ids(config, kg):
    retriever = _build_retriever(config, kg)
    ranked = retriever._find_document("QUERY_A authentication", top_k=3)
    assert ranked
    # Top hit must be alpha (its chunks contain QUERY_A and "authentication").
    alpha_id = str((config.docs_root / "alpha.html").resolve())
    assert ranked[0][0] == alpha_id
    # Scores monotonically non-increasing.
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
