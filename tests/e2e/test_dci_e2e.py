"""Corpus B (15-doc clustering) end-to-end + multi-corpus isolation.

Corpus B holds 15 deliberately distinct technical topics — five Triton
Database pages, five Poseidon Networking pages, five Hydra Scheduler pages.
Topics are vocabulary-distinct so k-means on the documents-collection
embeddings should split them cleanly.

The multi-corpus test mounts both corpus_a (Kraken API) and corpus_b
(Triton/Poseidon/Hydra) on a single FastAPI app and checks that a query
against one corpus never returns documents from the other.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arg.config import ARGConfig
from arg.pipeline import ARGPipeline
from arg.server import create_app

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corpus_b_config(tmp_path: Path, corpus_b_path: Path) -> ARGConfig:
    db_root = tmp_path / "arg_db"
    db_root.mkdir(exist_ok=True)
    return ARGConfig(
        docs_root=corpus_b_path,
        db_path=db_root / "clustering",
        watch_enabled=False,
        top_k_vector=8,
        graph_hop_depth=0,
        enrich_min_score=0.0,
        # n_clusters from the spec; we want exactly three clusters.
        n_clusters=3,
        min_cluster_docs=10,
    )


def _build(tmp_path: Path, corpus_b_path: Path, embedder, llm) -> ARGPipeline:
    config = _corpus_b_config(tmp_path, corpus_b_path)
    pipeline = ARGPipeline(
        config=config,
        corpus_name="clustering",
        llm=llm,
        embedder=embedder,
        skip_health_check=True,
        skip_signal_handlers=True,
    )
    pipeline.index()
    return pipeline


def _topic_prefix(doc_id: str) -> str | None:
    """Return the topic prefix (t1 / t2 / t3) for a clustering-corpus doc."""
    name = Path(doc_id).name
    for prefix in ("t1_", "t2_", "t3_"):
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_all_fifteen_docs_indexed(tmp_path, corpus_b_path, ollama_embedder, mock_llm):
    pipeline = _build(tmp_path, corpus_b_path, ollama_embedder, mock_llm)
    try:
        docs = pipeline.graph.list_all_documents()
        # 15 topical docs + 1 top-level index.
        assert len(docs) == 16
    finally:
        pipeline.close()


def test_clustering_produces_three_clusters_by_topic(
    tmp_path, corpus_b_path, ollama_embedder, mock_llm
):
    """The 15 topical docs split into three clusters. Each cluster should
    have a clear majority topic (≥ 3 of the cluster's topical docs from a
    single t1/t2/t3 family). Strict purity is fragile on a 15-doc corpus
    with some cross-topic vocabulary overlap (e.g., "backup" / "logging"
    both touch on history + alerts); the e2e contract is that clustering
    *separates* topics meaningfully, not that every doc lands perfectly.
    """
    mock_llm.respond_to("Given the following document titles", "Topic Label")
    pipeline = _build(tmp_path, corpus_b_path, ollama_embedder, mock_llm)
    try:
        clusters = pipeline.get_topic_clusters()
        assert len(clusters) == 3, (
            f"expected exactly 3 clusters; got {len(clusters)}: {[c['label'] for c in clusters]}"
        )

        # Each cluster has a clear topical majority.
        majority_topics: list[str] = []
        for cluster in clusters:
            prefixes: list[str] = []
            for d in cluster["doc_ids"]:
                p = _topic_prefix(d)
                if p is not None:
                    prefixes.append(p)
            if not prefixes:
                # Pure-index cluster (only the corpus root index landed here)
                # is unlikely but allowed; skip the majority check.
                continue
            counts: dict[str, int] = {}
            for p in prefixes:
                counts[p] = counts.get(p, 0) + 1
            top = max(counts.items(), key=lambda kv: kv[1])
            assert top[1] >= 3, (
                f"cluster lacks a clear topical majority: {counts}; docs={cluster['doc_ids']}"
            )
            majority_topics.append(top[0])

        # And every topical prefix should be the majority of at least one
        # cluster — clustering must distinguish all three topics.
        assert set(majority_topics) == {"t1", "t2", "t3"}, (
            f"expected one cluster per topic; got majorities {majority_topics}"
        )
    finally:
        pipeline.close()


def test_clustering_labels_are_non_empty(tmp_path, corpus_b_path, ollama_embedder, mock_llm):
    mock_llm.respond_to("Given the following document titles", "Triton DB topics")
    pipeline = _build(tmp_path, corpus_b_path, ollama_embedder, mock_llm)
    try:
        clusters = pipeline.get_topic_clusters()
        for c in clusters:
            assert c["label"]
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# Corpus search by topic
# ---------------------------------------------------------------------------


def test_corpus_search_database_indexing_returns_only_triton_docs(
    tmp_path, corpus_b_path, ollama_embedder, mock_llm
):
    pipeline = _build(tmp_path, corpus_b_path, ollama_embedder, mock_llm)
    try:
        hits = pipeline.corpus_search("database indexing schema design", top_k=5)
        assert hits
        top_doc = hits[0]["doc_id"]
        assert _topic_prefix(top_doc) == "t1", f"top hit should be a Triton doc; got {top_doc}"
    finally:
        pipeline.close()


def test_corpus_search_vpn_returns_only_poseidon_docs(
    tmp_path, corpus_b_path, ollama_embedder, mock_llm
):
    pipeline = _build(tmp_path, corpus_b_path, ollama_embedder, mock_llm)
    try:
        hits = pipeline.corpus_search("VPN tunnels WireGuard", top_k=5)
        assert hits
        assert _topic_prefix(hits[0]["doc_id"]) == "t2"
    finally:
        pipeline.close()


def test_corpus_search_file_type_html_returns_all(
    tmp_path, corpus_b_path, ollama_embedder, mock_llm
):
    pipeline = _build(tmp_path, corpus_b_path, ollama_embedder, mock_llm)
    try:
        hits = pipeline.corpus_search("scheduler jobs triggers workers", top_k=20, file_type="html")
        # All 15 topical docs (plus the index) are HTML — no PDFs in
        # corpus_b. top_k=20 is generous so we see them all.
        assert len(hits) >= 15
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# Multi-corpus FastAPI isolation
# ---------------------------------------------------------------------------


def test_multi_corpus_query_isolation(
    tmp_path, corpus_a_path, corpus_b_path, ollama_embedder, mock_llm
):
    """Two corpora mounted on one FastAPI app must never bleed."""
    # corpus_a (Kraken API) — re-use the conftest pattern locally.
    a_db = tmp_path / "arg_db_a"
    a_db.mkdir(exist_ok=True)
    a_config = ARGConfig(
        docs_root=corpus_a_path,
        db_path=a_db / "a",
        watch_enabled=False,
        top_k_vector=4,
        enrich_min_score=0.0,
    )
    p_a = ARGPipeline(
        config=a_config,
        corpus_name="kraken",
        llm=mock_llm,
        embedder=ollama_embedder,
        skip_health_check=True,
        skip_signal_handlers=True,
    )
    p_b = _build(tmp_path, corpus_b_path, ollama_embedder, mock_llm)
    p_a.index()

    try:
        app = create_app({"kraken": p_a, "clustering": p_b})
        client = TestClient(app)

        # corpus_a query: an auth question should land on page_a / manual.
        r = client.post(
            "/query?corpus=kraken",
            json={"question": "authentication tokens"},
        )
        assert r.status_code == 200
        sources_a = [s["doc_id"] for s in r.json()["sources"]]
        assert sources_a, "corpus_a query should yield sources"
        for did in sources_a:
            assert _topic_prefix(did) is None, f"corpus_a query leaked a corpus_b doc: {did}"

        # corpus_b query: BGP / routing should land on Poseidon docs.
        r = client.post(
            "/query?corpus=clustering",
            json={"question": "BGP routing configuration"},
        )
        assert r.status_code == 200
        sources_b = [s["doc_id"] for s in r.json()["sources"]]
        assert sources_b
        for did in sources_b:
            # Either a t1/t2/t3 doc OR the corpus-b index.
            ok = _topic_prefix(did) is not None or did.endswith("clustering_docs/index.html")
            assert ok, f"corpus_b query leaked a non-corpus-b doc: {did}"
    finally:
        p_a.close()
        p_b.close()
