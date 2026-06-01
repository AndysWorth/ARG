"""ARGPipeline tests — Section 10 ``test_pipeline.py`` test points.

The pipeline is exercised end-to-end with a fake embedder + fake LLM so the
suite stays offline (``skip_health_check=True`` short-circuits the Ollama
health check; injecting both backends bypasses the LlamaIndex-Ollama adapter
code paths that need a real daemon).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from arg.config import ARGConfig
from arg.pipeline import ARGPipeline

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


_VEC_DIM = 32


class _TagEmbedder:
    """Same QUERY_<TAG> embedder used elsewhere in the suite."""

    _BASE = ord("A")

    def embed(self, text: str) -> list[float]:
        import math
        import re

        vec = [0.001] * _VEC_DIM
        for m in re.finditer(r"QUERY_([A-Z])", text):
            idx = (ord(m.group(1)) - self._BASE) % _VEC_DIM
            vec[idx] += 1.0
        vec[abs(hash(text)) % _VEC_DIM] += 0.05
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _ScriptedLLM:
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "ANSWER FROM LLM",
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for trigger, response in self.responses.items():
            if trigger in prompt:
                return response
        return self.default

    def complete_structured(self, prompt: str, schema: dict) -> str:
        return self.complete(prompt)

    def stream_complete(self, prompt: str) -> Iterator[str]:
        yield from self.complete(prompt)


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
        top_k_vector=3,
        top_k_graph=1,
        graph_hop_depth=1,
        enrich_min_score=0.0,
        watch_enabled=False,
    )


def _write_corpus(docs_dir: Path) -> None:
    """Write a small corpus straight to disk (the pipeline crawls it)."""
    (docs_dir / "index.html").write_text(
        "<html><head><title>Index | Site</title></head><body>"
        "<h1>Index</h1>"
        "<p>Welcome to the docs. QUERY_A authentication and OAuth flow are documented.</p>"
        '<a href="page_a.html">A</a><a href="page_b.html">B</a>'
        "</body></html>",
        encoding="utf-8",
    )
    (docs_dir / "page_a.html").write_text(
        "<html><head><title>Page A | Site</title></head><body>"
        "<h1>Page A</h1>"
        "<p>Page A discusses QUERY_A authentication tokens in detail.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (docs_dir / "page_b.html").write_text(
        "<html><head><title>Page B | Site</title></head><body>"
        "<h1>Page B</h1>"
        "<p>Page B documents QUERY_B database migrations and schemas.</p>"
        "<table><tr><th>Column</th><th>Type</th></tr><tr><td>id</td><td>int</td></tr></table>"
        "</body></html>",
        encoding="utf-8",
    )


def _build_pipeline(
    config: ARGConfig,
    *,
    llm: _ScriptedLLM | None = None,
    embedder: _TagEmbedder | None = None,
    skip_signal_handlers: bool = True,
) -> ARGPipeline:
    return ARGPipeline(
        config=config,
        corpus_name="default",
        llm=llm or _ScriptedLLM(),
        embedder=embedder or _TagEmbedder(),
        skip_health_check=True,
        skip_signal_handlers=skip_signal_handlers,
    )


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def test_health_check_failure_raises(config):
    """No Ollama running on the bogus URL → RuntimeError at construction."""
    config.ollama_base_url = "http://localhost:1"  # nobody listens here
    with pytest.raises(RuntimeError, match="Ollama daemon not reachable"):
        ARGPipeline(
            config=config,
            corpus_name="default",
            llm=_ScriptedLLM(),
            embedder=_TagEmbedder(),
            skip_health_check=False,
            skip_signal_handlers=True,
        )


def test_skip_health_check_bypasses(config):
    """Tests must be able to build a pipeline without Ollama."""
    pipeline = _build_pipeline(config)
    try:
        assert pipeline.graph is not None
    finally:
        pipeline.close()


def test_schema_drift_detection(config, tmp_path):
    """A change to the indexed config raises RuntimeError on construction."""
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        pipeline.index()
    finally:
        pipeline.close()

    # Mutate a hash field; a second pipeline on the same DB must refuse.
    drifted = ARGConfig(
        docs_root=config.docs_root,
        db_path=config.db_path,
        chunk_size=512,  # was 1024 — drift
        watch_enabled=False,
    )
    with pytest.raises(RuntimeError, match="Schema drift detected"):
        ARGPipeline(
            config=drifted,
            corpus_name="default",
            llm=_ScriptedLLM(),
            embedder=_TagEmbedder(),
            skip_health_check=True,
            skip_signal_handlers=True,
        )


def test_no_schema_check_on_first_run(config):
    """First-ever construction on an empty corpus_root must not raise."""
    pipeline = _build_pipeline(config)
    pipeline.close()
    # No config_hash.json should exist yet (index() hasn't run).
    assert not config.config_hash_path("default").exists()


# ---------------------------------------------------------------------------
# Components instantiated in __init__
# ---------------------------------------------------------------------------


def test_query_processor_present_on_pipeline(config):
    pipeline = _build_pipeline(config)
    try:
        from arg.generator import QueryProcessor

        assert isinstance(pipeline.query_processor, QueryProcessor)
    finally:
        pipeline.close()


def test_all_subcomponents_present(config):
    pipeline = _build_pipeline(config)
    try:
        assert pipeline.graph is not None
        assert pipeline.indexer is not None
        assert pipeline.retriever is not None
        assert pipeline.generator is not None
        assert pipeline.analyst is not None
        assert pipeline.explorer is not None
        assert pipeline.query_processor is not None
        # watch_enabled=False → watcher is None.
        assert pipeline.watcher is None
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# Index + query
# ---------------------------------------------------------------------------


def test_index_writes_bm25_and_schema_hash(config):
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        stats = pipeline.index()
        assert stats["documents_indexed"] >= 1
        assert stats["chunks_written"] >= 1
        assert config.bm25_index_path("default").is_file()
        assert config.config_hash_path("default").is_file()
    finally:
        pipeline.close()


def test_query_returns_argresult_with_sources(config):
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        pipeline.index()
        result = pipeline.query("QUERY_A authentication tokens", enrich=False)
        assert result.answer
        assert result.sources, "non-empty corpus should yield sources"
        for ref in result.sources:
            assert ref.chunk_id
            assert ref.doc_id
    finally:
        pipeline.close()


def test_query_passes_filters_to_retriever(config):
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        pipeline.index()
        result = pipeline.query("QUERY_B migrations", enrich=False, filters={"has_table": True})
        # Every returned chunk must satisfy the filter.
        for ref in result.sources:
            # Verify via the indexer's Chroma collection.
            chroma = pipeline.indexer._chunks_coll.get(ids=[ref.chunk_id], include=["metadatas"])
            assert chroma["metadatas"][0]["has_table"] is True
    finally:
        pipeline.close()


def test_query_rewrite_populated_for_conversational(config):
    _write_corpus(config.docs_root)
    llm = _ScriptedLLM(
        responses={
            "Rewrite the following": "Auth methods and API key configuration",
            "Does the following question contain": "Auth methods",
        }
    )
    pipeline = _build_pipeline(config, llm=llm)
    try:
        pipeline.index()
        result = pipeline.query("how do I log in to my account")
        assert result.rewritten_query == "Auth methods and API key configuration"
    finally:
        pipeline.close()


def test_query_sub_queries_populated_for_compound(config):
    _write_corpus(config.docs_root)
    llm = _ScriptedLLM(
        responses={
            "Rewrite the following": "auth and migrations",
            "Does the following question contain": (
                '{"sub_questions": ["How does authentication work?", "What are the database migrations?"]}'
            ),
        }
    )
    pipeline = _build_pipeline(config, llm=llm)
    try:
        pipeline.index()
        result = pipeline.query("how does auth work and what about migrations")
        assert result.sub_queries == [
            "How does authentication work?",
            "What are the database migrations?",
        ]
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# Incremental + cache invalidation
# ---------------------------------------------------------------------------


def test_reindex_is_idempotent(config):
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        first = pipeline.index()
        second = pipeline.index()
        assert second["documents_skipped"] >= 1
        assert second["documents_indexed"] == 0
        _ = first  # silence ruff
    finally:
        pipeline.close()


def test_index_recomputes_cluster_cache(config):
    import time

    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        pipeline.index()
        # Cluster computation is async; close() joins the thread.
        pipeline.close()

        cache_path = config.cluster_cache_path("default")
        assert cache_path.is_file()
        # Overwrite with a stale sentinel.
        cache_path.write_text(
            json.dumps({"doc_to_cluster": {}, "cluster_members": {}, "labels": {}})
        )

        pipeline = _build_pipeline(config)
        # Re-index → background thread replaces stale cache with a fresh one.
        pipeline.index()
        # Poll until the background thread writes the fresh cache (3 s budget).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if cache_path.is_file():
                data = json.loads(cache_path.read_text())
                if data != {"doc_to_cluster": {}, "cluster_members": {}, "labels": {}}:
                    break
            time.sleep(0.02)
        else:
            pytest.fail("Cluster cache not refreshed within 3 s")
    finally:
        pipeline.close()


def test_remove_document_clears_chunks(config):
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        pipeline.index()
        doc_id = str((config.docs_root / "page_a.html").resolve())
        pipeline.remove_document(doc_id)
        assert pipeline.graph.get_doc_metadata(doc_id) == {}
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# Corpus stats + DCI public API
# ---------------------------------------------------------------------------


def test_corpus_stats_shape(config):
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        pipeline.index()
        stats = pipeline.corpus_stats()
        for key in (
            "documents",
            "chunks",
            "link_edges",
            "most_linked",
            "orphaned",
            "by_size_preview",
        ):
            assert key in stats
        assert stats["documents"] >= 3
    finally:
        pipeline.close()


def test_get_topic_clusters_returns_list(config):
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        pipeline.index()
        clusters = pipeline.get_topic_clusters()
        assert isinstance(clusters, list)
        for c in clusters:
            assert "label" in c
            assert "doc_ids" in c
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_is_idempotent(config):
    pipeline = _build_pipeline(config)
    pipeline.close()
    pipeline.close()  # second call is a no-op


def test_context_manager(config):
    _write_corpus(config.docs_root)
    with _build_pipeline(config) as pipeline:
        pipeline.index()
        result = pipeline.query("QUERY_A authentication", enrich=False)
        assert result.sources
    # After __exit__, the graph is closed.


# ---------------------------------------------------------------------------
# Feature 0003 — async cluster computation
# ---------------------------------------------------------------------------


def test_index_returns_before_cluster_completes(config):
    """index() must return before the background cluster thread finishes."""
    import threading
    import time
    from unittest.mock import patch

    gate = threading.Event()
    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    original_compute = pipeline.explorer._compute_clusters

    def _slow_compute():
        gate.wait(timeout=10.0)
        return original_compute()

    try:
        with patch.object(pipeline.explorer, "_compute_clusters", side_effect=_slow_compute):
            start = time.monotonic()
            pipeline.index()
            elapsed = time.monotonic() - start
            # index() must return before the gate is released (cluster still blocked).
            assert elapsed < 1.0, f"index() took {elapsed:.2f}s — should be near-instant"
    finally:
        gate.set()  # unblock so close() can join cleanly
        pipeline.close()


def test_cluster_eventually_populated(config):
    """Cluster cache is populated by the background thread within 3 s."""
    import time

    _write_corpus(config.docs_root)
    pipeline = _build_pipeline(config)
    try:
        pipeline.index()
        cache_path = config.cluster_cache_path("default")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if cache_path.is_file():
                data = json.loads(cache_path.read_text())
                if data.get("cluster_members"):
                    break
            time.sleep(0.02)
        else:
            pytest.fail("Cluster cache not populated within 3 s")
        clusters = pipeline.get_topic_clusters()
        assert isinstance(clusters, list)
        assert clusters
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# embed_batch batching tests (Feature 0005)
# ---------------------------------------------------------------------------


def _make_embedder_under_test(config: ARGConfig):
    """Return an _OllamaEmbedderAdapter with a mocked ollama.Client.

    The mock client's embed() returns fake embeddings so no real Ollama
    connection is needed.
    """
    from unittest.mock import MagicMock, patch

    dim = 768
    mock_client = MagicMock()

    def _fake_embed(**kwargs):
        n = len(kwargs.get("input", ["x"]))
        return MagicMock(embeddings=[[0.1] * dim for _ in range(n)])

    mock_client.embed.side_effect = _fake_embed

    pipeline = ARGPipeline(
        config=config,
        corpus_name="default",
        llm=_ScriptedLLM(),
        embedder=_TagEmbedder(),
        skip_health_check=True,
        skip_signal_handlers=True,
    )

    with (
        patch("ollama.Client", return_value=mock_client),
        patch("tiktoken.get_encoding", return_value=MagicMock(encode=lambda t: [1, 2, 3])),
    ):
        embedder = pipeline._default_embedder()

    return embedder, mock_client


def test_embed_batch_empty_returns_empty(config):
    """embed_batch([]) must return [] without calling Ollama."""
    embedder, mock_client = _make_embedder_under_test(config)
    result = embedder.embed_batch([])
    assert result == []
    mock_client.embed.assert_not_called()


def test_embed_batch_single_text_uses_list_input(config):
    """embed_batch with 1 text must call _client.embed with input=[text], not a bare string."""
    embedder, mock_client = _make_embedder_under_test(config)
    embedder.embed_batch(["hello world"])
    assert mock_client.embed.call_count == 1
    call_kwargs = mock_client.embed.call_args.kwargs
    assert isinstance(call_kwargs["input"], list), "input must be a list, not a bare string"
    assert call_kwargs["input"] == ["hello world"]


def test_embed_batch_sub_batches_by_embed_batch_size(config):
    """100 texts with embed_batch_size=64 must produce exactly 2 calls to _client.embed."""
    small_batch_config = ARGConfig(
        docs_root=config.docs_root,
        db_path=config.db_path,
        embed_batch_size=64,
    )
    embedder, mock_client = _make_embedder_under_test(small_batch_config)
    texts = [f"text_{i}" for i in range(100)]
    results = embedder.embed_batch(texts)
    assert mock_client.embed.call_count == 2  # ceil(100/64) = 2
    assert len(results) == 100
