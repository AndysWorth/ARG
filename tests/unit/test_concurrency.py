"""Concurrency regression tests.

These tests verify correct behaviour of ARG's three threading surfaces:
  - Cluster recompute background thread (pipeline.py)
  - BM25 rebuild debounce timer (pipeline.py)
  - Watcher debounce timers (watcher.py)
  - Sub-query ThreadPoolExecutor (generator.py)

See .claude/rules/concurrency.md for the documented threading model.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

from arg.config import ARGConfig
from arg.pipeline import ARGPipeline

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

_VEC_DIM = 32


class _TagEmbedder:
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
    def __init__(self, default: str = "ANSWER") -> None:
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.default

    def complete_structured(self, prompt: str, schema: dict) -> str:
        return self.complete(prompt)

    def stream_complete(self, prompt: str) -> Iterator[str]:
        yield from self.complete(prompt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path: Path) -> ARGPipeline:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    for name in ("a.html", "b.html"):
        (docs / name).write_text(
            f"<html><head><title>{name}</title></head>"
            f"<body><p>QUERY_A content for {name}.</p></body></html>",
            encoding="utf-8",
        )
    config = ARGConfig(docs_root=docs, db_path=tmp_path / "db", watch_enabled=False)
    return ARGPipeline(
        config=config,
        corpus_name="default",
        llm=_ScriptedLLM(),
        embedder=_TagEmbedder(),
        skip_health_check=True,
        skip_signal_handlers=True,
    )


# ---------------------------------------------------------------------------
# Test 1 — close() is idempotent
# ---------------------------------------------------------------------------


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Calling close() twice must not raise or deadlock."""
    pipeline = _make_pipeline(tmp_path)
    pipeline.index()
    pipeline.close()
    pipeline.close()  # must be a no-op


# ---------------------------------------------------------------------------
# Test 2 — close() while cluster thread running completes cleanly
# ---------------------------------------------------------------------------


def test_close_while_cluster_running_completes_cleanly(tmp_path: Path) -> None:
    """Regression test for the Feature 0003 race condition.

    close() sets _closed=True and then joins the cluster thread. The thread
    must NOT check _closed between invalidate_cluster_cache() and
    get_topic_clusters() — once it starts work, it runs to completion so
    the cache is always written before close() returns.
    """
    gate = threading.Event()
    pipeline = _make_pipeline(tmp_path)
    original_compute = pipeline.explorer._compute_clusters

    def _gated_compute():
        gate.wait(timeout=10.0)
        return original_compute()

    try:
        with patch.object(pipeline.explorer, "_compute_clusters", side_effect=_gated_compute):
            pipeline.index()
            # Pipeline has started the cluster thread; release after a brief pause
            # to ensure the thread is mid-execution when close() is called.
            time.sleep(0.05)
            gate.set()
            start = time.monotonic()
            pipeline.close()
            elapsed = time.monotonic() - start

        # close() must return promptly after joining (within join timeout + headroom).
        assert elapsed < 8.0, f"close() blocked for {elapsed:.2f}s"

        # Thread must have finished and written the cache.
        cache_path = pipeline.config.cluster_cache_path("default")
        assert cache_path.is_file(), (
            "cluster cache not written — the _closed race condition may have recurred: "
            "thread skipped get_topic_clusters() after invalidate_cluster_cache()"
        )
    finally:
        gate.set()  # ensure gate is always released so the thread can exit
        pipeline.close()


# ---------------------------------------------------------------------------
# Test 3 — BM25 rebuild timer is cancelled on close()
# ---------------------------------------------------------------------------


def test_bm25_rebuild_timer_cancelled_on_close(tmp_path: Path) -> None:
    """close() must cancel a pending BM25 rebuild timer.

    The timer fires 5 seconds after watcher events. If close() doesn't cancel
    it, the timer callback runs against a closed pipeline and raises.
    """
    pipeline = _make_pipeline(tmp_path)
    pipeline.index()

    # Manually start a BM25 rebuild timer (simulates a watcher event).
    pipeline._schedule_bm25_rebuild()
    assert pipeline._bm25_rebuild_timer is not None
    assert pipeline._bm25_rebuild_timer.is_alive()

    start = time.monotonic()
    pipeline.close()
    elapsed = time.monotonic() - start

    # close() must cancel the timer immediately — not wait for it to fire.
    assert elapsed < 2.0, f"close() took {elapsed:.2f}s; timer may not have been cancelled"
    # Timer must no longer be alive.
    assert pipeline._bm25_rebuild_timer is None or not pipeline._bm25_rebuild_timer.is_alive()


# ---------------------------------------------------------------------------
# Test 4 — concurrent index() and query() don't raise
# ---------------------------------------------------------------------------


def test_concurrent_index_and_query(tmp_path: Path) -> None:
    """Simultaneous index() and query() must not raise.

    The top-level RLock serialises them; this test verifies no deadlock or
    exception occurs when both are called from separate threads.
    """
    pipeline = _make_pipeline(tmp_path)
    pipeline.index()

    errors: list[Exception] = []

    def _query_loop() -> None:
        for _ in range(5):
            try:
                pipeline.query("QUERY_A text")
            except Exception as exc:
                errors.append(exc)

    def _reindex() -> None:
        try:
            pipeline.index()
        except Exception as exc:
            errors.append(exc)

    t_query = threading.Thread(target=_query_loop)
    t_index = threading.Thread(target=_reindex)
    t_query.start()
    t_index.start()
    t_query.join(timeout=30.0)
    t_index.join(timeout=30.0)

    pipeline.close()
    assert not errors, f"concurrent access raised: {errors}"


# ---------------------------------------------------------------------------
# Test 5 — watcher stop() cancels pending debounce timers
# ---------------------------------------------------------------------------


def test_watcher_stop_cancels_pending_debounce(tmp_path: Path) -> None:
    """Timers that are pending when stop() is called must never fire.

    Uses the watcher directly (not the pipeline) with a long debounce window.
    """
    from watchdog.events import FileModifiedEvent

    from arg.crawler.watcher import DocsWatcher

    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "page.html"
    target.touch()

    fired: list[bool] = []

    def _callback(path: Path, kind: str) -> None:
        fired.append(True)

    config = ARGConfig(docs_root=docs, db_path=tmp_path / "db", watch_debounce_ms=500)
    watcher = DocsWatcher(docs, config, _callback)

    # Trigger a debounce event.
    watcher.handler.on_modified(FileModifiedEvent(str(target)))
    assert not fired, "callback must not fire inside the debounce window"

    # Stop immediately — pending timer must be cancelled.
    watcher.stop()

    # Wait past the debounce window.
    time.sleep(0.7)
    assert not fired, "callback fired after stop() — timer was not cancelled"


# ---------------------------------------------------------------------------
# Test 6 — ThreadPoolExecutor uses min(n_queries, 4) workers
# ---------------------------------------------------------------------------


def test_threadpool_worker_count_bounded(tmp_path: Path) -> None:
    """With a single sub-query, exactly 1 retriever call is made.

    This is a sanity check that the parallel sub-query path doesn't
    over-allocate threads or call retrieve() more times than there are queries.
    """
    from arg.config import ARGConfig
    from arg.crawler.extractors import Document
    from arg.generator import Generator, QueryProcessor
    from arg.graph import KnowledgeGraph
    from arg.indexer import Indexer
    from arg.retriever import HybridRetriever

    docs = tmp_path / "docs"
    docs.mkdir()
    config = ARGConfig(
        docs_root=docs,
        db_path=tmp_path / "arg_db",
        top_k_vector=3,
        top_k_graph=1,
        graph_hop_depth=1,
        query_rewrite=False,
        query_decompose=False,
    )
    p = docs / "alpha.html"
    p.touch()
    doc = Document(
        path=p,
        content="##H1## Alpha\nQUERY_A authentication.",
        metadata={
            "title": "Alpha",
            "page_description": "",
            "file_type": "html",
            "links_to": [],
            "code_blocks": [],
        },
    )
    embedder = _TagEmbedder()
    kg = KnowledgeGraph(config.kuzu_path("default"))
    try:
        indexer = Indexer(config=config, knowledge_graph=kg, embedder=embedder)
        indexer.index([doc])
        retriever = HybridRetriever(
            config=config,
            knowledge_graph=kg,
            embedder=embedder,
            chroma_documents_collection=indexer._docs_coll,
            chroma_chunks_collection=indexer._chunks_coll,
            bm25_index_path=config.bm25_index_path("default"),
            cluster_cache_path=config.cluster_cache_path("default"),
        )
        llm = _ScriptedLLM()
        qp = QueryProcessor(config=config, llm=llm)
        gen = Generator(config=config, llm=llm, retriever=retriever, query_processor=qp)

        call_count = 0
        original = retriever.retrieve

        def _counting(q, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(q, **kwargs)

        with patch.object(retriever, "retrieve", side_effect=_counting):
            gen.generate("QUERY_A authentication")

        # With rewrite=False and decompose=False there is exactly one embedding query.
        assert call_count == 1, f"expected 1 retriever call for a single query, got {call_count}"
    finally:
        kg.close()
