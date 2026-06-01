"""Structural invariant tests.

These tests verify load-bearing constraints that are easy to break silently.
They are deliberately separate from the per-module test files so they are
immediately visible and are NEVER modified to accommodate implementation
changes — only to add new invariants.

See .claude/rules/ for the documented invariants each test covers.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from arg.config import ARGConfig
from arg.crawler.extractors import Document, _extract_links, _strip_invisible_and_boilerplate
from arg.generator import Generator, QueryProcessor
from arg.graph import KnowledgeGraph
from arg.indexer import Indexer
from arg.indexer.chunker import chunk_document
from arg.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Shared fakes (minimal — just enough to build components)
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
    def __init__(self, responses: dict[str, str] | None = None, default: str = "ANSWER") -> None:
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
    )


@pytest.fixture
def kg(config: ARGConfig):
    g = KnowledgeGraph(config.kuzu_path("default"))
    yield g
    g.close()


def _make_doc(docs_dir: Path, name: str, content: str, *, title: str | None = None) -> Document:
    p = docs_dir / name
    p.touch()
    return Document(
        path=p,
        content=content,
        metadata={
            "title": title or name,
            "page_description": f"Description for {name}",
            "file_type": "html",
            "links_to": [],
            "code_blocks": [],
        },
    )


def _build_retriever(config: ARGConfig, kg: KnowledgeGraph) -> tuple[HybridRetriever, Indexer]:
    embedder = _TagEmbedder()
    indexer = Indexer(config=config, knowledge_graph=kg, embedder=embedder)
    docs = [
        _make_doc(config.docs_root, "alpha.html", "##H1## Alpha\nQUERY_A body.", title="Alpha"),
        _make_doc(config.docs_root, "beta.html", "##H1## Beta\nQUERY_B body.", title="Beta"),
    ]
    indexer.index(docs)
    retriever = HybridRetriever(
        config=config,
        knowledge_graph=kg,
        embedder=embedder,
        chroma_documents_collection=indexer._docs_coll,
        chroma_chunks_collection=indexer._chunks_coll,
        bm25_index_path=config.bm25_index_path("default"),
        cluster_cache_path=config.cluster_cache_path("default"),
    )
    return retriever, indexer


# ---------------------------------------------------------------------------
# Invariant 1 — _extract_links before _strip_invisible_and_boilerplate
# ---------------------------------------------------------------------------


def test_extract_links_called_before_strip(config: ARGConfig) -> None:
    """_extract_links must see <nav> elements; stripping first silently drops those links.

    This test proves the ordering matters by demonstrating the data loss that
    occurs when the order is reversed.
    """
    html = (
        "<html><head><title>T</title></head><body>"
        "<nav><a href='other.html'>Other</a></nav>"
        "<main><p>Body text.</p></main>"
        "</body></html>"
    )
    # Correct order: links collected before stripping
    soup_correct = BeautifulSoup(html, features="lxml")
    links_before_strip = _extract_links(soup_correct)
    _strip_invisible_and_boilerplate(soup_correct, config)
    assert "other.html" in links_before_strip, "link should be found before stripping"

    # Wrong order: strip first, then collect — link is silently lost
    soup_wrong = BeautifulSoup(html, features="lxml")
    _strip_invisible_and_boilerplate(soup_wrong, config)
    links_after_strip = _extract_links(soup_wrong)
    assert "other.html" not in links_after_strip, (
        "link should be lost when strip runs first — this proves ordering is load-bearing"
    )


# ---------------------------------------------------------------------------
# Invariant 2 — position counter is global across sections
# ---------------------------------------------------------------------------


def test_chunk_position_is_global_not_per_section(tmp_path: Path) -> None:
    """Chunk positions must be strictly monotone across section boundaries.

    A reset to 0 at each section would produce duplicate position values and
    break Kuzu queries that sort or filter by position.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    # Use a tiny chunk_size so each section produces multiple chunks.
    small_config = ARGConfig(
        docs_root=docs,
        db_path=tmp_path / "db",
        chunk_size=50,
        chunk_overlap=0,
    )
    long_body = " ".join(f"word{i}" for i in range(200))
    content = f"##H1## Section One\n{long_body}\n##H2## Section Two\n{long_body}\n"
    p = tmp_path / "doc.html"
    p.touch()
    doc = Document(
        path=p,
        content=content,
        metadata={
            "title": "Test",
            "page_description": "",
            "file_type": "html",
            "links_to": [],
            "code_blocks": [],
        },
    )
    chunks = chunk_document(doc, small_config)
    assert len(chunks) >= 3, "need multiple chunks across at least two sections"

    positions = [cs.metadata["position"] for cs in chunks]
    # Positions must be strictly increasing — no resets, no duplicates.
    for i in range(1, len(positions)):
        assert positions[i] > positions[i - 1], (
            f"position[{i}]={positions[i]} not > position[{i - 1}]={positions[i - 1]}: "
            "position counter was reset between sections"
        )


# ---------------------------------------------------------------------------
# Invariant 3 — pipeline.index() schedules cluster recompute
# ---------------------------------------------------------------------------


def test_index_schedules_cluster_recompute(tmp_path: Path) -> None:
    """After index() + close(), the cluster cache file must exist."""
    from arg.pipeline import ARGPipeline

    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.html", "b.html"):
        (docs / name).write_text(
            f"<html><head><title>{name}</title></head><body><p>{name} body.</p></body></html>",
            encoding="utf-8",
        )
    config = ARGConfig(docs_root=docs, db_path=tmp_path / "db", watch_enabled=False)
    pipeline = ARGPipeline(
        config=config,
        corpus_name="default",
        llm=_ScriptedLLM(),
        embedder=_TagEmbedder(),
        skip_health_check=True,
        skip_signal_handlers=True,
    )
    pipeline.index()
    pipeline.close()  # joins the cluster thread
    assert config.cluster_cache_path("default").is_file(), (
        "cluster cache must exist after index() + close(); "
        "_recompute_clusters_bg() was not called or did not complete"
    )


# ---------------------------------------------------------------------------
# Invariant 4 — BM25 index is NOT written by the retriever
# ---------------------------------------------------------------------------


def test_bm25_index_not_written_by_retriever(config: ARGConfig, kg: KnowledgeGraph) -> None:
    """Calling retriever.retrieve() must never recreate a deleted BM25 file.

    The BM25 index is write-only from the indexer. The retriever is read-only.
    If the file disappears, the retriever must degrade gracefully (empty BM25
    results) but must not write a new file.
    """
    retriever, _ = _build_retriever(config, kg)
    bm25_path = config.bm25_index_path("default")
    assert bm25_path.is_file(), "indexer should have written the BM25 file"

    bm25_path.unlink()
    assert not bm25_path.exists()

    # Reload the retriever to reflect the missing file (simulates a fresh start
    # against an index where the BM25 file was manually deleted).
    retriever.reload()
    retriever.retrieve("QUERY_A")

    assert not bm25_path.exists(), (
        "retriever.retrieve() must not write a BM25 file; "
        "only Indexer.index() is permitted to call BM25Index.build()/save()"
    )


# ---------------------------------------------------------------------------
# Invariant 5 — raw query (not rewritten) is passed to the LLM
# ---------------------------------------------------------------------------


def test_raw_query_not_rewritten_query_in_llm_prompt(config: ARGConfig, kg: KnowledgeGraph) -> None:
    """The generation prompt must contain the raw query, never the rewritten form."""
    raw = "what is the way to log in"
    rewritten = "authentication methods and session management"

    llm = _ScriptedLLM(
        responses={
            "Rewrite the following": rewritten,
            "Does the following question contain": rewritten,
        }
    )
    embedder = _TagEmbedder()
    indexer = Indexer(config=config, knowledge_graph=kg, embedder=embedder)
    indexer.index(
        [_make_doc(config.docs_root, "a.html", "##H1## Auth\nQUERY_A body text.", title="Auth")]
    )
    retriever = HybridRetriever(
        config=config,
        knowledge_graph=kg,
        embedder=embedder,
        chroma_documents_collection=indexer._docs_coll,
        chroma_chunks_collection=indexer._chunks_coll,
        bm25_index_path=config.bm25_index_path("default"),
        cluster_cache_path=config.cluster_cache_path("default"),
    )
    qp = QueryProcessor(config=config, llm=llm)
    gen = Generator(config=config, llm=llm, retriever=retriever, query_processor=qp)
    gen.generate(raw)

    answer_prompts = [p for p in llm.calls if "You are Archivist" in p]
    assert answer_prompts, "LLM must be called for answer generation"
    for prompt in answer_prompts:
        assert raw in prompt, "raw query must appear in the generation prompt"
        assert rewritten not in prompt, (
            "rewritten query must NOT appear in the generation prompt; "
            "rewrites are retrieval-only signals"
        )


# ---------------------------------------------------------------------------
# Invariant 6 — close() joins the cluster thread
# ---------------------------------------------------------------------------


def test_close_joins_cluster_thread(tmp_path: Path) -> None:
    """close() must join the cluster background thread so callers see clean state."""
    from arg.pipeline import ARGPipeline

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "page.html").write_text(
        "<html><head><title>Page</title></head><body><p>QUERY_A body.</p></body></html>",
        encoding="utf-8",
    )
    config = ARGConfig(docs_root=docs, db_path=tmp_path / "db", watch_enabled=False)
    pipeline = ARGPipeline(
        config=config,
        corpus_name="default",
        llm=_ScriptedLLM(),
        embedder=_TagEmbedder(),
        skip_health_check=True,
        skip_signal_handlers=True,
    )
    pipeline.index()

    start = time.monotonic()
    pipeline.close()
    elapsed = time.monotonic() - start

    # close() must return within the join timeout (5s) plus modest headroom.
    assert elapsed < 8.0, f"close() took {elapsed:.2f}s — cluster thread may not have joined"

    # The cluster thread must be finished after close() returns.
    t = pipeline._cluster_thread
    if t is not None:
        assert not t.is_alive(), "cluster thread still alive after close() returned"


# ---------------------------------------------------------------------------
# Invariant 7 — find_document is public on HybridRetriever
# ---------------------------------------------------------------------------


def test_find_document_is_public_on_retriever(config: ARGConfig, kg: KnowledgeGraph) -> None:
    """find_document must be a public method; renaming it to _find_document would
    break CorpusAnalyst and CorpusExplorer without a compile-time error."""
    retriever, _ = _build_retriever(config, kg)
    assert hasattr(retriever, "find_document"), (
        "HybridRetriever.find_document must be a public method"
    )
    assert not hasattr(retriever, "_find_document"), (
        "HybridRetriever must not have a _find_document method; "
        "CorpusAnalyst calls retriever.find_document (public)"
    )
    # Callable with the expected signature.
    results = retriever.find_document("QUERY_A", top_k=3)
    assert isinstance(results, list)
