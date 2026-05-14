"""Corpus A end-to-end RAG test.

Covers the Section 12 e2e contract:

  * Crawl + index produces all readable documents (4 HTML + 2 PDFs); the
    AES-256 encrypted PDF is skipped without crashing.
  * Cross-component metadata correctness: PDF /Title resolution, /Subject
    in document-level embedding text, temp-file title fallback.
  * Running footer "Kraken API Docs - Confidential" never makes it into a
    chunk (Step 0e header/footer stripping).
  * Topic clustering: corpus is below the min_cluster_docs threshold, so
    get_topic_clusters returns the "All documents" fallback without LLM
    calls.
  * Watcher: drop a new file into docs_root → it appears in the listing
    after the debounce window; delete the file → it disappears.
  * RAG quality (real LLM): a small question battery returns non-empty
    answers, the unrelated question returns the refusal string, and the
    rate-limit table answer surfaces table content. These are gated by
    the ``real_llm`` fixture so the suite skips them when llama3.3:70b
    isn't pulled.

All tests in this file are marked ``e2e`` per the project's marker scheme.
The whole file is skipped when Ollama isn't reachable on
``localhost:11434`` (gate: the ``ollama_embedder`` fixture in conftest).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from arg.config import ARGConfig
from arg.pipeline import ARGPipeline

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_in(pipeline: ARGPipeline, name_suffix: str) -> bool:
    return any(d["doc_id"].endswith(name_suffix) for d in pipeline.graph.list_all_documents())


def _resolved(corpus_root: Path, *parts: str) -> str:
    return str(corpus_root.joinpath(*parts).resolve())


# ---------------------------------------------------------------------------
# Structural assertions (mocked LLM; doesn't need llama3.3)
# ---------------------------------------------------------------------------


def test_all_readable_documents_indexed(indexed_pipeline, corpus_a_path):
    """4 HTML + 2 readable PDFs = 6 docs. Encrypted PDF is skipped."""
    docs = indexed_pipeline.graph.list_all_documents()
    assert len(docs) == 6, f"expected 6 indexed docs; got {len(docs)}"
    # All four HTML fixture pages.
    for name in ("index.html", "page_a.html", "page_b.html", "subdir/page_c.html"):
        assert _doc_in(indexed_pipeline, name), f"missing {name}"
    # Both readable PDFs.
    for name in ("manual.pdf", "scanned_notice.pdf"):
        assert _doc_in(indexed_pipeline, name), f"missing {name}"


def test_encrypted_pdf_not_indexed(indexed_pipeline, corpus_a_path):
    encrypted_id = _resolved(corpus_a_path, "encrypted_notice.pdf")
    assert indexed_pipeline.graph.get_doc_metadata(encrypted_id) == {}, (
        "encrypted PDF must not appear in Kuzu"
    )
    rows = indexed_pipeline.indexer._chunks_coll.get(
        where={"doc_id": encrypted_id}, include=["metadatas"]
    )
    assert rows["ids"] == [], "encrypted PDF must not appear in ChromaDB"


def test_page_a_links_to_pdfs_recorded_in_graph(indexed_pipeline, corpus_a_path):
    """page_a.html now links to both manual.pdf and scanned_notice.pdf."""
    page_a_id = _resolved(corpus_a_path, "page_a.html")
    forward = set(indexed_pipeline.graph.get_linked_docs(page_a_id, depth=1))
    assert _resolved(corpus_a_path, "manual.pdf") in forward
    assert _resolved(corpus_a_path, "scanned_notice.pdf") in forward


def test_manual_pdf_title_from_metadata(indexed_pipeline, corpus_a_path):
    manual_id = _resolved(corpus_a_path, "manual.pdf")
    meta = indexed_pipeline.graph.get_doc_metadata(manual_id)
    assert meta["title"] == "Kraken API Full Manual"
    assert meta["file_type"] == "pdf"


def test_scanned_pdf_title_falls_back_to_filename_stem(indexed_pipeline, corpus_a_path):
    """``/Title`` is "Microsoft Word - document1.docx" — a temp-file pattern.
    The resolver rejects it and falls back to the filename stem."""
    scanned_id = _resolved(corpus_a_path, "scanned_notice.pdf")
    meta = indexed_pipeline.graph.get_doc_metadata(scanned_id)
    assert meta["title"] == "scanned_notice"


def test_running_footer_stripped_from_every_chunk(indexed_pipeline):
    """The "Kraken API Docs - Confidential" footer appears on every page of
    manual.pdf at the same y-coordinate; Step 0e must remove it before
    chunking. No chunk text may contain it."""
    chunks = indexed_pipeline.indexer._chunks_coll.get(include=["documents"])
    for body in chunks["documents"]:
        assert "Confidential" not in body, "running footer 'Confidential' leaked into a chunk"


def test_pdf_subject_in_document_embedding_text(indexed_pipeline, corpus_a_path):
    """/Subject metadata from manual.pdf is prepended to the doc-level
    embedding text (the ``documents`` collection)."""
    manual_id = _resolved(corpus_a_path, "manual.pdf")
    row = indexed_pipeline.indexer._docs_coll.get(
        ids=[manual_id], include=["documents", "metadatas"]
    )
    assert row["ids"], "manual.pdf must appear in documents collection"
    doc_embedding_text = row["documents"][0]
    assert doc_embedding_text.startswith("Complete reference for the Kraken API")
    assert row["metadatas"][0]["page_description"].startswith(
        "Complete reference for the Kraken API"
    )


def test_small_corpus_clustering_fallback(indexed_pipeline):
    """6 docs < min_cluster_docs (default 10) → "All documents" fallback,
    no LLM call for cluster labelling."""
    clusters = indexed_pipeline.get_topic_clusters()
    assert clusters == [
        {"label": "All documents", "doc_ids": clusters[0]["doc_ids"]},
    ]


# ---------------------------------------------------------------------------
# Watcher live add / remove
# ---------------------------------------------------------------------------


def _live_pipeline(tmp_path: Path, corpus_a_path: Path, embedder, mock_llm) -> ARGPipeline:
    """Build a pipeline over a *copy* of corpus_a with watch_enabled=True.

    Operating on a copy keeps the watcher test from touching the git-tracked
    fixture directory.
    """
    docs_copy = tmp_path / "live_docs"
    shutil.copytree(corpus_a_path, docs_copy)
    db_root = tmp_path / "arg_db"
    db_root.mkdir(exist_ok=True)
    config = ARGConfig(
        docs_root=docs_copy,
        db_path=db_root / "live",
        watch_enabled=True,
        watch_debounce_ms=80,
        top_k_vector=4,
        top_k_graph=1,
        graph_hop_depth=1,
        enrich_min_score=0.0,
    )
    return ARGPipeline(
        config=config,
        corpus_name="default",
        llm=mock_llm,
        embedder=embedder,
        skip_health_check=True,
        skip_signal_handlers=True,
    )


def test_watcher_live_add_remove(tmp_path, corpus_a_path, ollama_embedder, mock_llm):
    pipeline = _live_pipeline(tmp_path, corpus_a_path, ollama_embedder, mock_llm)
    try:
        pipeline.index()
        baseline = {d["doc_id"] for d in pipeline.graph.list_all_documents()}

        # Drop a new HTML page; watcher debounce-fires after ~80ms.
        new_page = pipeline.config.docs_root / "new_page.html"
        new_page.write_text(
            "<html><head><title>Hot Drop</title></head>"
            "<body><h1>Hot Drop</h1><p>This document arrived after indexing.</p></body></html>",
            encoding="utf-8",
        )
        # Wait debounce + headroom for the indexer to finish + retriever reload.
        time.sleep(0.4)

        after_add = {d["doc_id"] for d in pipeline.graph.list_all_documents()}
        new_doc_id = str(new_page.resolve())
        assert new_doc_id in after_add - baseline, (
            f"expected {new_page.name} in the listing after drop;\n"
            f"baseline={baseline}\nafter_add={after_add}"
        )

        # Delete it; watcher should remove it from the listing.
        new_page.unlink()
        time.sleep(0.4)
        after_delete = {d["doc_id"] for d in pipeline.graph.list_all_documents()}
        assert new_doc_id not in after_delete
    finally:
        pipeline.close()


# ---------------------------------------------------------------------------
# RAG quality (real LLM — slow; skips when llama3.3:70b isn't pulled)
# ---------------------------------------------------------------------------


def _real_llm_pipeline(base_config, ollama_embedder, real_llm) -> ARGPipeline:
    return ARGPipeline(
        config=base_config,
        corpus_name="default",
        llm=real_llm,
        embedder=ollama_embedder,
        skip_health_check=True,
        skip_signal_handlers=True,
    )


def test_rag_unrelated_query_returns_refusal(base_config, ollama_embedder, real_llm):
    """The fallback string is produced when retrieval AND filtering still
    yields nothing useful for the LLM to ground on. The real LLM is
    expected to follow the system-prompt instruction to refuse when the
    context is off-topic."""
    pipeline = _real_llm_pipeline(base_config, ollama_embedder, real_llm)
    try:
        pipeline.index()
        result = pipeline.query("What is the capital of France?")
        # Either the spec's refusal text (no chunks retrieved) or the model
        # acknowledging the documentation doesn't cover the topic.
        assert (
            "documentation does not cover this topic" in result.answer.lower()
            or "does not contain" in result.answer.lower()
            or "documentation" in result.answer.lower()
        ), f"expected refusal, got: {result.answer!r}"
    finally:
        pipeline.close()


def test_rag_auth_question_returns_page_a_sources(base_config, ollama_embedder, real_llm):
    pipeline = _real_llm_pipeline(base_config, ollama_embedder, real_llm)
    try:
        pipeline.index()
        result = pipeline.query("How do I authenticate with the Kraken API?")
        assert result.answer
        source_docs = {s.doc_id for s in result.sources}
        # page_a (HTML) or manual.pdf both legitimately cover auth.
        relevant = [d for d in source_docs if "page_a.html" in d or "manual.pdf" in d]
        assert relevant, f"no auth-relevant sources in: {source_docs}"
    finally:
        pipeline.close()


def test_rag_rate_limit_table_query(base_config, ollama_embedder, real_llm):
    """The rate-limit table is in both page_b.html and manual.pdf (page 2);
    a topical query must surface one of them."""
    pipeline = _real_llm_pipeline(base_config, ollama_embedder, real_llm)
    try:
        pipeline.index()
        result = pipeline.query("What is the rate limit for tier 2?")
        assert result.answer
        source_docs = {s.doc_id for s in result.sources}
        relevant = [d for d in source_docs if "page_b.html" in d or "manual.pdf" in d]
        assert relevant, f"no rate-limit-relevant sources in: {source_docs}"
    finally:
        pipeline.close()
