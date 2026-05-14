"""Integration: crawler → indexer (Chroma documents + chunks + Kuzu + BM25).

Uses the real Ollama embedder so on-disk shape matches production exactly.
Skips when Ollama isn't reachable.
"""

from __future__ import annotations

import pytest

from arg.crawler.crawler import crawl
from arg.graph import KnowledgeGraph
from arg.indexer import Indexer

pytestmark = pytest.mark.integration


def test_indexer_produces_consistent_collections_and_files(
    base_config,
    ollama_embedder,
):
    """End-to-end indexer run: chunks + documents + BM25 + KG all in sync."""
    kg = KnowledgeGraph(base_config.kuzu_path("default"))
    try:
        indexer = Indexer(
            config=base_config,
            knowledge_graph=kg,
            embedder=ollama_embedder,
        )
        documents = list(crawl(base_config.docs_root, base_config))
        stats = indexer.index(documents)

        # 4 HTML + 2 readable PDFs (manual + scanned). encrypted_notice.pdf
        # is found by the crawler but extract_pdf_to_document returns None
        # for it, so it never reaches the indexer.
        assert stats.documents_indexed == 6
        assert stats.chunks_written >= 6  # each doc yields at least one chunk

        # documents collection: one row per source file
        assert indexer._docs_coll.count() == 6

        # chunks collection: matches `chunks_written`
        assert indexer._chunks_coll.count() == stats.chunks_written

        # Each chunk's doc_id metadata matches a graph node.
        chunks = indexer._chunks_coll.get(include=["metadatas"])
        graph_doc_ids = {d["doc_id"] for d in kg.list_all_documents()}
        for meta in chunks["metadatas"]:
            assert meta["doc_id"] in graph_doc_ids

        # Graph `chunk_count` matches ChromaDB count per document.
        for d in kg.list_all_documents():
            actual = sum(1 for m in chunks["metadatas"] if m["doc_id"] == d["doc_id"])
            assert d["chunk_count"] == actual, (
                f"{d['doc_id']!r}: graph says {d['chunk_count']} chunks, Chroma stored {actual}"
            )

        # BM25 index file exists.
        assert base_config.bm25_index_path("default").is_file()

        # Contextual enrichment: embedding_text carries the heading prefix.
        for meta in chunks["metadatas"]:
            embedding_text = meta.get("embedding_text", "")
            heading_path = meta.get("heading_path", "")
            assert heading_path, "every chunk must have a heading_path"
            assert embedding_text.startswith(f"{heading_path}: "), (
                f"chunk {meta.get('doc_id')}: embedding_text does not start "
                f"with the contextual prefix"
            )
    finally:
        kg.close()
