"""Indexer tests — covers every Section 7 ``test_indexer.py`` test point.

Embeddings are produced by a deterministic fake :class:`_FakeEmbedder` so the
suite stays offline (no Ollama). The fake hashes input text into a 16-dim
vector — collisions don't matter for these tests; what matters is that
ChromaDB upserts succeed and the vectors round-trip.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from arg.config import ARGConfig
from arg.crawler.extractors import Document
from arg.graph import KnowledgeGraph
from arg.indexer import Indexer
from arg.retriever.bm25_index import BM25Index

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_VEC_DIM = 16


class _FakeEmbedder:
    """Deterministic 16-dim hash-based embedder. No network, no Ollama."""

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # 2 bytes per dimension; normalise into [-1, 1].
        return [
            (int.from_bytes(h[i * 2 : (i + 1) * 2], "big") / 65535.0) * 2 - 1
            for i in range(_VEC_DIM)
        ]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def config(tmp_path: Path) -> ARGConfig:
    docs = tmp_path / "docs"
    docs.mkdir()
    return ARGConfig(docs_root=docs, db_path=tmp_path / "arg_db")


@pytest.fixture
def kg(config: ARGConfig):
    g = KnowledgeGraph(config.kuzu_path("default"))
    yield g
    g.close()


@pytest.fixture
def indexer(config: ARGConfig, kg: KnowledgeGraph) -> Indexer:
    return Indexer(config=config, knowledge_graph=kg, embedder=_FakeEmbedder())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    docs_dir: Path,
    name: str,
    *,
    content: str,
    title: str | None = None,
    page_description: str = "",
    links_to: list[str] | None = None,
    file_type: str = "html",
) -> Document:
    p = docs_dir / name
    p.touch()
    metadata: dict[str, Any] = {
        "title": title or name,
        "page_description": page_description,
        "file_type": file_type,
        "links_to": list(links_to or []),
        "code_blocks": [],
    }
    return Document(path=p, content=content, metadata=metadata)


def _simple_doc(docs_dir: Path, name: str, body_paragraphs: int = 1) -> Document:
    content = f"##H1## {name} Heading\n" + "\n".join(
        f"Paragraph {i} body text here." for i in range(body_paragraphs)
    )
    return _make_doc(
        docs_dir,
        name,
        content=content,
        title=name.removesuffix(".html"),
        page_description=f"Description for {name}",
    )


# ---------------------------------------------------------------------------
# Collection counts
# ---------------------------------------------------------------------------


def test_after_index_chunks_collection_has_correct_count(indexer, config):
    docs = [
        _simple_doc(config.docs_root, "a.html", body_paragraphs=2),
        _simple_doc(config.docs_root, "b.html", body_paragraphs=2),
    ]
    stats = indexer.index(docs)
    assert stats.documents_indexed == 2
    assert stats.chunks_written == indexer._chunks_coll.count()


def test_after_index_documents_collection_has_one_entry_per_doc(indexer, config):
    docs = [
        _simple_doc(config.docs_root, "a.html"),
        _simple_doc(config.docs_root, "b.html"),
        _simple_doc(config.docs_root, "c.html"),
    ]
    indexer.index(docs)
    assert indexer._docs_coll.count() == 3


# ---------------------------------------------------------------------------
# Metadata shape
# ---------------------------------------------------------------------------


def test_doc_embedding_text_is_page_description_plus_body(indexer, config):
    doc = _make_doc(
        config.docs_root,
        "a.html",
        content="##H1## Heading\n" + ("body text. " * 50),
        title="Doc A",
        page_description="API authentication overview.",
    )
    indexer.index([doc])
    got = indexer._docs_coll.get(ids=[str(doc.path.resolve())], include=["documents"])
    assert got["documents"][0].startswith("API authentication overview.")


def test_chunk_embedding_text_stored_separately_from_chunk_text(indexer, config):
    doc = _make_doc(
        config.docs_root,
        "a.html",
        content=(
            "##H1## Auth\nSection about authentication.\n##H2## OAuth\nThe flow exchanges tokens.\n"
        ),
        title="API Docs",
        page_description="API auth overview",
    )
    indexer.index([doc])
    got = indexer._chunks_coll.get(include=["documents", "metadatas"])
    raw_texts = got["documents"]
    metadatas = got["metadatas"]
    assert raw_texts
    # Every chunk's embedding_text must be prefixed; the prefix matches its
    # own heading_path, not necessarily the deepest one in the doc.
    for raw, meta in zip(raw_texts, metadatas, strict=True):
        prefix = f"{meta['heading_path']}: "
        assert meta["embedding_text"].startswith(prefix)
        assert prefix not in raw  # raw stays clean
        assert meta["embedding_text"] != raw
    # And at least one chunk reached the OAuth section.
    assert any(m["heading_path"] == "API Docs > Auth > OAuth" for m in metadatas)


def test_chunk_metadata_carries_spec_fields(indexer, config):
    doc = _make_doc(
        config.docs_root,
        "a.html",
        content=("##H1## Tables\nIntro line.\n| A | B |\n|---|---|\n| 1 | 2 |\n"),
        title="Doc A",
        page_description="some description",
    )
    indexer.index([doc])
    got = indexer._chunks_coll.get(include=["metadatas"])
    metas = got["metadatas"]
    assert metas
    for m in metas:
        for required in ("doc_id", "title", "heading_path", "position", "file_type"):
            assert required in m, f"missing chunk metadata key: {required}"
    assert any(m["has_table"] for m in metas)


def test_doc_metadata_carries_spec_fields(indexer, config):
    doc = _make_doc(
        config.docs_root,
        "a.html",
        content="##H1## H\nbody.",
        title="My Title",
        page_description="A description.",
    )
    indexer.index([doc])
    got = indexer._docs_coll.get(ids=[str(doc.path.resolve())], include=["metadatas"])
    m = got["metadatas"][0]
    for required in ("doc_id", "title", "file_type", "page_description"):
        assert required in m
    assert m["title"] == "My Title"
    assert m["page_description"] == "A description."


# ---------------------------------------------------------------------------
# Incremental hashing
# ---------------------------------------------------------------------------


def test_reindexing_unchanged_doc_is_a_noop(indexer, config):
    doc = _simple_doc(config.docs_root, "a.html", body_paragraphs=3)
    indexer.index([doc])
    second = indexer.index([doc])
    assert second.documents_indexed == 0
    assert second.documents_skipped == 1
    assert second.chunks_written == 0


def test_hashes_persisted_incrementally_after_each_doc(indexer, config):
    """Mid-run interruption recovery: hashes must hit disk per doc, not
    just at the end of index(). A Ctrl-C / kill after doc N can then be
    resumed by re-running index() — the first N docs are skipped."""
    docs = [
        _simple_doc(config.docs_root, "a.html"),
        _simple_doc(config.docs_root, "b.html"),
        _simple_doc(config.docs_root, "c.html"),
    ]
    # Capture the hash file state during the run by spying on _save_hashes.
    save_counts: list[int] = []
    original_save = indexer._save_hashes

    def spy_save() -> None:
        save_counts.append(len(indexer._hashes))
        original_save()

    indexer._save_hashes = spy_save  # type: ignore[method-assign, unused-ignore]
    indexer.index(docs)
    # Hashes should be saved at least once per doc (3) plus a final save at
    # the end of index() — so >= 4 calls, each carrying a non-decreasing
    # hash count.
    assert len(save_counts) >= len(docs) + 1
    assert save_counts == sorted(save_counts), "hash count must be monotonically non-decreasing"
    assert save_counts[-1] == len(docs)


def test_index_processes_documents_streamed(indexer, config):
    """Streaming contract: each doc's hash is on disk before the next doc is yielded.

    Passing a generator (not a list) confirms the indexer does not
    materialise the input via list() before processing.
    """
    import json

    docs = [
        _simple_doc(config.docs_root, "a.html"),
        _simple_doc(config.docs_root, "b.html"),
        _simple_doc(config.docs_root, "c.html"),
    ]
    hash_path = config.corpus_root("default") / "doc_hashes.json"

    def streaming_docs():
        prev_id: str | None = None
        for doc in docs:
            if prev_id is not None:
                assert hash_path.is_file(), "hash file must exist after first doc"
                hashes = json.loads(hash_path.read_text())
                assert prev_id in hashes, f"hash for {prev_id} not saved before next doc yielded"
            yield doc
            prev_id = str(doc.path.resolve())

    stats = indexer.index(streaming_docs())
    assert stats.documents_indexed == len(docs)


def test_simulated_interruption_resumes_cleanly(indexer, config, monkeypatch):
    """If the indexer dies after doc 1, a fresh indexer over the same
    arg_db skips that doc on the next run."""
    docs = [
        _simple_doc(config.docs_root, "a.html"),
        _simple_doc(config.docs_root, "b.html"),
    ]
    # Wire indexer._index_one to crash on the second doc.
    real_index_one = indexer._index_one
    seen: list[str] = []

    def crashy_index_one(doc):
        if "b.html" in str(doc.path):
            raise RuntimeError("simulated crash mid-run")
        seen.append(str(doc.path))
        return real_index_one(doc)

    monkeypatch.setattr(indexer, "_index_one", crashy_index_one)
    with pytest.raises(RuntimeError, match="simulated crash"):
        indexer.index(docs)
    # The first doc was processed and its hash persisted.
    assert len(seen) == 1
    monkeypatch.undo()

    # Fresh indexer over the same on-disk state: a.html is skipped, b.html
    # gets indexed.
    fresh = Indexer(config=config, knowledge_graph=indexer.kg, embedder=_FakeEmbedder())
    stats = fresh.index(docs)
    assert stats.documents_skipped == 1
    assert stats.documents_indexed == 1


def test_reindexing_changed_doc_updates_both_collections(indexer, config):
    doc = _make_doc(
        config.docs_root,
        "a.html",
        content="##H1## H\nFirst version body content.",
        title="A",
        page_description="initial",
    )
    indexer.index([doc])
    chunks_before = indexer._chunks_coll.count()

    # Edit the content.
    doc.content = "##H1## H\nSecond version body content totally different."
    doc.metadata["page_description"] = "updated"
    stats = indexer.index([doc])
    assert stats.documents_indexed == 1

    # Old chunks gone, new chunks present.
    got = indexer._chunks_coll.get(include=["documents"])
    assert any("Second version body content" in d for d in got["documents"])
    assert not any("First version body content" in d for d in got["documents"])

    # documents collection metadata refreshed.
    got_doc = indexer._docs_coll.get(ids=[str(doc.path.resolve())], include=["metadatas"])
    assert got_doc["metadatas"][0]["page_description"] == "updated"
    # Total chunks won't necessarily change, but the doc count stays 1.
    assert indexer._docs_coll.count() == 1
    _ = chunks_before  # silence ruff unused-var if collapsed by formatter


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def test_remove_document_deletes_from_both_collections(indexer, config):
    a = _simple_doc(config.docs_root, "a.html")
    b = _simple_doc(config.docs_root, "b.html")
    indexer.index([a, b])

    a_id = str(a.path.resolve())
    indexer.remove_document(a_id)

    # a is gone from both collections.
    assert indexer._docs_coll.get(ids=[a_id], include=[])["ids"] == []
    chunks_after = indexer._chunks_coll.get(include=["metadatas"])
    remaining_doc_ids = {m["doc_id"] for m in chunks_after["metadatas"]}
    assert a_id not in remaining_doc_ids
    # b survives.
    b_id = str(b.path.resolve())
    assert indexer._docs_coll.get(ids=[b_id], include=[])["ids"] == [b_id]


def test_update_document_replaces_index_footprint(indexer, config):
    doc = _make_doc(
        config.docs_root,
        "a.html",
        content="##H1## H\noriginal text in body.",
        title="A",
    )
    indexer.index([doc])
    doc.content = "##H1## H\ncompletely fresh body content."
    indexer.update_document(doc)
    got = indexer._chunks_coll.get(include=["documents"])
    assert any("completely fresh body content" in d for d in got["documents"])
    assert not any("original text in body" in d for d in got["documents"])


# ---------------------------------------------------------------------------
# BM25 index written by indexer
# ---------------------------------------------------------------------------


def test_bm25_index_written_after_index(indexer, config):
    # BM25Okapi's IDF is non-positive when a term appears in >= N/2 of N docs,
    # so a 2-doc corpus often returns zero-score everywhere. Use 4 docs to
    # keep the topical-query check meaningful.
    docs = [
        _make_doc(
            config.docs_root,
            "auth.html",
            content="##H1## H\nauthentication and authorisation tokens flow",
            title="A",
        ),
        _make_doc(
            config.docs_root,
            "db.html",
            content="##H1## H\ndatabase migrations and schema design",
            title="B",
        ),
        _make_doc(
            config.docs_root,
            "net.html",
            content="##H1## H\nnetwork topology firewall rules dns",
            title="C",
        ),
        _make_doc(
            config.docs_root,
            "ui.html",
            content="##H1## H\nuser interface form layout css",
            title="D",
        ),
    ]
    indexer.index(docs)
    bm25_path = config.bm25_index_path("default")
    assert bm25_path.is_file(), "indexer must write bm25_index.pkl"
    assert bm25_path.stat().st_size > 0

    idx = BM25Index.load(bm25_path)
    assert not idx.is_empty
    hits = idx.query("authentication tokens", top_k=5)
    assert hits, "BM25 should retrieve a chunk on a topical query"
    top_chunk_id = hits[0][0]
    assert "auth.html" in top_chunk_id


def test_bm25_index_updates_on_remove(indexer, config):
    a = _make_doc(
        config.docs_root,
        "a.html",
        content="##H1## H\nunique-marker-token-zzz appears here.",
        title="A",
    )
    b = _simple_doc(config.docs_root, "b.html")
    indexer.index([a, b])

    indexer.remove_document(str(a.path.resolve()))
    idx = BM25Index.load(config.bm25_index_path("default"))
    hits = idx.query("unique-marker-token-zzz", top_k=5)
    # All chunks containing this token were removed.
    assert hits == [] or all("a.html" not in cid for cid, _ in hits)


# ---------------------------------------------------------------------------
# Links recorded by indexer (two-pass)
# ---------------------------------------------------------------------------


def test_index_records_links_to_indexed_targets(indexer, config, kg):
    a = _make_doc(
        config.docs_root,
        "a.html",
        content="##H1## A\nbody",
        title="A",
    )
    b = _make_doc(
        config.docs_root,
        "b.html",
        content="##H1## B\nbody",
        title="B",
    )
    # Have to populate links_to BEFORE indexing.
    a.metadata["links_to"] = [str(b.path.resolve())]
    stats = indexer.index([a, b])
    assert stats.links_recorded == 1
    # Verify in the graph.
    linked = kg.get_linked_docs(str(a.path.resolve()), depth=1)
    assert linked == [str(b.path.resolve())]


def test_index_ignores_links_to_unindexed_targets(indexer, config, kg):
    a = _make_doc(
        config.docs_root,
        "a.html",
        content="##H1## A\nbody",
        title="A",
        links_to=["/some/external/path.html"],  # not in this index run
    )
    stats = indexer.index([a])
    assert stats.links_recorded == 0


# ---------------------------------------------------------------------------
# 0-chunk warnings
# ---------------------------------------------------------------------------


def test_zero_chunk_doc_logs_warning(indexer, config, caplog):
    import logging

    doc = _make_doc(config.docs_root, "empty.html", content="", title="Empty Doc")
    with caplog.at_level(logging.WARNING, logger="arg.indexer.indexer"):
        indexer._index_one(doc)
    messages = [r.getMessage() for r in caplog.records]
    assert any("0 chunks" in m and "empty.html" in m for m in messages)


def test_index_summary_zero_chunk_warning(indexer, config, caplog):
    import logging

    good = _make_doc(
        config.docs_root,
        "good.html",
        content="##H1## Good\nThis document has real content.",
        title="Good Doc",
    )
    empty = _make_doc(config.docs_root, "empty.html", content="", title="Empty Doc")
    with caplog.at_level(logging.WARNING, logger="arg.indexer.indexer"):
        indexer.index([good, empty])
    messages = [r.getMessage() for r in caplog.records]
    assert any("index complete" in m and "1 doc(s)" in m for m in messages)
