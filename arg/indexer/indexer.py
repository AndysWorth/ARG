"""Indexer — orchestrates chunking, embedding, ChromaDB writes, Kuzu writes,
and BM25 index construction.

Two ChromaDB collections live side-by-side in ``config.chroma_path(corpus)``:

  ``documents``
      One row per source file. Embedding text =
      ``page_description + first 512 tokens of body``. Used for doc-level
      similarity (corpus_search, DCI enrichment Stage 0).

  ``chunks``
      One row per `ChunkedSection`. Embedding text is the contextually
      enriched form (``"{title} > {heading_path}: {chunk_text}"``);
      ``chunk_text`` is the raw LLM-facing form. Used by the retriever for
      dense vector search.

Both collections are kept in sync — every add/update/remove touches both,
and the Kuzu graph, in one logical step.

BM25 index — ownership note
---------------------------
The spec is explicit: **the BM25 index is built by the indexer at the end of
``index()``, not lazily by the retriever.** This module writes
``bm25_index.pkl`` to ``config.bm25_index_path(corpus)`` after every full
indexing pass; the retriever (Section 8) only ever loads.

Incremental re-indexing
-----------------------
``_load_hashes`` reads ``doc_hashes.json`` (per-corpus). For each Document,
``_is_unchanged`` compares the current SHA-256 of ``content + title`` against
the previously recorded hash. Unchanged documents are skipped end-to-end
(no embeddings, no Kuzu writes, no Chroma upserts). Changed documents have
their chunks fully removed and re-inserted so stale chunks never linger.

Locality
--------
ChromaDB is instantiated with ``anonymized_telemetry=False``; Ollama
embedding goes through the injected :class:`Embedder` (production uses an
Ollama-backed implementation pointed at ``config.ollama_base_url``, which the
config validator already restricts to localhost). The hash file and the BM25
pickle are local files.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import chromadb
import tiktoken

from arg.config import ARGConfig
from arg.crawler.extractors import Document
from arg.embeddings import Embedder
from arg.graph import KnowledgeGraph
from arg.indexer.chunker import chunk_document
from arg.retriever.bm25_index import BM25Index

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")
_DOC_EMBED_TOKEN_BUDGET = 512


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


@dataclass
class IndexStats:
    """Counts returned from :meth:`Indexer.index`."""

    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_written: int = 0
    links_recorded: int = 0


class Indexer:
    """Sync three stores — Chroma (documents+chunks), Kuzu, BM25 — in one place."""

    def __init__(
        self,
        config: ARGConfig,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        corpus_name: str = "default",
    ) -> None:
        self.config = config
        self.kg = knowledge_graph
        self.embedder = embedder
        self.corpus_name = corpus_name

        chroma_path = config.chroma_path(corpus_name)
        chroma_path.mkdir(parents=True, exist_ok=True)
        # NEVER instantiate ChromaDB without ``anonymized_telemetry=False`` —
        # this is the only line that wires the spec's "ChromaDB telemetry off"
        # rule into runtime behaviour.
        self._chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        # Disable Chroma's automatic embedding function — we provide vectors
        # directly through the injected Embedder. _NoEmbeddingFunction raises
        # if Chroma ever tries to invoke it, which it shouldn't since every
        # upsert here passes ``embeddings=...`` explicitly. The cast is
        # needed because Chroma's EmbeddingFunction protocol is generic and
        # cannot be satisfied without pulling in chromadb's internal types.
        from typing import cast as _cast

        from chromadb.api.types import EmbeddingFunction

        no_embed = _cast(EmbeddingFunction, _NoEmbeddingFunction())
        self._docs_coll = self._chroma_client.get_or_create_collection(
            "documents", embedding_function=no_embed
        )
        self._chunks_coll = self._chroma_client.get_or_create_collection(
            "chunks", embedding_function=no_embed
        )

        self._hash_path = config.corpus_root(corpus_name) / "doc_hashes.json"
        self._hashes: dict[str, str] = self._load_hashes()

    # ------------------------------------------------------------------
    # Top-level indexing
    # ------------------------------------------------------------------

    def index(self, documents: Iterable[Document]) -> IndexStats:
        """Run a full indexing pass.

        Streams the input one document at a time: each file is embedded,
        written to all three stores, and its hash persisted before the
        iterator advances. A Ctrl-C at any point leaves every file
        processed so far durably in the index.

        Link edges are recorded in a second pass after the iterator is
        exhausted, using accumulated (src, target) tuples — no need to
        retain Document objects in memory.

        The BM25 index is rebuilt once at the end of a complete run.
        """
        import time as _time

        stats = IndexStats()
        indexed_doc_ids: set[str] = set()
        link_records: list[tuple[str, str]] = []

        for i, doc in enumerate(documents, start=1):
            src = str(doc.path.resolve())
            # Accumulate link pairs now while the Document is in scope.
            for target in doc.metadata.get("links_to", []) or []:
                link_records.append((src, target))
            indexed_doc_ids.add(src)

            if self._is_unchanged(doc):
                stats.documents_skipped += 1
                logger.info("[#%d] skip (unchanged): %s", i, doc.path)
                continue

            t0 = _time.perf_counter()
            n_chunks = self._index_one(doc)
            elapsed_ms = int((_time.perf_counter() - t0) * 1000)
            stats.documents_indexed += 1
            stats.chunks_written += n_chunks
            logger.info(
                "[#%d] indexed %s (%d chunks, %d ms)",
                i,
                doc.path,
                n_chunks,
                elapsed_ms,
            )
            # Persist hash immediately so the next run can skip this doc
            # even if the process is killed before the iterator finishes.
            self._save_hashes()

        # Link pass: union the current run's doc ids with any already in
        # the graph from prior runs, then record edges whose target is known.
        known: set[str] = set(indexed_doc_ids)
        for entry in self.kg.list_all_documents():
            known.add(entry["doc_id"])

        for src, target in link_records:
            if target in known and target != src:
                self.kg.add_link(src, target, "")
                stats.links_recorded += 1

        self._write_bm25_index()
        self._save_hashes()
        return stats

    # ------------------------------------------------------------------
    # Single-document ops (for the watcher)
    # ------------------------------------------------------------------

    def add_document(self, doc: Document) -> int:
        """Re-index one document. Returns the number of chunks written."""
        n = self._index_one(doc)
        self._write_bm25_index()
        self._save_hashes()
        return n

    def remove_document(self, doc_id: str) -> None:
        """Remove a document end-to-end: Chroma chunks, Chroma doc, Kuzu."""
        chunk_ids = self.kg.get_chunks_for_doc(doc_id)
        if chunk_ids:
            self._chunks_coll.delete(ids=chunk_ids)
        # Chroma's delete on a missing id is a no-op.
        self._docs_coll.delete(ids=[doc_id])
        self.kg.remove_document(doc_id)
        self._hashes.pop(doc_id, None)
        self._write_bm25_index()
        self._save_hashes()

    def update_document(self, doc: Document) -> int:
        """Wholesale replace one document's index footprint."""
        doc_id = str(doc.path.resolve())
        self.remove_document(doc_id)
        return self.add_document(doc)

    # ------------------------------------------------------------------
    # Internal: per-doc work
    # ------------------------------------------------------------------

    def _index_one(self, doc: Document) -> int:
        """Index a single document. Returns number of chunks written."""
        doc_id = str(doc.path.resolve())

        # Wipe any prior chunks for this doc before re-inserting so a
        # shrinking document doesn't leave stale chunk rows behind.
        prior_chunk_ids = self.kg.get_chunks_for_doc(doc_id)
        if prior_chunk_ids:
            self._chunks_coll.delete(ids=prior_chunk_ids)

        # Kuzu Document node.
        self.kg.add_document(doc)

        # Doc-level Chroma row.
        doc_embed_text = self._doc_embedding_text(doc)
        doc_vec = self.embedder.embed(doc_embed_text)
        doc_meta = _clean_meta(
            {
                "doc_id": doc_id,
                "title": doc.metadata.get("title", ""),
                "file_type": doc.metadata.get("file_type", "html"),
                "page_description": doc.metadata.get("page_description", ""),
            }
        )
        # The arg-type ignore is for chromadb's overly-narrow embeddings
        # typing: `list[list[float]]` is runtime-valid but
        # invariance-incompatible with the typed signature.
        self._docs_coll.upsert(
            ids=[doc_id],
            embeddings=[doc_vec],  # type: ignore[arg-type, unused-ignore]
            documents=[doc_embed_text],
            metadatas=[doc_meta],
        )

        # Chunk rows.
        sections = chunk_document(doc, self.config)
        if sections:
            embed_texts = [s.embedding_text for s in sections]
            vecs = self.embedder.embed_batch(embed_texts)
            ids = [s.chunk.chunk_id for s in sections]
            chunk_metas = [
                _clean_meta({**s.metadata, "embedding_text": s.embedding_text}) for s in sections
            ]
            chunk_texts = [s.chunk_text for s in sections]
            self._chunks_coll.upsert(
                ids=ids,
                embeddings=vecs,  # type: ignore[arg-type, unused-ignore]
                documents=chunk_texts,
                metadatas=chunk_metas,  # type: ignore[arg-type, unused-ignore]
            )
            for s in sections:
                self.kg.add_chunk(s.chunk, doc_id, s.metadata["position"])

        # Hash now so re-runs see this document as unchanged.
        self._hashes[doc_id] = self._compute_hash(doc)
        return len(sections)

    def _doc_embedding_text(self, doc: Document) -> str:
        """``page_description`` + first 512 tokens of body. Spec-mandated shape."""
        desc = (doc.metadata.get("page_description") or "").strip()
        tokens = _ENCODER.encode(doc.content)
        body_excerpt = _ENCODER.decode(tokens[:_DOC_EMBED_TOKEN_BUDGET])
        return (desc + " " + body_excerpt).strip() if desc else body_excerpt.strip()

    # ------------------------------------------------------------------
    # Incremental: hashing
    # ------------------------------------------------------------------

    def _is_unchanged(self, doc: Document) -> bool:
        doc_id = str(doc.path.resolve())
        existing = self._hashes.get(doc_id)
        return existing is not None and existing == self._compute_hash(doc)

    @staticmethod
    def _compute_hash(doc: Document) -> str:
        h = hashlib.sha256()
        h.update(doc.content.encode("utf-8"))
        # Title changes alone should also invalidate the cached embedding —
        # doc-level embedding text includes ``page_description`` and title via
        # the chunker's enrichment, so a metadata-only edit must re-index.
        h.update(str(doc.metadata.get("title", "")).encode("utf-8"))
        h.update(str(doc.metadata.get("page_description", "")).encode("utf-8"))
        return h.hexdigest()

    def _load_hashes(self) -> dict[str, str]:
        if not self._hash_path.is_file():
            return {}
        try:
            with self._hash_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return {}
            return {str(k): str(v) for k, v in data.items()}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load %s: %s", self._hash_path, exc)
            return {}

    def _save_hashes(self) -> None:
        self._hash_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._hash_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._hashes, fh, indent=2, sort_keys=True)
        tmp.replace(self._hash_path)

    # ------------------------------------------------------------------
    # BM25 build
    # ------------------------------------------------------------------

    def _write_bm25_index(self) -> None:
        """Rebuild the BM25 index from the full ``chunks`` collection."""
        chunks = self._chunks_coll.get(include=["documents"])
        ids = list(chunks.get("ids") or [])
        docs = list(chunks.get("documents") or [])
        bm25 = BM25Index()
        bm25.build(list(zip(ids, docs, strict=True)))
        bm25.save(self.config.bm25_index_path(self.corpus_name))
        logger.info("BM25 index written with %d chunks", len(ids))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_meta(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Drop keys whose values aren't Chroma-safe (None / list / dict).

    ChromaDB rejects ``None`` values in metadata and serialises only
    str/int/float/bool. We drop unrepresentable entries rather than try to
    coerce them.
    """
    out: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, bool | int | float | str):
            out[k] = v
        else:
            # Lists like ``code_blocks`` and dicts like ``page_metadata`` are
            # excluded from Chroma metadata. Callers that need them keep them
            # on the Document object.
            continue
    return out


class _NoEmbeddingFunction:
    """ChromaDB's collection API rejects ``embedding_function=None``.

    We always supply vectors explicitly to ``upsert(embeddings=...)``, so the
    collection-level embedding function should never be called. This stub
    raises if Chroma ever tries to use it — which would mean we forgot to
    pass embeddings somewhere.
    """

    def name(self) -> str:  # pragma: no cover - chromadb diagnostic only
        return "arg-no-embedding-function"

    def __call__(self, input: Any) -> list[list[float]]:
        raise RuntimeError(
            "ARG indexer must always pass embeddings explicitly; "
            "ChromaDB tried to invoke its default embedding function."
        )
