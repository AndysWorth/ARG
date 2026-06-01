"""Embedded Kuzu knowledge graph.

`KnowledgeGraph` owns the per-corpus graph database. Documents and chunks are
nodes; cross-document HTML/PDF links are ``LINKS_TO`` edges; ``CONTAINS`` edges
connect a document to its chunks. The graph backs three layers of ARG:

  * the **retriever** (Section 8) — Stage 2 graph traversal expands the
    candidate set via outgoing links;
  * **CorpusExplorer + CorpusAnalyst** (Sections 9/10) — reverse-link lookup,
    cluster reporting, and corpus-wide analytics;
  * the **web UI** — :meth:`get_graph_json` returns a D3-compatible
    ``{nodes, edges}`` payload.

The class is intentionally synchronous — Kuzu is embedded, fast, and runs in
the calling thread. Callers that need concurrency wrap operations in their own
locks (see :class:`ARGPipeline` in Section 10).

Locality
--------
Kuzu is a library, not a server: no socket is opened. The graph lives in a
single directory under ``db_path``. Closing and reopening is non-destructive
and is exercised by :func:`test_persistence_across_reopen`.

Kuzu quirks worth recording (verified against 0.11.3)
-----------------------------------------------------
  * Variable-length paths (``*1..N``) refuse to bind ``N`` to a query parameter
    — the parser rejects ``*1..$depth``. :meth:`get_linked_docs` formats
    ``depth`` into the query after validating it as a positive int.
  * Variable-length paths can revisit the starting node on a cycle
    (``A->B->C->A`` returns ``A`` at depth 3). Queries explicitly filter out
    the source ``doc_id`` to give callers the linked-set semantics they expect.
  * ``CREATE NODE TABLE IF NOT EXISTS`` and ``CREATE REL TABLE IF NOT EXISTS``
    are supported, so the schema-init step is fully idempotent.

# Implements: docs/spec/section-06-knowledge-graph.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kuzu

from arg.crawler.extractors import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk dataclass — used by the chunker (Section 7) and the retriever
# (Section 8). Defined here because :meth:`KnowledgeGraph.add_chunk` is its
# only direct consumer in this section.
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One chunk of a document's text.

    Attributes
    ----------
    chunk_id:
        Stable identifier — by convention ``"{doc_id}::chunk::{position}"``,
        though :class:`KnowledgeGraph` doesn't enforce the format.
    text:
        Chunk text. Stored in the graph so that callers without Chroma access
        can still recover chunk content from ``chunk_id``.
    token_count:
        Approximate token count, computed by the chunker.
    """

    chunk_id: str
    text: str
    token_count: int


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_DDL: tuple[str, ...] = (
    "CREATE NODE TABLE IF NOT EXISTS Document("
    "  doc_id STRING,"
    "  title STRING,"
    "  file_type STRING,"
    "  chunk_count INT64,"
    "  PRIMARY KEY (doc_id))",
    "CREATE NODE TABLE IF NOT EXISTS Chunk("
    "  chunk_id STRING,"
    "  text STRING,"
    "  token_count INT64,"
    "  PRIMARY KEY (chunk_id))",
    "CREATE REL TABLE IF NOT EXISTS LINKS_TO(  FROM Document TO Document,  anchor_text STRING)",
    "CREATE REL TABLE IF NOT EXISTS CONTAINS(  FROM Document TO Chunk,  position INT64)",
)


_MAX_TRAVERSAL_DEPTH = 64  # Sanity cap on get_linked_docs to keep Kuzu happy.


def _doc_id_from_document(doc: Document) -> str:
    """Canonical doc_id for graph storage: the document's absolute path string."""
    return str(doc.path.resolve())


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Kuzu-backed graph of documents, chunks, and their relationships."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        # Kuzu insists on a non-existent path OR a file path it owns — directly
        # nominating a directory like ``arg_db/default/kuzu`` works as long as
        # the path isn't an existing on-disk directory it didn't create.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        for ddl in _SCHEMA_DDL:
            self._conn.execute(ddl)

    def close(self) -> None:
        """Release the connection and database handle.

        Kuzu finalises its WAL when both the Connection and Database objects
        are garbage-collected; calling ``close`` makes that deterministic so a
        process can re-open the same path immediately after. Idempotent —
        a second call is a no-op so signal handlers can call it safely.
        """
        # Order matters: connection before database. After close(), any
        # subsequent method call on the graph raises AttributeError — that's
        # the intended failure mode; callers shouldn't reuse a closed graph.
        if hasattr(self, "_conn"):
            del self._conn
        if hasattr(self, "_db"):
            del self._db

    def __enter__(self) -> KnowledgeGraph:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def add_document(self, doc: Document) -> None:
        """Insert or update a Document node. Idempotent."""
        doc_id = _doc_id_from_document(doc)
        title = str(doc.metadata.get("title", "") or "")
        file_type = str(doc.metadata.get("file_type", "html") or "html")

        # MERGE keeps add_document idempotent: a re-index doesn't create
        # duplicate nodes. chunk_count is initialised on CREATE and preserved
        # on MATCH so subsequent add_chunk calls can keep it consistent.
        self._conn.execute(
            (
                "MERGE (d:Document {doc_id: $doc_id}) "
                "ON CREATE SET d.title = $title, d.file_type = $file_type, d.chunk_count = 0 "
                "ON MATCH SET d.title = $title, d.file_type = $file_type"
            ),
            {"doc_id": doc_id, "title": title, "file_type": file_type},
        )

    def add_chunk(self, chunk: Chunk, doc_id: str, position: int) -> None:
        """Insert/update a Chunk and connect it to ``doc_id`` via CONTAINS."""
        if position < 0:
            raise ValueError(f"position must be >= 0 (got {position})")
        # MERGE the chunk (idempotent on chunk_id).
        self._conn.execute(
            (
                "MERGE (c:Chunk {chunk_id: $cid}) "
                "ON CREATE SET c.text = $text, c.token_count = $tc "
                "ON MATCH SET c.text = $text, c.token_count = $tc"
            ),
            {"cid": chunk.chunk_id, "text": chunk.text, "tc": int(chunk.token_count)},
        )
        # MERGE the CONTAINS edge. Kuzu's MERGE on a relationship only matches
        # on the endpoint pattern, not edge properties, so re-adding a chunk
        # at a new position would silently keep the old position; we update
        # the edge's position explicitly below.
        self._conn.execute(
            (
                "MATCH (d:Document {doc_id: $did}), (c:Chunk {chunk_id: $cid}) "
                "MERGE (d)-[r:CONTAINS]->(c) "
                "ON CREATE SET r.position = $pos "
                "ON MATCH SET r.position = $pos"
            ),
            {"did": doc_id, "cid": chunk.chunk_id, "pos": int(position)},
        )
        self._recompute_chunk_count(doc_id)

    def add_link(self, source_doc_id: str, target_doc_id: str, anchor_text: str) -> None:
        """Insert or update a LINKS_TO edge from ``source`` to ``target``.

        MERGE semantics prevent duplicate edges from accumulating across
        re-indexes: only one edge per (source, target) pair is kept.
        """
        self._conn.execute(
            (
                "MATCH (s:Document {doc_id: $src}), (t:Document {doc_id: $tgt}) "
                "MERGE (s)-[r:LINKS_TO]->(t) "
                "ON CREATE SET r.anchor_text = $anchor "
                "ON MATCH SET r.anchor_text = $anchor"
            ),
            {"src": source_doc_id, "tgt": target_doc_id, "anchor": anchor_text or ""},
        )

    def remove_document(self, doc_id: str) -> None:
        """Delete a Document and all of its chunks + outgoing/incoming edges."""
        # Delete chunks owned by this doc first. Kuzu's DETACH DELETE removes
        # the CONTAINS edges along with each chunk node. Chunks belong to one
        # document, so this never strands chunks elsewhere.
        self._conn.execute(
            "MATCH (d:Document {doc_id: $did})-[:CONTAINS]->(c:Chunk) DETACH DELETE c",
            {"did": doc_id},
        )
        # Now delete the document; remaining LINKS_TO edges go with it.
        self._conn.execute(
            "MATCH (d:Document {doc_id: $did}) DETACH DELETE d",
            {"did": doc_id},
        )

    def get_chunks_for_doc(self, doc_id: str) -> list[str]:
        """Return chunk_ids belonging to ``doc_id`` ordered by CONTAINS.position."""
        result = self._conn.execute(
            (
                "MATCH (d:Document {doc_id: $did})-[r:CONTAINS]->(c:Chunk) "
                "RETURN c.chunk_id ORDER BY r.position"
            ),
            {"did": doc_id},
        )
        return [row[0] for row in _iter_rows(result)]

    def get_doc_metadata(self, doc_id: str) -> dict[str, Any]:
        """Return the Document node as a dict; empty dict if not found."""
        result = self._conn.execute(
            (
                "MATCH (d:Document {doc_id: $did}) "
                "RETURN d.doc_id, d.title, d.file_type, d.chunk_count"
            ),
            {"did": doc_id},
        )
        for row in _iter_rows(result):
            return {
                "doc_id": row[0],
                "title": row[1],
                "file_type": row[2],
                "chunk_count": row[3],
            }
        return {}

    def stats(self) -> dict[str, int]:
        """Diagnostic counts: documents, chunks, link edges, contains edges."""
        return {
            "documents": self._scalar_int("MATCH (d:Document) RETURN count(d)"),
            "chunks": self._scalar_int("MATCH (c:Chunk) RETURN count(c)"),
            "link_edges": self._scalar_int("MATCH ()-[r:LINKS_TO]->() RETURN count(r)"),
            "contains_edges": self._scalar_int("MATCH ()-[r:CONTAINS]->() RETURN count(r)"),
        }

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_linked_docs(self, doc_id: str, depth: int) -> list[str]:
        """Return distinct downstream doc_ids reachable within ``depth`` hops.

        The starting ``doc_id`` is never included even when a cycle would
        otherwise bring it back. ``depth`` must be a positive int <= 64.
        """
        if depth < 1:
            return []
        if depth > _MAX_TRAVERSAL_DEPTH:
            raise ValueError(f"depth must be <= {_MAX_TRAVERSAL_DEPTH} (got {depth})")
        # Kuzu's Cypher parser does NOT accept a parameter inside the
        # variable-length range, so the validated integer is formatted in.
        query = (
            f"MATCH (d:Document {{doc_id: $did}})-[:LINKS_TO*1..{int(depth)}]->"
            "(n:Document) "
            "WHERE n.doc_id <> $did "
            "RETURN DISTINCT n.doc_id"
        )
        result = self._conn.execute(query, {"did": doc_id})
        return [row[0] for row in _iter_rows(result)]

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def get_reverse_links(self, doc_id: str) -> list[dict[str, Any]]:
        """Documents that link TO ``doc_id``, with their anchor text."""
        result = self._conn.execute(
            (
                "MATCH (s:Document)-[r:LINKS_TO]->(t:Document {doc_id: $did}) "
                "RETURN s.doc_id, s.title, r.anchor_text "
                "ORDER BY s.doc_id"
            ),
            {"did": doc_id},
        )
        return [
            {"doc_id": row[0], "title": row[1], "anchor_text": row[2]} for row in _iter_rows(result)
        ]

    def list_all_documents(self, limit: int = 0, offset: int = 0) -> list[dict[str, Any]]:
        """All Document nodes, sorted by doc_id.

        ``limit=0`` means no limit (return all).  ``offset`` is applied
        independently of ``limit`` so callers can page without a hard cap.
        """
        query = (
            "MATCH (d:Document) "
            "RETURN d.doc_id, d.title, d.file_type, d.chunk_count "
            "ORDER BY d.doc_id"
        )
        if offset > 0:
            query += f" SKIP {int(offset)}"
        if limit > 0:
            query += f" LIMIT {int(limit)}"
        result = self._conn.execute(query)
        return [
            {
                "doc_id": row[0],
                "title": row[1],
                "file_type": row[2],
                "chunk_count": row[3],
            }
            for row in _iter_rows(result)
        ]

    def list_documents_by_chunk_count(
        self, limit: int = 0, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Documents ordered by chunk_count DESC with server-side pagination."""
        query = (
            "MATCH (d:Document) "
            "RETURN d.doc_id, d.title, d.chunk_count "
            "ORDER BY d.chunk_count DESC, d.doc_id ASC"
        )
        if offset > 0:
            query += f" SKIP {int(offset)}"
        if limit > 0:
            query += f" LIMIT {int(limit)}"
        result = self._conn.execute(query)
        return [
            {"doc_id": row[0], "title": row[1], "chunk_count": row[2]} for row in _iter_rows(result)
        ]

    def count_documents(self) -> int:
        """Fast document count without materialising all rows."""
        return self._scalar_int("MATCH (d:Document) RETURN count(d)")

    def get_graph_json(self, max_nodes: int = 500, max_edges: int = 2000) -> dict[str, Any]:
        """Return ``{"nodes": [...], "edges": [...]}`` for D3 rendering.

        Caps at ``max_nodes`` documents (ranked by chunk_count desc so the
        most content-rich docs appear first) and ``max_edges`` link edges.
        A ``"truncated": true`` key is added only when the cap is hit, so
        callers that don't check it keep working without changes.
        """
        nodes_result = self._conn.execute(
            "MATCH (d:Document) "
            "RETURN d.doc_id, d.title, d.file_type, d.chunk_count "
            "ORDER BY d.chunk_count DESC, d.doc_id "
            f"LIMIT {int(max_nodes + 1)}"
        )
        all_nodes = [
            {
                "id": row[0],
                "title": row[1],
                "file_type": row[2],
                "chunk_count": row[3],
            }
            for row in _iter_rows(nodes_result)
        ]
        truncated_nodes = len(all_nodes) > max_nodes
        nodes = all_nodes[:max_nodes]
        included = {n["id"] for n in nodes}

        # Fetch more edges than the cap so filtering to included nodes leaves
        # enough candidates; the Python filter handles the final selection.
        edges_result = self._conn.execute(
            "MATCH (s:Document)-[r:LINKS_TO]->(t:Document) "
            "RETURN s.doc_id, t.doc_id, r.anchor_text "
            f"LIMIT {int(max_edges * 4)}"
        )
        filtered_edges = [
            {"source": row[0], "target": row[1], "anchor_text": row[2]}
            for row in _iter_rows(edges_result)
            if row[0] in included and row[1] in included
        ]
        truncated_edges = len(filtered_edges) > max_edges
        edges = filtered_edges[:max_edges]

        result: dict[str, Any] = {"nodes": nodes, "edges": edges}
        if truncated_nodes or truncated_edges:
            result["truncated"] = True
        return result

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def most_linked_docs(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Documents ranked by inbound LINKS_TO count (descending)."""
        if top_n < 1:
            return []
        # Same parameter-binding quirk as variable-length paths — LIMIT N is
        # statically parsed in Kuzu 0.11, so we format the validated int.
        query = (
            "MATCH (d:Document) "
            "OPTIONAL MATCH (other:Document)-[:LINKS_TO]->(d) "
            "WITH d, count(other) AS inbound "
            "RETURN d.doc_id, d.title, inbound "
            "ORDER BY inbound DESC, d.doc_id ASC "
            f"LIMIT {int(top_n)}"
        )
        result = self._conn.execute(query)
        return [
            {"doc_id": row[0], "title": row[1], "inbound": row[2]} for row in _iter_rows(result)
        ]

    def orphaned_docs(self) -> list[str]:
        """Documents with zero inbound LINKS_TO edges."""
        result = self._conn.execute(
            "MATCH (d:Document) "
            "OPTIONAL MATCH (other:Document)-[:LINKS_TO]->(d) "
            "WITH d, count(other) AS inbound "
            "WHERE inbound = 0 "
            "RETURN d.doc_id "
            "ORDER BY d.doc_id"
        )
        return [row[0] for row in _iter_rows(result)]

    def docs_by_chunk_count(self, descending: bool = True) -> list[dict[str, Any]]:
        """All documents sorted by chunk_count. No truncation."""
        direction = "DESC" if descending else "ASC"
        result = self._conn.execute(
            "MATCH (d:Document) "
            "RETURN d.doc_id, d.title, d.chunk_count "
            f"ORDER BY d.chunk_count {direction}, d.doc_id ASC"
        )
        return [
            {"doc_id": row[0], "title": row[1], "chunk_count": row[2]} for row in _iter_rows(result)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _recompute_chunk_count(self, doc_id: str) -> None:
        """Sync ``Document.chunk_count`` with actual CONTAINS edges.

        Cheaper than incrementing on every add_chunk and immune to double-add /
        delete edge cases — the count always matches reality after the call
        returns.
        """
        self._conn.execute(
            (
                "MATCH (d:Document {doc_id: $did}) "
                "OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk) "
                "WITH d, count(c) AS cnt "
                "SET d.chunk_count = cnt"
            ),
            {"did": doc_id},
        )

    def _scalar_int(self, query: str) -> int:
        result = self._conn.execute(query)
        for row in _iter_rows(result):
            return int(row[0])
        return 0


def _iter_rows(result: Any) -> Any:
    """Yield rows from a Kuzu QueryResult one at a time.

    Kuzu's QueryResult is a forward-only iterator that supports
    ``has_next()`` / ``get_next()`` rather than the Python iter protocol.
    """
    while result.has_next():
        yield result.get_next()
