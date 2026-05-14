"""HybridRetriever — the 5-stage retrieval pipeline.

Stages, in order:

  0. **Context enrichment.** Doc-level dense search picks the top
     ``config.enrich_top_docs`` documents (gated by
     ``config.enrich_min_score``). Each one's link neighbours (forward +
     reverse) and topic-cluster mates expand a candidate doc-id set.
  1. **Dense vector search.** Embed the query and pull ``top_k_vector``
     chunks from ChromaDB. When the enrichment set is non-empty, the search
     is filtered to it; a Stage-1 yield smaller than ``top_k_vector // 2``
     triggers an unfiltered re-run that is merged with the filtered hits so
     the user never sees an empty list when good chunks live outside the
     enrichment set.
  1.5. **Sparse BM25 search.** Independent keyword/exact-term retrieval
     against the BM25 pickle the indexer wrote. Honours the same metadata
     filters as Stage 1.
  2. **Graph expansion.** For each Stage-1/1.5 chunk's parent document,
     traverse ``LINKS_TO`` to ``graph_hop_depth`` and pull a few extra
     chunks from each linked document (ranked by the same query embedding,
     scoped to that doc_id via Chroma's ``where``).
  3. **RRF fusion.** Deduplicate across stages by chunk_id; RRF score
     ``Σ 1/(k + rank)`` with ``k = 60`` (standard).
  4. **Lost-in-middle reordering.** U-shape bookend arrangement so the two
     most relevant chunks sit at positions 0 and -1 of the returned list.

``scope_doc_id`` short-circuits the pipeline to a single document: Stages 0
and 2 are skipped, Stages 1 and 1.5 are filtered to that doc_id, and the
reordering still applies.

The returned ``list[NodeWithScore]`` uses LlamaIndex's schema types so the
generator (Section 9) can hand them directly to a query engine.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode

from arg.config import ARGConfig
from arg.embeddings import Embedder
from arg.graph import KnowledgeGraph
from arg.retriever.bm25_index import BM25Index

logger = logging.getLogger(__name__)


# Standard RRF constant (Cormack et al., 2009). Larger ``k`` flattens the
# influence of high-rank chunks; 60 is the value the original paper used.
_RRF_K = 60


# ---------------------------------------------------------------------------
# Internal helpers — typed candidate row used during fusion
# ---------------------------------------------------------------------------


@dataclass
class _ChunkHit:
    """One chunk's data as it travels through the retrieval pipeline."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    stage_scores: dict[str, float] = field(default_factory=dict)
    stage_ranks: dict[str, int] = field(default_factory=dict)
    rrf_score: float = 0.0


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Five-stage hybrid retriever.

    Construct once per pipeline; safe to call :meth:`retrieve` repeatedly.
    BM25 is loaded eagerly from disk; the cluster cache is read lazily on
    each call so a freshly-written cache picks up immediately.
    """

    def __init__(
        self,
        *,
        config: ARGConfig,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        chroma_documents_collection: Any,
        chroma_chunks_collection: Any,
        bm25_index_path: Path,
        cluster_cache_path: Path | None = None,
    ) -> None:
        self.config = config
        self.kg = knowledge_graph
        self.embedder = embedder
        self._docs_coll = chroma_documents_collection
        self._chunks_coll = chroma_chunks_collection
        self._bm25_path = bm25_index_path
        self._cluster_cache_path = cluster_cache_path
        # Load BM25 once at startup. Re-indexing rebuilds the file on disk;
        # callers that want the freshly-written index call :meth:`reload`.
        self._bm25 = BM25Index.load(bm25_index_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read the BM25 pickle. Call after a re-index from the indexer."""
        self._bm25 = BM25Index.load(self._bm25_path)

    def retrieve(
        self,
        query: str,
        *,
        enrich: bool = True,
        scope_doc_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[NodeWithScore]:
        """Return top-N chunks for ``query`` after the full 5-stage pipeline."""
        # `scope_doc_id` overrides every other narrowing: skip enrichment +
        # graph; constrain Stages 1 and 1.5 to one document.
        if scope_doc_id is not None:
            return self._retrieve_scoped(query, scope_doc_id, filters or {})

        # Stage 0 — enrichment.
        candidate_doc_ids: set[str] | None = None
        if enrich and self.config.enrich_enabled:
            candidate_doc_ids = self._stage0_enrichment(query)

        chroma_filters = self._chroma_where(filters) if filters else None

        # Stage 1 — dense.
        dense_hits = self._stage1_dense(
            query=query,
            candidate_doc_ids=candidate_doc_ids,
            top_k=self.config.top_k_vector,
            chroma_filters=chroma_filters,
        )

        # Stage 1.5 — BM25.
        bm25_hits = self._stage1_5_bm25(
            query=query,
            top_k=self.config.top_k_vector,
            candidate_doc_ids=candidate_doc_ids,
            filters=filters,
        )

        # Stage 2 — graph expansion.
        graph_hits: list[_ChunkHit] = []
        if self.config.graph_hop_depth > 0:
            graph_hits = self._stage2_graph(
                query=query,
                seed_hits=dense_hits + bm25_hits,
                chroma_filters=chroma_filters,
            )

        # Stage 3 — RRF fusion.
        stage_results = {
            "dense": dense_hits,
            "bm25": bm25_hits,
            "graph": graph_hits,
        }
        fused = _rrf_fuse(stage_results)

        # Stage 4 — Lost-in-middle reordering.
        return _lost_in_middle_reorder(fused, target_n=self.config.top_k_vector)

    # ------------------------------------------------------------------
    # Stage 0 — Context Enrichment
    # ------------------------------------------------------------------

    def _stage0_enrichment(self, query: str) -> set[str] | None:
        """Return a candidate doc-id set, or ``None`` to mean "use full corpus"."""
        # 0.1 doc-level dense search.
        ranked = self._find_document(query, top_k=self.config.enrich_top_docs)
        if not ranked:
            return None
        # Apply threshold; if none clear it, skip enrichment.
        seeds = [doc_id for doc_id, score in ranked if score >= self.config.enrich_min_score]
        if not seeds:
            return None

        candidates: set[str] = set(seeds)

        # 0.2 link expansion: outgoing + reverse neighbours of each seed doc.
        for seed in seeds:
            for outbound in self.kg.get_linked_docs(seed, depth=1):
                candidates.add(outbound)
            for rev in self.kg.get_reverse_links(seed):
                candidates.add(rev["doc_id"])

        # 0.3 cluster expansion (only if the cache exists AND we have enough
        # documents to make clustering meaningful).
        cluster_data = self._load_cluster_cache()
        if cluster_data is not None:
            total_docs = len(self.kg.list_all_documents())
            if total_docs >= self.config.min_cluster_docs:
                cluster_for_seed = cluster_data["doc_to_cluster"].get(seeds[0])
                if cluster_for_seed is not None:
                    members = cluster_data["cluster_members"].get(str(cluster_for_seed), [])
                    candidates.update(members)
        return candidates

    def _find_document(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        """Search the ``documents`` Chroma collection. Returns ``[(doc_id, score)]``."""
        if top_k <= 0:
            return []
        qvec = self.embedder.embed(query)
        n_in = self._docs_coll.count()
        if n_in == 0:
            return []
        result = self._docs_coll.query(
            query_embeddings=[qvec],  # type: ignore[arg-type, unused-ignore]
            n_results=min(top_k, n_in),
        )
        ids = result["ids"][0] if result["ids"] else []
        dists = result["distances"][0] if result.get("distances") else [1.0] * len(ids)
        return [(doc_id, _distance_to_score(dist)) for doc_id, dist in zip(ids, dists, strict=True)]

    def _load_cluster_cache(self) -> dict[str, Any] | None:
        if self._cluster_cache_path is None:
            return None
        if not Path(self._cluster_cache_path).is_file():
            return None
        try:
            with Path(self._cluster_cache_path).open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if (
                not isinstance(data, dict)
                or "doc_to_cluster" not in data
                or "cluster_members" not in data
            ):
                return None
            return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load cluster cache %s: %s", self._cluster_cache_path, exc)
            return None

    # ------------------------------------------------------------------
    # Stage 1 — Dense
    # ------------------------------------------------------------------

    def _stage1_dense(
        self,
        *,
        query: str,
        candidate_doc_ids: set[str] | None,
        top_k: int,
        chroma_filters: dict[str, Any] | None,
    ) -> list[_ChunkHit]:
        """Top ``top_k`` chunks by embedding similarity, filtered as needed."""
        if top_k <= 0:
            return []
        qvec = self.embedder.embed(query)

        # Combine candidate-doc filter with caller-supplied filters.
        effective_filter = _combine_where(chroma_filters, _doc_ids_filter(candidate_doc_ids))

        primary = self._chroma_query(qvec, top_k=top_k, where=effective_filter)

        # If we filtered and the primary yield is sparse, run an unfiltered
        # pass and merge — preserves recall when the enrichment guess was off.
        if candidate_doc_ids is not None and len(primary) < max(1, top_k // 2):
            unfiltered = self._chroma_query(qvec, top_k=top_k, where=chroma_filters)
            merged: dict[str, _ChunkHit] = {h.chunk_id: h for h in primary}
            for h in unfiltered:
                merged.setdefault(h.chunk_id, h)
            return list(merged.values())[:top_k]

        return primary

    def _chroma_query(
        self,
        qvec: list[float],
        *,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[_ChunkHit]:
        n_in = self._chunks_coll.count()
        if n_in == 0 or top_k <= 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [qvec],
            "n_results": min(top_k, n_in),
        }
        if where:
            kwargs["where"] = where
        result = self._chunks_coll.query(**kwargs)
        ids = result["ids"][0] if result["ids"] else []
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        dists = result.get("distances") or [[1.0] * len(ids)]
        hits: list[_ChunkHit] = []
        for rank, (cid, text, meta, dist) in enumerate(
            zip(ids, docs[0] or [], metas[0] or [], dists[0] or [], strict=True)
        ):
            score = _distance_to_score(dist)
            hits.append(
                _ChunkHit(
                    chunk_id=cid,
                    text=text or "",
                    metadata=dict(meta) if meta else {},
                    stage_scores={"dense": score},
                    stage_ranks={"dense": rank + 1},
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Stage 1.5 — BM25
    # ------------------------------------------------------------------

    def _stage1_5_bm25(
        self,
        *,
        query: str,
        top_k: int,
        candidate_doc_ids: set[str] | None,
        filters: dict[str, Any] | None,
    ) -> list[_ChunkHit]:
        if not self.config.bm25_enabled or top_k <= 0:
            return []
        if self._bm25.is_empty:
            return []
        # Over-fetch so post-filter trimming doesn't starve the result.
        raw = self._bm25.query(query, top_k=top_k * 4)
        if not raw:
            return []
        # Map chunk_id → score; we still need each chunk's text + metadata
        # from Chroma so RRF + downstream code can treat hits uniformly.
        ids = [cid for cid, _ in raw]
        chroma_rows = self._chunks_coll.get(ids=ids, include=["documents", "metadatas"])
        by_id_text = dict(
            zip(chroma_rows.get("ids", []), chroma_rows.get("documents", []), strict=False)
        )
        by_id_meta = dict(
            zip(chroma_rows.get("ids", []), chroma_rows.get("metadatas", []), strict=False)
        )
        out: list[_ChunkHit] = []
        for rank, (cid, score) in enumerate(raw):
            meta = dict(by_id_meta.get(cid) or {})
            # Apply post-hoc filtering — BM25Index doesn't know about metadata.
            if not _meta_matches(meta, filters):
                continue
            if candidate_doc_ids is not None and meta.get("doc_id") not in candidate_doc_ids:
                continue
            out.append(
                _ChunkHit(
                    chunk_id=cid,
                    text=by_id_text.get(cid) or "",
                    metadata=meta,
                    stage_scores={"bm25": score},
                    stage_ranks={"bm25": rank + 1},
                )
            )
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------------
    # Stage 2 — Graph expansion
    # ------------------------------------------------------------------

    def _stage2_graph(
        self,
        *,
        query: str,
        seed_hits: Sequence[_ChunkHit],
        chroma_filters: dict[str, Any] | None,
    ) -> list[_ChunkHit]:
        if self.config.top_k_graph <= 0 or not seed_hits:
            return []
        # Collect distinct seed doc_ids; traverse from each one.
        seed_doc_ids: list[str] = []
        seen_docs: set[str] = set()
        for hit in seed_hits:
            did = hit.metadata.get("doc_id")
            if did and did not in seen_docs:
                seen_docs.add(did)
                seed_doc_ids.append(did)

        linked_doc_ids: set[str] = set()
        for did in seed_doc_ids:
            for linked in self.kg.get_linked_docs(did, depth=self.config.graph_hop_depth):
                if linked not in seen_docs:
                    linked_doc_ids.add(linked)
        if not linked_doc_ids:
            return []

        qvec = self.embedder.embed(query)
        graph_hits: list[_ChunkHit] = []
        rank_counter = 0
        for did in sorted(linked_doc_ids):
            scoped = _combine_where(chroma_filters, {"doc_id": did})
            doc_hits = self._chroma_query(qvec, top_k=self.config.top_k_graph, where=scoped)
            for hit in doc_hits:
                rank_counter += 1
                # Re-tag the hit with the graph stage so RRF sees it as a
                # separate signal.
                hit.stage_scores = {"graph": hit.stage_scores.get("dense", 0.0)}
                hit.stage_ranks = {"graph": rank_counter}
                graph_hits.append(hit)
        return graph_hits

    # ------------------------------------------------------------------
    # Scoped retrieve (scope_doc_id)
    # ------------------------------------------------------------------

    def _retrieve_scoped(
        self,
        query: str,
        scope_doc_id: str,
        filters: dict[str, Any],
    ) -> list[NodeWithScore]:
        chroma_filters = _combine_where(
            self._chroma_where(filters) if filters else None,
            {"doc_id": scope_doc_id},
        )
        dense = self._stage1_dense(
            query=query,
            candidate_doc_ids=None,
            top_k=self.config.top_k_vector,
            chroma_filters=chroma_filters,
        )
        bm25 = self._stage1_5_bm25(
            query=query,
            top_k=self.config.top_k_vector,
            candidate_doc_ids={scope_doc_id},
            filters=filters,
        )
        fused = _rrf_fuse({"dense": dense, "bm25": bm25, "graph": []})
        return _lost_in_middle_reorder(fused, target_n=self.config.top_k_vector)

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chroma_where(filters: dict[str, Any]) -> dict[str, Any]:
        """Translate a simple ``{key: value}`` filter dict into Chroma's where syntax.

        Pass-through for values that are already Chroma-shaped (``{"$contains": ...}``,
        ``{"$in": [...]}``).
        """
        out: dict[str, Any] = {}
        for key, val in filters.items():
            out[key] = val
        return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _distance_to_score(distance: float) -> float:
    """Monotonic distance → score map. Bounded in ``(0, 1]``."""
    if distance is None or math.isnan(distance):
        return 0.0
    return 1.0 / (1.0 + max(0.0, float(distance)))


def _doc_ids_filter(doc_ids: set[str] | None) -> dict[str, Any] | None:
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": next(iter(doc_ids))}
    return {"doc_id": {"$in": sorted(doc_ids)}}


def _combine_where(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    """Combine two Chroma ``where`` clauses with implicit AND.

    Chroma supports ``$and`` for multi-clause filters; when only one side
    is non-empty we return that side directly to keep simple queries readable.
    """
    if not a:
        return b or None
    if not b:
        return a or None
    return {"$and": [a, b]}


def _meta_matches(meta: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Pure-Python evaluation of a Chroma-style filter dict against one row.

    Used by Stage 1.5 since BM25 hits don't go through Chroma's `where`.
    Supports plain ``{key: value}`` equality, ``{"$in": [...]}``, and
    ``{"$contains": substring}`` — the operators the spec calls out.
    """
    if not filters:
        return True
    for key, condition in filters.items():
        actual = meta.get(key)
        if isinstance(condition, dict):
            if "$in" in condition:
                if actual not in condition["$in"]:
                    return False
            elif "$contains" in condition:
                if not isinstance(actual, str) or condition["$contains"] not in actual:
                    return False
            else:
                # Unrecognised operator — fail closed.
                return False
        else:
            if actual != condition:
                return False
    return True


def _rrf_fuse(stage_results: dict[str, list[_ChunkHit]]) -> list[_ChunkHit]:
    """RRF deduplicate + score across stages. Returns chunks sorted by RRF desc."""
    merged: dict[str, _ChunkHit] = {}
    for stage_name, hits in stage_results.items():
        for rank, hit in enumerate(hits):
            existing = merged.get(hit.chunk_id)
            if existing is None:
                existing = _ChunkHit(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    metadata=dict(hit.metadata),
                )
                merged[hit.chunk_id] = existing
            existing.stage_scores[stage_name] = hit.stage_scores.get(stage_name, 0.0)
            existing.stage_ranks[stage_name] = rank + 1
            existing.rrf_score += 1.0 / (_RRF_K + rank + 1)
    return sorted(merged.values(), key=lambda h: (-h.rrf_score, h.chunk_id))


def _lost_in_middle_reorder(hits: list[_ChunkHit], *, target_n: int) -> list[NodeWithScore]:
    """U-shape arrangement: best at positions 0 and -1, decreasing toward middle."""
    top = hits[:target_n] if target_n > 0 else hits
    if not top:
        return []
    front: list[_ChunkHit] = []
    back: list[_ChunkHit] = []
    for i, hit in enumerate(top):
        if i % 2 == 0:
            front.append(hit)
        else:
            back.append(hit)
    reordered = front + list(reversed(back))
    return [_to_node_with_score(h) for h in reordered]


def _to_node_with_score(hit: _ChunkHit) -> NodeWithScore:
    node = TextNode(id_=hit.chunk_id, text=hit.text, metadata=hit.metadata)
    return NodeWithScore(node=node, score=hit.rrf_score)
