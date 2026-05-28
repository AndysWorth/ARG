"""CorpusExplorer — DCI navigation, clustering, and corpus-wide analytics.

Wraps `KnowledgeGraph` for cheap graph reads, `CorpusAnalyst.find_document`
for corpus-wide dense search, and `scikit-learn` k-means for topic clustering
over the ``documents`` Chroma collection.

Small-corpus clustering rule
----------------------------
The spec is explicit: if ``doc_count < config.min_cluster_docs``, clustering
returns ``[{"label": "All documents", "doc_ids": [...]}]`` without calling
the LLM or running k-means. The threshold is configurable but defaults to
10. The cluster cache is still written (so the retriever's Stage 0.3 can
short-circuit) but with a single entry — invalidation works the same way.

Cluster cache
-------------
Persisted to ``cluster_cache_path`` (per-corpus). Shape:

    {
        "doc_to_cluster": {doc_id: cluster_id, ...},
        "cluster_members": {cluster_id: [doc_id, ...], ...},
        "labels": {cluster_id: "human-readable label", ...},
    }

Computed at the end of ``ARGPipeline.index()`` and on demand via
:meth:`get_topic_clusters`. :meth:`invalidate_cluster_cache` deletes the
file so the next call recomputes.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from arg.config import ARGConfig
from arg.dci.analyst import CorpusAnalyst
from arg.graph import KnowledgeGraph
from arg.llm import LLM

logger = logging.getLogger(__name__)


_CLUSTER_LABEL_PROMPT = """\
Given the following document titles, write a single short topic label
(2-5 words) that describes what they have in common. Output only the label.

Titles:
{titles}"""


class CorpusExplorer:
    """Browsing + clustering + analytics surface for the DCI layer."""

    def __init__(
        self,
        *,
        config: ARGConfig,
        knowledge_graph: KnowledgeGraph,
        analyst: CorpusAnalyst,
        llm: LLM,
        chroma_documents_collection: Any,
        corpus_name: str = "default",
    ) -> None:
        self.config = config
        self.kg = knowledge_graph
        self.analyst = analyst
        self.llm = llm
        self._docs_coll = chroma_documents_collection
        self.corpus_name = corpus_name

    # ------------------------------------------------------------------
    # Trivial passthrough surfaces (1, 2, 3)
    # ------------------------------------------------------------------

    def list_all_documents(self) -> list[dict[str, Any]]:
        return self.kg.list_all_documents()

    def get_reverse_links(self, doc_id: str) -> list[dict[str, Any]]:
        return self.kg.get_reverse_links(doc_id)

    def get_graph_json(self) -> dict[str, list[dict[str, Any]]]:
        return self.kg.get_graph_json()

    # ------------------------------------------------------------------
    # Analytics (6)
    # ------------------------------------------------------------------

    def most_linked_docs(self, top_n: int = 10) -> list[dict[str, Any]]:
        return self.kg.most_linked_docs(top_n=top_n)

    def orphaned_docs(self) -> list[str]:
        return self.kg.orphaned_docs()

    def docs_by_chunk_count(
        self, page: int = 1, page_size: int = 25, order: str = "desc"
    ) -> dict[str, Any]:
        """Paginated chunk-count ranking; pagination lives at this layer."""
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 25
        ranked = self.kg.docs_by_chunk_count(descending=(order != "asc"))
        total = len(ranked)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max(1, math.ceil(total / page_size)),
            "order": "asc" if order == "asc" else "desc",
            "items": ranked[start:end],
        }

    # ------------------------------------------------------------------
    # Corpus search (5)
    # ------------------------------------------------------------------

    def corpus_search(
        self, query: str, file_type: str | None = None, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Doc-level BM25 search; optional file_type filter."""
        return self.analyst.find_document(query, top_k=top_k, file_type=file_type)

    # ------------------------------------------------------------------
    # Topic clustering (4)
    # ------------------------------------------------------------------

    def get_topic_clusters(self) -> list[dict[str, Any]]:
        """Return the cluster list, computing + caching when missing."""
        cache = self._load_cluster_cache()
        if cache is not None:
            return _cache_to_list(cache)

        clusters = self._compute_clusters()
        self._save_cluster_cache(clusters)
        return _cache_to_list(clusters)

    def invalidate_cluster_cache(self) -> None:
        """Delete ``cluster_cache.json`` so the next call recomputes."""
        path = self.config.cluster_cache_path(self.corpus_name)
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Could not delete cluster cache %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Cluster cache I/O
    # ------------------------------------------------------------------

    def _load_cluster_cache(self) -> dict[str, Any] | None:
        path = self.config.cluster_cache_path(self.corpus_name)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return None
            return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read cluster cache %s: %s", path, exc)
            return None

    def _save_cluster_cache(self, cache: dict[str, Any]) -> None:
        path = self.config.cluster_cache_path(self.corpus_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, sort_keys=True)
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Cluster computation
    # ------------------------------------------------------------------

    def _compute_clusters(self) -> dict[str, Any]:
        """k-means on documents collection; LLM-label per cluster.

        Small-corpus fallback: when there are fewer than
        ``config.min_cluster_docs`` documents we skip k-means + LLM entirely
        and emit a single "All documents" cluster. The retriever's Stage 0.3
        still consumes the file the same way.
        """
        all_docs = self.kg.list_all_documents()
        doc_ids = [d["doc_id"] for d in all_docs]
        if not doc_ids:
            return {"doc_to_cluster": {}, "cluster_members": {}, "labels": {}}

        if len(doc_ids) < self.config.min_cluster_docs:
            logger.info(
                "Topic clustering skipped: %d docs < min_cluster_docs (%d)",
                len(doc_ids),
                self.config.min_cluster_docs,
            )
            return {
                "doc_to_cluster": dict.fromkeys(doc_ids, "all"),
                "cluster_members": {"all": list(doc_ids)},
                "labels": {"all": "All documents"},
            }

        # Pull embeddings for every document. ChromaDB returns embeddings
        # as a numpy ndarray, so we avoid ``rows.get(...) or []`` (that
        # short-circuit triggers numpy's truthiness ambiguity).
        rows = self._docs_coll.get(ids=doc_ids, include=["embeddings", "metadatas"])
        embeddings_obj = rows.get("embeddings")
        ids = list(rows.get("ids") or [])
        metas = list(rows.get("metadatas") or [])
        if embeddings_obj is None or len(embeddings_obj) == 0 or len(embeddings_obj) != len(ids):
            logger.warning("Cluster compute aborted: documents collection has no embeddings")
            return {
                "doc_to_cluster": dict.fromkeys(doc_ids, "all"),
                "cluster_members": {"all": list(doc_ids)},
                "labels": {"all": "All documents"},
            }
        # Normalise to list-of-lists so k-means and the zip() below behave
        # the same whether Chroma returns ndarray or list.
        embeddings = [list(map(float, row)) for row in embeddings_obj]

        from sklearn.cluster import KMeans

        n_clusters = max(2, min(self.config.n_clusters, len(ids)))
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        labels = km.fit_predict(embeddings)

        doc_to_cluster: dict[str, str] = {}
        cluster_members: dict[str, list[str]] = {}
        for doc_id, cluster_idx in zip(ids, labels, strict=True):
            cid = f"c{int(cluster_idx)}"
            doc_to_cluster[doc_id] = cid
            cluster_members.setdefault(cid, []).append(doc_id)

        # Build a lookup of frozenset(doc_ids) -> existing label from the old
        # cache so unchanged clusters can reuse their labels without an LLM call.
        old_cache = self._load_cluster_cache()
        old_members: dict[str, list[str]] = {}
        old_labels: dict[str, str] = {}
        if old_cache is not None:
            old_members = old_cache.get("cluster_members") or {}
            old_labels = old_cache.get("labels") or {}
        reuse_label: dict[frozenset[str], str] = {
            frozenset(v): old_labels[k] for k, v in old_members.items() if k in old_labels
        }

        labels_map: dict[str, str] = {}
        # Build a title map for nicer cluster-label prompts.
        title_by_id: dict[str, str] = {}
        for doc_id, meta in zip(ids, metas, strict=True):
            if isinstance(meta, dict):
                title_by_id[doc_id] = str(meta.get("title", "") or "")
        for cid, member_doc_ids in cluster_members.items():
            fs = frozenset(member_doc_ids)
            if fs in reuse_label:
                labels_map[cid] = reuse_label[fs]
                continue
            sample_titles = [
                title_by_id.get(d, "") for d in member_doc_ids[:8] if title_by_id.get(d)
            ]
            if not sample_titles:
                labels_map[cid] = "Cluster"
                continue
            prompt = _CLUSTER_LABEL_PROMPT.format(titles="\n".join(sample_titles))
            logger.info("explorer: labeling cluster %s (%d docs) via LLM", cid, len(member_doc_ids))
            label = self.llm.complete(prompt).strip().strip('"').strip("'")
            labels_map[cid] = label or "Cluster"

        return {
            "doc_to_cluster": doc_to_cluster,
            "cluster_members": cluster_members,
            "labels": labels_map,
        }


def _cache_to_list(cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Reshape the on-disk cache into the API's ``[{label, doc_ids}]`` list."""
    members = cache.get("cluster_members") or {}
    labels = cache.get("labels") or {}
    out: list[dict[str, Any]] = []
    for cluster_id, doc_ids in members.items():
        out.append(
            {
                "label": labels.get(cluster_id, "Cluster"),
                "doc_ids": list(doc_ids),
            }
        )
    return out
