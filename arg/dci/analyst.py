"""CorpusAnalyst — DCI capabilities that operate on whole documents.

Six capabilities live on this class. They share the same LLM and retriever
the generator uses, so a single Ollama client and a single retriever
reference cover the full RAG + DCI surface.

  * :meth:`summarize_document`     — short 3-5 sentence digest, map-reduce
    when the doc exceeds the LLM's context window.
  * :meth:`extract_key_points`     — bulleted key-point list.
  * :meth:`compare_documents`      — structured side-by-side comparison.
  * :meth:`scoped_search`          — vector search inside one document.
  * :meth:`get_chunks`             — raw chunk records from KG + Chroma.
  * :meth:`find_document`          — doc-level BM25 search (internal;
    used by Stage 0 enrichment and CorpusExplorer.corpus_search).

The summary cache writes to ``{corpus_root}/summaries/{hash}.json`` when
``config.summary_cache`` is True. Cache invalidation is the caller's
responsibility (pipeline.update_document re-summarises on the next call).

# Implements: docs/spec/section-10-pipeline.md
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import tiktoken
from llama_index.core.schema import NodeWithScore

from arg.config import ARGConfig
from arg.graph import KnowledgeGraph
from arg.llm import LLM
from arg.retriever.retriever import HybridRetriever

logger = logging.getLogger(__name__)


_ENCODER = tiktoken.get_encoding("cl100k_base")
# Approximate budget for a single LLM call. qwen3.6:35b has a 256k context,
# but we keep batches small to leave headroom for the prompt template + response.
_BATCH_TOKEN_BUDGET = 4000

_SUMMARY_PROMPT = """\
Summarise the following document in 3-5 sentences, preserving the key facts
and structural points. Do not add information that is not in the text.

Document:
{text}"""

_KEY_POINTS_PROMPT = """\
List the {max_points} most important key points from the following document.
Output ONE point per line, with no numbering, bullets, or commentary. If
fewer than {max_points} key points exist, output only as many as you find.

Document:
{text}"""

_COMPARE_PROMPT = """\
Compare the following two documents. Structure the comparison as:

  Topics covered by both:
  Topics unique to Document A:
  Topics unique to Document B:
  Contradictions or conflicts:

Be concrete and quote document phrasing where useful. If a section has
no content, write "(none)".

--- Document A ({title_a}) ---
{text_a}

--- Document B ({title_b}) ---
{text_b}"""

_MAP_REDUCE_PROMPT = """\
Combine the following per-section summaries into one coherent 3-5 sentence
overview of the whole document. Do not add information that is not in the
section summaries.

Section summaries:
{summaries}"""


class CorpusAnalyst:
    """DCI methods for whole-document operations."""

    def __init__(
        self,
        *,
        config: ARGConfig,
        llm: LLM,
        retriever: HybridRetriever,
        knowledge_graph: KnowledgeGraph,
        chroma_chunks_collection: Any,
        chroma_documents_collection: Any,
        corpus_name: str = "default",
    ) -> None:
        self.config = config
        self.llm = llm
        self.retriever = retriever
        self.kg = knowledge_graph
        self._chunks_coll = chroma_chunks_collection
        self._docs_coll = chroma_documents_collection
        self.corpus_name = corpus_name

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarize_document(self, doc_id: str) -> str:
        """Return a 3-5 sentence summary of ``doc_id``.

        Long documents are summarised in batches and the partial summaries
        are reduced into one final summary. The result is cached to disk
        when ``config.summary_cache`` is True.
        """
        cache_path = self._summary_cache_path(doc_id)
        if self.config.summary_cache and cache_path.is_file():
            try:
                with cache_path.open("r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                if isinstance(cached, dict) and "summary" in cached:
                    logger.info("analyst: summary cache hit for %s", Path(doc_id).name)
                    return str(cached["summary"])
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read summary cache %s: %s", cache_path, exc)

        chunks = self.get_chunks(doc_id)
        if not chunks:
            return ""
        joined_text = "\n\n".join(c["text"] for c in chunks)
        token_count = len(_ENCODER.encode(joined_text))

        logger.info("analyst: summarizing %s via LLM", Path(doc_id).name)
        if token_count <= _BATCH_TOKEN_BUDGET:
            summary = self.llm.complete(_SUMMARY_PROMPT.format(text=joined_text)).strip()
        else:
            batches = _split_into_batches(chunks, budget=_BATCH_TOKEN_BUDGET)
            partials = [
                self.llm.complete(
                    _SUMMARY_PROMPT.format(text="\n\n".join(c["text"] for c in batch))
                ).strip()
                for batch in batches
            ]
            summary = self.llm.complete(
                _MAP_REDUCE_PROMPT.format(summaries="\n\n".join(partials))
            ).strip()

        if self.config.summary_cache:
            self._write_summary_cache(cache_path, doc_id, summary)
        return summary

    def extract_key_points(self, doc_id: str, max_points: int = 10) -> list[str]:
        """Return up to ``max_points`` key points as a list of strings."""
        if max_points < 1:
            return []
        chunks = self.get_chunks(doc_id)
        if not chunks:
            return []
        joined_text = "\n\n".join(c["text"] for c in chunks)
        # For very long docs, summarise first to keep the prompt within budget.
        token_count = len(_ENCODER.encode(joined_text))
        if token_count > _BATCH_TOKEN_BUDGET:
            joined_text = self.summarize_document(doc_id)

        raw = self.llm.complete(_KEY_POINTS_PROMPT.format(text=joined_text, max_points=max_points))
        points = [line.strip().lstrip("-*•1234567890. \t") for line in raw.splitlines()]
        return [p for p in points if p][:max_points]

    # ------------------------------------------------------------------
    # Document comparison
    # ------------------------------------------------------------------

    def compare_documents(self, doc_id_a: str, doc_id_b: str) -> str:
        """Side-by-side comparison. Map-reduces if combined text exceeds budget."""
        meta_a = self.kg.get_doc_metadata(doc_id_a)
        meta_b = self.kg.get_doc_metadata(doc_id_b)
        title_a = str(meta_a.get("title", "") or doc_id_a)
        title_b = str(meta_b.get("title", "") or doc_id_b)

        text_a, text_b = self._compare_inputs(doc_id_a, doc_id_b)
        prompt = _COMPARE_PROMPT.format(
            title_a=title_a,
            title_b=title_b,
            text_a=text_a,
            text_b=text_b,
        )
        return self.llm.complete(prompt).strip()

    def _compare_inputs(self, doc_id_a: str, doc_id_b: str) -> tuple[str, str]:
        chunks_a = self.get_chunks(doc_id_a)
        chunks_b = self.get_chunks(doc_id_b)
        text_a = "\n\n".join(c["text"] for c in chunks_a)
        text_b = "\n\n".join(c["text"] for c in chunks_b)
        combined_tokens = len(_ENCODER.encode(text_a)) + len(_ENCODER.encode(text_b))
        if combined_tokens > _BATCH_TOKEN_BUDGET:
            # Too big — substitute summaries so the comparison fits.
            text_a = self.summarize_document(doc_id_a)
            text_b = self.summarize_document(doc_id_b)
        return text_a, text_b

    # ------------------------------------------------------------------
    # Scoped retrieval
    # ------------------------------------------------------------------

    def scoped_search(self, query: str, doc_id: str, top_k: int = 5) -> list[NodeWithScore]:
        """Vector search within one document only.

        Stages 0 and 2 are skipped (the retriever handles that when
        ``scope_doc_id`` is set). Results are truncated to ``top_k``.
        """
        if top_k < 1:
            return []
        results = self.retriever.retrieve(query, scope_doc_id=doc_id)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Raw chunk inspection
    # ------------------------------------------------------------------

    def get_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        """Return ``[{chunk_id, position, text, token_count, heading_path}]``.

        Ordered by ``CONTAINS.position``. Combines Kuzu (which owns the
        ordering) with ChromaDB metadata (which carries heading_path).
        """
        chunk_ids = self.kg.get_chunks_for_doc(doc_id)
        if not chunk_ids:
            return []
        chroma = self._chunks_coll.get(ids=chunk_ids, include=["documents", "metadatas"])
        by_id_text = dict(zip(chroma.get("ids", []), chroma.get("documents", []), strict=False))
        by_id_meta = dict(zip(chroma.get("ids", []), chroma.get("metadatas", []), strict=False))
        out: list[dict[str, Any]] = []
        for pos, cid in enumerate(chunk_ids):
            meta = by_id_meta.get(cid) or {}
            text = by_id_text.get(cid) or ""
            out.append(
                {
                    "chunk_id": cid,
                    "position": pos,
                    "text": text,
                    "token_count": len(_ENCODER.encode(text)),
                    "heading_path": str(meta.get("heading_path", "") or ""),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Document-level dense search (internal)
    # ------------------------------------------------------------------

    def find_document(
        self, query: str, top_k: int = 5, file_type: str | None = None
    ) -> list[dict[str, Any]]:
        """BM25 document search. Returns ranked ``[{doc_id, title, similarity_score}]``.

        Aggregates chunk-level BM25 scores to the document level (max score
        per doc), normalised to [0, 1]. Used by the retriever's Stage 0.1
        enrichment and by :class:`CorpusExplorer.corpus_search` (Section 10).
        Not exposed on the pipeline's public API.
        """
        ranked = self.retriever.find_document(query, top_k=top_k)
        if not ranked:
            return []
        meta_map: dict[str, str] = {}
        if file_type is not None:
            # Pull metadata for filtering. The retriever's _find_document
            # doesn't carry file_type back so we re-look-up via Chroma.
            ids = [doc_id for doc_id, _ in ranked]
            rows = self._docs_coll.get(ids=ids, include=["metadatas"])
            for doc_id, meta in zip(rows.get("ids", []), rows.get("metadatas", []), strict=False):
                if isinstance(meta, dict):
                    meta_map[doc_id] = str(meta.get("file_type", "") or "")
        out: list[dict[str, Any]] = []
        for doc_id, score in ranked:
            if file_type is not None and meta_map.get(doc_id) != file_type:
                continue
            kg_meta = self.kg.get_doc_metadata(doc_id)
            out.append(
                {
                    "doc_id": doc_id,
                    "title": str(kg_meta.get("title", "") or ""),
                    "similarity_score": score,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Summary cache helpers
    # ------------------------------------------------------------------

    def _summary_cache_path(self, doc_id: str) -> Path:
        digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16]
        return self.config.summary_path(self.corpus_name) / f"{digest}.json"

    def _write_summary_cache(self, path: Path, doc_id: str, summary: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump({"doc_id": doc_id, "summary": summary}, fh, indent=2)
        except OSError as exc:
            logger.warning("Could not write summary cache %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_into_batches(chunks: list[dict[str, Any]], *, budget: int) -> list[list[dict[str, Any]]]:
    """Greedily group chunks into batches whose joined text fits ``budget`` tokens."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for chunk in chunks:
        ctok = int(chunk.get("token_count", 0))
        if current_tokens + ctok > budget and current:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += ctok
    if current:
        batches.append(current)
    return batches
