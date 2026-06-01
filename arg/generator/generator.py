"""RAG answer generator.

Composes :class:`QueryProcessor`, :class:`HybridRetriever`, and the LLM into
a single ``generate(query)`` entry point. Returns an :class:`ARGResult` with
everything the FastAPI server needs to render an answer:

  * ``answer``       — the LLM's response, post-template.
  * ``sources``      — one :class:`SourceRef` per chunk used in context.
  * ``latency_ms``   — total wall-clock for the call.
  * ``enriched_doc_ids``  — doc_ids surfaced by Stage 0 enrichment.
  * ``rewritten_query``   — populated when QueryProcessor rewrote the query.
  * ``sub_queries``       — populated when the query was decomposed.

The LLM always sees the **raw** query at generation time. Rewriting and
HyDE shape retrieval only.

Empty-context contract
----------------------
If retrieval returns zero chunks across all sub-queries, ``generate`` short-
circuits and returns the spec-mandated fallback string
("The documentation does not cover this topic.") without calling the LLM.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.schema import NodeWithScore

from arg.config import ARGConfig
from arg.generator.query_processor import ProcessedQuery, QueryProcessor
from arg.llm import LLM
from arg.retriever.retriever import HybridRetriever

logger = logging.getLogger(__name__)


_NO_CONTEXT_ANSWER = "The documentation does not cover this topic."

# System prompt — verbatim from the Section 9 spec.
_SYSTEM_PROMPT = """\
You are Archivist, an expert assistant that answers questions using only
the provided documentation. Follow these rules for every answer:

SOURCING:
- Base your answer only on the context provided. Do not use outside knowledge.
- If the answer is not in the context, respond exactly:
  "The documentation does not cover this topic."
- Do not speculate, infer, or extrapolate beyond what the documents say.

CITATIONS:
- After each key claim, cite the source document title in parentheses.
  Example: "API keys expire after 90 days (Kraken API - Authentication)."

FORMAT - choose the format that matches the question type:
- Procedure / how-to: numbered steps. Each step on its own line.
- Reference / lookup (error codes, config values, limits): a compact table or
  bulleted list with key -> value pairs.
- Concept / explanation: 2-4 sentences of plain prose. No bullet points.
- Comparison: a two-column table (Feature | Doc A | Doc B).
- Code syntax: a fenced code block with the appropriate language tag.

LENGTH:
- Answer only what is asked. No preamble, no closing remarks.
- Procedures: include all steps. Do not truncate.
- Explanations: maximum 4 sentences unless the topic genuinely requires more.

Context:
{context_str}

Question: {query_str}"""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SourceRef:
    """One chunk used as RAG context, in a form safe to surface to the UI."""

    doc_id: str
    title: str
    chunk_id: str
    heading_path: str


@dataclass
class ARGResult:
    """The full answer record returned by :meth:`Generator.generate`."""

    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    latency_ms: int = 0
    enriched_doc_ids: list[str] = field(default_factory=list)
    rewritten_query: str | None = None
    sub_queries: list[str] | None = None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator:
    """Compose QueryProcessor + retriever + LLM into a single ``generate`` call."""

    def __init__(
        self,
        *,
        config: ARGConfig,
        llm: LLM,
        retriever: HybridRetriever,
        query_processor: QueryProcessor,
    ) -> None:
        self.config = config
        self.llm = llm
        self.retriever = retriever
        self.query_processor = query_processor

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        *,
        enrich: bool = True,
        scope_doc_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> ARGResult:
        """Non-streaming RAG answer."""
        start = time.perf_counter()
        processed = self.query_processor.process(query)
        chunks, enriched_doc_ids = self._retrieve_union(
            processed, enrich=enrich, scope_doc_id=scope_doc_id, filters=filters
        )
        if not chunks:
            return ARGResult(
                answer=_NO_CONTEXT_ANSWER,
                sources=[],
                latency_ms=_elapsed_ms(start),
                enriched_doc_ids=enriched_doc_ids,
                rewritten_query=processed.rewritten_query,
                sub_queries=processed.sub_queries,
            )

        prompt = self._build_prompt(processed.raw_query, chunks)
        answer = self.llm.complete(prompt).strip()
        logger.info("generator: query answered in %dms — %r", _elapsed_ms(start), query[:80])
        return ARGResult(
            answer=answer,
            sources=_to_source_refs(chunks),
            latency_ms=_elapsed_ms(start),
            enriched_doc_ids=enriched_doc_ids,
            rewritten_query=processed.rewritten_query,
            sub_queries=processed.sub_queries,
        )

    def stream_generate(
        self,
        query: str,
        *,
        enrich: bool = True,
        scope_doc_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield answer text in arrival order. Caller can re-run :meth:`generate`
        afterwards if it needs the full :class:`ARGResult` (sources, latency)."""
        processed = self.query_processor.process(query)
        chunks, _enriched = self._retrieve_union(
            processed, enrich=enrich, scope_doc_id=scope_doc_id, filters=filters
        )
        if not chunks:
            yield _NO_CONTEXT_ANSWER
            return
        prompt = self._build_prompt(processed.raw_query, chunks)
        yield from self.llm.stream_complete(prompt)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _retrieve_union(
        self,
        processed: ProcessedQuery,
        *,
        enrich: bool,
        scope_doc_id: str | None,
        filters: dict[str, Any] | None,
    ) -> tuple[list[NodeWithScore], list[str]]:
        """Run the retriever for every embedding query; union by chunk_id.

        Returns ``(chunks, enriched_doc_ids)``. The enriched doc-id list is
        populated from the union across sub-queries (Stage 0 may surface
        different docs per sub-query) — currently passed through for the
        web UI to expose; not used for re-ranking.
        """
        seen_chunk_ids: set[str] = set()
        union: list[NodeWithScore] = []
        enriched_doc_ids_set: set[str] = set()

        # When the processor didn't actually decompose, fall back to a single
        # retrieval round using the raw query — equivalent semantics, fewer
        # LLM calls when query_decompose=False.
        if not processed.embedding_queries:
            processed.embedding_queries = [processed.raw_query]

        queries = processed.embedding_queries

        def _retrieve_one(q: str) -> list[NodeWithScore]:
            return self.retriever.retrieve(
                q, enrich=enrich, scope_doc_id=scope_doc_id, filters=filters
            )

        max_workers = min(len(queries), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            all_results = list(pool.map(_retrieve_one, queries))

        for chunks in all_results:
            for node_with_score in chunks:
                cid = node_with_score.node.id_
                if cid in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(cid)
                union.append(node_with_score)
                doc_id = node_with_score.node.metadata.get("doc_id")
                if doc_id:
                    enriched_doc_ids_set.add(str(doc_id))
        return union, sorted(enriched_doc_ids_set)

    @staticmethod
    def _build_prompt(query: str, chunks: list[NodeWithScore]) -> str:
        context_str = "\n\n".join(
            _format_chunk_for_context(node_with_score) for node_with_score in chunks
        )
        return _SYSTEM_PROMPT.format(context_str=context_str, query_str=query)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_chunk_for_context(nws: NodeWithScore) -> str:
    """Render one chunk as a labelled block the LLM can cite from."""
    meta = nws.node.metadata or {}
    title = str(meta.get("title", "") or "")
    heading_path = str(meta.get("heading_path", "") or "")
    header = " > ".join(filter(None, [title, heading_path])) if heading_path else title
    body = nws.node.get_content() or ""
    if header:
        return f"[Source: {header}]\n{body}"
    return body


def _to_source_refs(chunks: list[NodeWithScore]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    for nws in chunks:
        meta = nws.node.metadata or {}
        refs.append(
            SourceRef(
                doc_id=str(meta.get("doc_id", "") or ""),
                title=str(meta.get("title", "") or ""),
                chunk_id=nws.node.id_,
                heading_path=str(meta.get("heading_path", "") or ""),
            )
        )
    return refs


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
