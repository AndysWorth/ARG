"""Pre-retrieval query processing: rewrite, decompose, optional HyDE.

The raw user query passes through three optional transformations before the
retriever sees an embedding:

  1. **Rewrite.** Conversational queries ("How do I log in?") become
     documentation-shaped ("Authentication methods and API key
     configuration"). Skipped when the query already looks technical
     (uppercase tokens, HTTP codes, version paths, function-call parens).
  2. **Decompose.** Compound questions split into independent sub-queries
     that the retriever runs in parallel; the resulting chunk sets union
     before the generator sees them.
  3. **HyDE (opt-in).** Replace the embedding text with a short
     hypothetical answer paragraph — semantically closer to real docs than
     the question itself.

The LLM always sees the *raw* query at generation time. Rewrites and HyDE
paragraphs are retrieval-only artefacts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from arg.config import ARGConfig
from arg.llm import LLM

logger = logging.getLogger(__name__)


# --- Heuristic: does the query already look technical? ----------------------
#
# Any of these matches and we skip rewriting — the query is already in the
# kind of shape the docs use.
_TECHNICAL_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[A-Z_]{3,}"),  # e.g. CONFIG_VAR, X-RATE-LIMIT
    re.compile(r"\b\d{3}\b"),  # HTTP status codes (200, 404, 503)
    re.compile(r"/v\d"),  # API version path (/v2)
    re.compile(r"\([^)]*\)"),  # function call: foo(), bar(x)
)


# --- Prompts ----------------------------------------------------------------

_REWRITE_PROMPT = """\
You are a technical documentation assistant. Rewrite the following user question
into precise technical language that would appear in software documentation.
Keep the same meaning. Output only the rewritten question, nothing else.

User question: {query}"""

_DECOMPOSE_PROMPT = """\
Does the following question contain multiple independent sub-questions that should
be researched separately?

Return a JSON object with a single key "sub_questions" whose value is an array of
strings. If the question should be split, list each independent sub-question as a
separate array element. If not, return a single-element array with the original
question unchanged.

Question: {query}"""

# JSON schema passed to Ollama's format parameter — grammar-based sampling
# guarantees the output matches this structure.
_DECOMPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["sub_questions"],
    "additionalProperties": False,
}

_HYDE_PROMPT = """\
Write a short paragraph (3-5 sentences) that would be a plausible answer to the
following question, as if it came from technical documentation. Be specific and
use technical language. Do not say "I don't know."

Question: {query}"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class ProcessedQuery:
    """Bundle the pre-retrieval transforms together for the generator to consume.

    Attributes
    ----------
    raw_query:
        The user's original input, unchanged. The LLM is shown this at
        generation time.
    rewritten_query:
        The technical-shape rewrite, or ``None`` if rewriting was skipped
        (either by config or by the technical-marker heuristic).
    sub_queries:
        ``None`` for a single retrieval; a list with > 1 entry when the LLM
        decomposed a compound question.
    embedding_queries:
        The actual strings the retriever embeds. One per sub_query when
        decomposed; replaced by the HyDE paragraph when ``hyde_enabled``.
    """

    raw_query: str
    rewritten_query: str | None = None
    sub_queries: list[str] | None = None
    embedding_queries: list[str] = field(default_factory=list)


class QueryProcessor:
    """Run the rewrite → decompose → (HyDE) pipeline on a raw query."""

    def __init__(self, config: ARGConfig, llm: LLM) -> None:
        self.config = config
        self.llm = llm

    def process(self, raw_query: str) -> ProcessedQuery:
        rewritten = self._maybe_rewrite(raw_query)
        retrieval_query = rewritten if rewritten is not None else raw_query

        sub_queries = self._maybe_decompose(retrieval_query)
        # If decomposition returned exactly one entry equal to the input, we
        # represent that as "no decomposition" so downstream code can branch.
        decomposed = sub_queries if len(sub_queries) > 1 else None

        if self.config.hyde_enabled:
            embedding_queries = [self._hyde_paragraph(sq) for sq in sub_queries]
        else:
            embedding_queries = list(sub_queries)

        return ProcessedQuery(
            raw_query=raw_query,
            rewritten_query=rewritten,
            sub_queries=decomposed,
            embedding_queries=embedding_queries,
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _maybe_rewrite(self, query: str) -> str | None:
        """Rewrite a conversational query into doc-shape. Returns None if skipped."""
        if not self.config.query_rewrite:
            return None
        if _looks_technical(query):
            logger.debug("query_processor: rewrite skipped (technical markers)")
            return None
        result = self.llm.complete(_REWRITE_PROMPT.format(query=query)).strip()
        return result or None

    def _maybe_decompose(self, query: str) -> list[str]:
        """Split a compound query into sub-queries. Returns ``[query]`` if no split."""
        if not self.config.query_decompose:
            return [query]
        raw = self.llm.complete_structured(
            _DECOMPOSE_PROMPT.format(query=query),
            schema=_DECOMPOSE_SCHEMA,
        )
        try:
            parts = [s.strip() for s in json.loads(raw).get("sub_questions", []) if s.strip()]
        except (json.JSONDecodeError, AttributeError):
            return [query]
        return parts if len(parts) > 1 else [query]

    def _hyde_paragraph(self, query: str) -> str:
        """Generate a hypothetical answer paragraph for embedding."""
        return self.llm.complete(_HYDE_PROMPT.format(query=query)).strip() or query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_technical(query: str) -> bool:
    """Spec Section 9 heuristic — fast regex sniff for technical content."""
    return any(p.search(query) for p in _TECHNICAL_MARKERS)
