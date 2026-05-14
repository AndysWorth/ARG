"""Semantic chunker: heading-aware section splits + token-based sliding window.

The chunker is the bridge from a `Document` (one HTML page or one whole PDF) to
the per-chunk records that get embedded by ChromaDB and recorded in Kuzu.

Strategy (Section 7 spec)
-------------------------
1. **Heading-aware section splits.** Both extractors inject ``##H1## ``,
   ``##H2## ``, ``##H3## `` sentinels at the start of heading lines. The
   chunker walks the text line-by-line tracking the live heading hierarchy and
   uses these markers as section boundaries. H4-H6 are plain text and do not
   force a split.
2. **Sliding window within a section.** Each section is tokenised with
   tiktoken (``cl100k_base`` — used as a stable, model-agnostic stand-in for
   real token counts) and chopped into windows of ``config.chunk_size`` tokens
   with ``config.chunk_overlap`` tokens carried into the next window.
3. **Contextual enrichment.** When ``config.contextual_enrichment`` is True
   (the default), each chunk's ``embedding_text`` is prefixed with
   ``"{title} > {heading_path}: "`` so the embedder sees the chunk in its
   semantic context. The ``chunk_text`` field stays clean — it is what the
   LLM eventually receives.
4. **Per-chunk metadata.** Every chunk carries ``doc_id``, ``title``,
   ``page_description``, ``heading_path``, ``position``, ``file_type``,
   ``page_number`` (PDFs only — derived from ``Document.metadata['page_offsets']``),
   ``has_table``, ``has_code``.

The chunker does NOT touch ChromaDB or Kuzu — it returns plain data. The
indexer (`arg.indexer.indexer`) is what writes the chunks out to both stores.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from arg.config import ARGConfig
from arg.crawler.extractors import Document
from arg.graph import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Heading sentinel patterns. The matched group captures the level (1/2/3) and
# the heading text. Anchored to the start of a line.
_HEADING_RE = re.compile(r"^##H([123])## (.*)$", re.MULTILINE)

# Code-fence detection inside chunk text. Both Markdown fences and raw code
# blocks from the HTML extractor's <pre> output will trip this — the test for
# `has_code` cross-references against the original document's ``code_blocks``
# metadata for robustness.
_FENCED_CODE_RE = re.compile(r"```")

# Pipe-delimited Markdown table — match the alignment row, the most specific
# part of a table.
_MARKDOWN_TABLE_RE = re.compile(r"^\s*\|\s*-{3,}", re.MULTILINE)

# Tokeniser. cl100k_base is OpenAI's GPT-4 BPE; we use it as a stable token
# counter, not for any model-specific behaviour. nomic-embed-text uses a
# different tokeniser, but for chunk-size purposes the difference is
# negligible.
_ENCODER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# ChunkedSection
# ---------------------------------------------------------------------------


@dataclass
class ChunkedSection:
    """One chunk produced by the chunker — ready for the indexer.

    Attributes
    ----------
    chunk:
        The :class:`arg.graph.Chunk` written to Kuzu.
    chunk_text:
        Raw chunk content. This is what the LLM receives as retrieved context.
    embedding_text:
        Either the same as ``chunk_text`` (when contextual enrichment is off)
        or ``chunk_text`` prefixed with the contextual header. This is what
        the embedder sees.
    metadata:
        Per-chunk metadata dict written into ChromaDB alongside the embedding.
    """

    chunk: Chunk
    chunk_text: str
    embedding_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_document(doc: Document, config: ARGConfig) -> list[ChunkedSection]:
    """Split a Document into ``ChunkedSection`` objects.

    Section boundaries are derived from the ``##H1##``/``##H2##``/``##H3##``
    sentinels the extractors inject. PDF page numbers are recovered from
    ``doc.metadata['page_offsets']`` when present.
    """
    doc_id = str(doc.path.resolve())
    title = str(doc.metadata.get("title", "") or "")
    page_description = str(doc.metadata.get("page_description", "") or "")
    file_type = str(doc.metadata.get("file_type", "html") or "html")
    code_blocks: list[str] = list(doc.metadata.get("code_blocks") or [])
    # `page_offsets[i]` = char offset at which page i+1 begins in ``doc.content``.
    page_offsets: list[int] = list(doc.metadata.get("page_offsets") or [])

    sections = _split_into_sections(doc.content, title)
    out: list[ChunkedSection] = []
    position = 0
    for section in sections:
        for window_text in _sliding_window(section.text, config):
            chunk_id = f"{doc_id}::chunk::{position}"
            cleaned_text = window_text.strip()
            if not cleaned_text:
                continue

            page_number = _page_for_offset(
                section.start_offset + section.text.find(window_text),
                page_offsets,
            )

            metadata = {
                "doc_id": doc_id,
                "title": title,
                "page_description": page_description,
                "heading_path": section.heading_path,
                "position": position,
                "file_type": file_type,
                "page_number": page_number,
                "has_table": bool(_MARKDOWN_TABLE_RE.search(cleaned_text)),
                "has_code": _detect_code(cleaned_text, code_blocks),
            }
            embedding_text = _build_embedding_text(
                cleaned_text,
                title=title,
                heading_path=section.heading_path,
                enrich=config.contextual_enrichment,
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                text=cleaned_text,
                token_count=_token_count(cleaned_text),
            )
            out.append(
                ChunkedSection(
                    chunk=chunk,
                    chunk_text=cleaned_text,
                    embedding_text=embedding_text,
                    metadata=metadata,
                )
            )
            position += 1
    return out


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    """One logical section between heading sentinels."""

    text: str
    heading_path: str
    start_offset: int  # char offset of this section in the original content


def _split_into_sections(content: str, title: str) -> list[_Section]:
    """Walk content line-by-line, accumulating sections at H1/H2/H3 boundaries.

    The leading title-less prose (if any — uncommon when the extractor sets a
    title) becomes a section with heading_path = title.
    """
    if not content:
        return []

    # Build the live heading hierarchy as we walk.
    h1, h2, h3 = "", "", ""

    def current_path() -> str:
        parts = [title] if title else []
        if h1:
            parts.append(h1)
        if h2:
            parts.append(h2)
        if h3:
            parts.append(h3)
        return " > ".join(parts) if parts else ""

    sections: list[_Section] = []
    buf: list[str] = []
    section_start = 0
    current_heading = current_path()
    char_cursor = 0

    def flush(end_offset: int) -> None:
        nonlocal buf
        if buf:
            text = "\n".join(buf).strip()
            if text:
                sections.append(
                    _Section(text=text, heading_path=current_heading, start_offset=section_start)
                )
            buf = []
        # update section_start for the NEXT section
        # (caller updates section_start = end_offset)

    for line in content.split("\n"):
        line_len = len(line) + 1  # +1 for the \n we split on
        m = _HEADING_RE.match(line)
        if m is None:
            buf.append(line)
            char_cursor += line_len
            continue

        # Heading line — flush the current section first, then start a new one.
        flush(char_cursor)
        section_start = char_cursor
        level = int(m.group(1))
        heading_text = m.group(2).strip()
        if level == 1:
            h1, h2, h3 = heading_text, "", ""
        elif level == 2:
            h2, h3 = heading_text, ""
        else:
            h3 = heading_text
        current_heading = current_path()
        # Include the heading line itself as the first line of the new section.
        buf.append(heading_text)
        char_cursor += line_len

    flush(char_cursor)
    return sections


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def _sliding_window(text: str, config: ARGConfig) -> list[str]:
    """Token-bucket the section text into ``chunk_size`` windows with overlap."""
    if not text:
        return []
    tokens = _ENCODER.encode(text)
    if len(tokens) <= config.chunk_size:
        return [text]

    step = config.chunk_size - config.chunk_overlap
    if step <= 0:
        # __post_init__ already validates 0 <= overlap < chunk_size, so this
        # should be unreachable — keep as a defensive fallback.
        step = config.chunk_size
    windows: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + config.chunk_size, len(tokens))
        windows.append(_ENCODER.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return windows


def _token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Contextual enrichment
# ---------------------------------------------------------------------------


def _build_embedding_text(chunk_text: str, *, title: str, heading_path: str, enrich: bool) -> str:
    if not enrich:
        return chunk_text
    # heading_path already includes the title (see `_split_into_sections`), so
    # the prefix is just "{heading_path}: {chunk_text}". If there is no
    # heading_path (rare — title-less doc), fall back to bare chunk_text.
    prefix = heading_path if heading_path else title
    if not prefix:
        return chunk_text
    return f"{prefix}: {chunk_text}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page_for_offset(offset: int, page_offsets: list[int]) -> int | None:
    """Return the 1-based page number whose region of ``content`` contains ``offset``.

    Returns ``None`` for HTML documents (which have no ``page_offsets``).
    """
    if not page_offsets:
        return None
    # page_offsets is the START offset of each page. The page containing offset
    # X is the LAST page whose start is <= X.
    idx = bisect_right(page_offsets, offset) - 1
    if idx < 0:
        idx = 0
    return idx + 1  # 1-based page number


def _detect_code(chunk_text: str, code_blocks: list[str]) -> bool:
    """``has_code`` heuristic.

    Returns True if the chunk contains a Markdown fence OR overlaps with any
    of the document's recorded code blocks (HTML extractor records full
    ``<pre>`` content in ``metadata['code_blocks']``).
    """
    if _FENCED_CODE_RE.search(chunk_text):
        return True
    if not code_blocks:
        return False
    # Cheap overlap heuristic: check whether any code block substring of >= 20
    # chars appears in the chunk. The threshold avoids false positives from
    # tiny snippets like "()" that show up everywhere.
    for block in code_blocks:
        stripped = block.strip()
        if len(stripped) < 20:
            continue
        if stripped[:80] in chunk_text:
            return True
    return False
