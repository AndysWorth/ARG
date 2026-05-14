"""HTML and PDF text extractors.

The HTML side is implemented here. The PDF side is built in the second pass of
Section 5; this module currently exposes a stub that raises NotImplementedError
so callers fail loudly rather than silently.

The extractor's job is to convert one source file into a single `Document`
holding cleaned body text + metadata. URL resolution / cross-file traversal /
de-duplication is the crawler's job (`arg.crawler.crawler.crawl`) — extractors
just report the raw `<a href>` values they saw in the page; the crawler
normalises them into absolute paths inside `docs_root`.

Heading boundary sentinels
--------------------------
The chunker (Section 7) detects chunk boundaries by looking for `##H1##`,
`##H2##`, `##H3##` markers. The HTML extractor and (forthcoming) PDF extractor
both inject these markers at the same conceptual level so the chunker handles
the two file types through identical logic.

Locality
--------
This module reads files from disk only. It does not make HTTP requests; the
crawler enforces the same guarantee for cross-file traversal.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from arg.config import ARGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# CSS selectors for `<div>`-based navigation that older / template-based doc
# generators use instead of semantic <nav>/<aside>. `extract_html` strips these
# in addition to the semantic tags listed in Step 1.
DEFAULT_STRIP_SELECTORS: list[str] = [
    "div.sidebar",
    "div.nav",
    "div.navigation",
    "div.breadcrumb",
    "div.breadcrumbs",
    "div#nav",
    "div#sidebar",
    "div#header",
    "div#footer",
    "div.toc",
    "div#toc",
    "div.related",
    "div.sphinxsidebar",
    "div.rst-footer-buttons",
    "div.wy-nav-side",
    "div.md-sidebar",
]

# Standard separators for title-suffix stripping. The user's
# `config.title_separator` is added to this set at call time if not already
# present, so a custom value never silently disables the standard cases.
STANDARD_TITLE_SEPARATORS: tuple[str, ...] = (" | ", " — ", " - ", " :: ")

# Tags decomposed before text extraction (Step 1).
_SEMANTIC_BOILERPLATE_TAGS: tuple[str, ...] = (
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "iframe",
)

# Heading sentinels — must stay identical to the PDF extractor.
_HEADING_SENTINELS: dict[str, str] = {
    "h1": "##H1## ",
    "h2": "##H2## ",
    "h3": "##H3## ",
}


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """One indexable file's extracted content + metadata.

    Attributes
    ----------
    path:
        Absolute path to the source file on disk.
    content:
        Cleaned body text with heading sentinels injected (`##H1## …`).
    metadata:
        Dictionary with the keys promised by Section 5:
          * ``title`` (str)
          * ``page_description`` (str, "" if absent)
          * ``heading_path`` (str)
          * ``links_to`` (list[str]) — raw hrefs when produced by the extractor,
            absolute path strings after the crawler has normalised them.
          * ``file_type`` ("html" or "pdf")
          * ``code_blocks`` (list[str]) — full text of any truncated `<pre>` blocks.
    """

    path: Path
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTML extractor
# ---------------------------------------------------------------------------


def extract_html(path: Path, config: ARGConfig) -> Document:
    """Extract one HTML file into a `Document`."""
    raw_bytes = path.read_bytes()
    # lxml is mandatory per spec — it decodes entities correctly and is more
    # forgiving of malformed HTML than the stdlib parser.
    soup = BeautifulSoup(raw_bytes, features="lxml")

    _strip_invisible_and_boilerplate(soup, config)

    title = _extract_title(soup, config)
    page_description = _extract_page_description(soup)

    # Links must be collected BEFORE we discard the soup; raw href values are
    # returned and the crawler normalises them into absolute paths.
    links_to = _extract_links(soup)

    code_blocks_full, _ = _extract_and_truncate_code_blocks(soup, config)
    _convert_tables_to_markdown(soup)
    _inject_heading_sentinels(soup)

    body_text = soup.get_text(separator="\n")
    body_text = _normalise_whitespace(body_text)

    metadata: dict[str, Any] = {
        "title": title,
        "page_description": page_description,
        "heading_path": title,
        "links_to": links_to,
        "file_type": "html",
        "code_blocks": code_blocks_full,
    }
    return Document(path=path.resolve(), content=body_text, metadata=metadata)


# ---------------------------------------------------------------------------
# Step 1: strip invisible / boilerplate
# ---------------------------------------------------------------------------

_DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_VISIBILITY_HIDDEN_RE = re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE)


def _strip_invisible_and_boilerplate(soup: BeautifulSoup, config: ARGConfig) -> None:
    for tag_name in _SEMANTIC_BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(attrs={"style": True}):
        style = tag.get("style", "")
        if not isinstance(style, str):
            continue
        if _DISPLAY_NONE_RE.search(style) or _VISIBILITY_HIDDEN_RE.search(style):
            tag.decompose()

    strip_selectors = config.strip_selectors or DEFAULT_STRIP_SELECTORS
    for selector in strip_selectors:
        for tag in soup.select(selector):
            tag.decompose()


# ---------------------------------------------------------------------------
# Step 2: title
# ---------------------------------------------------------------------------


def _extract_title(soup: BeautifulSoup, config: ARGConfig) -> str:
    title_tag = soup.find("title")
    raw_title = title_tag.get_text(strip=True) if title_tag else ""

    if not raw_title:
        h1 = soup.find("h1")
        raw_title = h1.get_text(strip=True) if h1 else ""

    if not raw_title:
        return ""

    separators = list(STANDARD_TITLE_SEPARATORS)
    if config.title_separator and config.title_separator not in separators:
        separators.append(config.title_separator)

    # Take the first segment after splitting on whichever separator appears
    # earliest. Greedy left-most split: most-specific text wins.
    best = raw_title
    earliest = len(raw_title)
    for sep in separators:
        idx = raw_title.find(sep)
        if idx != -1 and idx < earliest:
            earliest = idx
            best = raw_title[:idx]
    return best.strip()


# ---------------------------------------------------------------------------
# Step 3: page description
# ---------------------------------------------------------------------------


def _extract_page_description(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and isinstance(meta, Tag):
        content = meta.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

    og = soup.find("meta", attrs={"property": "og:description"})
    if og and isinstance(og, Tag):
        content = og.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


def _extract_links(soup: BeautifulSoup) -> list[str]:
    hrefs: list[str] = []
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = a.get("href", "")
        if isinstance(href, str) and href:
            hrefs.append(href)
    return hrefs


# ---------------------------------------------------------------------------
# Step 4: tables → pipe-delimited Markdown
# ---------------------------------------------------------------------------


def _convert_tables_to_markdown(soup: BeautifulSoup) -> None:
    for table in list(soup.find_all("table")):
        markdown = _table_to_markdown(table)
        # Replace the <table> tag in-place with a NavigableString carrying the
        # markdown text so it lands at the same position in document flow.
        table.replace_with(NavigableString("\n" + markdown + "\n"))


def _table_to_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    header: list[str] | None = None

    for tr in table.find_all("tr"):
        if not isinstance(tr, Tag):
            continue
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        cell_texts = [_squash_inline_whitespace(c.get_text(" ", strip=True)) for c in cells]
        if header is None and all(isinstance(c, Tag) and c.name == "th" for c in cells):
            header = cell_texts
        else:
            rows.append(cell_texts)

    if header is None and rows:
        # No <th> row — treat the first row as header so the Markdown is well-formed.
        header = rows[0]
        rows = rows[1:]
    if header is None:
        return ""

    width = max(len(header), max((len(r) for r in rows), default=0))
    header = header + [""] * (width - len(header))
    norm_rows = [r + [""] * (width - len(r)) for r in rows]

    out: list[str] = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * width) + "|")
    for r in norm_rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _squash_inline_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Step 5: heading sentinels
# ---------------------------------------------------------------------------


def _inject_heading_sentinels(soup: BeautifulSoup) -> None:
    for tag_name, sentinel in _HEADING_SENTINELS.items():
        for tag in soup.find_all(tag_name):
            if not isinstance(tag, Tag):
                continue
            text = tag.get_text(" ", strip=True)
            tag.clear()
            tag.append(NavigableString(sentinel + text))


# ---------------------------------------------------------------------------
# Step 6: code blocks
# ---------------------------------------------------------------------------


def _extract_and_truncate_code_blocks(
    soup: BeautifulSoup, config: ARGConfig
) -> tuple[list[str], list[str]]:
    """Return (full_block_texts, truncated_in_place_blocks)."""
    full_texts: list[str] = []
    truncated_in_place: list[str] = []
    limit = config.max_code_block_tokens

    for pre in soup.find_all("pre"):
        if not isinstance(pre, Tag):
            continue
        text = pre.get_text()
        full_texts.append(text)

        tokens = text.split()
        if len(tokens) > limit:
            truncated = " ".join(tokens[:limit]) + " [... truncated ...]"
            pre.clear()
            pre.append(NavigableString(truncated))
            truncated_in_place.append(text)
    return full_texts, truncated_in_place


# ---------------------------------------------------------------------------
# Step 8: whitespace normalisation
# ---------------------------------------------------------------------------

_TRIPLE_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ ]{2,}")


def _normalise_whitespace(text: str) -> str:
    # Defensive: BeautifulSoup with lxml usually decodes entities, but if the
    # raw bytes contained a numeric reference that the parser left intact,
    # unescape here so chunk text never contains `&amp;` etc.
    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\t", " ")
    text = _TRIPLE_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PDF extractor — stub, implemented in pass 2 of Section 5
# ---------------------------------------------------------------------------


def extract_pdf(
    path: Path, config: ARGConfig
) -> Iterator[tuple[int, str, dict[str, Any]]]:  # pragma: no cover - second pass
    """Yield ``(page_number, page_text, page_metadata)`` for each PDF page.

    Pass-1 stub: raises NotImplementedError. The crawler currently records PDF
    file paths as link-graph edges but does not invoke this function.
    """
    raise NotImplementedError(
        "PDF extraction is implemented in pass 2 of Section 5; "
        "the HTML pass records PDFs as graph edges only."
    )
