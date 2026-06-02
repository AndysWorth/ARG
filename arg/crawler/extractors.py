"""HTML and PDF text extractors.

Both the HTML and PDF sides land here. The extractor's job is to convert one
source file into a single `Document` holding cleaned body text + metadata. URL
resolution / cross-file traversal / de-duplication is the crawler's job
(`arg.crawler.crawler.crawl`) — extractors just report the raw `<a href>`
values they saw in the page; the crawler normalises them into absolute paths
inside `docs_root`.

PDF extraction is broken into three entry points:
  * `extract_pdf_metadata(path, config)`  — doc-level pre-flight only (title,
    description, keywords, links, page count). Returns `None` if encrypted or
    corrupt.
  * `extract_pdf(path, config)`           — generator yielding
    `(page_number, page_text, page_metadata)` tuples for the indexer.
  * `extract_pdf_to_document(path, config)` — convenience wrapper that
    consumes the generator and produces a single `Document`.

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

# Implements: docs/spec/section-05-crawler.md
"""

from __future__ import annotations

import html
import json
import logging
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf as fitz
from bs4 import BeautifulSoup, NavigableString, Tag

from arg.config import ARGConfig

logger = logging.getLogger(__name__)

# pdfminer and pdfplumber emit spurious WARNING-level lines for normal PDF
# quirks (missing FontBBox, non-standard color spaces, inline images). These
# are not actionable and drown out real warnings at ~288 lines per corpus run.
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


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

    # Links must be collected BEFORE stripping navigation/boilerplate elements,
    # because <nav> and sidebar divs are removed by the strip pass and would
    # otherwise silently drop cross-document links from index pages.
    links_to = _extract_links(soup)

    _strip_invisible_and_boilerplate(soup, config)

    title = _extract_title(soup, config)
    page_description = _extract_page_description(soup)

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
# Plain-text extractor (Feature 0001)
# ---------------------------------------------------------------------------


def extract_text(path: Path, config: ARGConfig) -> Document:
    """Extract one ``.txt`` / ``.md`` / ``.markdown`` file into a `Document`.

    Decoding: try UTF-8 (with BOM stripping via ``utf-8-sig``); fall back to
    latin-1 on ``UnicodeDecodeError``. latin-1 is byte-clean so the fallback
    never raises. Mojibake on rare encodings (CP932 / GB18030 / etc.) is
    acceptable for v1 — operators with those corpora can convert upstream.

    Title: the filename stem (no ``<title>`` to parse). No heading-sentinel
    injection is performed — the chunker already handles the "no headings"
    case by emitting one section covering the whole document. Markdown
    semantic structure (``# H1`` / ``## H2``) is intentionally NOT parsed in
    this feature; see docs/features/0001-plain-text-indexing.md for the
    deferral rationale.
    """
    _ = config  # currently unused; kept for symmetry with the other extractors
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        # latin-1 maps every byte to a valid code point; this branch
        # never raises further.
        text = raw_bytes.decode("latin-1")
    text = _normalise_whitespace(text)

    title = path.stem
    metadata: dict[str, Any] = {
        "title": title,
        "page_description": "",
        "heading_path": title,
        "links_to": [],
        "file_type": "text",
        "code_blocks": [],
    }
    return Document(path=path.resolve(), content=text, metadata=metadata)


# ---------------------------------------------------------------------------
# Step 1: strip invisible / boilerplate
# ---------------------------------------------------------------------------

_DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_VISIBILITY_HIDDEN_RE = re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE)


def _strip_invisible_and_boilerplate(soup: BeautifulSoup, config: ARGConfig) -> None:
    for tag_name in _SEMANTIC_BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            if isinstance(tag, Tag):
                tag.decompose()

    # Malformed HTML can surface nodes from ``find_all(attrs=...)`` where
    # ``.attrs is None`` — calling ``.get()`` on them raises AttributeError
    # mid-iteration. Guard with isinstance + attrs presence.
    for tag in soup.find_all(attrs={"style": True}):
        if not isinstance(tag, Tag) or tag.attrs is None:
            continue
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
# PDF extractor
# ---------------------------------------------------------------------------
#
# Per the Section 5 spec, PDF extraction is a multi-stage per-page pipeline:
#
#   Step 0 — document-level pre-flight (encryption, form fields, title,
#            subject/keywords, running header/footer detection across pages).
#   Step 1 — per-page three-stage extraction:
#              1a pdfplumber (primary)  →  1b pymupdf text  →  1c pymupdf OCR
#            Stage choice is per-page, not per-document: a single PDF can mix
#            native-text and scanned pages.
#   Step 2 — font-based heading detection via pymupdf's get_text("dict")
#            structure; injects ##H1##/##H2##/##H3## sentinels identical to
#            the HTML extractor so the chunker uses one boundary algorithm
#            for both formats.
#   Step 3 — text cleaning (soft-hyphen rejoin, ligatures, typographic
#            characters, Unicode NFC, whitespace normalisation).
#   Step 4 — page metadata assembly.
#   Step 5 — incremental yield (generator) so the indexer can checkpoint.
#
# `extract_pdf_metadata` is the cheap path used by the crawler to build a
# `Document` for a PDF (title + page_description + keywords + links).
# `extract_pdf` is the full streaming generator used by the indexer.
# `extract_pdf_to_document` calls both and assembles a `Document` with the
# concatenated page text — convenient for callers that don't need streaming.

# Title strings the spec wants treated as "no title" (Step 0c.1). Match anchored
# at start, case-insensitive. Anything that begins with these patterns falls
# back to the largest-font line on page 1, then to the filename stem.
_PDF_TITLE_TEMP_RE = re.compile(
    r"^\s*("
    r"Microsoft Word|Microsoft PowerPoint|Microsoft Excel|"
    r"Untitled|"
    r"document\d*|Presentation\d*|Worksheet\d*"
    r")",
    re.IGNORECASE,
)

# Ligatures decoded per spec Step 3b.
_PDF_LIGATURES: dict[str, str] = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬆ": "st",
    "Ĳ": "IJ",
    "ĳ": "ij",
}

# Typographic characters normalised per spec Step 3b. Keys use explicit
# \uXXXX escapes so the source file stays free of homoglyph-flagged literals.
_PDF_TYPOGRAPHIC: dict[str, str] = {
    "\u2014": " \u2014 ",  # em dash, surrounded by spaces for tokenisation
    "\u2013": " - ",  # en dash -> ASCII hyphen
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u2026": "...",  # ellipsis
    "\u00ad": "",  # soft hyphen removed entirely
}

# pymupdf span flags bit-positions.
_FITZ_FLAG_BOLD = 1 << 4  # value 16

# Defaults for running-line detection (Step 0e).
_RUNNING_LINE_Y_TOLERANCE_PX = 3.0
_RUNNING_LINE_MIN_PAGES = 3

# Pages whose OCR output is below this character count are flagged as likely
# blank or image-only (distinct from ocr_char_threshold, which controls whether
# OCR *fires*; this threshold flags low-quality OCR *results*).
_OCR_LOW_QUALITY_CHARS: int = 25


# ---- helper: title resolution -----------------------------------------------


def _resolve_pdf_title(
    pdf_metadata: dict[str, str], largest_first_page_line: str | None, filename_stem: str
) -> str:
    """Apply the three-step rule from Step 0c: /Title → largest line → filename stem."""
    raw = (pdf_metadata.get("title") or "").strip()
    if raw and not _PDF_TITLE_TEMP_RE.match(raw):
        return raw
    if largest_first_page_line:
        return largest_first_page_line.strip()
    return filename_stem


# ---- helper: text cleaning --------------------------------------------------


def _rejoin_soft_hyphens(text: str) -> str:
    """Rejoin words split across line wraps when the next line is lowercase."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and line.endswith("-"):
            nxt = lines[i + 1]
            if nxt and nxt[:1].islower():
                out.append(line[:-1] + nxt)
                i += 2
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _clean_pdf_text(text: str) -> str:
    """Apply spec Step 3 (a, b, c) to one page's text."""
    text = _rejoin_soft_hyphens(text)
    text = unicodedata.normalize("NFC", text)
    for lig, rep in _PDF_LIGATURES.items():
        text = text.replace(lig, rep)
    for char, rep in _PDF_TYPOGRAPHIC.items():
        text = text.replace(char, rep)
    text = text.replace("\u00a0", " ")
    text = text.replace("\t", " ")
    text = _TRIPLE_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ---- helper: running header / footer detection ------------------------------


def _detect_running_lines(
    pages: list[list[tuple[float, str]]],
    min_pages: int = _RUNNING_LINE_MIN_PAGES,
    y_tolerance: float = _RUNNING_LINE_Y_TOLERANCE_PX,
) -> set[tuple[float, str]]:
    """Return `(y_bucket, text)` pairs that appear on `>= min_pages` pages.

    `pages` is per-page list of `(y_position, text)` tuples. Each page
    contributes only one occurrence of any given (y_bucket, text) pair so a
    duplicate line on the same page doesn't tip the threshold.
    """
    counter: Counter[tuple[float, str]] = Counter()
    for page in pages:
        seen_this_page: set[tuple[float, str]] = set()
        for y, text in page:
            stripped = text.strip()
            if not stripped:
                continue
            bucket = round(y / y_tolerance) * y_tolerance
            seen_this_page.add((bucket, stripped))
        for key in seen_this_page:
            counter[key] += 1
    return {k for k, v in counter.items() if v >= min_pages}


# ---- helper: font-based heading sentinels -----------------------------------


def _heading_sentinel_map(fitz_page: fitz.Page) -> dict[str, str]:
    """Walk a pymupdf page and return ``{line_text: sentinel}`` for headings.

    Heuristics (Section 5 Step 2):
      * size > 1.5x body-median  -> ``##H1## ``
      * size > 1.2x body-median and <= 1.5x  -> ``##H2## ``
      * bold AND within 10% of body-median size  -> ``##H3## ``
    """
    lines: list[tuple[str, float, bool]] = []  # (text, max_span_size, any_bold)
    for block in fitz_page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            size = max(s["size"] for s in spans)
            bold = any((s["flags"] & _FITZ_FLAG_BOLD) for s in spans)
            lines.append((text, size, bold))

    if not lines:
        return {}

    body_median = statistics.median(s for _, s, _ in lines)
    if body_median <= 0:
        return {}

    sentinels: dict[str, str] = {}
    for text, size, bold in lines:
        if size > 1.5 * body_median:
            sentinels[text] = "##H1## "
        elif size > 1.2 * body_median:
            sentinels[text] = "##H2## "
        elif bold and abs(size - body_median) / body_median <= 0.1:
            sentinels[text] = "##H3## "
    return sentinels


def _inject_pdf_heading_sentinels(body_text: str, heading_map: dict[str, str]) -> str:
    """Prepend sentinels to lines that match a heading-text key."""
    if not heading_map:
        return body_text
    out: list[str] = []
    for line in body_text.split("\n"):
        stripped = line.strip()
        prefix = heading_map.get(stripped, "")
        out.append(prefix + line if prefix else line)
    return "\n".join(out)


def _largest_first_page_line(fitz_page: fitz.Page) -> str | None:
    """Return the line on the page with the largest font size (for title fallback)."""
    best_size = -1.0
    best_text: str | None = None
    for block in fitz_page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            size = max(s["size"] for s in spans)
            if size > best_size:
                best_size = size
                best_text = text
    return best_text


# ---- helper: sidecar config -------------------------------------------------


def _read_pdf_sidecar(pdf_path: Path) -> dict[str, Any]:
    """Read ``{pdf_path}.argconfig`` JSON if present; otherwise return ``{}``."""
    sidecar = pdf_path.with_suffix(pdf_path.suffix + ".argconfig")
    if not sidecar.is_file():
        return {}
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("PDF sidecar %s is not a JSON object; ignoring", sidecar)
            return {}
        return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read PDF sidecar %s: %s", sidecar, exc)
        return {}


# ---- helper: pdfplumber + table splicing ------------------------------------


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Convert pdfplumber's rows-of-cells output to pipe-Markdown."""
    cleaned: list[list[str]] = []
    for row in rows:
        cleaned.append([_squash_inline_whitespace(c or "") for c in row])
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    header = cleaned[0]
    body = cleaned[1:]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * width) + "|"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _group_chars_to_lines(
    chars: list[dict[str, Any]], y_tolerance: float = 3.0
) -> list[tuple[float, str]]:
    """Group pdfplumber `chars` into lines by y-proximity. Returns [(y_top, text)]."""
    if not chars:
        return []
    sorted_chars = sorted(chars, key=lambda c: (c["top"], c["x0"]))
    lines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_y: float | None = None
    for c in sorted_chars:
        if current_y is None or abs(c["top"] - current_y) <= y_tolerance:
            current.append(c)
            current_y = c["top"] if current_y is None else current_y
        else:
            lines.append(current)
            current = [c]
            current_y = c["top"]
    if current:
        lines.append(current)

    out: list[tuple[float, str]] = []
    for line in lines:
        line.sort(key=lambda c: c["x0"])
        text = ""
        prev_x1: float | None = None
        for c in line:
            if prev_x1 is not None and c["x0"] - prev_x1 > 1.5:
                text += " "
            text += c["text"]
            prev_x1 = c["x1"]
        out.append((line[0]["top"], _squash_inline_whitespace(text)))
    return out


def _char_in_any_bbox(c: dict[str, Any], bboxes: list[tuple[float, float, float, float]]) -> bool:
    x0, top, x1, bottom = c["x0"], c["top"], c["x1"], c["bottom"]
    for bx0, by0, bx1, by1 in bboxes:
        if x0 >= bx0 and x1 <= bx1 and top >= by0 and bottom <= by1:
            return True
    return False


def _pdfplumber_extract_page(page: Any) -> tuple[list[tuple[float, str]], list[str], int]:
    """Stage 1a — pdfplumber. Returns (lines_with_y, markdown_tables, total_chars)."""
    tables = list(page.find_tables())
    bboxes = [t.bbox for t in tables]
    markdown_tables: list[str] = []
    items: list[tuple[float, str]] = []
    for t in tables:
        try:
            rows = t.extract()
        except Exception:
            rows = []
        if rows:
            md = _rows_to_markdown(rows)
            if md:
                markdown_tables.append(md)
                items.append((t.bbox[1], md))

    chars_outside_tables = [c for c in page.chars if not _char_in_any_bbox(c, bboxes)]
    body_lines = _group_chars_to_lines(chars_outside_tables)
    items.extend(body_lines)
    items.sort(key=lambda x: x[0])

    body_text = "\n".join(text for _, text in items if text)
    total_chars = len(body_text)
    return body_lines, markdown_tables, total_chars


# ---- helper: pymupdf text + OCR stages --------------------------------------


def _pymupdf_extract_text(fitz_page: fitz.Page) -> tuple[list[tuple[float, str]], int]:
    """Stage 1b — pymupdf text. Returns (lines_with_y, total_chars)."""
    lines: list[tuple[float, str]] = []
    for block in fitz_page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            y = line["bbox"][1]
            lines.append((y, text))
    total_chars = sum(len(text) for _, text in lines)
    return lines, total_chars


def _pymupdf_extract_ocr(fitz_page: fitz.Page) -> tuple[list[tuple[float, str]], int]:
    """Stage 1c — pymupdf OCR. Returns (lines_with_y, total_chars).

    Wrapped in try/except: if Tesseract isn't installed, OCR fails and we
    return empty so the caller can fall through gracefully rather than crash
    the whole indexing run.
    """
    try:
        tp = fitz_page.get_textpage_ocr(full=True)
        text = tp.extractText()
    except Exception as exc:
        logger.warning(
            "OCR failed on page %s (tesseract not installed?): %s",
            fitz_page.number,
            exc,
        )
        return [], 0
    # OCR doesn't give us per-line y positions cheaply — synthesise rising y
    # values so the running-line detector can still see them in y order.
    lines: list[tuple[float, str]] = []
    for i, line in enumerate(text.split("\n")):
        if line.strip():
            lines.append((float(i), line))
    total_chars = sum(len(text) for _, text in lines)
    return lines, total_chars


# ---- main entry points ------------------------------------------------------


def _open_pdf(path: Path) -> fitz.Document | None:
    try:
        doc = fitz.open(path)
    except (fitz.FileDataError, fitz.EmptyFileError, RuntimeError, FileNotFoundError) as exc:
        logger.warning("Skipping unreadable PDF: %s — %s", path, exc)
        return None
    except Exception as exc:
        logger.warning("Skipping unreadable PDF: %s — %s", path, exc)
        return None
    if doc.is_encrypted:
        logger.warning("Skipping encrypted PDF: %s", path)
        doc.close()
        return None
    return doc


def extract_pdf_metadata(path: Path, config: ARGConfig) -> dict[str, Any] | None:
    """Return doc-level PDF metadata: title, page_description, keywords, links_to.

    Returns ``None`` if the PDF is encrypted or unreadable.
    """
    doc = _open_pdf(path)
    if doc is None:
        return None
    try:
        meta = dict(doc.metadata or {})

        # Largest first-page line as a fallback title source.
        largest = _largest_first_page_line(doc[0]) if doc.page_count > 0 else None
        title = _resolve_pdf_title(meta, largest, path.stem)

        page_description = (meta.get("subject") or "").strip()
        keywords = (meta.get("keywords") or "").strip()

        # Internal links: pymupdf gives each annotation a kind. URI links go
        # straight into the link graph; named-destination links (kind == LINK_GOTO)
        # stay inside the PDF and are not useful for cross-doc traversal.
        # Iterate by index — pymupdf's type stubs do not expose Document.__iter__,
        # so `for page in doc:` upsets CI mypy even though it works at runtime.
        links_to: list[str] = []
        for page_index in range(doc.page_count):
            page = doc[page_index]
            for link in page.get_links():
                uri = link.get("uri")
                if uri:
                    links_to.append(uri)

        # Form-field detection — warn but continue.
        if doc.is_form_pdf:
            logger.warning(
                "PDF %s contains AcroForm fields — extracted text may be incomplete",
                path,
            )

        return {
            "title": title,
            "page_description": page_description,
            "keywords": keywords,
            "links_to": links_to,
            "page_count": doc.page_count,
            "is_form_pdf": bool(doc.is_form_pdf),
        }
    finally:
        doc.close()


def extract_pdf(path: Path, config: ARGConfig) -> Iterator[tuple[int, str, dict[str, Any]]]:
    """Stream per-page extraction results.

    Yields ``(page_number, page_text, page_metadata)`` for each page where
    ``page_metadata`` has keys ``tables`` (list[str]), ``ocr_used`` (bool),
    ``char_count`` (int), ``heading_sentinels`` (list[str]).

    Skips the whole document and yields nothing if the PDF is encrypted or
    unreadable (a warning has already been logged).

    Per-document sidecar overrides: a JSON file
    ``{path}.argconfig`` next to the PDF may override ``pdf_layout_analysis``
    for this document only.
    """
    doc = _open_pdf(path)
    if doc is None:
        return
    try:
        # Sidecar overrides apply only inside this generator.
        sidecar = _read_pdf_sidecar(path)
        layout_analysis = bool(sidecar.get("pdf_layout_analysis", config.pdf_layout_analysis))

        is_form = bool(doc.is_form_pdf)

        # Pre-flight text-density check: sample up to 5 pages with pymupdf (fast,
        # C-level) to measure average extractable chars per page. Image-dominated
        # PDFs (scanned maps, photo archives) produce near-zero text via pdfplumber
        # while taking hours to process; skipping pdfplumber for them avoids the
        # worst-case hangs. OCR-enabled runs still process these files via Stage 1c.
        sample_pages = min(doc.page_count, 5)
        sample_chars = sum(len(doc[i].get_text("text")) for i in range(sample_pages))
        avg_chars = sample_chars / sample_pages if sample_pages else 0
        is_image_dominated = avg_chars < config.pdf_min_chars_per_page

        if is_image_dominated:
            logger.info(
                "PDF %s has avg %.1f chars/page (threshold %d) — skipping pdfplumber",
                path,
                avg_chars,
                config.pdf_min_chars_per_page,
            )

        # Single-pass pdfplumber: buffer (lines, tables, nchars) for every page,
        # run running-header detection on the full buffer, then process pages
        # using the buffered data. Eliminates the previous second open() call.
        # AcroForm PDFs skip pdfplumber entirely — it is slow and produces
        # incomplete text for form fields; pymupdf's text layer is faster and
        # more complete for this file type.
        PageBuf = tuple[list[tuple[float, str]], list[str], int]
        page_buffer: list[PageBuf] = []
        if not is_form and not is_image_dominated:
            with pdfplumber.open(str(path)) as pdf:
                for pp in pdf.pages:
                    page_buffer.append(_pdfplumber_extract_page(pp))
        else:
            # AcroForm or image-dominated: skip pdfplumber; pymupdf (Stages 1b/1c)
            # will fill each page in the per-page loop below.
            page_buffer = [([], [], 0)] * doc.page_count

        all_pages_lines = [lines for lines, _, _ in page_buffer]
        running_lines = _detect_running_lines(all_pages_lines)

        # Step 1 — per-page extraction using buffered pdfplumber data + fitz.
        for page_index, (lines, markdown_tables, total_chars) in enumerate(page_buffer):
            fitz_page = doc[page_index]
            ocr_used = False

            # 1a pdfplumber data already buffered above.
            # layout_analysis is acknowledged; chars-based path is layout-agnostic.
            _ = layout_analysis

            # 1b pymupdf text
            if total_chars < config.ocr_char_threshold:
                lines, total_chars = _pymupdf_extract_text(fitz_page)
                markdown_tables = []
            # 1c pymupdf OCR
            if total_chars < config.ocr_char_threshold and config.ocr_enabled:
                logger.info("OCR used for page %s of %s", page_index + 1, path)
                lines, total_chars = _pymupdf_extract_ocr(fitz_page)
                markdown_tables = []
                ocr_used = True
                if total_chars < _OCR_LOW_QUALITY_CHARS:
                    logger.warning(
                        "Low OCR quality: page %d of %s yielded only %d chars",
                        page_index + 1,
                        path,
                        total_chars,
                    )

            # Strip running headers/footers
            stripped_lines = [
                (y, text)
                for y, text in lines
                if (
                    round(y / _RUNNING_LINE_Y_TOLERANCE_PX) * _RUNNING_LINE_Y_TOLERANCE_PX,
                    text.strip(),
                )
                not in running_lines
            ]

            # Step 2 — font-based heading sentinels
            sentinel_map = _heading_sentinel_map(fitz_page)
            body_text = "\n".join(text for _, text in stripped_lines)
            if markdown_tables:
                body_text = body_text + "\n" + "\n\n".join(markdown_tables)

            # Step 1d — AcroForm widget field values (form PDFs only)
            if is_form:
                widget_parts: list[str] = []
                for widget in fitz_page.widgets():
                    label = (widget.field_label or "").strip()
                    value = str(widget.field_value or "").strip()
                    if value:
                        widget_parts.append(f"{label}: {value}" if label else value)
                if widget_parts:
                    body_text = (body_text + "\n" + "\n".join(widget_parts)).strip()

            body_text = _inject_pdf_heading_sentinels(body_text, sentinel_map)

            # Step 3 — text cleaning
            body_text = _clean_pdf_text(body_text)

            # Step 4 — page metadata assembly
            heading_sentinels = [sent + text for text, sent in sentinel_map.items()]
            page_metadata: dict[str, Any] = {
                "tables": markdown_tables,
                "ocr_used": ocr_used,
                "char_count": len(body_text),
                "heading_sentinels": heading_sentinels,
            }
            yield (page_index + 1, body_text, page_metadata)
    finally:
        doc.close()


def _make_encrypted_pdf_stub(path: Path) -> Document | None:
    """Return a minimal stub Document for a password-protected PDF.

    Opens the file without a password to read XMP metadata only.
    Returns None if the file is unreadable for a reason other than encryption.
    """
    try:
        doc = fitz.open(path)
    except Exception:
        return None
    if not doc.is_encrypted:
        doc.close()
        return None
    try:
        meta = dict(doc.metadata or {})
        title = (meta.get("title") or "").strip() or path.stem
        parts = [
            "[Encrypted PDF — content inaccessible]",
            f"Filename: {path.name}",
            f"Directory: {path.parent.name}",
        ]
        if meta.get("author"):
            parts.append(f"Author: {meta['author']}")
        if meta.get("creationDate"):
            parts.append(f"Creation Date: {meta['creationDate']}")
    finally:
        doc.close()
    return Document(
        path=path.resolve(),
        content="\n".join(parts),
        metadata={
            "title": title,
            "page_description": f"Encrypted PDF: {path.name}",
            "keywords": [],
            "heading_path": title,
            "links_to": [],
            "file_type": "pdf",
            "code_blocks": [],
            "page_count": 0,
            "is_form_pdf": False,
            "page_metadata": [],
            "page_offsets": [],
            "is_encrypted_stub": True,
        },
    )


def extract_pdf_to_document(path: Path, config: ARGConfig) -> Document | None:
    """Assemble a `Document` for a PDF by consuming `extract_pdf`.

    Returns ``None`` when the PDF is corrupt or otherwise unreadable.
    Encrypted PDFs return a minimal stub Document (see `_make_encrypted_pdf_stub`)
    so they appear in search results by filename.
    """
    meta = extract_pdf_metadata(path, config)
    if meta is None:
        return _make_encrypted_pdf_stub(path)

    pages_text: list[str] = []
    per_page_meta: list[dict[str, Any]] = []
    # Track the character offset at which each page begins inside ``content``.
    # ``page_offsets[i] == start of page (i+1) inside content``. Chunker uses
    # this to attribute chunks back to PDF page numbers (Section 7).
    page_offsets: list[int] = []
    separator = "\n\n"
    cursor = 0
    for page_num, text, page_meta in extract_pdf(path, config):
        page_offsets.append(cursor)
        pages_text.append(text)
        per_page_meta.append({"page_number": page_num, **page_meta})
        cursor += len(text) + len(separator)

    # Skip the trailing `.strip()` from the previous implementation — stripping
    # leading whitespace would shift every subsequent page_offset.
    content = separator.join(pages_text)

    metadata: dict[str, Any] = {
        "title": meta["title"],
        "page_description": meta["page_description"],
        "keywords": meta["keywords"],
        "heading_path": meta["title"],
        "links_to": list(meta["links_to"]),
        "file_type": "pdf",
        "code_blocks": [],
        "page_count": meta["page_count"],
        "is_form_pdf": meta["is_form_pdf"],
        "page_metadata": per_page_meta,
        "page_offsets": page_offsets,
    }
    return Document(path=path.resolve(), content=content, metadata=metadata)
