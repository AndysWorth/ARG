# Section 5: Crawler & Extractors

> **Prompt to Claude:** "Build Section 5 of ARG: the crawler and extractors — complete"

### What Claude will produce:
- `arg/crawler/crawler.py` — recursive document crawler
- `arg/crawler/extractors.py` — HTML→text and PDF→text extractors

---

### Crawler behaviour:

1. Starts at `docs_root/index.html`
2. Parses all `<a href>` links — **normalises every href before evaluation** (see URL rules below)
3. Also walks the directory tree to catch any `.html`, `.htm`, `.pdf` files not reachable via links
4. Deduplicates by resolved absolute path
5. Records every `(source_path, target_path)` link pair for the knowledge graph
6. Yields `Document` objects: `{path, content, metadata{title, page_description, links_to, file_type}}`
7. Handles: `.html`, `.htm`, `.pdf` — logs and skips everything else

**URL normalisation rules (applied to every href before following):**

Skip entirely if the href:
- Is an anchor-only link (`href="#..."`)
- Has a non-file scheme: `mailto:`, `javascript:`, `tel:`, `ftp:`
- Is protocol-relative (`href="//..."`)

Resolve to absolute path, then skip if:
- The resolved path is outside `docs_root` (e.g. `../sibling-project/page.html`)
- The resolved path has an `http://` or `https://` scheme (external site)

Follow if:
- The resolved absolute path is inside `docs_root` and has extension `.html`, `.htm`, or `.pdf`

This replaces the naive `no http://` check in the original plan, which missed
protocol-relative URLs, mailto links, anchor-only links, and path-escape attacks.

---

### HTML extractor behaviour (`extractors.py`):

**Parser:** always call `BeautifulSoup(content, features="lxml")` — never `html.parser`.
lxml handles malformed HTML more robustly and decodes HTML entities correctly.

**Step 1 — Strip invisible and boilerplate elements:**

Remove all of the following before extracting any text:

| What | How |
|---|---|
| `<script>` | `tag.decompose()` |
| `<style>` | `tag.decompose()` |
| `<nav>` | `tag.decompose()` |
| `<header>` | `tag.decompose()` |
| `<footer>` | `tag.decompose()` |
| `<aside>` | `tag.decompose()` |
| `<iframe>` | `tag.decompose()` (content not indexable; see Known Limitations) |
| Any tag with `style` attr containing `display:none` | `tag.decompose()` |
| Any tag with `style` attr containing `visibility:hidden` | `tag.decompose()` |
| Elements matching `strip_selectors` config list | `tag.decompose()` |

`strip_selectors` is a configurable list of CSS selectors for `<div>`-based navigation
that older documentation generators use instead of semantic tags. Default value:
```python
strip_selectors: list[str] = [
    "div.sidebar", "div.nav", "div.navigation", "div.breadcrumb",
    "div.breadcrumbs", "div#nav", "div#sidebar", "div#header",
    "div#footer", "div.toc", "div#toc", "div.related",
    "div.sphinxsidebar",   # Sphinx
    "div.rst-footer-buttons",  # Read the Docs
    "div.wy-nav-side",     # Read the Docs
    "div.md-sidebar",      # MkDocs Material
]
```
Add to `ARGConfig` as `strip_selectors: list[str]` with the above default.

**Step 2 — Extract title:**

1. Try `soup.find("title").get_text()`
2. Clean title: split on ` | `, ` — `, ` - `, ` :: ` and keep the **first** segment
   (most specific). Example: `"Authentication | Kraken API Docs"` → `"Authentication"`.
   Configurable via `title_separator: str = " | "` in `ARGConfig` (default covers most cases;
   user can set to `" — "` or `" - "` for their doc generator's pattern).
3. If `<title>` absent or empty, fall back to the first `<h1>` text.
4. Store cleaned title as `metadata["title"]`.

**Step 3 — Extract page description:**

1. Try `soup.find("meta", {"name": "description"})["content"]`
2. Also try `soup.find("meta", {"property": "og:description"})["content"]` as fallback
3. Store as `metadata["page_description"]` (empty string if neither present)
4. Prepend description to the text fed into the `documents` ChromaDB embedding
   (before the first 512 body tokens), so the doc-level vector reflects the
   page's own summary rather than just its opening sentences.

**Step 4 — Extract and convert tables:**

Before extracting body text, find all `<table>` elements and replace each in-place
with a pipe-delimited Markdown representation:
```
| Column A | Column B |
|---|---|
| value 1  | value 2  |
```
- Use `<th>` for header row; `<td>` for data rows
- Strip any nested tags within cells; keep only their text
- Replace the original `<table>` tag with a `NavigableString` of the Markdown text
- This preserves row/column relationships so the LLM reads "Tier 2 → 500 req/min"
  correctly rather than flattened "2 500 req/min"

**Step 5 — Extract heading structure:**

Process H1–H6 in document order to build `heading_path` metadata for each section:
- H1–H3: included in body text **and** tracked as section boundaries for the chunker.
  Their text is preserved in the output at their position.
- H4–H6: included in body text only; not used as chunk boundaries.
- The current heading path (e.g. `"API Reference > Authentication > OAuth Flow"`)
  is stored as `metadata["heading_path"]` on each extracted section and passed
  through to chunk metadata.

**Step 6 — Handle code blocks:**

`<pre>` and `<code>` blocks are kept in body text. However:
- A `<pre>` block whose text exceeds `max_code_block_tokens` (default 256, configurable)
  is truncated to 256 tokens with a `[... truncated ...]` marker appended inline.
- The full untruncated text of each `<pre>` block is stored in `metadata["code_blocks"]`
  as a list of strings, for potential future use.
- This prevents a 500-line code example from dominating an entire chunk and
  pushing out the prose explanation that gives it context.

**Step 7 — Extract body text:**

Call `soup.get_text(separator="\n")` on the remaining (stripped) document.

**Step 8 — Whitespace normalisation (mandatory):**

Apply in this order:
1. Replace `\u00a0` (non-breaking space) with regular space
2. Replace `\t` (tab) with single space
3. Collapse runs of 3+ newlines to exactly 2 newlines
4. Collapse runs of 2+ spaces to single space (except inside code blocks)
5. Strip leading and trailing whitespace from the entire result

Without this step, chunks contain irregular whitespace that wastes tokens and
confuses the tokeniser's boundary detection.

**Output document metadata (complete set):**
```python
metadata = {
    "title": str,              # cleaned page title
    "page_description": str,   # from <meta name="description">; empty if absent
    "heading_path": str,       # top-level heading hierarchy at extraction time
    "links_to": list[str],     # absolute paths of internal links found on this page
    "file_type": "html",
    "code_blocks": list[str],  # full text of any truncated <pre> blocks
}
```

**Known limitations (document in README):**
- `<iframe>` content is not indexed. If your documentation embeds content via
  iframes (e.g. embedded API explorers, embedded PDFs), that content will not
  be searchable in ARG.
- `<script>` text content is not indexed, including `<script type="application/ld+json">`
  structured data. If your docs embed important content in JSON-LD, it will not be retrieved.
- Dynamically rendered content (JavaScript-rendered text that requires a browser to execute)
  is not indexed. ARG reads the static HTML file; any content that only appears after
  JS execution will be absent.

---

### PDF extractor behaviour (`extractors.py`):

---

#### Overview

PDF extraction is significantly more complex than HTML because PDFs have no semantic
markup — structure must be inferred from visual properties (font size, position, repetition).
The extractor is a multi-stage pipeline applied **per page independently**, not once per
document. A single PDF can mix native-text pages, scanned pages, table-heavy pages, and
form pages; each page is evaluated and handled on its own merits.

---

#### Step 0 — Document-level pre-flight

Before processing any pages:

**0a. Encryption check:**
```python
try:
    doc = fitz.open(path)
    if doc.is_encrypted:
        raise PDFEncryptedError
except (fitz.EmptyFileError, PDFPasswordIncorrect, Exception):
    log WARNING: f"Skipping unreadable PDF: {path} — encrypted or corrupt"
    return None   # caller skips this file; indexing continues
```
An encrypted or corrupt PDF logs a clear warning and is skipped entirely. It never
crashes the indexing run.

**0b. Form PDF detection:**
```python
if doc.is_pdf and len(doc.get_form_fields()) > 0:
    log WARNING: f"PDF {path} contains AcroForm fields — extracted text may be incomplete"
    # continue extraction; warn user that form field values may be missing
```

**0c. PDF title resolution** (stored as `metadata["title"]`):
1. Try `doc.metadata.get("title", "")` — use if non-empty AND not matching
   temp-file patterns: `^(Microsoft Word|Untitled|document\d*|Presentation\d*|Worksheet\d*)` (case-insensitive)
2. Fallback: largest-font line on page 1 (detected in Step 2 font analysis)
3. Final fallback: `Path(path).stem` (filename without extension)

**0d. PDF description and keywords** (used in `documents` collection embedding):
- `page_description`: `doc.metadata.get("subject", "")` — PDF `/Subject` field
- `keywords`: `doc.metadata.get("keywords", "")` — PDF `/Keywords` field
- Both stored in document metadata; `page_description` prepended to doc-level
  embedding text (same pattern as HTML `<meta name="description">`)

**0e. Running header/footer detection:**
Scan all pages to find lines that appear at the same vertical position (within ±3px)
on ≥ 3 pages with identical or near-identical text. These are running headers/footers.
Build a set of `(y_position, text_pattern)` pairs to exclude during per-page extraction.
This removes page numbers, document titles, "CONFIDENTIAL" stamps, and copyright
lines that would otherwise appear in every chunk.

---

#### Step 1 — Per-page three-stage extraction

Each page is processed independently through three stages. The stage used for one
page does not affect the stage used for the next.

**Stage 1a — pdfplumber (primary):**

1. Extract tables first via `page.extract_tables()`:
   - Record each table's bounding box
   - Convert each table to pipe-delimited Markdown (same format as HTML tables)
   - Store tables as `(bbox, markdown_text)` pairs

2. Extract body text excluding table bounding boxes:
   ```python
   body_text = page.extract_text(
       layout=config.pdf_layout_analysis,
       bbox_exclude=[t.bbox for t in tables]   # prevent cell text appearing twice
   )
   ```
   This is critical — without `bbox_exclude`, every table cell appears twice: once
   in the Markdown table and once in the raw text flow.

3. Splice tables into body text at their vertical position (by `bbox.y0`)

4. Strip running headers/footers detected in Step 0e

5. Count total chars: `total_chars = len(body_text) + sum(len(t.markdown) for t in tables)`

6. If `total_chars >= ocr_char_threshold`: use this result. Move to Step 2 (font analysis).
   If `total_chars < ocr_char_threshold`: proceed to Stage 1b.

**Stage 1b — pymupdf text extraction (first fallback):**

```python
page_fitz = fitz_doc[page_number]
body_text = page_fitz.get_text("text")
```

Strip running headers/footers. Count chars. If `>= ocr_char_threshold`: use this result.
If still `< ocr_char_threshold`: proceed to Stage 1c.

**Stage 1c — pymupdf OCR (final fallback):**

```python
body_text = page_fitz.get_textpage_ocr(full=True).extractText()
```

Only called when both Stage 1a and 1b yielded fewer than `ocr_char_threshold` chars.
Logs at INFO level: `"OCR used for page {n} of {path}"`.

Note: OCR runs on the entire page image. If Stage 1b found some text but below threshold,
that text is discarded and OCR result is used instead — OCR on the whole page is more
reliable than mixing partial native text with partial OCR text.

---

#### Step 2 — Font-based heading detection

After Stage 1 produces body text, use pdfplumber's character-level `chars` data
(if Stage 1a was used) or pymupdf's `get_text("dict")` block data (if Stage 1b/1c was
used) to infer heading hierarchy from font metrics:

1. Compute the **median font size** of all body text characters on the page
2. For each line:
   - If median font size of line chars > `1.5 × body_median`: inject `##H1##` sentinel before line
   - If median font size of line chars > `1.2 × body_median` and ≤ `1.5 ×`: inject `##H2##` sentinel
   - If bold weight AND font size within 10% of body: inject `##H3##` sentinel
   - Otherwise: plain body text
3. These sentinels are identical to those injected by the HTML extractor, so the chunker
   (Section 7) handles PDFs and HTML through the same boundary-detection logic

**First-page title detection fallback (for Step 0c.2):**
On page 1, the line with the largest font size (before any body text lines) is the
document title candidate. Used only if PDF `/Title` metadata is absent or invalid.

---

#### Step 3 — Post-extraction text cleaning

Apply to the combined text of each page after extraction and heading injection:

**3a. Hyphen line-break rejoining:**
```
If line ends with `-` AND next line starts with a lowercase letter:
    remove `-` and join lines (soft hyphen from line-wrap)
Else if line ends with `-` AND next line starts with uppercase:
    keep hyphen (intentional compound word or list item)
```

**3b. Unicode normalisation:**
```python
import unicodedata
text = unicodedata.normalize("NFC", text)

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
             "ﬆ": "st", "Ĳ": "IJ", "ĳ": "ij"}
for lig, rep in LIGATURES.items():
    text = text.replace(lig, rep)

TYPOGRAPHIC = {"\u2014": " — ", "\u2013": " - ",   # em dash, en dash
               "\u201c": '"',  "\u201d": '"',        # smart double quotes
               "\u2018": "'",  "\u2019": "'",        # smart single quotes
               "\u2026": "...",                       # ellipsis
               "\u00ad": ""}                          # soft hyphen (remove)
for char, rep in TYPOGRAPHIC.items():
    text = text.replace(char, rep)
```

**3c. Whitespace normalisation** (same rules as HTML):
- Collapse 3+ newlines to 2
- Collapse 2+ spaces to 1 (outside any remaining code regions)
- Strip leading/trailing whitespace per page

---

#### Step 4 — Page metadata assembly

Each page yields:
```python
{
    "page_number": int,
    "text": str,                  # cleaned body text with ##H1##/##H2##/##H3## sentinels
    "tables": list[str],          # Markdown tables found on this page
    "ocr_used": bool,
    "char_count": int,
    "heading_sentinels": list[str]  # e.g. ["##H1## Getting Started", "##H2## Prerequisites"]
}
```

---

#### Step 5 — Document-level assembly and incremental yielding

The PDF extractor is a **generator** — it yields `(page_number, page_text, page_metadata)`
tuples one at a time rather than returning the entire document as a single string.

Benefits:
- The indexer can process and checkpoint pages in batches (write every N pages)
- If indexing is interrupted mid-PDF, a restart can skip already-processed pages
- Avoids loading a 500-page manual entirely into memory before writing anything
- Watcher events from other documents are not blocked waiting for a large PDF to finish

The indexer (Section 7) receives pages as a stream and assembles chunks across
page boundaries (a section heading on page 12 whose body continues onto page 13
is stitched into a single chunk).

---

#### Per-document layout override (Problem 4 fix)

`pdf_layout_analysis` is a global default but can be overridden per-document via a
**sidecar config file**: place a file named `{pdf_filename}.argconfig` in the same
directory as the PDF, containing JSON:
```json
{"pdf_layout_analysis": false}
```
The extractor checks for this sidecar before processing each PDF. If found, its
settings override the global config for that document only.

Example: `docs/manual.pdf` → extractor checks for `docs/manual.pdf.argconfig`

---

#### `documents` collection embedding text for PDFs

Mirrors the HTML approach exactly:
```
embedding_text = page_description + " " + first_512_tokens_of_page_1_body_text
```
Where `page_description` = PDF `/Subject` metadata (empty string if absent).
This ensures PDFs and HTML documents are comparable in the same embedding space.

---

#### Known limitations for PDFs (document in README):

- **Encrypted / password-protected PDFs:** skipped with a warning. Not supported.
- **AcroForm / XFA fillable forms:** form field values may not be extracted. Text
  visible in the static page view will be extracted; field default values may be absent.
- **Right-to-left text (Arabic, Hebrew):** extraction order may be incorrect.
  Not tested or supported.
- **Embedded fonts with non-standard encodings:** some PDFs use private font encodings
  that produce garbled characters even with native text extraction. If a document
  produces obviously garbled text, try `pdf_layout_analysis: false` in its sidecar
  config. If still garbled, OCR is the only reliable fallback.
- **PDF portfolios (collections of files in one container):** only the container
  shell is indexed; embedded sub-documents are not extracted.

---

---

### Tests (unit):

**`test_crawler.py`:**
- Crawler finds all linked documents from a fixture `index.html`
- Crawler finds unlinked files via directory walk
- Crawler does not follow `http://` external links
- Crawler does not follow `mailto:`, `javascript:`, `tel:` hrefs
- Crawler does not follow anchor-only `href="#section"` links
- Crawler does not follow protocol-relative `href="//cdn.example.com/..."` links
- Crawler does not follow hrefs that resolve outside `docs_root`
- Crawler deduplicates circular links (A→B→A)
- Crawler respects `max_file_depth`

**`test_extractors.py`:**
- `<style>` tag content does not appear in extracted text
- `<nav>` tag content does not appear in extracted text
- `<aside>` tag content does not appear in extracted text
- Element with `style="display:none"` content does not appear in extracted text
- Element with `style="visibility:hidden"` content does not appear in extracted text
- Element matching a `strip_selectors` entry does not appear in extracted text
- `BeautifulSoup` is called with `features="lxml"` (assert parser used)
- `<title>` is extracted and suffix-stripped correctly: `"Auth | Docs"` → `"Auth"`
- `<meta name="description">` content stored in `metadata["page_description"]`
- `<table>` rendered as pipe-delimited Markdown with correct column alignment
- H1–H3 text appears in body text at correct position
- H4 text appears in body text (not used as chunk boundary)
- `<pre>` block under `max_code_block_tokens`: full text in body text
- `<pre>` block over `max_code_block_tokens`: truncated in body text; full text in `metadata["code_blocks"]`
- Whitespace normalised: no `\u00a0`, no triple newlines, no double spaces
- HTML entities decoded: `&lt;` → `<`, `&amp;` → `&`, `&nbsp;` → ` `
**`test_extractors.py` — PDF tests:**

*Pre-flight:*
- Encrypted PDF returns `None`; no exception raised; warning logged
- Corrupt/empty PDF returns `None`; warning logged
- PDF with AcroForm fields logs warning but continues extraction
- PDF `/Title` used when non-empty and not a temp-file pattern
- PDF `/Title` skipped when it matches `"Microsoft Word - document1.docx"` → fallback to font detection
- PDF `/Subject` stored as `page_description`; empty string when absent
- PDF `/Keywords` stored in `keywords` metadata field

*Per-page extraction:*
- Native-text PDF: pdfplumber stage used; OCR not called
- pdfplumber-yields-below-threshold page: pymupdf text stage used
- Both-below-threshold page: OCR stage used; `ocr_used=True` in page metadata
- Pages within same PDF can use different stages independently
  (page 1 native-text → pdfplumber; page 2 scanned → OCR)

*Table handling:*
- Table cells do NOT appear twice (once in Markdown, once in body text)
- Table Markdown is correctly spliced at vertical position of original table
- Page with only tables and no body text: total_chars counts table text;
  does not trigger unnecessary OCR

*Running header/footer stripping:*
- Text appearing at same vertical position on ≥ 3 pages is stripped from all pages
- Page numbers stripped (digit-only lines at consistent y-position)
- Unique page content at same y-position is NOT stripped

*Font-based heading detection:*
- Line with font size > 1.5× body median: `##H1##` sentinel injected
- Line with font size 1.2–1.5× body median: `##H2##` sentinel injected
- Bold line at body font size: `##H3##` sentinel injected
- Regular body text: no sentinel

*Text cleaning:*
- Soft hyphen line break rejoined: `"config-\nuration"` → `"configuration"`
- Intentional hyphen preserved: `"self-hosted"` not modified
- Ligatures decoded: `"ﬁle"` → `"file"`, `"ﬂow"` → `"flow"`
- Em dash normalised: `"\u2014"` → `" — "`
- Smart quotes normalised: `"\u201ctext\u201d"` → `'"text"'`
- Ellipsis normalised: `"\u2026"` → `"..."`
- Soft hyphens removed: `"\u00ad"` → `""`
- Unicode NFC normalisation applied
- Whitespace: no triple newlines, no double spaces, no leading/trailing whitespace per page

*Incremental yielding:*
- PDF extractor is a generator; yields `(page_number, text, metadata)` tuples
- Does not materialise the entire document in memory before yielding
- Indexer receives pages one at a time (integration test)

*Per-document layout override:*
- Global `pdf_layout_analysis=True`; sidecar `.argconfig` with `false` overrides for that doc only
- Other docs in same corpus not affected by the sidecar

*`documents` collection embedding text:*
- For PDF with `/Subject`: embedding text = `/Subject` + first 512 tokens of page 1 body
- For PDF without `/Subject`: embedding text = first 512 tokens of page 1 body only
