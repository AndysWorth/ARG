# Feature 0001: Plain-text file indexing

**Status:** shipped
**Created:** 2026-05-14
**Shipped:** 2026-05-14 (see commit log: `feat(crawler): plain-text indexing (Feature 0001)`)

---

## Motivation

ARG currently indexes only HTML and PDF. Real documentation corpora carry
a long tail of plain-text files — `.txt` release notes, `.md` READMEs in
subdirectories, ad-hoc notes — that operators want pulled into the same
RAG surface. Excluding them forces operators to pre-convert files to HTML
just to make them searchable, which is exactly the kind of friction ARG
is meant to remove.

## Scope

**In scope:**
- Crawler discovers `.txt` and `.md` files via the existing directory walk
  and through `<a href>` links from HTML pages (relative-path links to
  plain-text files in real corpora are common).
- A new `extract_text(path, config) -> Document` extractor that:
  - Reads the file as UTF-8, falling back to latin-1 on `UnicodeDecodeError`.
  - Uses the filename stem as the document title.
  - Returns a `Document` with `file_type="text"`, empty `page_description`,
    empty `code_blocks`, empty `links_to`, content = the cleaned file
    text.
- Watcher includes `.txt` and `.md` in its `_INDEXABLE_SUFFIXES` so live
  add / modify / delete fires the same callback path as HTML and PDF.
- The chunker handles "no headings" already (single section, sliding
  window). No chunker changes required.

**Out of scope:**
- Markdown-aware heading detection (`# H1` / `## H2` → sentinel injection).
  Plain text and Markdown both flow through `extract_text` for v1; Markdown
  semantics are a separate feature (call it 0002 if/when needed).
- Other plain-text dialects (`.rst`, `.org`, `.adoc`, `.tex`). Same shape;
  pick one of them as Feature 0003 if demand surfaces.
- Encoding auto-detection via `chardet` / `charset-normalizer`. The
  UTF-8 → latin-1 fallback covers most real-world cases; full encoding
  detection is a dependency we don't want to add yet.

## Deliverables

- `arg/crawler/extractors.py`
  - New `extract_text(path: Path, config: ARGConfig) -> Document` function.
  - Re-exported from `arg.crawler` via `__init__.py`.
- `arg/crawler/crawler.py`
  - Extend `_INDEXABLE_SUFFIXES` with `.txt`, `.md`, `.markdown`.
  - Extend `_HTML_SUFFIXES` is **NOT** modified — text isn't HTML.
    Introduce a new `_TEXT_SUFFIXES` constant and a corresponding branch in
    `_extract_for_path`.
- `arg/crawler/watcher.py`
  - Mirror the suffix update; text files trigger the same callback flow.
- `arg/pipeline.py`
  - `_extract_one` dispatch gets a text branch (mirrors `_extract_for_path`).
- `tests/unit/test_extractors.py`
  - Unit tests for `extract_text`: title-from-stem, UTF-8 round-trip,
    latin-1 fallback, empty file, chunk_text shape.
- `tests/unit/test_crawler.py`
  - `normalise_href` accepts `.txt` / `.md` targets.
  - `crawl` walks a directory containing a `.txt` file and yields it.
- `tests/unit/test_watcher.py`
  - Watcher fires on a created `.txt` file.
- `tests/fixtures/docs/`
  - Add `release_notes.txt` (linked from `page_a.html`) and `NOTES.md`
    (found via directory walk) so the integration + e2e suites exercise
    the path end-to-end without separate fixtures.
- `tests/integration/test_graph_to_indexer.py`
  - Indexed-doc count updates from 6 → 8 (4 HTML + 2 PDFs + 2 text).
- `tests/e2e/test_full_rag.py`
  - Same indexed-doc count update; one new assertion that a text-only
    query surfaces a `.txt` source.
- `README.md` and `CLAUDE.md` — see "CLAUDE.md impact" below.

## Design notes

**Why introduce `_TEXT_SUFFIXES` instead of folding text into
`_HTML_SUFFIXES`:** the crawler's BFS-from-index logic and the watcher's
filter both already pivot on the suffix set. Treating text as a third
category keeps the dispatch in `_extract_for_path` clean and makes the
distinction visible to future readers — text doesn't go through
BeautifulSoup, so the suffix sets shouldn't lie.

**Heading sentinels:** plain text has no headings. The chunker already
handles the "no `##H1##` sentinels found" case by emitting one section
covering the whole document. No chunker changes needed; the chunker test
suite confirms this path works.

**`heading_path` metadata:** for HTML and PDF this is the document title.
For text files it's the title as well — consistent with the chunker's
top-level-heading fallback used when no sentinels are present.

**Encoding policy:** the extractor tries UTF-8 first, falls back to
latin-1 on `UnicodeDecodeError`. latin-1 is byte-clean (every byte is a
valid code point), so the fallback never raises. Mojibake on rare
encodings (CP932, GB18030) is acceptable for v1; users with those corpora
can convert to UTF-8 upstream.

**Locality:** `extract_text` reads files via `Path.read_bytes()` and
decodes in memory. No new network calls.

**mypy:** no new third-party imports; no stub-availability quirks expected.

## Test points

Unit:
- `extract_text` reads a UTF-8 file and returns content correctly.
- `extract_text` falls back to latin-1 on bytes that aren't valid UTF-8.
- Title is the filename stem (`release_notes.txt` → `"release_notes"`).
- Empty file yields `content == ""` and the chunker produces zero chunks.
- `Document.metadata` carries `file_type="text"`, empty `links_to`,
  empty `code_blocks`, empty `page_description`.
- `normalise_href` accepts a `.txt` target.
- `crawl` finds a `.txt` file via directory walk and via link-following.
- Watcher fires its callback when a `.txt` file is created / modified /
  deleted.

Integration (Ollama-dependent):
- After `pipeline.index()`, the documents collection contains a row for
  every `.txt` / `.md` file in the corpus.
- BM25 query for a distinctive token in a text file surfaces that chunk.

E2E (real LLM):
- A question whose answer lives only in a text file returns that file as
  a source.

## Open questions / risks

- **`.markdown` extension** — accept it alongside `.md`? Cheap to include;
  no downside. Going with yes.
- **Watcher debounce on rapid editor saves** — text editors like vim
  rename-replace on save, generating create + delete + create events.
  The existing watcher's per-path debounce window (default 500 ms)
  already handles this for HTML files; no extra work expected.
- **BOM-prefixed UTF-8** — `path.read_text(encoding="utf-8-sig")` would
  strip a leading BOM cleanly. Cheap; do it.

## CLAUDE.md impact

This expands ARG's product scope, which deserves CLAUDE.md changes. Three
diff hunks proposed:

**1. CLAUDE.md preamble (line ~1-3):**

```diff
 # ARG — Archivist RAG Graph
-### A fully local, knowledge-graph-augmented RAG for HTML + PDF documentation
+### A fully local, knowledge-graph-augmented RAG for HTML, PDF, and plain-text documentation
 #### Optimized for Apple M1 Max · 64GB Unified Memory · Open Source Only
```

**2. CLAUDE.md Section 1 stack table — add a row after the PDF parser:**

```diff
 | **PDF Parsing** | pdfplumber (primary) + pymupdf (fallback + OCR) | ... |
+| **Text Parsing** | Python stdlib `pathlib.Path.read_text()` | UTF-8 with latin-1 fallback; no new dependency |
 | **OCR** | pymupdf built-in OCR + Tesseract data files | ... |
```

**3. CLAUDE.md Section 13 architectural-decisions table — append:**

```diff
 | Sparse retrieval | BM25 via rank_bm25; fused with dense via RRF (k=60) |
 | Context reordering | Lost-in-middle U-shape; rank 1 → position 0; rank 2 → position -1 |
 | Query processing | Rewrite → decompose → (optional HyDE); raw query used for generation |
 | DCI integration | Fully integrated into Sections 6–10; not a post-RAG layer |
+| Plain-text indexing | Feature 0001: `.txt` / `.md` / `.markdown` accepted via `extract_text`; UTF-8 → latin-1 fallback; no markdown heading detection (out of scope until a separate feature). |
```

**4. README.md "What ARG Does" paragraph (line ~10):**

```diff
-ARG indexes a directory of HTML and PDF documentation, follows links recursively
+ARG indexes a directory of HTML, PDF, and plain-text documentation, follows links recursively
 to build a knowledge graph of the corpus, ...
```

**5. README.md "Known Limitations":**

```diff
+- **Markdown structure** is not parsed — `.md` and `.markdown` files index
+  as plain text. Atx-style headings (`# H1`, `## H2`) are not recognised
+  as chunk boundaries. A future feature may add Markdown-aware extraction.
 - **iframes** are not followed or indexed ...
```

The CLAUDE.md edits are small but load-bearing: they tell future Claude
sessions reading the project spec that text is in scope, and Section 13
gains a one-line ADR pointing back at this feature doc for the why.

---

## Implementation plan

1. Branch `feature/0001-plain-text-indexing` off `main`.
2. Add `extract_text` to `arg/crawler/extractors.py` + unit tests for the
   extractor in isolation. Verify mypy + lint clean.
3. Wire it into `arg/crawler/crawler.py` (suffix sets, dispatch) +
   `arg/crawler/watcher.py` (same suffix set) + `arg/pipeline.py`
   (`_extract_one` dispatch). Add crawler / watcher tests for the text
   path.
4. Add `release_notes.txt` and `NOTES.md` fixtures to
   `tests/fixtures/docs/`. Update the linked-to attribute on
   `page_a.html` to reach `release_notes.txt`.
5. Update affected integration + e2e doc-count assertions. Add one new
   e2e assertion: a text-only query surfaces a text source.
6. Apply the CLAUDE.md + README edits proposed above.
7. Run `pytest tests/unit/` + `pytest tests/integration/` +
   `pytest tests/e2e/ --deselect <real-LLM-tests>` + `mypy arg/` +
   locality grep — all clean.
8. Commit `feat(crawler): plain-text indexing (Feature 0001)` referencing
   this doc.
9. Push, ff-merge to `main`, delete branch.
