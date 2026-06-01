# Plan: PDF extraction efficiency

Findings from live log analysis of a 1500+ document corpus run on 2026-05-28/29.

**COMPLETE — merged to main 2026-06-01 as Feature 0004.**
`parallel-extraction-pipeline` is now unblocked (Feature 0005).

---

## Background

A 22.5-hour indexing run on a real corpus revealed that 97% of wall-clock time
was spent in PDF extraction, with the embedding+writing phase taking only 45
minutes of the total. Three categories of PDF cause most of the time loss:

1. **Image-dominated PDFs** (geological/topo maps, scanned docs with no text):
   pdfplumber spends hours attempting to extract text from large raster images.
   The two worst cases: 8,750s and 4,129s for two map PDFs that yielded only 18
   and 36 chunks respectively.

2. **AcroForm PDFs** (fillable forms): already warned in the log (52 files in
   one run). pdfplumber spends extra time on form fields before giving up with
   incomplete text. One thermostat rebate form took 2,052s to extract 5 chunks.

3. **Long pdfplumber hangs on specific PDF structures**: a Schlage keypad
   programming guide took 2,702s to extract 13 chunks. No OCR trigger, no
   AcroForm — just a structure pdfplumber is slow on.

Additionally, **288 spurious WARNING-level lines** per run from `pdfminer` and
`pdfplumber` internals (FontBBox, color components, inline images) clutter the
log and obscure real issues.

**Note:** The double `pdfplumber.open()` bug is fixed by Feature 0003 (single-pass
extraction, merged 2026-06-01). The items below are independent of that fix.

**Test infrastructure:** The unit suite now has 401 tests. New tests for this
feature go in `tests/unit/test_extractors.py`. See `.claude/rules/testing.md`
for test discipline rules — notably, `test_invariants.py` and `test_concurrency.py`
are off-limits for modification.

---

## Item 1: Pre-flight text-density check

**File:** `arg/crawler/extractors.py`
**Where:** At the top of `extract_pdf()`, before the first `pdfplumber.open()`.
**Prerequisite:** none (pymupdf/fitz is already imported).

**Problem:** `extract_pdf` runs the full pdfplumber path on every PDF regardless
of whether the file contains any machine-readable text. Image-only PDFs (scanned
maps, photo archives) spend hours in pdfplumber with near-zero text yield.

**Fix:** Use pymupdf's fast text extraction to compute average characters per
page before opening pdfplumber. pymupdf's `get_text("text")` is C-level and
completes in milliseconds even for large files.

```python
# At the top of extract_pdf(), after _open_pdf():
total_chars = sum(
    len(doc[i].get_text("text"))
    for i in range(min(doc.page_count, 5))  # sample first 5 pages
)
avg_chars = total_chars / min(doc.page_count, 5)

if avg_chars < config.pdf_min_chars_per_page:
    # Image-dominated: skip pdfplumber entirely.
    if config.ocr_enabled:
        # Go straight to the OCR path (Stage 1c) for all pages.
        ...
    else:
        # No OCR: yield a single "image-only" page with a note.
        yield 0, f"[Image-only PDF — OCR disabled. File: {path.name}]", {...}
        return
```

**New config field in `arg/config.py`:**
```python
pdf_min_chars_per_page: int = 30  # below this → treat as image-dominated
```
Add `"PDF_MIN_CHARS_PER_PAGE": ("pdf_min_chars_per_page", int)` to the env-var
mapping.

**Expected impact:** The two geo-map PDFs (8,750s + 4,129s = 3.6 hours) would
each complete in seconds, yielding either OCR content or a stub chunk.

**Tests to add** (`tests/unit/test_extractors.py`):
- A PDF whose first 5 pages have zero extractable text triggers the image-only
  path (OCR disabled: yields stub chunk; OCR enabled: goes to Stage 1c).
- A PDF with mixed pages (some text, some images) exceeds the threshold and
  proceeds normally.

---

## Item 2: Per-document extraction timeout

**File:** `arg/crawler/extractors.py` (wrapper) and `arg/config.py`
**Where:** `_extract_for_path()` in `crawler.py` is the call site, but the
timeout should wrap the extractor itself.

**Problem:** Pathological PDFs can hang pdfplumber for hours. The Schlage
programming guide spent 2,702s for 13 chunks; no other signal indicated it was
slow until the run finished. A configurable timeout caps the worst case and lets
the user decide whether to skip or fall back.

**Fix:** Wrap the call to `extract_pdf_to_document()` in a thread + join with
timeout. Use `concurrent.futures.ThreadPoolExecutor(max_workers=1)` with
`future.result(timeout=N)` — this is macOS/Linux safe and does not use
`signal.alarm` (which is not safe in multi-threaded contexts).

```python
# In crawler.py _extract_for_path(), replace the PDF branch:
if suffix in _PDF_SUFFIXES:
    timeout = config.pdf_extract_timeout_seconds
    if timeout <= 0:
        return extract_pdf_to_document(path, config)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(extract_pdf_to_document, path, config)
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            logger.warning(
                "crawler: PDF extraction timed out after %ds, skipping: %s",
                timeout, path,
            )
            fut.cancel()
            return None
```

**New config field:**
```python
pdf_extract_timeout_seconds: int = 300  # 0 = no timeout
```
Add `"PDF_EXTRACT_TIMEOUT": ("pdf_extract_timeout_seconds", int)` to env-var map.

**Note:** `ollama_timeout: float = 300.0` already exists in config for LLM calls.
The PDF timeout is separate and applies to extraction only, not embedding.

**Caveats:**
- `ThreadPoolExecutor(max_workers=1)` for the timeout wrapper is not the same
  as the extraction worker pool in the parallel pipeline plan. These are
  independent and can coexist.
- Cancelling a `future` does not kill the running thread. pdfplumber may
  continue consuming CPU in the background. This is acceptable for the timeout
  use case; the document is skipped and indexing continues. The thread will
  eventually complete or be reaped when the process exits.

**Tests to add** (`tests/unit/test_extractors.py`):
- Mock `extract_pdf_to_document` to sleep longer than `pdf_extract_timeout_seconds`;
  assert `_extract_for_path` returns `None` and logs a WARNING.
- `pdf_extract_timeout_seconds=0` disables timeout; extractor runs to completion.

---

## Item 3: AcroForm fast-path

**File:** `arg/crawler/extractors.py`
**Where:** `extract_pdf_metadata()` already detects `doc.is_form_pdf` and
logs a warning (line ~874). `extract_pdf()` runs immediately after.

**Problem:** When `is_form_pdf` is True, pdfplumber still runs its full
extraction path. For a complex fillable form (e.g., lease agreements, insurance
applications), this is slow and produces incomplete text anyway. pymupdf's
`get_text()` is faster and produces the same or better output for form PDFs
because it reads the text layer directly without trying to reconstruct layout
from character positions.

**Fix:** Pass `is_form_pdf` from `extract_pdf_metadata()` to `extract_pdf()` via
the document metadata (it's already stored: `"is_form_pdf": bool(doc.is_form_pdf)`).
At the start of the per-page loop in `extract_pdf()`, if `is_form_pdf` is True,
skip pdfplumber (Stage 1a) and go directly to Stage 1b (pymupdf text):

```python
# Inside the per-page loop in extract_pdf(), before Stage 1a:
is_form = metadata.get("is_form_pdf", False)
if not is_form:
    lines, markdown_tables, total_chars = _pdfplumber_extract_page(plumber_page)
else:
    # AcroForm: pdfplumber is slow and incomplete; go straight to pymupdf.
    lines, total_chars = _pymupdf_extract_text(fitz_page)
    markdown_tables = []
    total_chars = max(total_chars, 1)  # bypass the OCR threshold check
```

The `is_form_pdf` flag is already available in the `Document.metadata` dict
passed into `extract_pdf_to_document()`.

**Expected impact:** The 52 AcroForm PDFs in the test corpus include lease
agreements, insurance applications, and rebate forms — a mix of simple and
complex documents. Eliminating the slow pdfplumber path for all of them would
recover a significant portion of the ~2,052s example case.

**Tests to add** (`tests/unit/test_extractors.py`):
- A PDF with `is_form_pdf=True` never calls `_pdfplumber_extract_page`.
- The text yielded for a form PDF comes from `_pymupdf_extract_text`.

---

## Item 4: Suppress pdfminer/pdfplumber library log noise

**File:** `arg/crawler/extractors.py` (module level) or `arg/logging/json_formatter.py`

**Problem:** Each run produces ~288 WARNING-level lines from pdfminer and
pdfplumber internals:
- 230 × `Could not get FontBBox from font descriptor because None cannot be
  parsed as 4 floats` — font metadata issue in the underlying C library.
- 56 × `Cannot set non-stroke color: 2 components specified, but only 1
  (grayscale), 3 (RGB)...` — PDF color space issue.
- 2 × `Execute called on non-indirect object (inline image?)` — rare edge case.

These are not actionable and originate from `pdfminer.six` and `pdfplumber`
loggers, not ARG code. They clutter the log and drown out real warnings.

**Fix:** At the top of `extractors.py`, after the imports:

```python
import logging as _logging
# pdfminer and pdfplumber emit spurious WARNING-level lines for normal PDF
# quirks (missing FontBBox, non-standard color spaces). Silence them.
_logging.getLogger("pdfminer").setLevel(_logging.ERROR)
_logging.getLogger("pdfplumber").setLevel(_logging.ERROR)
```

This is local to the module, does not affect ARG's own loggers, and takes effect
when the module is first imported.

**Tests:** No new tests needed. Existing extractor tests are unaffected (they
don't assert on pdfminer/pdfplumber log output).

---

## Item 5: Fix `num_ctx` for nomic-embed-text

**File:** `arg/pipeline.py`
**Where:** `_default_embedder()`, the `embed()` method of
`_OllamaEmbedderAdapter`, line ~283 — currently passes `options={"num_ctx": 8192}`.

**Problem:** `nomic-embed-text` has a 2048-token context window. Telling Ollama
`num_ctx: 8192` causes it to allocate a KV cache 4× larger than needed for every
single embedding call — wasted memory and initialization overhead on each HTTP
round-trip.

**Fix:**
```python
options={"num_ctx": config.embed_num_ctx},
```

**New config field:**
```python
embed_num_ctx: int = 2048  # nomic-embed-text context window
```
Add `"EMBED_NUM_CTX": ("embed_num_ctx", int)` to env-var map. Keeping it
configurable allows switching to a model with a larger context (e.g.,
`mxbai-embed-large` has 512 tokens, `nomic-embed-text-v1.5` has 8192) without
a code change.

**Tests:** The existing embedder unit tests use a fake embedder that ignores
options, so no test changes are needed. Integration test: verify that a query
returns the same results before and after the config change (the embedding values
should be identical for texts under 2048 tokens).

---

## Implementation order

These items are independent and can be done in any order, but the suggested
sequence minimises risk:

1. **Item 4** (log noise) — zero logic risk, immediate observability improvement.
2. **Item 5** (num_ctx) — zero logic risk, free performance gain.
3. **Item 3** (AcroForm fast-path) — small, well-contained.
4. **Item 1** (text density check) — adds new config; test with known image PDFs.
5. **Item 2** (extraction timeout) — most complex; test with mocked slow extractor.

All five can ship in a single branch. Suggested commit messages:
- `perf(extractors): silence pdfminer/pdfplumber log noise`
- `perf(embedder): lower num_ctx to 2048 for nomic-embed-text`
- `perf(extractors): AcroForm fast-path via pymupdf`
- `perf(extractors): pre-flight text-density check for image-dominated PDFs`
- `perf(crawler): per-document PDF extraction timeout`
