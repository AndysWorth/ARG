# Feature 0004: PDF extraction efficiency

**Status:** shipped (commit: `14ea07a`)
**Created:** 2026-05-29
**Merged:** 2026-06-01

---

## Motivation

A 22.5-hour indexing run on a 1500+ document real corpus revealed that 97% of
wall-clock time was spent in PDF extraction, with the embedding + writing phase
taking only 45 minutes total. Three categories of PDF cause most of the time loss:

1. **Image-dominated PDFs** (geological/topo maps, scanned docs with no text):
   pdfplumber spends hours attempting to extract text from large raster images.
   The two worst cases: 8,750s and 4,129s for two map PDFs that yielded only 18
   and 36 chunks respectively — 3.6 hours combined for 54 chunks.

2. **AcroForm PDFs** (fillable forms): pdfplumber runs its full extraction path on
   form fields and produces incomplete text. One thermostat rebate form took 2,052s
   to yield 5 chunks.

3. **Long pdfplumber hangs on specific PDF structures**: a Schlage keypad programming
   guide took 2,702s to extract 13 chunks. No OCR trigger, no AcroForm — just a
   structure pdfplumber is slow on.

Additionally, **288 spurious WARNING-level lines** per run from `pdfminer` and
`pdfplumber` internals (FontBBox, color components, inline images) cluttered the
log and obscured real issues.

A secondary inefficiency: `nomic-embed-text` has a 2048-token context window, but
the embedder was passing `num_ctx: 8192` to Ollama, causing it to allocate a KV
cache 4× larger than needed on every single embedding call.

Note: the double `pdfplumber.open()` bug was fixed by Feature 0003
(single-pass extraction). The items here are independent of that fix.

## Scope

**In scope:**

- Suppress `pdfminer` and `pdfplumber` library log noise by setting those loggers
  to ERROR level at module import time in `extractors.py`.
- Fix `embed_num_ctx` to 2048 (matching nomic-embed-text's actual context window)
  via a new config field, replacing the hardcoded 8192.
- Add an AcroForm fast-path: when `is_form_pdf` is True, skip pdfplumber entirely
  and use pymupdf's `get_text()` directly for all pages.
- Add a pre-flight text-density check: sample the first 5 pages with pymupdf
  (`get_text("text")`); if average characters per page is below a configurable
  threshold (`pdf_min_chars_per_page`), treat the file as image-dominated and skip
  pdfplumber.
- Add a per-document extraction timeout: wrap `extract_pdf_to_document()` in a
  `ThreadPoolExecutor(max_workers=1)` with `future.result(timeout=N)`; timed-out
  files are skipped with a WARNING.

**Out of scope:**

- Page-level parallelism within a single large PDF (see Feature 0005 plan,
  Item 4 — deferred until after parallel dirwalk is profiled).
- OCR improvements or tesseract dependency management.
- Incremental/resumable extraction across runs.

## Deliverables

**`arg/config.py`**
- New field `pdf_min_chars_per_page: int = 30` — avg chars/page below this triggers
  image-dominated path. Env var `PDF_MIN_CHARS_PER_PAGE`.
- New field `pdf_extract_timeout_seconds: int = 300` — 0 disables timeout. Env var
  `PDF_EXTRACT_TIMEOUT`.
- New field `embed_num_ctx: int = 2048` — nomic-embed-text context window. Env var
  `EMBED_NUM_CTX`.
- Removed dead `pdf_batch_size: int = 10` field and its `PDF_BATCH_SIZE` env var
  entry (was never read anywhere in the codebase).

**`arg/crawler/extractors.py`**
- After logger initialisation: set `pdfminer` and `pdfplumber` loggers to ERROR.
- In `extract_pdf()`: pre-flight text-density check using pymupdf on first 5 pages.
- In `extract_pdf()`: AcroForm detection — if `is_form_pdf` or `is_image_dominated`,
  skip pdfplumber and fill `page_buffer` with empty `([], [], 0)` tuples so the
  per-page loop falls through to pymupdf Stage 1b.

**`arg/crawler/crawler.py`**
- New `_extract_pdf_with_timeout(path, config) → Document | None`: wraps
  `extract_pdf_to_document` in a single-worker thread pool with configurable timeout.
- `_extract_for_path`: PDF branch calls `_extract_pdf_with_timeout` instead of
  `extract_pdf_to_document` directly.

**`arg/pipeline.py`**
- `_OllamaEmbedderAdapter.embed()`: `options={"num_ctx": self.config.embed_num_ctx}`
  (was hardcoded 8192).

**`tests/unit/test_extractors.py`** — 3 new tests:
- `test_image_dominated_pdf_skips_pdfplumber`: blank PDF, assert `pdfplumber.open`
  not called.
- `test_text_rich_pdf_uses_pdfplumber`: native-text PDF, assert `pdfplumber.open`
  is called.
- `test_acroform_pdf_skips_pdfplumber`: form PDF with `is_form_pdf=True`, assert
  `pdfplumber.open` not called.

**`tests/unit/test_crawler.py`** — 2 new tests:
- `test_pdf_extraction_timeout_skips_slow_pdf`: mock extractor sleeps 5s, timeout=1,
  assert returns None + WARNING logged.
- `test_pdf_extraction_timeout_zero_disables_timeout`: timeout=0, assert
  `extract_pdf_to_document` called directly (no pool).

## Design notes

**Pre-flight text-density check — why pymupdf:** pymupdf's `get_text("text")` is
C-level and completes in milliseconds even for large files (it reads the existing
text layer without rendering). For the two geo-map PDFs (8,750s + 4,129s combined),
the pre-flight check returns in under 1s and routes them to OCR or a stub chunk.

**AcroForm fast-path — why skip pdfplumber:** pdfplumber attempts to reconstruct
text from character positions; for AcroForm fields it produces incomplete output and
is slow. pymupdf's `get_text()` reads the text layer directly without layout
reconstruction, producing the same or better output for forms faster.

**Extraction timeout — why `ThreadPoolExecutor(max_workers=1)` not `signal.alarm`:**
`signal.alarm` is not safe in multi-threaded contexts (only the main thread can
receive SIGALRM, and it's not available on Windows). The single-worker pool avoids
both constraints. Cancelling a future does not kill the running thread — pdfplumber
may continue consuming CPU in the background; this is acceptable because the document
is skipped and indexing continues unblocked. The thread is reaped when the process
exits or the pool shuts down.

**Timeout pool vs. extraction pool:** the `ThreadPoolExecutor(max_workers=1)` used
here for the timeout is independent of the extraction worker pool added in Feature 0005.
These can coexist: the timeout pool serialises a single extraction call; the extraction
pool (Feature 0005) runs multiple files concurrently.

**embed_num_ctx — why 2048:** nomic-embed-text has a 2048-token context window.
Passing 8192 caused Ollama to allocate a KV cache 4× larger than needed on every
embedding HTTP call — wasted initialisation overhead at scale. The field is kept
configurable (`embed_num_ctx`) to support switching to models with different context
windows (e.g., `nomic-embed-text-v1.5` has 8192 tokens) without a code change.

**Locality:** no change — all operations remain local.

## Test points

Unit:
- Blank PDF (0 avg chars/page) → `pdfplumber.open` not called.
- Native-text PDF (chars/page >> threshold) → `pdfplumber.open` is called.
- AcroForm PDF → `pdfplumber.open` not called.
- Extractor sleeps longer than timeout → returns None, WARNING logged.
- `pdf_extract_timeout_seconds=0` → extractor called directly (no pool).

Integration (Ollama-dependent):
- Existing PDF extraction integration tests pass unchanged.

E2E:
- Existing e2e tests pass unchanged.

## Open questions / risks

- **Thread not killed on timeout:** cancelling a `concurrent.futures.Future` does
  not interrupt a running thread. A timed-out pdfplumber call keeps consuming CPU
  until it finishes naturally. For pathological PDFs this may be minutes. Acceptable
  for now — indexing continues and the worst offenders are caught by the text-density
  pre-flight check anyway.
- **AcroForm detection accuracy:** `fitz.Document.is_form_pdf` is a heuristic.
  Some PDFs with AcroForm structure contain substantial text (multi-page PDF forms
  with static content). The pymupdf `get_text()` fast-path produces the same or
  better output for those anyway, so a false-positive is not harmful.

## CLAUDE.md impact

No product-scope or user-facing changes. Section 13 ADR table gains a row:

```diff
+| PDF extraction efficiency | Feature 0004: pre-flight text-density check skips pdfplumber for image-dominated PDFs; AcroForm fast-path via pymupdf; per-document extraction timeout; embed_num_ctx corrected to 2048; pdfminer/pdfplumber log noise suppressed. See `docs/features/0004-pdf-extraction-efficiency.md`. |
```

---

## Implementation plan

Shipped as a single branch `feature/0004-pdf-extraction-efficiency` (later
re-committed directly to main).

1. `arg/config.py`: add `pdf_min_chars_per_page`, `pdf_extract_timeout_seconds`,
   `embed_num_ctx`; remove dead `pdf_batch_size`.
2. `arg/pipeline.py`: `embed()` options use `config.embed_num_ctx`.
3. `arg/crawler/extractors.py`: silence pdfminer/pdfplumber; pre-flight density
   check; AcroForm fast-path.
4. `arg/crawler/crawler.py`: `_extract_pdf_with_timeout`; update `_extract_for_path`.
5. `tests/unit/test_extractors.py`: 3 new tests.
6. `tests/unit/test_crawler.py`: 2 new tests.
7. `pytest tests/unit/ -x` — all pass.
8. Commit `perf(extractors): Feature 0004 — PDF extraction efficiency`.
9. Push, ff-merge to `main`, delete branch.
