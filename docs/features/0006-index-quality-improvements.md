# Feature 0006: Index and Extraction Quality Improvements

**Status:** draft
**Created:** 2026-06-01

---

## Motivation

A structured analysis of the May 28–31 2026 indexing run (5,865 docs, 92,905
chunks, 113,731 log records — see `reports/indexing_report.html`) surfaced ten
actionable recommendations. Two were already implemented:

- **Parallel batch embedding** — Feature 0005 added `embed_batch_size=64` and
  native Ollama batch input; the pipeline no longer issues one HTTP request per
  chunk.
- **pdfminer noise suppression** — `extractors.py:59` already sets
  `logging.getLogger("pdfminer").setLevel(logging.ERROR)`, eliminating the 2,816
  benign font/color-space warnings that were burying real warnings.

This feature implements the remaining six recommendations. They fall into two
groups — indexer quality signals and PDF extraction coverage — shipped on two
branches to stay within the 7-file branch limit.

### What the report found

| Issue | Count | Impact |
|---|---|---|
| 0-chunk documents | 20 | Completely unsearchable; P&S contracts, tax forms, image receipts |
| AcroForm PDFs with missing field text | 127 | Filled-in names, addresses, dollar amounts invisible |
| Encrypted PDFs silently skipped | 32 | No searchable footprint at all; includes tax returns, IRA applications |
| Cluster granularity | 8 clusters / 5,865 docs | ~733 docs/cluster average; "Personal Life" and "Rental Property" are catch-alls |
| Documents exceeding 200 chunks | 57 docs | RAV4 manual (2,275 chunks), Ashley Book of Knots (1,872 chunks) skew BM25 |
| Low-confidence OCR pages | Unknown | 1,382 docs needed OCR; no quality signal logged |

The 20 zero-chunk documents include the two fully-executed Purchase & Sale
agreements for 3 Seal Harbor Rd PH312 (the Winthrop, MA property) — the most
critical documents in the corpus that are entirely unsearchable.

---

## Scope

**In scope:**

- Warn (at `WARNING` level) when a document is indexed with 0 chunks, both
  per-document and as a summary count at the end of `pipeline.index()`.
- Add a `max_chunks_per_doc` config field (default 0 = unlimited) that caps how
  many chunks any single document contributes. Documents that hit the cap log a
  warning. Prevents large reference manuals from dominating BM25 and ChromaDB
  frequency statistics.
- Extract filled AcroForm field values from PDFs by calling `page.widgets()` on
  each page, appending `"Label: Value"` lines to the page text. Recovers content
  from P&S contracts, leases, W-9s, 1099s, and voter registration forms that
  currently produce 0 or very few chunks.
- Raise the default `n_clusters` from 8 to 16 so the topic cluster graph has
  finer granularity (~366 docs/cluster vs. ~733) for a 5,865-doc corpus.
- Index a minimal stub Document for encrypted PDFs instead of silently skipping
  them. The stub contains the filename, parent directory name, and any XMP
  metadata (title, author, creation date) that pymupdf can read without
  decryption. Makes 32 encrypted tax returns and IRA applications findable by
  filename search.
- Log a `WARNING` when OCR extraction yields fewer than 25 characters for a page,
  flagging likely image-only or blank pages that produced near-empty results.

**Out of scope:**

- Decrypting password-protected PDFs. Stub indexing is the extent of what can be
  done without a password store.
- AcroForm extraction via pdfplumber `page.annots` — `page.widgets()` from
  pymupdf is already available (pymupdf is the primary fallback engine) and
  avoids a second library dependency.
- Hierarchical clustering (coarse + fine levels). Raising `n_clusters` to 16 is
  the minimum viable improvement; a hierarchical UI is a separate feature.
- True OCR confidence from Tesseract's hOCR output. PyMuPDF's `get_textpage_ocr`
  does not expose per-word Tesseract confidence in a stable, documented way; the
  char-count heuristic (`< 25 chars`) is a reliable proxy for pathological cases.
- Fixing the 65 dangling HTML links — those require editing the source HTML files
  outside the corpus root, which is operational, not a code change.
- Resolving the `Apartments_and_Houses` / `Apartments_and_houses`
  case-sensitivity split — that is a filesystem rename, not a code change.

---

## Deliverables

### Branch A — `feature-0006-index-quality` (5 files)

**`arg/config.py`**
- Change `n_clusters: int = 8` → `n_clusters: int = 16`
- Add `max_chunks_per_doc: int = 0` field (after `n_clusters`; 0 = unlimited)
- Add `"MAX_CHUNKS_PER_DOC": ("max_chunks_per_doc", int)` to `env_map`

**`arg/indexer/chunker.py`**
- Add `import logging` and `logger = logging.getLogger(__name__)` if not present
- In `chunk_document()`, add `hit_cap = False` before the outer `for section` loop
- After `position += 1` (currently the last line of the inner loop body), add:
  ```python
  if config.max_chunks_per_doc > 0 and len(out) >= config.max_chunks_per_doc:
      hit_cap = True
      break
  ```
- After the inner `for` loop ends, add `if hit_cap: break` to exit the outer loop
- After both loops, add warning and return:
  ```python
  if hit_cap:
      logger.warning(
          "chunk cap (%d) reached for %s — %d chunks total",
          config.max_chunks_per_doc,
          doc.path,
          len(out),
      )
  return out
  ```
  (The existing `return out` at line 170 is replaced by this block.)

**`arg/indexer/indexer.py`**
- In `_index_one()`, after `sections = chunk_document(doc, self.config)` (line 274):
  ```python
  if not sections:
      logger.warning("indexed with 0 chunks: %s", doc.path)
  ```
- In `index()`, add `zero_chunk_paths: list[Path] = []` after `stats = IndexStats()`
- After `n_chunks = self._index_one(doc)` (line 169), add:
  ```python
  if n_chunks == 0:
      zero_chunk_paths.append(doc.path)
  ```
- Before `self._write_bm25_index()` (line 195), add summary:
  ```python
  if zero_chunk_paths:
      logger.warning(
          "index complete: %d doc(s) produced 0 chunks: %s",
          len(zero_chunk_paths),
          ", ".join(str(p) for p in zero_chunk_paths),
      )
  ```

**`tests/unit/test_chunker.py`**
- `test_max_chunks_per_doc_cap` — build a document whose content produces >3
  chunks; set `max_chunks_per_doc=3` in config; assert `len(result) == 3` and
  that the invariant rule (global `position` counter) still holds for the
  retained chunks.
- `test_max_chunks_per_doc_zero_unlimited` — same content, `max_chunks_per_doc=0`;
  assert all chunks are returned (no truncation).

**`tests/unit/test_indexer.py`**
- `test_zero_chunk_doc_logs_warning` — create a `Document` with empty content;
  call `_index_one`; use `caplog` at `WARNING` level and assert the
  "0 chunks" warning is emitted with the document path.
- `test_index_summary_zero_chunk_warning` — feed `index()` two documents, one
  with content and one empty; assert the summary "index complete: 1 doc(s)"
  WARNING fires after the loop.

### Branch B — `feature-0006-pdf-extraction` (2 files)

**`arg/crawler/extractors.py`**

*AcroForm widget extraction* — inside `extract_pdf` generator, in the per-page
loop, after `body_text` is assembled from `stripped_lines` + `markdown_tables`
and **before** `body_text = _inject_pdf_heading_sentinels(body_text, sentinel_map)`:

```python
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
```

The sentinel injection and text cleaning then run on the combined body text.
`is_form` is already set at the top of the generator (`is_form = bool(doc.is_form_pdf)`).

*Encrypted PDF stub* — add helper function before `extract_pdf_to_document`:

```python
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
```

Modify `extract_pdf_to_document` — replace the early return on `meta is None`
(currently line 1021–1022):
```python
if meta is None:
    return _make_encrypted_pdf_stub(path)
```

*OCR quality logging* — add module-level constant near other PDF thresholds:
```python
_OCR_LOW_QUALITY_CHARS: int = 25
```

In `extract_pdf` per-page loop, after `lines, total_chars = _pymupdf_extract_ocr(fitz_page)`
(currently inside the Stage 1c block), add:
```python
if total_chars < _OCR_LOW_QUALITY_CHARS:
    logger.warning(
        "Low OCR quality: page %d of %s yielded only %d chars",
        page_index + 1,
        path,
        total_chars,
    )
```

**`tests/unit/test_extractors.py`**
- `test_encrypted_stub_returns_none_for_nonpdf` — call `_make_encrypted_pdf_stub`
  with a `.txt` file path; assert it returns `None`.
- `test_encrypted_stub_structure` — patch `fitz.open` to return a mock with
  `is_encrypted=True` and `metadata={"title": "Test", "author": "A"}`;
  assert returned Document has `is_encrypted_stub=True` in metadata, correct
  title, and "[Encrypted PDF" in content.
- `test_acroform_widgets_appended_to_page_text` — patch `fitz.Page.widgets` to
  yield a mock widget (`field_label="Buyer"`, `field_value="Andy Worth"`); call
  `extract_pdf_to_document` on an AcroForm-flagged PDF mock; assert "Buyer: Andy
  Worth" appears in the document content. (Marked `@pytest.mark.pdf` since it
  requires a fitz mock or fixture PDF.)

---

## Design notes

**AcroForm: why `page.widgets()` and not pdfplumber `page.annots`:** pymupdf is
already the engine for AcroForm PDFs (pdfplumber is skipped when `is_form=True` —
see the `if not is_form and not is_image_dominated:` guard). `page.widgets()` is
the natural pymupdf API for form fields and returns typed objects with
`field_label` and `field_value` attributes. No additional dependency.

**Encrypted stub: why re-open the file:** `_open_pdf` already closed the doc
before returning `None`. Re-opening inside `_make_encrypted_pdf_stub` is a
separate concern and keeps the existing `_open_pdf` / `extract_pdf_metadata`
contract unchanged. The 32 encrypted files are a one-time cost at index time.

**Chunk cap: why 0 = unlimited (not a sentinel like -1):** `int` defaults are
natural at 0; the sentinel is documented in the field comment. The cap is enforced
in the chunker rather than the indexer so `chunk_document` is self-contained and
testable in isolation.

**n_clusters 8 → 16:** With 5,865 documents and `min_cluster_docs=10`, 8 clusters
averaged 733 docs/cluster. Two clusters ("Personal Life Categories" and "Rental
Property Records") each held 17–19% of the corpus — too coarse for useful
navigation. 16 clusters targets ~366 docs/cluster and should split those two broad
clusters. The next natural increment (24) can be revisited after a re-index.

**OCR quality threshold 25 chars:** The `ocr_char_threshold` config (default 100)
is the threshold below which OCR *fires*. A page that OCR'd but still produced
fewer than 25 chars is almost certainly blank or image-only — the warning surfaces
these for manual inspection. The 25-char constant is a module-level sentinel
(`_OCR_LOW_QUALITY_CHARS`) so it can be found and adjusted without touching
config. A configurable threshold was considered but deferred (low priority).

**Locality:** no change — all operations remain local. The encrypted stub uses
`fitz.open` which is a local file read, not a network call.

**mypy:** `fitz.Page.widgets()` is typed in pymupdf stubs; `Widget.field_label`
and `Widget.field_value` may require `# type: ignore[attr-defined]` if stubs are
incomplete — verify at implementation time.

---

## Test points

Unit:
- `chunk_document` with `max_chunks_per_doc=3` and content producing 5 chunks →
  returns exactly 3 chunks; global `position` values are 0, 1, 2.
- `chunk_document` with `max_chunks_per_doc=0` → all chunks returned.
- `_index_one` with empty-content doc → WARNING "0 chunks" logged.
- `index()` with one empty doc → summary WARNING "1 doc(s) produced 0 chunks".
- `_make_encrypted_pdf_stub` with non-PDF → returns `None`.
- `_make_encrypted_pdf_stub` with mocked encrypted fitz doc → returns `Document`
  with correct metadata and `"[Encrypted PDF"` in content.
- AcroForm field values appear in page text when `page.widgets()` yields them.

Integration (Ollama-dependent):
- Re-index corpus after this change; confirm previously-0-chunk AcroForm docs
  now produce ≥1 chunk.
- Confirm encrypted PDFs appear in `pipeline.stats()` document count.
- Confirm `get_topic_clusters()` returns 16 clusters after a fresh `pipeline.index()`.
- Query "Andy Worth" or "3 Seal Harbor" and confirm results include the P&S
  contracts (previously invisible due to 0 chunks).

---

## Open questions / risks

- **AcroForm widget values for multi-value fields (checkboxes, lists):** `field_value`
  for a checkbox is typically `"Yes"` or `"Off"`. These are useful values; no special
  handling needed. List boxes return the selected option string. Drop-down fields
  similarly. No risk identified.
- **pymupdf Widget stubs:** if `Widget.field_label` / `Widget.field_value` are not in
  the installed pymupdf stubs, mypy will fail. Add `# type: ignore[attr-defined]`
  at the access sites and note in the commit message.
- **n_clusters change triggers full cluster recompute:** changing the default does
  not automatically re-cluster existing indexed corpora. The user must run
  `pipeline._recompute_clusters_bg()` or re-index. Document this in the commit
  message or operational notes.
- **Encrypted stub + doc_hashes.json:** the stub Document is hashed like any other
  doc. If the encrypted file is later decrypted and re-indexed, its hash will
  change and the indexer will replace the stub. This is correct behavior.

---

## CLAUDE.md impact

Add a row to the Section 13 Architectural Decision Records table:

| Decision | Summary |
|---|---|
| Index quality (Feature 0006) | Warns on 0-chunk docs; `max_chunks_per_doc` cap prevents large manuals from dominating BM25; AcroForm field values extracted via `page.widgets()`; encrypted PDFs produce searchable stubs; `n_clusters` default raised 8→16; OCR quality logged when page yields <25 chars post-OCR. See `docs/features/0006-index-quality-improvements.md`. |

---

## Implementation plan

### Branch A — `feature-0006-index-quality`

1. `git switch main && git pull && git switch -c feature-0006-index-quality`
2. Edit `arg/config.py` — change `n_clusters` default, add `max_chunks_per_doc`
3. Edit `arg/indexer/chunker.py` — add cap logic with break + warning
4. Edit `arg/indexer/indexer.py` — add per-doc and summary 0-chunk warnings
5. Add tests to `tests/unit/test_chunker.py` and `tests/unit/test_indexer.py`
6. `pytest tests/unit/test_chunker.py tests/unit/test_indexer.py tests/unit/test_config.py -xq`
7. `pytest tests/unit/ -q` — full unit suite must stay green
8. `ruff check arg/ && mypy arg/ --ignore-missing-imports`
9. Locality grep (no new outbound calls)
10. Stage explicit paths, commit `feat(indexer): 0-chunk warning, chunk cap, n_clusters 16`
11. Push branch; ff-merge to main; delete branch

### Branch B — `feature-0006-pdf-extraction`

1. `git switch main && git pull && git switch -c feature-0006-pdf-extraction`
2. Edit `arg/crawler/extractors.py` — add AcroForm widget extraction (Step 1d),
   `_make_encrypted_pdf_stub`, modify `extract_pdf_to_document`, add
   `_OCR_LOW_QUALITY_CHARS` constant and warning
3. Add tests to `tests/unit/test_extractors.py`
4. `pytest tests/unit/test_extractors.py -xq`
5. `pytest tests/unit/ -q`
6. `ruff check arg/ && mypy arg/ --ignore-missing-imports`
7. Locality grep
8. Stage explicit paths, commit `feat(extractors): AcroForm fields, encrypted stubs, OCR quality warn`
9. Push branch; ff-merge to main; delete branch
