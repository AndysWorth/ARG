# Feature 0002: Stream the crawl → indexer pipeline

**Status:** complete
**Created:** 2026-05-15

---

## Motivation

`pipeline.index()` currently runs in two strictly sequential phases:

```python
documents = list(crawl(self.config.docs_root, self.config))   # ① extract every file
stats = self.indexer.index(documents)                         # ② write to DB
```

Step ① consumes the entire crawler generator before any database writes
happen. On a large corpus this is slow — every HTML / PDF / text file is
parsed and held in memory as a `Document` before the indexer touches Chroma
or Kuzu. The previously-shipped per-doc hash-save (Fix
`baf2a8a`) only helps once we're inside step ②.

The failure mode we hit in the wild: indexing a multi-thousand-file
corpus, the crawler got stuck mid-extraction on a scanned PDF whose pages
contained dozens of tiny embedded images. pymupdf's tesseract-backed OCR
chewed on it for many minutes. Ctrl-C at that point lost every minute of
prior crawl work — nothing had been written, no hashes saved. The next
run started from zero on the same problematic PDF.

Streaming the crawl → indexer handoff fixes this: each file is crawled,
extracted, embedded, written to all three stores, and its hash persisted
before the crawler advances to the next file. Ctrl-C at any moment leaves
every file processed up to that point durably in the index.

## Scope

**In scope:**
- Indexer accepts a streaming iterator and processes documents one at a
  time, instead of materialising the input via `list()`.
- Pipeline passes the crawl generator directly to the indexer (no
  intermediate list).
- Link-graph recording continues to work: accumulate `(src, target)`
  pairs in memory as documents stream through, then run a single link
  pass at the end of the run from those accumulated tuples (no need to
  retain `Document` objects).
- Progress log format adapts to the unknown total: `[#N]` instead of
  `[N/M]`.

**Out of scope:**
- Incremental BM25 rebuild. BM25 is still rebuilt once at the end of a
  full run. Mid-stream interrupts leave the previous run's BM25 file on
  disk (or no file on first run); the next full run rebuilds. A future
  feature could rebuild every K documents, but it's a meaningful complexity
  bump and the streaming-resumability win doesn't depend on it.
- Counting docs upfront so we can keep the `[N/M]` format. The whole
  point of streaming is to skip the up-front walk — a two-pass version
  would re-introduce part of the latency we're removing. `[#N]` is fine.
- Watcher-driven add/remove paths. Those are already per-file (the watcher
  fires one event at a time) and not affected by this change.

## Deliverables

- `arg/indexer/indexer.py`
  - Drop `docs = list(documents)` in `Indexer.index()`.
  - Iterate the input directly with a 1-indexed counter.
  - Accumulate `indexed_doc_ids: set[str]` and `link_records:
    list[tuple[str, str]]` as we stream; use them for the link pass
    after the iterator is exhausted.
  - Update the per-doc progress log format from `[N/M]` to `[#N]`.
- `arg/pipeline.py`
  - `index()` passes the crawl generator straight into `indexer.index()`
    instead of `list()`-wrapping it.
  - Move `self.explorer.invalidate_cluster_cache()` to BEFORE the
    streaming loop — once we start writing new docs, any existing
    cluster cache is stale regardless of how the run terminates.
- `tests/unit/test_indexer.py`
  - New `test_index_processes_documents_streamed` — pass a generator
    that asserts each `Document` has been written to disk before the
    next one is yielded. Confirms the streaming contract from the
    indexer's side.
  - Existing tests that pass lists still pass unchanged (a `list` is an
    `Iterable`).

No new dependencies. No public API changes.

## Design notes

**Why accumulate `(src, target)` tuples instead of keeping `Document`
references**: tuples of two strings are cheap (a few hundred bytes each).
A `Document` carries the full chunked text, metadata dict, and code-block
list — on a multi-megabyte HTML file that's a real footprint. The link
pass only needs source path + target path, so we keep only those.

**Why a separate `indexed_doc_ids` set even though the same data is in
the graph**: `kg.list_all_documents()` works but does a Kuzu query per
invocation. We do the union once at the end of the link pass. Cheap.

**Cluster cache invalidation moved**: currently `pipeline.index()`
invalidates the cluster cache AFTER `indexer.index()` returns. Streaming
makes a partial run plausible — the cluster cache is stale the moment any
new chunk lands, so invalidate before the loop starts. The
`invalidate_cluster_cache()` call is a single `unlink()` of a JSON file;
no perf concern.

**BM25 stays end-of-run**: rebuilding the BM25 pickle after every doc is
prohibitive (O(N) work per doc → O(N²) total). End-of-run rebuild is
cheap (single sort + write) and the temporarily-stale BM25 between an
interrupted run and the next full run is an acceptable degradation
(dense + graph retrieval still work; BM25 just under-reports).

**Schema-hash write stays end-of-run**: only useful once a "complete"
index has finished. Mid-stream interrupts already inhibit it. No change.

**Locality**: no change — same components, same network surface (Ollama
embed calls, ChromaDB local writes).

**mypy**: existing types already accept `Iterable[Document]`. No churn.

## Test points

Unit:
- New `test_index_processes_documents_streamed`: pass a generator whose
  `__next__` checks the hash file on disk; assert the previous doc's
  hash exists in the file before the next doc is yielded.
- Existing `test_hashes_persisted_incrementally_after_each_doc` continues
  to pass — it doesn't depend on `list()` materialisation.
- Existing `test_simulated_interruption_resumes_cleanly` continues to
  pass — interruption + resume semantics are unchanged at the indexer
  level; the win is moved to the pipeline level.

No integration / e2e changes needed; the contracts those tests assert
(doc counts, source surfacing) are end-state assertions and don't care
about the order of crawl / index work.

## Open questions / risks

- **Crawler exceptions mid-stream.** The crawler's existing try/except in
  `_extract_for_path` catches per-file errors and yields nothing for that
  file. A crash deeper in the crawler (extremely unlikely — `crawl()`
  itself is a thin generator) would propagate up out of `indexer.index()`
  and abort the run partway. The per-doc hash-save means the user's
  next run still resumes cleanly. No change required.
- **Generator re-entry.** The link-pass code currently iterates `docs`
  twice (once to build `known`, once to record edges). Streaming forces
  us to consume the iterator exactly once. Resolved by accumulating
  `(src, target)` tuples — no re-iteration needed.
- **A bug in `_index_one` that raises after the doc was partially written
  to Chroma but before the hash save.** Today this is rare and the doc
  becomes idempotently re-processed on next run (chunks are deleted and
  re-inserted by `_index_one`). Streaming doesn't make this worse; if
  anything, the more granular per-file resumability narrows the blast
  radius.

## CLAUDE.md impact

No CLAUDE.md / README changes. This is an internal implementation
refactor — no new behaviour visible to operators beyond "Ctrl-C is now
genuinely cheap mid-run".

Section 13 ADR table gets one new row (the doc-feature pointer):

```diff
 | Plain-text indexing | Feature 0001: ... |
+| Streaming indexer   | Feature 0002: crawler output streams directly into the indexer; per-file durability + lower memory footprint. See `docs/features/0002-streaming-indexing.md`. |
```

---

## Implementation plan

1. Branch `feature/0002-streaming-indexing` off `main`.
2. Update `arg/indexer/indexer.py`:
   - Drop `docs = list(documents)` in `index()`.
   - Loop with a 1-indexed counter; emit `[#N]` progress lines.
   - Accumulate `indexed_doc_ids` and `link_records` during the loop.
   - Replace the existing two-pass-over-`docs` link block with a single
     pass over `link_records`.
3. Update `arg/pipeline.py`:
   - Pass the crawl iterator directly to `self.indexer.index(...)`.
   - Move `self.explorer.invalidate_cluster_cache()` before the loop.
   - Keep the "crawl yielded N docs" log line, but emit it AFTER
     `indexer.index()` returns (we don't know N up front any more) —
     or rephrase to "crawl + index complete".
4. Add `test_index_processes_documents_streamed` to
   `tests/unit/test_indexer.py`.
5. Run `pytest tests/unit/` + `mypy arg/` — all clean.
6. Run `pytest tests/integration/` to confirm the on-disk shape is
   unchanged.
7. Run `pytest tests/e2e/ --deselect <real-LLM-tests>` to confirm the
   end-state assertions still pass.
8. Apply the Section 13 ADR row in CLAUDE.md.
9. Commit `feat(pipeline,indexer): stream crawl → indexer (Feature 0002)`
   referencing this doc.
10. Push, ff-merge to `main`, delete branch.
