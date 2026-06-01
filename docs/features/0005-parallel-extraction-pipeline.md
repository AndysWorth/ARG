# Feature 0005: Parallel extraction pipeline

**Status:** shipped (commit: `c4d7fe9`)
**Created:** 2026-05-29
**Merged:** 2026-06-01

---

## Motivation

The 22.5-hour indexing run that motivated Feature 0004 also revealed two
underutilised hardware resources:

**Idle CPU cores:** Even with Feature 0004's pre-flight checks eliminating the
worst-offending PDFs, extraction is still CPU-bound (pdfplumber + pymupdf
rendering). The M1 Max has 8 performance cores. The current serial
extract→embed→extract→embed loop uses one core at a time. With pdfplumber and
pymupdf both releasing the Python GIL during I/O and rendering, `ThreadPoolExecutor`
gives true multi-core parallelism for extraction.

**Idle Neural Engine:** `embed_batch()` previously called `ollama.Client.embed()`
once per text in a Python loop. Ollama's `embed()` has accepted
`input: Union[str, Sequence[str]]` (a list) for the current version, allowing all
chunks of a document to be sent in a single HTTP call so the 16-core Neural Engine
can process them in parallel. For a 2,275-chunk PDF at ~80ms per call:
- Before: 2275 calls × 80ms ≈ 182 seconds
- After: 36 batches of 64 × 80ms ≈ 3 seconds

There was also a dead config field `pdf_batch_size: int = 10` that had a
`PDF_BATCH_SIZE` env-var mapping but was never read anywhere in the codebase
(removed in Feature 0004).

**Rough projections for M1 Max (22.5-hour baseline):**

| Change | Extraction | Embed | Total |
|---|---|---|---|
| Baseline | 21.75h | 0.75h | 22.5h |
| + embed_batch fix | 21.75h | ~0.1h | ~21.85h |
| + extraction_workers=5 | ~4.4h | ~0.1h | ~4.5h |
| + text-density check (F0004) | ~1h | ~0.1h | ~1.1h |
| + extraction timeout (F0004) | ~1h | ~0.1h | ~1.1h |

## Scope

**In scope:**

- Fix `embed_batch()` in `_OllamaEmbedderAdapter` to sub-batch texts using the
  native Ollama list input, sending one HTTP request per sub-batch of
  `embed_batch_size` texts (default 64).
- Add `embed_batch_size: int = 64` config field.
- Add `extraction_workers: int = 1` config field (1 = serial, preserves existing
  behaviour by default).
- Refactor `crawl()` to split the dirwalk phase into path collection followed by
  optional parallel extraction: when `extraction_workers > 1`, dispatch paths to
  `ThreadPoolExecutor(max_workers=N)` with a bounded `Queue(maxsize=N*2)` for
  backpressure.
- The BFS phase (HTML link-following) remains serial — each HTML file must be read
  to discover the next path to visit, so parallel BFS is not possible without
  doubling HTML reads.

**Out of scope:**

- Page-level parallelism within a single large PDF (Item 4 from the plan — deferred
  until after Items 1–3 are profiled in production; the text-density check from
  Feature 0004 already eliminates the worst-offending image-dominated PDFs).
- `ProcessPoolExecutor` for extraction (ThreadPoolExecutor is sufficient because
  pdfplumber and pymupdf release the GIL; pickling large `Document` objects across
  process boundaries would add overhead with no benefit).
- Parallelising the BFS phase — requires reading HTML twice (once for links, once
  for content) and provides little benefit since HTML extraction is fast.
- Incremental BM25 updates during parallel indexing.

## Deliverables

**`arg/config.py`**
- New field `embed_batch_size: int = 64` — chunks per Ollama embed call. Env var
  `EMBED_BATCH_SIZE`.
- New field `extraction_workers: int = 1` — dirwalk extraction concurrency. Env var
  `EXTRACTION_WORKERS`.

**`arg/pipeline.py`**
- `_OllamaEmbedderAdapter.embed_batch()`: sub-batches `texts` by `config.embed_batch_size`,
  calls `_client.embed(model=…, input=sub_batch, truncate=True, options=…)` once per
  sub-batch, extends results list. Returns `[]` immediately for empty input.

**`arg/crawler/crawler.py`**
- `crawl()`: dirwalk phase restructured to collect all paths first (adding to `seen`),
  then either extract serially (`extraction_workers ≤ 1`) or dispatch to
  `_parallel_dirwalk()`.
- New `_parallel_dirwalk(paths, workers, config, docs_root, seen, bfs_queue)`:
  producer thread submits all paths to `ThreadPoolExecutor(max_workers=workers)`,
  feeds completed `Document` objects into a `Queue(maxsize=workers*2)` for
  backpressure; consumer (main generator) normalises `links_to` and yields each
  document.

**`tests/unit/test_pipeline.py`** — 3 new tests:
- `test_embed_batch_empty_returns_empty`: `embed_batch([])` returns `[]`, no Ollama
  call made.
- `test_embed_batch_single_text_uses_list_input`: 1 text → `_client.embed` called
  once with `input=[text]` (list, not bare string).
- `test_embed_batch_sub_batches_by_embed_batch_size`: 100 texts with `embed_batch_size=64`
  → exactly 2 calls to `_client.embed`.

**`tests/unit/test_crawler.py`** — 3 new tests:
- `test_extraction_workers_default_is_serial`: `extraction_workers=1` produces the
  same doc set as the pre-feature serial path.
- `test_parallel_crawl_produces_same_doc_set_as_serial`: `extraction_workers=4`
  yields the same set of paths as `extraction_workers=1`.
- `test_parallel_dirwalk_bounded_queue_no_deadlock`: 10 paths with `workers=2`
  (maxsize=4) completes without deadlock, yields all 10 documents.

**`tests/conftest.py`** (CI fix bundled with this feature)
- `_FakeEmbedder.embed()`: replaced `hash(text)` with `hashlib.md5` per word.
  `hash()` is randomised per process by `PYTHONHASHSEED`; ~30% of seeds caused
  hash collisions that made `page_b.html` unreachable in integration test retrieval.
  The bag-of-words approach is stable across platforms and keyword-aware.

## Design notes

**Why `ThreadPoolExecutor` (not `ProcessPoolExecutor`):** pdfplumber and pymupdf
both have C extensions that release the Python GIL during file I/O, rendering, and
font parsing — the GIL is not the bottleneck. `ThreadPoolExecutor` avoids the
serialisation cost of pickling large `Document` objects across process boundaries,
and all workers share the same process memory so config and logger are naturally
shared.

**Why the consumer must be single-threaded:** ChromaDB's SQLite backend is not safe
for concurrent writes. Kuzu similarly expects a single writer per connection. The
`_hashes` dict and `_write_bm25_index()` at end of run are not thread-safe. The
producer-consumer design keeps extraction parallel and indexing serial.

**BFS phase stays serial:** HTML link-following is inherently sequential — the next
path to visit depends on links found in the current file. Making BFS parallel would
require reading each HTML file twice (once for links in a discovery pass, once for
full content extraction in the parallel pass). Since HTML extraction is fast
(< 5ms/file) and PDFs are the bottleneck, this trade-off is not worth it.

**Bounded queue for backpressure:** `Queue(maxsize=workers*2)` means the producer
blocks when the consumer (Ollama embedding) falls behind. Without backpressure, a
fast extraction phase on a large corpus could accumulate hundreds of large `Document`
objects in memory before embedding catches up.

**Recommended worker count for M1 Max:** 5 extraction workers + 1 consumer thread =
6 active threads. Leaves 2 of 8 P-cores for Ollama's process and the OS. With 64 GB
RAM, 5 simultaneous pdfplumber instances on large PDFs is comfortable (each uses
~200–500 MB at peak).

**`embed_batch` retry logic:** the per-text retry/halving loop in `embed()` handles
context-exceeded errors for single texts. For batch calls, `truncate=True` is set so
Ollama handles truncation server-side. The retry loop is only in `embed()` (used for
query embedding at retrieval time); `embed_batch` does not need it.

**Timeout pool coexistence:** `_extract_pdf_with_timeout` (Feature 0004) uses a
separate `ThreadPoolExecutor(max_workers=1)` per PDF call. When
`extraction_workers > 1`, each worker thread in the extraction pool may itself create
a single-worker timeout pool. These nest cleanly — Python thread pools do not have
a global limit.

**Locality:** no change — all operations remain local.

## Test points

Unit:
- `embed_batch([])` → `[]`, 0 Ollama calls.
- `embed_batch(["x"])` → `_client.embed` called with `input=["x"]` (list).
- 100 texts, `embed_batch_size=64` → exactly 2 Ollama calls.
- `extraction_workers=1` → same doc set as serial (regression guard).
- `extraction_workers=4` → same doc set as `extraction_workers=1`.
- `_parallel_dirwalk` with 10 paths, `workers=2` → completes, 10 docs yielded.

Integration (Ollama-dependent):
- Existing integration tests pass with both `extraction_workers=1` and `extraction_workers=4`.
- After `pipeline.index()` with `extraction_workers=4`, query returns correct results.

E2E:
- Existing e2e tests pass unchanged.

## Open questions / risks

- **Thread abandonment on early generator exit:** if the caller of `crawl()` breaks
  out of the iteration early, the producer thread will eventually block on
  `doc_queue.put(doc)` with no consumer. The thread is `daemon=True` so it's reaped
  on process exit. For long-running server processes this could leave a stuck thread
  per abandoned crawl. Acceptable for the current use case (crawls always run to
  completion during `pipeline.index()`).
- **Nested thread pools:** each worker in the extraction pool may spawn a
  `ThreadPoolExecutor(max_workers=1)` for the PDF timeout. With `extraction_workers=5`,
  this could create up to 10 threads simultaneously (5 extraction + up to 5 timeout
  pools). Well within the OS thread limit; no action needed.
- **Item 4 (page-level parallelism):** still deferred. If profiling after deploying
  Items 1–3 shows a single large PDF is still blocking a worker for hours, revisit.
  The text-density check from Feature 0004 eliminates the worst offenders.

## CLAUDE.md impact

No product-scope or user-facing changes. Section 13 ADR table gains a row:

```diff
+| Parallel extraction pipeline | Feature 0005: native Ollama batch input for embed_batch (embed_batch_size config); dirwalk phase parallelised with bounded producer-consumer pool (extraction_workers config). See `docs/features/0005-parallel-extraction-pipeline.md`. |
```

---

## Implementation plan

Shipped as a single branch `feature/0005-parallel-extraction`.

1. `arg/config.py`: add `embed_batch_size`, `extraction_workers`.
2. `arg/pipeline.py`: rewrite `embed_batch()` to sub-batch.
3. `arg/crawler/crawler.py`: restructure dirwalk; add `_parallel_dirwalk()`.
4. `tests/conftest.py`: fix `_FakeEmbedder` hash randomisation.
5. `tests/unit/test_pipeline.py`: 3 embed_batch tests.
6. `tests/unit/test_crawler.py`: 3 parallel extraction tests.
7. `pytest tests/unit/ tests/integration/ -x` — all 428 pass.
8. Commit messages:
   - `fix(ci): make _FakeEmbedder deterministic across PYTHONHASHSEED values`
   - `perf(embedder): native Ollama batch input for embed_batch (Feature 0005 Item 1)`
   - `perf(crawler): parallel dirwalk extraction pool (Feature 0005 Items 2+3)`
9. Push, ff-merge to `main`, delete branch.
