# Plan: Parallel extraction pipeline

Findings from live log analysis and hardware analysis of a 1500+ document corpus
run on Apple M1 Max (10-core CPU, 32-core GPU, 16-core Neural Engine, 64GB RAM).
To be converted into a feature doc once Feature 0003 is complete.

**Not yet a feature.** Do not implement until Feature 0003 is merged and the
`pdf-extraction-efficiency` plan has been converted and implemented first
(that plan's items are easier and reduce the extraction problem before adding
concurrency to it).

---

## Background

The log analysis showed:
- 97% of a 22.5-hour indexing run was PDF extraction (CPU-bound).
- 3% (45 minutes) was embedding + writing to ChromaDB/Kuzu.
- The M1 Max has 8 performance cores and a 16-core Neural Engine, nearly all of
  which sit idle during the current serial extract→embed→extract→embed loop.
- `embed_batch()` in `pipeline.py` currently calls Ollama once per chunk in a
  Python loop — the Ollama client's `embed()` has accepted
  `input: Union[str, Sequence[str]]` (a list) since the current version, meaning
  all chunks of a document can be sent in a single HTTP call and the Neural
  Engine can process them in parallel.
- `pdf_batch_size: int = 10` exists in `ARGConfig` and is wired to the
  `PDF_BATCH_SIZE` env var but is **never used anywhere in the codebase**.

---

## Item 1: Fix `embed_batch` to use Ollama's native batch input

**File:** `arg/pipeline.py`
**Where:** `_OllamaEmbedderAdapter.embed_batch()`, currently ~line 291.

**Current code:**
```python
def embed_batch(self_inner, texts: list[str]) -> list[list[float]]:
    return [self_inner.embed(t) for t in texts]
```

This is a sequential HTTP loop: 2275 chunks × 80ms = 182 seconds for the RAV4
Owners Manual. The Ollama client already supports passing a list:
```
ollama.Client.embed(input: Union[str, Sequence[str]])
```

**Fix:** Sub-batch into groups of `config.embed_batch_size` (new config field,
default 64) and call `_client.embed` once per sub-batch. Each sub-batch is one
HTTP round-trip; the Neural Engine processes the whole batch in parallel:

```python
def embed_batch(self_inner, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    batch_size = config.embed_batch_size  # default 64
    results: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        sub = texts[start : start + batch_size]
        resp = _client.embed(
            model=_model,
            input=sub,            # list, not a single string
            truncate=True,
            options={"num_ctx": config.embed_num_ctx},
        )
        results.extend(list(e) for e in resp.embeddings)
    return results
```

The per-text retry/halving logic in `embed()` handles context-exceeded errors for
single texts. For batch calls, `truncate=True` is set, so Ollama handles
truncation server-side without erroring. The retry loop in `embed()` is only
needed for the single-text `embed()` method (used for query embedding at retrieval
time); `embed_batch` does not need it.

**New config field in `arg/config.py`:**
```python
embed_batch_size: int = 64   # chunks per Ollama embed call
```
Add `"EMBED_BATCH_SIZE": ("embed_batch_size", int)` to env-var map.

**Repurpose `pdf_batch_size`:** `pdf_batch_size: int = 10` has been dead config
since it was added. Remove it and its env-var entry `PDF_BATCH_SIZE` to avoid
confusion. Replace with `embed_batch_size` above.

**Expected impact:** For a 2275-chunk PDF, reduces embed phase from 182s to
~36 calls at ~80ms each = ~3s (rough estimate; actual Neural Engine batch
throughput on M1 Max may be even faster).

**Tests to add** (`tests/unit/test_pipeline.py` or `tests/unit/test_embedder.py`):
- `embed_batch` with 100 texts and `embed_batch_size=64` makes exactly 2 calls
  to `_client.embed` (assert via mock).
- `embed_batch` with 0 texts returns `[]`.
- `embed_batch` with 1 text makes 1 call with `input=[text]` (not a bare string).

---

## Item 2: Split crawler into path discovery and extraction

**File:** `arg/crawler/crawler.py`
**Why:** The parallel extraction pool (Item 3) needs to dispatch extraction jobs
to worker threads. Currently `crawl()` combines BFS path discovery and document
extraction in a single generator; you cannot extract documents in parallel
without restructuring this.

**Fix:** Extract an inner function `_discover_paths()` that yields `Path` objects
in the same BFS + dirwalk order as today, without calling `_extract_for_path`.
The public `crawl()` function becomes a thin wrapper that calls `_discover_paths`
and then either runs `_extract_for_path` inline (current behaviour, used when
`extraction_workers=1`) or dispatches to the thread pool (used when
`extraction_workers > 1`).

```python
def _discover_paths(
    docs_root: Path,
    config: ARGConfig,
    path_filter: Callable[[Path], bool] | None,
) -> Iterator[Path]:
    """BFS from index.html + directory walk. Yields absolute Paths in order.

    Link enqueueing still happens here because it requires reading the HTML
    (to find <a href> targets) — but extraction into Document objects is
    deferred to the caller.
    """
    ...
```

**Important:** Link discovery (following `<a href>` links to find the next HTML
file to crawl) requires reading each HTML file — this is unavoidably serial in
the BFS phase because the next path to visit depends on links found in the
current file. PDF and text files do not contribute links to the BFS queue.

The split therefore works as follows:
- **BFS phase (serial):** read HTML files to follow links; enqueue linked paths;
  yield HTML paths for extraction.
- **Dirwalk phase (parallel):** all PDF and text files found by directory walk
  are independent of each other and can be extracted in parallel.

An alternative: keep the BFS phase serial but make the dirwalk phase use the
thread pool. This is simpler and still captures most of the parallelism gain
(PDFs are the slow files; most HTML files are fast anyway).

**Tests:** Existing `test_crawler.py` tests should pass without modification if
`crawl()` preserves its current yielded-document order and behaviour.

---

## Item 3: Producer-consumer extraction pool

**File:** `arg/pipeline.py` (`index()` method) and `arg/crawler/crawler.py`
**Prerequisite:** Item 2 (path/extraction split) for full parallelism; or a
simpler version (see below) that doesn't require the split.

**Architecture:**
```
[Path discovery — BFS + dirwalk, single thread]
         ↓  path queue (unbounded for discovery, bounds set by doc queue)
[Extraction pool — ThreadPoolExecutor(max_workers=N)]
    worker 1: _extract_for_path(path1) → Document
    worker 2: _extract_for_path(path2) → Document
    ...
         ↓  document queue (bounded, maxsize=2×N for backpressure)
[Indexer consumer — single thread]
    embed_batch → ChromaDB upsert → Kuzu writes → hash save → log
```

**Why `ThreadPoolExecutor` and not `ProcessPoolExecutor`:**
- pdfplumber and pymupdf both have C extensions that release the Python GIL
  during file I/O, rendering, and font parsing — the GIL is not the bottleneck.
- `ThreadPoolExecutor` avoids the serialization cost of pickling large `Document`
  objects across process boundaries.
- All workers share the same process memory, so config and logger are naturally
  shared.

**Why the consumer must be single-threaded:**
- ChromaDB's SQLite backend is not safe for concurrent writes.
- Kuzu similarly expects a single writer per connection.
- The `_hashes` dict and `_write_bm25_index()` at end of run are not
  thread-safe.

**New config field:**
```python
extraction_workers: int = 1   # 1 = current serial behaviour; 4–6 recommended on M1 Max
```
Add `"EXTRACTION_WORKERS": ("extraction_workers", int)` to env-var map.

Default of 1 preserves current behaviour; no existing tests break.

**Simplified implementation (no crawler split required):**

If splitting the crawler is deferred, a simpler approach wraps just the
dirwalk phase in parallel while keeping BFS serial:

```python
# In pipeline.index() or in crawl():
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue as _queue

def _parallel_crawl(docs_root, config, path_filter, workers):
    doc_queue = _queue.Queue(maxsize=workers * 2)

    def producer():
        paths = list(_discover_paths(docs_root, config, path_filter))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_for_path, p, config): p for p in paths}
            for fut in as_completed(futures):
                doc = fut.result()
                if doc:
                    doc_queue.put(doc)
        doc_queue.put(None)  # sentinel

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    while True:
        doc = doc_queue.get()
        if doc is None:
            break
        yield doc
```

**Recommended worker count for M1 Max:**
- 5 extraction workers + 1 consumer thread = 6 active threads
- Leaves 2 of 8 P-cores for Ollama's process and the OS
- With 64 GB RAM, 5 simultaneous pdfplumber instances on large PDFs is
  comfortable (each uses ~200–500 MB at peak)

**Backpressure:** The bounded document queue (`maxsize=workers*2`) naturally
throttles the extraction pool if the consumer (Ollama embedding) is slower than
extraction. Without backpressure, a fast extraction phase could fill RAM with
hundreds of large `Document` objects.

**Tests to add:**
- `test_parallel_crawl_produces_same_docs_as_serial`: with `extraction_workers=4`,
  assert the set of indexed doc_ids matches `extraction_workers=1`.
- `test_extraction_workers_default_is_serial`: `extraction_workers=1` produces
  identical output to current behaviour (regression guard).
- `test_backpressure`: mock `_extract_for_path` to be slow; assert the producer
  thread blocks rather than accumulating unbounded documents.

---

## Item 4: Page-level parallelism within large PDFs (optional, future)

**File:** `arg/crawler/extractors.py`

This is a targeted optimization for the specific worst-offender files (large
scanned maps, thick technical manuals). It is more invasive than Items 1–3 and
should be deferred until those are in place and profiled.

**Problem:** Even with extraction_workers=5, a single 8,750-second PDF blocks
one worker for 2.4 hours. The other 4 workers may finish all remaining PDFs
while this one is still running.

**Approach:** When `doc.page_count > config.pdf_parallel_page_threshold` (e.g.,
100 pages), split the PDF into N page-range jobs:

```python
def extract_pdf_parallel(path: Path, config: ARGConfig) -> Document | None:
    import fitz
    doc = fitz.open(str(path))
    n_pages = doc.page_count
    doc.close()
    chunk_size = 50  # pages per worker
    ranges = [(i, min(i + chunk_size, n_pages))
              for i in range(0, n_pages, chunk_size)]

    with ProcessPoolExecutor(max_workers=4) as pool:
        page_docs = list(pool.map(
            _extract_page_range,
            [(path, start, end, config) for start, end in ranges]
        ))
    # merge page_docs in order → assemble into single Document
    ...
```

`ProcessPoolExecutor` (not Thread) is used here because pymupdf's `fitz.open()`
is not thread-safe when multiple threads open the same file path simultaneously.

**New config fields:**
```python
pdf_parallel_page_threshold: int = 100   # pages; below this, use single-threaded
pdf_page_workers: int = 4
```

**Deferral rationale:** Items 1–3 plus the `pdf-extraction-efficiency` plan
items together reduce the worst cases significantly. The two 3.6-hour geo maps
would be eliminated by the text-density check (Item 1 of the other plan). The
HVAC manuals would be 2× faster from the single-pass fix (Feature 0003) and
covered by the timeout. Page-level parallelism is only needed if large PDFs
remain a bottleneck after all those changes.

---

## Implementation order

**Phase 1 — no architecture change required (implement first):**
1. Item 1: `embed_batch` → native Ollama batch input. Self-contained 20-line
   change with high ROI.
2. Remove dead `pdf_batch_size` config; add `embed_batch_size` and
   `embed_num_ctx` (from the extraction efficiency plan).

**Phase 2 — requires crawler refactor:**
3. Item 2: `_discover_paths()` split in `crawler.py`.
4. Item 3: Producer-consumer extraction pool in `pipeline.py`. Start with
   `extraction_workers=1` default; increment and profile.

**Phase 3 — only if profiling shows need after Phases 1–2:**
5. Item 4: Page-level parallelism for very large PDFs.

---

## Estimated impact on M1 Max (rough projections)

| Change | Extraction time | Embed time | Total |
|---|---|---|---|
| Baseline (current) | 21.75h | 0.75h | 22.5h |
| + embed_batch fix | 21.75h | ~0.1h | ~21.85h |
| + extraction_workers=5 | ~4.4h | ~0.1h | ~4.5h |
| + text-density check (other plan) | ~1h | ~0.1h | ~1.1h |
| + extraction timeout (other plan) | ~1h | ~0.1h | ~1.1h |

The dominant win is parallel extraction (5× speedup). The batch embedding fix
is small in absolute terms but effectively free (3 lines of code).
