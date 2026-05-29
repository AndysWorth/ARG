# Feature 0003: Large-corpus hardening

**Status:** draft
**Created:** 2026-05-29

---

## Motivation

ARG works well on small corpora (tens to low hundreds of documents) but has
several correctness bugs and performance bottlenecks that become problematic at
scale. Three categories of issues were identified in a review of the codebase:

1. **Correctness bugs** that are silent on small corpora but produce wrong results
   as the corpus grows: duplicate `LINKS_TO` edges accumulate across re-indexes;
   `list_all_documents` silently drops the `SKIP` clause in certain call patterns;
   nested `$and` filters passed to ChromaDB may be silently mishandled.

2. **Performance bottlenecks** that are O(total corpus) on every incremental
   update: the BM25 index is fully rebuilt from every chunk on every watcher
   event; topic cluster computation runs synchronously and blocks `index()` from
   returning; `_compute_clusters` loads all document embeddings into Python in a
   single allocation; `extract_pdf` opens pdfplumber twice per file; sub-query
   embedding is serial rather than parallel.

3. **Test and documentation gaps** that make the system fragile to maintain:
   `time.sleep` in watcher tests is flaky on slow CI; a hardcoded `/tmp` path
   in a unit test collides across parallel runs; private API coupling between
   `CorpusAnalyst` and `Retriever`; undocumented invariants that would mislead
   a future maintainer.

This feature addresses all three categories in a single planned build.

## Scope

**In scope:**

- Fix `add_link` to use MERGE semantics so duplicate edges do not accumulate
  across re-indexes.
- Fix `list_all_documents` to emit `SKIP` independently of `LIMIT`.
- Fix `_combine_where` to flatten nested `$and` clauses before passing to
  ChromaDB.
- Replace `rank_bm25` with `bm25s` (Rust-backed) so the BM25 full rebuild is
  ~50x faster; the full-rebuild-on-update pattern is preserved but made fast
  enough to be acceptable at large scale.
- Move topic cluster computation off the `index()` critical path into a daemon
  background thread; watcher-triggered recomputes also dispatch asynchronously.
- Batch the ChromaDB embedding fetch in `_compute_clusters` (2000-doc pages)
  to reduce peak memory pressure.
- Eliminate the second `pdfplumber.open()` call in `extract_pdf` with a
  single-pass approach that buffers raw page data before applying
  running-header filtering.
- Parallelize sub-query embedding in `_retrieve_union` using a thread pool so
  decomposed queries embed concurrently rather than sequentially.
- Push `ORDER BY chunk_count DESC SKIP/LIMIT` into Kuzu for
  `docs_by_chunk_count` pagination, replacing the all-docs-into-Python pattern.
- Replace `time.sleep` in watcher and e2e tests with polling loops.
- Fix the hardcoded `/tmp/dummy_docs` path in `test_chunker.py` to use
  `tmp_path`.
- Promote `Retriever._find_document` to a public method and update
  `CorpusAnalyst` to use the public API.
- Add a single-line comment to `chunker.py` documenting the global `pos`
  counter invariant.
- Document the route-ordering constraint in `server.py` with an enumerated list.
- Add `# Implements: docs/spec/section-NN-*.md` cross-references to all
  `arg/` modules.
- Add missing unit tests for each of the above correctness fixes.

**Out of scope:**

- True incremental BM25 updates (partial index merging). The `bm25s` swap
  makes a full rebuild fast enough; incremental merge is a separate feature.
- PCA before KMeans for very large corpora (>50k docs). Noted in design notes
  as a follow-up when needed.
- A read-write lock replacing the coarse `_lock` in `pipeline.py`. The
  background cluster thread removes the most painful contention; a proper
  RW lock is a separate improvement.
- Load/stress tests with synthetic large corpora. Out of scope for this
  feature; worth a dedicated performance benchmark suite later.
- Markdown-aware heading detection (that is Feature 0001 scope / future).

## Deliverables

**`arg/graph/knowledge_graph.py`**
- `add_link`: replace `CREATE (s)-[:LINKS_TO …]->(t)` with MERGE or an
  existence-guard `WHERE NOT EXISTS { … }` pattern to prevent duplicate edges.
- `list_all_documents`: move the `SKIP` clause outside the `limit > 0` branch
  so `offset` is always honoured when `> 0`, regardless of `limit`.
- New `list_documents_by_chunk_count(limit, offset)` method: Kuzu query with
  `ORDER BY d.chunk_count DESC`, server-side `SKIP`/`LIMIT`.

**`arg/retriever/retriever.py`**
- `_find_document` → `find_document` (public rename; update internal call site
  in `_stage0_enrichment`).
- `_combine_where`: flatten nested `$and` clauses instead of wrapping them.

**`arg/indexer/indexer.py`**
- No logic changes — `_write_bm25_index` continues to fetch all chunks and
  rebuild; the performance gain comes from the library swap in `bm25_index.py`.

**`arg/retriever/bm25_index.py`**
- Replace `rank_bm25.BM25Okapi` with `bm25s.BM25`. Preserve the public API:
  `build(id_text_pairs)`, `save(path)`, `load(path)`, `query(text, top_k)`,
  `score_all(text)`, `is_empty`.

**`pyproject.toml`**
- Remove `rank-bm25` dependency; add `bm25s>=0.2`.

**`arg/pipeline.py`**
- Add `_cluster_lock: threading.Lock` and `_cluster_thread` to `__init__`.
- New `_recompute_clusters_bg()` method: dispatches cluster recompute to a
  daemon thread; skips dispatch if a thread is already in flight.
- New `_run_cluster_recompute()`: called on the background thread; guards on
  `self._closed`.
- Replace all `self.explorer.get_topic_clusters()` and
  `self._recompute_clusters()` call sites with `self._recompute_clusters_bg()`.

**`arg/dci/explorer.py`**
- `_compute_clusters`: fetch document embeddings in batches of 2000 using
  `np.vstack` to assemble the final matrix.
- `docs_by_chunk_count`: call `kg.list_documents_by_chunk_count(limit, offset)`
  instead of `kg.list_all_documents()` + Python sort/slice.

**`arg/dci/analyst.py`**
- `find_document`: call `self.retriever.find_document(...)` (public) instead of
  `self.retriever._find_document(...)`.

**`arg/crawler/extractors.py`**
- `extract_pdf`: single-pass approach — one `pdfplumber.open()` context that
  buffers `(lines, tables, nchars)` for all pages, then applies
  `_detect_running_lines`, then processes pages with fitz. Second
  `pdfplumber.open()` call eliminated.

**`arg/generator/generator.py`**
- `_retrieve_union`: wrap the per-sub-query loop in a
  `concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 4))`.

**`arg/indexer/chunker.py`**
- Add one comment to the `pos = 0` initialisation:
  `# global across sections — Kuzu orders chunks by this; do not reset per section`

**`arg/server.py`**
- Replace the route-ordering comment with an enumerated list of all affected
  keyword routes (stats, compare, search, topics, graph, file, query, stream).

**`arg/` modules (all) — doc-only**
- Add `# Implements: docs/spec/section-NN-*.md` to each module-level docstring.
  See the module-to-spec mapping in the implementation plan.

**`tests/unit/test_knowledge_graph.py`**
- `test_add_link_idempotent`: same `(src, tgt, anchor)` twice → `get_linked_docs`
  returns target once.
- `test_add_link_distinct_anchors`: two different anchors → target returned once
  (graph expansion cares about reachability, not anchor text multiplicity).
- `test_list_all_documents_offset_without_limit`: `limit=0, offset=2` with 4 docs
  → 2 results.
- `test_list_documents_by_chunk_count_pagination`: 5 docs with varying chunk
  counts → correct 3-item page returned in descending order.

**`tests/unit/test_retriever.py`**
- `test_combine_where_flat_and`: both inputs are `$and` → single flat `$and`.
- `test_combine_where_mixed`: one `$and` + one plain filter → flat `$and`.
- `test_retrieve_with_filters_and_candidate_ids`: filters + candidate_doc_ids
  simultaneously → no ChromaDB error, correct results.

**`tests/unit/test_pipeline.py`**
- `test_index_returns_before_cluster_completes`: mock cluster to sleep; assert
  `index()` returns before the sleep elapses.
- `test_cluster_eventually_populated`: after `index()` returns, poll until
  `get_topic_clusters()` returns non-empty (timeout 3 s).

**`tests/unit/test_explorer.py`**
- `test_compute_clusters_batched_fetch`: inject >2000 mock embedding rows;
  assert `_compute_clusters` returns a valid cluster dict without error.

**`tests/unit/test_generator.py`**
- `test_retrieve_union_parallel`: with a mock query processor returning 3
  sub-queries, assert `retriever.retrieve` is called 3 times and results are
  correctly unioned.

**`tests/unit/test_chunker.py`**
- `test_chunk_overlap_is_applied`: replace `/tmp/dummy_docs` and `/tmp/dummy_db`
  with `tmp_path / "dummy_docs"` and `tmp_path / "dummy_db"`.

**`tests/unit/test_watcher.py`** and **`tests/e2e/test_full_rag.py`**
- Replace all `time.sleep(N)` that wait for async events with a
  `_wait_for(condition_fn, timeout, interval)` polling helper.

## Design notes

**BM25 library swap — why `bm25s`:** `rank_bm25` is pure Python with no
incremental update path; a full rebuild is the only option. `bm25s` is Rust-backed
(via PyO3), offers a nearly identical query API (`tokenize` + `BM25` +
`retrieve`/`get_scores`), and is ~10–50x faster at indexing. The full-rebuild
pattern is preserved — the rebuild just becomes fast enough. `tantivy-py` (true
incremental) is an alternative but adds more API complexity; defer to a later
feature if rebuild cost is still unacceptable after this change.

**Background cluster thread — why not async:** `pipeline.py` is synchronous
throughout; `get_topic_clusters` calls ChromaDB (blocking) and then runs KMeans
(CPU-bound). An `asyncio`-based solution would require converting significant call
paths to async. A daemon `threading.Thread` is simpler and sufficient: KMeans
releases the GIL during its numpy operations, so it doesn't block other threads.

**Why a single-slot cluster thread (skip if in flight):** watcher events can fire
rapidly (e.g., `git checkout` touching many files). Queuing a new KMeans run for
each event would cause unbounded stacking. If a run is already in flight it will
read fresh data from ChromaDB when it starts (because `invalidate_cluster_cache`
is called first inside `_run_cluster_recompute`), so skipping redundant enqueues
is safe.

**Batched embedding fetch — peak memory:** batching reduces the single large
allocation to a series of smaller ones but the KMeans step still needs the full
matrix. Peak memory is halved (each batch is freed after `np.vstack`), not
eliminated. A PCA pre-step (config option `pca_before_cluster`, `pca_dims=64`)
is the right follow-up for corpora >50k docs; leave a TODO comment in
`_compute_clusters` pointing at this.

**Parallel sub-query embedding — thread safety:** `Retriever.retrieve()` is read-
only during execution (BM25 index and ChromaDB client are not mutated). The thread
pool is safe. Cap `max_workers=4` to avoid overwhelming a single Ollama instance.

**`pdfplumber` single-pass:** the two-pass approach exists because running-header
detection needs all pages' lines before any page can be processed. The fix buffers
`(lines, tables, nchars)` for all pages inside a single `pdfplumber.open()` context,
then runs `_detect_running_lines` on the buffered data, then iterates for extraction
using fitz. Peak memory increases slightly (all raw page lines held simultaneously)
but this was already the case in the first `pdfplumber.open()` pass.

**Locality:** no change — all operations remain local. No new network calls.

**mypy:** `bm25s` ships type stubs. Verify stub availability; if stubs are absent
add `# type: ignore[import-untyped]` at the import site per the project pattern.
`concurrent.futures` is stdlib; no stub issues.

## Test points

Unit:
- `add_link` called twice with same triple → `get_linked_docs` returns target once.
- `add_link` with two distinct anchors → target returned once.
- `list_all_documents(limit=0, offset=2)` → correct skip with no limit.
- `list_documents_by_chunk_count` → correct descending order with pagination.
- `_combine_where` with nested `$and` → flat output.
- `_combine_where` with mixed inputs → flat `$and`.
- Retrieval with both `filters` and `candidate_doc_ids` active → no error.
- `index()` returns before cluster computation completes (mocked sleep).
- Cluster cache is eventually populated after `index()` returns.
- `_compute_clusters` with >2000 mock rows → correct result, no memory error.
- `_retrieve_union` with 3 sub-queries → 3 `retrieve` calls, correct union.
- `test_chunk_overlap_is_applied` uses `tmp_path` (no hardcoded `/tmp`).
- All `time.sleep` test sites replaced with `_wait_for` polling.

Integration (Ollama-dependent):
- `pipeline.index()` on Corpus A completes and returns promptly; cluster cache
  is populated within 5 s of return.
- After watcher triggers `add_document`, BM25 query for a token unique to the
  new doc surfaces that chunk (confirms BM25 rebuild with `bm25s` is correct).
- Re-indexing a document with links does not produce duplicate `get_linked_docs`
  results.

E2E (real LLM):
- Existing e2e tests pass unchanged — end-state assertions are unaffected by
  the internal changes here.

## Open questions / risks

- **`bm25s` pickle compatibility.** Existing `bm25_index.pkl` files were written
  by `rank_bm25`. After the swap, loading an old pickle will fail. The right
  mitigation is to detect load failure, log a warning, and return an empty index
  (triggering a rebuild on next `index()`). The schema-hash mechanism already
  handles re-indexing detection; this is a one-time migration concern.
- **Kuzu MERGE on relationship tables.** Kuzu's MERGE for relationship properties
  is version-dependent. If `MERGE (s)-[:LINKS_TO {anchor_text: $anchor}]->(t)`
  is not supported, fall back to the `WHERE NOT EXISTS { … }` guard pattern. Test
  against the project's pinned Kuzu version before committing.
- **Cluster background thread and `close()`.** The daemon thread must check
  `self._closed` before touching any ChromaDB or Kuzu handle. Add the guard at
  the top of `_run_cluster_recompute`. The existing `_bm25_rebuild_timer` guard
  pattern (`if self._closed: return`) is the right model.
- **`bm25s` `score_all` equivalent.** Verify that `bm25s`'s `get_scores` returns
  a score for every document in the index (not just top-k), matching the
  `score_all` contract used by Stage 0 enrichment.

## CLAUDE.md impact

No product-scope or user-facing changes in this feature. CLAUDE.md needs one
update: the Section 13 ADR table gains a row, and the Section 1 stack table gains
an updated BM25 entry:

**Section 1 stack table — update BM25 row:**

```diff
-| **Sparse Retrieval** | rank_bm25 (BM25Okapi) | ... |
+| **Sparse Retrieval** | bm25s (Rust-backed BM25) | Replaced rank_bm25; Feature 0003. |
```

**Section 13 ADR table — append:**

```diff
 | Streaming indexer   | Feature 0002: ... |
+| Large-corpus hardening | Feature 0003: bm25s replaces rank_bm25; cluster compute async; batched embedding fetch; single-pass PDF extraction; parallel sub-query embedding; correctness fixes for duplicate edges, SKIP/LIMIT, and nested $and filters. See `docs/features/0003-large-corpus-hardening.md`. |
```

---

## Implementation plan

Work in five sequential branches to keep diffs reviewable. Each branch is
ff-merged to `main` before the next starts.

**Branch 1 — `feature/0003a-graph-correctness`**
1. `git switch -c feature/0003a-graph-correctness`
2. Fix `add_link` (MERGE / existence guard) in `knowledge_graph.py`.
3. Fix `list_all_documents` SKIP clause in `knowledge_graph.py`.
4. Add `list_documents_by_chunk_count` to `knowledge_graph.py`.
5. Update `explorer.docs_by_chunk_count` to call the new method.
6. Add tests: `test_add_link_idempotent`, `test_add_link_distinct_anchors`,
   `test_list_all_documents_offset_without_limit`,
   `test_list_documents_by_chunk_count_pagination`.
7. `pytest tests/unit/test_knowledge_graph.py tests/unit/test_explorer.py -x`
8. `mypy arg/` clean; locality grep clean.
9. Commit `fix(graph): MERGE add_link + fix SKIP clause + chunk-count pagination (Feature 0003)`.
10. Push, ff-merge to `main`, delete branch.

**Branch 2 — `feature/0003b-retriever-correctness`**
1. `git switch -c feature/0003b-retriever-correctness`
2. Rename `_find_document` → `find_document` in `retriever.py`; update
   internal call site.
3. Update `analyst.py` to call `self.retriever.find_document(...)`.
4. Fix `_combine_where` to flatten nested `$and`.
5. Add tests: `test_combine_where_flat_and`, `test_combine_where_mixed`,
   `test_retrieve_with_filters_and_candidate_ids`.
6. `pytest tests/unit/test_retriever.py tests/unit/test_analyst.py -x`
7. `mypy arg/` clean.
8. Commit `fix(retriever): flatten nested $and + public find_document (Feature 0003)`.
9. Push, ff-merge to `main`, delete branch.

**Branch 3 — `feature/0003c-bm25s-swap`**
1. `git switch -c feature/0003c-bm25s-swap`
2. `pip install bm25s` in the venv.
3. Update `pyproject.toml`: remove `rank-bm25`, add `bm25s>=0.2`.
4. Rewrite `arg/retriever/bm25_index.py` using `bm25s`. Keep the public
   API (`build`, `save`, `load`, `query`, `score_all`, `is_empty`) unchanged.
   Add load-failure guard: catch `Exception` on load, log a warning, return
   empty index.
5. Verify `score_all` contract: `bm25s.BM25.get_scores(query_tokens)` must
   return one score per indexed document.
6. `pytest tests/unit/test_bm25_index.py tests/unit/test_retriever.py
   tests/unit/test_indexer.py -x`
7. `mypy arg/` — add `# type: ignore[import-untyped]` if `bm25s` lacks stubs.
8. Commit `perf(indexer): replace rank_bm25 with bm25s (Feature 0003)`.
9. Push, ff-merge to `main`, delete branch.

**Branch 4 — `feature/0003d-async-clusters-and-perf`**
1. `git switch -c feature/0003d-async-clusters-and-perf`
2. `pipeline.py`: add `_cluster_lock`, `_cluster_thread`, `_recompute_clusters_bg`,
   `_run_cluster_recompute`. Replace all cluster recompute call sites.
3. `explorer.py`: batch the embedding fetch in `_compute_clusters` (2000-doc
   pages + `np.vstack`).
4. `crawler/extractors.py`: single-pass pdfplumber in `extract_pdf`.
5. `generator/generator.py`: parallelize `_retrieve_union` with ThreadPoolExecutor.
6. Add tests: `test_index_returns_before_cluster_completes`,
   `test_cluster_eventually_populated`,
   `test_compute_clusters_batched_fetch`,
   `test_retrieve_union_parallel`.
7. `pytest tests/unit/ -x`
8. `pytest tests/integration/ -x` (requires Ollama).
9. `mypy arg/` clean; locality grep clean.
10. Commit `perf(pipeline,explorer,generator): async clusters + batched embed fetch + single-pass PDF + parallel embed (Feature 0003)`.
11. Push, ff-merge to `main`, delete branch.

**Branch 5 — `feature/0003e-test-and-doc-cleanup`**
1. `git switch -c feature/0003e-test-and-doc-cleanup`
2. `test_chunker.py`: replace `/tmp/dummy_docs` and `/tmp/dummy_db` with
   `tmp_path`.
3. `test_watcher.py` and `test_full_rag.py`: replace `time.sleep` with
   `_wait_for` polling helper. Add the helper to `tests/conftest.py` so
   both unit and e2e tests can import it.
4. `chunker.py`: add `pos` counter comment.
5. `server.py`: update route-ordering comment with enumerated list.
6. All `arg/` modules: add `# Implements: docs/spec/…` lines.
   Module-to-spec mapping:
   - `config.py` → `section-04-config.md`
   - `crawler/crawler.py` → `section-05-crawler.md`
   - `crawler/extractors.py` → `section-05-crawler.md`
   - `crawler/watcher.py` → `section-05-crawler.md`
   - `graph/knowledge_graph.py` → `section-06-knowledge-graph.md`
   - `indexer/chunker.py` → `section-07-indexer.md`
   - `indexer/indexer.py` → `section-07-indexer.md`
   - `retriever/retriever.py` → `section-08-retriever.md`
   - `retriever/bm25_index.py` → `section-08-retriever.md`
   - `generator/generator.py` → `section-09-generator.md`
   - `generator/query_processor.py` → `section-09-generator.md`
   - `dci/explorer.py` → `section-10-pipeline.md`
   - `dci/analyst.py` → `section-10-pipeline.md`
   - `pipeline.py` → `section-10-pipeline.md`
   - `server.py` → `section-10-pipeline.md`
7. Apply CLAUDE.md Section 1 + Section 13 edits described above.
8. `pytest tests/unit/ tests/integration/ tests/e2e/ -x` — all clean.
9. `mypy arg/` clean; ruff clean; locality grep clean.
10. Commit `docs(all): spec cross-references + test cleanup (Feature 0003)`.
11. Push, ff-merge to `main`, delete branch.
