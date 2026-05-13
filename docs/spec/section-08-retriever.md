# Section 8: Retriever

> **Prompt to Claude:** "Build Section 8 of ARG: the complete hybrid retriever"

### What Claude will produce:
- `arg/retriever/retriever.py` — `HybridRetriever` implementing `BaseRetriever`
- `arg/retriever/bm25_index.py` — BM25 sparse index (built at index time, queried at retrieval time)

### Complete retrieval pipeline (all stages built here):

```
query (rewritten by QueryProcessor if query_rewrite=True — see Section 9)
  │
  ├─► [Stage 0 — Context Enrichment] (when enrich=True)
  │     0.1  _find_document(query, top_k=enrich_top_docs)
  │          → search `documents` ChromaDB collection by embedding similarity
  │          → if no hits above enrich_min_score: skip Stage 0 entirely
  │
  │     0.2  Link expansion: for each top doc:
  │          get_reverse_links(doc_id) + get_linked_docs(doc_id, depth=1)
  │          → add neighbours to enriched candidate set
  │
  │     0.3  Cluster expansion (if doc_count >= min_cluster_docs):
  │          read cluster_cache.json → find cluster of top hit doc
  │          → add all docs in that cluster to candidate set
  │          (if doc_count < min_cluster_docs: skip; use Stage 0.1+0.2 only)
  │
  ├─► [Stage 1 — Dense Vector Search]
  │     If enrichment fired: ChromaDB `chunks` search filtered to candidate set doc_ids
  │                          (also apply any metadata filters from retrieve() call)
  │     If enrichment skipped: ChromaDB `chunks` search over full corpus
  │                            (with metadata filters applied)
  │     → top_k_vector chunks by embedding similarity
  │     → if filtered result < top_k_vector/2: re-run unfiltered + merge
  │
  ├─► [Stage 1.5 — Sparse BM25 Search]
  │     Query BM25 index (rank_bm25) with raw query tokens
  │     → top_k_vector chunks by BM25 score (keyword/exact-term matching)
  │     → apply same metadata filters as Stage 1 if set
  │     → run independently of Stage 1; results merged in Stage 3
  │
  ├─► [Stage 2 — Graph Expansion]
  │     For each Stage 1 + Stage 1.5 chunk, traverse LINKS_TO edges (depth=graph_hop_depth)
  │     Fetch top_k_graph chunks from each linked document
  │
  ├─► [Stage 3 — RRF Fusion & Re-ranking]
  │     Deduplicate by chunk_id across all stages
  │     Reciprocal Rank Fusion:
  │       rrf_score(chunk) = Σ 1/(k + rank_in_stage)  for each stage that returned it
  │       k = 60  (standard RRF constant)
  │     Replaces the previous weighted formula (0.7 × vector + 0.3 × graph)
  │     → top-N chunks by RRF score
  │
  └─► [Stage 4 — Lost-in-Middle Reordering]
        Reorder top-N chunks using U-shape bookend arrangement:
        Position 0 (first) → rank 1 (highest RRF score)
        Position -1 (last)  → rank 2
        Position 1          → rank 3
        Position -2         → rank 4
        ... and so on, alternating front/back
        Rationale: LLMs attend best to context at the start and end of the window.
        Placing the two most relevant chunks at positions 0 and -1 maximises the
        chance that the most important information is in attended positions.
        Return final list[NodeWithScore] with positions updated.
```

### Key parameters:
```python
class HybridRetriever(BaseRetriever):
    def retrieve(
        self,
        query: str,
        enrich: bool = True,              # master switch for Stage 0
        scope_doc_id: str | None = None,  # if set: skip Stages 0+2, filter to one doc
        filters: dict | None = None,      # ChromaDB metadata filters, e.g. {"has_table": True}
    ) -> list[NodeWithScore]:
```

**`filters` examples:**
```python
# Only chunks containing tables (for "show me the rate limit table")
pipeline.query("show me the rate limit table", filters={"has_table": True})

# Only chunks from PDF documents
pipeline.query("what does the manual say about OAuth", filters={"file_type": "pdf"})

# Only chunks from a specific section heading
pipeline.query("token expiry", filters={"heading_path": {"$contains": "OAuth"}})
```
Filters are passed directly to ChromaDB's `where` clause and apply in both Stage 1
and Stage 1.5. `scope_doc_id` takes precedence over `filters`; if both are set,
`scope_doc_id` wins.

### BM25 index (`arg/retriever/bm25_index.py`):
- Built during `pipeline.index()` after all chunks are embedded
- Stores `{chunk_id: tokenized_text}` index using `rank_bm25.BM25Okapi`
- Persisted to `arg_db/{corpus_name}/bm25_index.pkl`
- Updated incrementally: add/remove individual chunk entries on `add_document()` / `remove_document()`
- Queried with raw query tokens (no stemming required; BM25Okapi handles it)
- Config: `bm25_enabled: bool = True` — set False to disable sparse retrieval entirely

### Notes:
- **`scope_doc_id`** bypasses all enrichment and graph expansion; returns chunks from
  one document only. Filters still apply within scoped mode.
- **`_find_document()`** private method: searches `documents` collection, called by
  Stage 0.1 and `corpus_search()`.
- **RRF replaces weighted scoring:** The old `0.7 × vector + 0.3 × graph` formula is
  replaced by RRF across all stages (dense, sparse, graph). RRF is more robust because
  it doesn't require normalising scores across different retrieval methods.
- **Lost-in-middle reordering** applies unconditionally after Stage 3. It is not
  configurable — the quality improvement is always positive and the cost is zero.

### New dependency:
```toml
"rank-bm25>=0.2.2",   # pure Python BM25; no server; ~3MB
```

### Tests (unit — `test_retriever.py`):
- Stage 1 only (`enrich=False`, `bm25_enabled=False`): returns correct chunks for known query
- Stage 1.5 BM25 returns chunks matching exact technical terms (e.g. "X-Rate-Limit-Retry-After")
  that dense search would score low on
- BM25 and dense results are combined by RRF; combined result beats either alone
  for a query combining semantic and exact-term aspects
- Stage 2 adds chunks from documents linked to Stage 1 + 1.5 hits
- RRF fusion deduplicates correctly; all scores positive
- `graph_hop_depth=0` disables Stage 2
- `scope_doc_id` set: only returns chunks from that document; no Stage 0, 2, or enrichment
- `filters={"has_table": True}`: only chunks with `has_table=True` returned
- `filters={"file_type": "pdf"}`: only PDF chunks returned
- `enrich=True`, `enrich_min_score=1.0` (impossible): falls back to unfiltered Stage 1
- Sparse Stage 1 result (< top_k_vector/2): triggers unfiltered re-run + merge
- Lost-in-middle reordering: highest-scored chunk is at position 0; second-highest at position -1
- `_find_document()` returns ranked doc_ids from `documents` collection
- Stage 0.3 skipped gracefully when `cluster_cache.json` absent
- BM25 index persists to disk; reloads correctly
- BM25 index updated correctly after `add_document()` and `remove_document()`
- `bm25_enabled=False`: Stage 1.5 skipped; only dense retrieval used
- Retriever handles zero hits from all stages gracefully (empty list, no crash)
