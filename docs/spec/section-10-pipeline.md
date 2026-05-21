# Section 10: Pipeline, CorpusExplorer, Web UI & Logging

> **Prompt to Claude:** "Build Section 10 of ARG: pipeline, CorpusExplorer, web UI, logging, and CLI — complete"

### What Claude will produce:
- `arg/dci/explorer.py` — `CorpusExplorer` (completes the DCI layer alongside Section 9's `CorpusAnalyst`)
- `arg/dci/__init__.py` — final version re-exporting both DCI classes
- `arg/pipeline.py` — complete `ARGPipeline`
- `arg/server.py` — FastAPI app with all RAG + DCI endpoints
- `arg/static/index.html` — single-page UI (vanilla HTML + JS + D3)
- `arg/logging/json_formatter.py` + `arg/logging/tracing.py`
- `scripts/index_docs.py` — full subcommand CLI
- `scripts/reset_corpus.py` — corpus teardown
- `scripts/eval_retrieval.py` — retrieval eval

### Why together:
`CorpusExplorer` depends on `KnowledgeGraph` (Section 6), `documents` ChromaDB
(Section 7), and `CorpusAnalyst.find_document()` (Section 9) — all now exist.
`ARGPipeline` depends on all seven components. `server.py` depends on `ARGPipeline`.
Building everything in one session means no dangling imports anywhere.

---

### CorpusExplorer (`arg/dci/explorer.py`):

**1. Document listing** — `list_all_documents() -> list[dict]`
Wraps `KnowledgeGraph.list_all_documents()`.

**2. Reverse link lookup** — `get_reverse_links(doc_id) -> list[dict]`
Wraps `KnowledgeGraph.get_reverse_links()`.

**3. Graph export** — `get_graph_json() -> dict`
Returns `{nodes: [{id, title, file_type, chunk_count}], edges: [{source, target, anchor_text}]}`.

**4. Topic clustering** — `get_topic_clusters() -> list[{label, doc_ids}]`
- Small corpus guard: if `doc_count < config.min_cluster_docs` → return
  `[{"label": "All documents", "doc_ids": [...]}]`; log reason; no LLM calls
- Normal path: k-means on `documents` embeddings with
  `n_clusters = max(2, min(config.n_clusters, doc_count))`;
  one LLM call per cluster for label generation
- Result cached to `cluster_cache.json`; `invalidate_cluster_cache()` deletes it
- Pre-built at end of `pipeline.index()` before returning

**5. Corpus search** — `corpus_search(query, file_type=None, top_k=10) -> list[dict]`
- Calls `analyst.find_document(query, top_k)` then applies optional `file_type` filter
- Sole owner of `GET /corpus/search`

**6. Analytics** (delegates to KnowledgeGraph + paginator):
- `most_linked_docs(top_n=10)`
- `orphaned_docs()`
- `docs_by_chunk_count(page, page_size, order)` — pagination at this layer

---

### `arg/dci/__init__.py` (final):
```python
from arg.dci.explorer import CorpusExplorer
from arg.dci.analyst import CorpusAnalyst
__all__ = ["CorpusExplorer", "CorpusAnalyst"]
```

---

### `ARGPipeline` — complete public API:
```python
pipeline = ARGPipeline(config, corpus_name="default")

# Sub-components (all instantiated in __init__)
pipeline.graph           # KnowledgeGraph
pipeline.indexer         # Indexer
pipeline.retriever       # HybridRetriever
pipeline.generator       # Generator
pipeline.query_processor # QueryProcessor (rewrite, decompose, HyDE)
pipeline.explorer        # CorpusExplorer
pipeline.analyst         # CorpusAnalyst
pipeline.watcher         # Watcher (None if watch_enabled=False)

# Indexing — docs_root from config; no path arg
pipeline.index()
pipeline.add_document(path: Path)        # invalidates cluster cache + summary
pipeline.remove_document(doc_id: str)    # invalidates caches
pipeline.update_document(path: Path)     # invalidates this doc's summary only

# Querying
result = pipeline.query(
    question: str,
    enrich: bool = True,
    stream: bool = False,
    filters: dict | None = None,   # metadata filters: has_table, file_type, etc.
) -> ARGResult
# ARGResult: {answer, sources[{doc_id,title,chunk_id,heading_path}], latency_ms,
#             enriched_doc_ids, rewritten_query, sub_queries}

# DCI (public)
pipeline.summarize_document(doc_id) -> str
pipeline.compare_documents(doc_id_a, doc_id_b) -> str
pipeline.corpus_search(query, file_type=None) -> list[dict]
pipeline.get_topic_clusters() -> list[dict]
pipeline.corpus_stats() -> dict
# find_document() is internal — NOT on this API

# Lifecycle
pipeline.close()   # flush logs, close Kuzu, stop watcher
```

**Startup sequence:**
1. Verify Ollama health → `RuntimeError` if down
2. Check `config_hash.json` for schema drift → `RuntimeError` if mismatch
3. Instantiate all sub-components
4. Register SIGTERM/SIGINT → `pipeline.close()`
5. Start watcher if `watch_enabled=True`

**Locking:** Write operations acquire `threading.RLock`. Cluster rebuilds run
async in background thread; stale cache served during rebuild (Section 19.4.4).

---

### FastAPI server (`arg/server.py`):
All endpoints accept `?corpus=<name>` (default `"default"`). Returns 404 for
unknown corpus name. Server dict: `{corpus_name: ARGPipeline}`.

```
POST /query                      → {answer, sources, latency_ms, enriched_doc_ids,
                                    rewritten_query, sub_queries}
GET  /health                     → {status, model, corpus_name, doc_count, chunk_count}
GET  /corpus                     → list all documents
POST /corpus/add                 → index a new document
DELETE /corpus/{doc_id}          → remove document
GET  /corpus/graph               → {nodes, edges} for D3
GET  /corpus/{doc_id}/linked-by  → reverse links
GET  /corpus/topics              → topic clusters
GET  /corpus/search              → ?query&file_type — doc-level search
GET  /corpus/{doc_id}            → metadata + key points
GET  /corpus/{doc_id}/summary    → LLM summary
GET  /corpus/{doc_id}/chunks     → raw chunks
GET  /corpus/compare             → ?a&b — LLM comparison
GET  /corpus/{doc_id}/search     → ?query — scoped chunk search
GET  /corpus/stats               → {most_linked, orphaned, by_size_preview}
GET  /corpus/stats/by-size       → ?page&page_size&order
```

### UI (`arg/static/index.html`):
Query box + streaming answer + citations; document list sidebar; document detail panel
(key points, summary, links, raw chunks); D3 force-directed graph (loads local `d3.min.js`);
topic clusters panel; corpus stats panel with paginated size ranking; scoped search;
watching status indicator.

### CLI (`scripts/index_docs.py`):
```bash
python scripts/index_docs.py index  --docs /path --db ./arg_db --corpus default [--no-watch]
                                     [--subset SUBDIR] [--include PATTERN] [--reset]
python scripts/index_docs.py query  --db ./arg_db --corpus default [--no-enrich]
python scripts/index_docs.py serve  --db ./arg_db --corpus default --port 8000
python scripts/index_docs.py stats  --db ./arg_db --corpus default
```

`--subset PATH` restricts the crawl to files at or under the given directory.
`--include PATTERN` filters by fnmatch pattern (repeatable; OR logic between patterns).
`--reset` deletes the corpus before indexing; no confirmation prompt is shown because the
explicit flag is sufficient.

### Tests:
- `test_explorer.py`: all CorpusExplorer methods; small-corpus clustering fallback;
  normal clustering (≥ min_cluster_docs); cache invalidation; corpus_search accuracy
- `test_pipeline.py`: Ollama health check failure; schema drift detection;
  index + query + enrich; `query_processor` instantiated in `__init__`;
  `pipeline.query(filters={"has_table": True})` passes filters to retriever;
  `ARGResult.rewritten_query` present on conversational query;
  `ARGResult.sub_queries` present on compound query;
  BM25 index written to disk after `pipeline.index()`;
  cache invalidation on mutations; idempotent re-index; clean close
- `test_server.py`: all endpoints; `?corpus=nonexistent` → 404; two-corpus isolation;
  paginated by-size; streaming query response
- `test_logging.py`: JSON lines valid; debug traces created with `--debug`
