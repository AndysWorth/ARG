# Section 19: Operational Notes

> These items do not require build Claude sessions — they are guidance for running,
> tuning, and maintaining ARG after it is built. Reference this section when you
> encounter the issues described.

### 4.1 — Ollama health check on startup
`ARGPipeline.__init__` must call `GET http://localhost:11434/api/tags` before
initialising any component. If the call fails or returns an empty model list:
```
RuntimeError: Ollama is not running or has no models loaded.
  Start Ollama: `ollama serve`
  Verify model: `ollama list`
```
This produces a clear actionable error instead of a cryptic connection refused
deep inside LlamaIndex.

### 4.2 — Memory pressure on M1 Max (64GB)
Expected memory footprint during normal operation:
- qwen3.6:35b-a3b-q4_K_M loaded in Ollama: ~22GB
- ChromaDB + numpy (Metal): ~1–2GB
- Kuzu graph: ~200MB for typical doc sets
- Python process + FastAPI: ~500MB
- Total steady state: ~24–25GB — well within 64GB

**Risk scenario:** cluster cache rebuild triggered during active LLM inference.
Both operations are memory-hungry. To prevent:
- Cluster rebuilds run in a background thread (4.4)
- Background thread checks `threading.Event` that is set while LLM inference is active;
  if set, the rebuild sleeps 5s and retries. This adds at most one retry cycle of latency
  to the rebuild, not to the query.

### 4.3 — SIGTERM / SIGINT shutdown handling
`scripts/serve.py` and `scripts/index_docs.py` must register signal handlers:
```python
import signal, sys

def shutdown(sig, frame):
    pipeline.close()   # flushes logs, closes Kuzu WAL, stops watcher thread
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)
```
Kuzu uses a write-ahead log. If the process is killed without `close()`, the next
startup will replay the WAL correctly, but registering the handler avoids this
delay and prevents any possibility of log corruption.

### 4.4 — Asynchronous cluster cache rebuild
When `add_document()`, `update_document()`, or `remove_document()` is called:
1. The mutation completes synchronously (Chroma + Kuzu updated, read lock released)
2. The old `cluster_cache.json` is renamed to `cluster_cache.json.stale`
   (queries continue to use the stale cache — never a cold miss)
3. A background `threading.Thread` starts the rebuild:
   - Calls `explorer.get_topic_clusters()` (LLM calls, ~30–120s for 8 clusters)
   - On completion, atomically replaces `cluster_cache.json.stale` with the new file
4. If another mutation arrives during rebuild, the in-progress thread is cancelled
   (via a `threading.Event`) and a new thread starts after the mutation completes

### 4.5 — Port conflict on startup
If port 8000 is in use (AirPlay Receiver, other dev servers), `scripts/serve.py`
catches the `OSError: [Errno 48] Address already in use` and prints:
```
Error: port 8000 is already in use.
  Try: python scripts/index_docs.py serve --port 8001
  Or disable AirPlay Receiver in System Settings → General → AirDrop & Handoff
```
The `--port` flag is the standard resolution. Port 8000 is the default but never
assumed to be available.

### 4.6 — Schema drift detection (chunk size or embedding dim changes)
On `pipeline.index()` startup, ARG writes a config fingerprint to
`arg_db/{corpus_name}/config_hash.json`:
```json
{"chunk_size": 1024, "chunk_overlap": 128, "embed_dim": 256, "embed_model": "nomic-embed-text"}
```
On subsequent startups, if the fingerprint does not match the current config, ARG logs:
```
WARNING: Index config has changed (chunk_size: 512 → 1024).
  Existing embeddings are incompatible. Run reset_corpus.py then re-index.
```
ARG does NOT silently mix old and new embeddings. It refuses to proceed until the
corpus is reset and re-indexed. This prevents silent quality degradation.

### 4.7 — PDF layout problems (per-document sidecar config)

If a specific PDF produces wrong column order or garbled text, create a sidecar
config file next to it:

```bash
# Example: fix column order for one PDF
echo '{"pdf_layout_analysis": false}' > docs/manual.pdf.argconfig
```

ARG detects this file automatically on the next index run. No restart required.
Other documents in the corpus are not affected.

### 4.8 — Interrupted indexing of large PDFs

If indexing is killed mid-way through a large PDF, the partially-written chunks
remain in ChromaDB and Kuzu. On the next `pipeline.index()` run, the hash-check
sees the PDF as unchanged and skips it — leaving the partial state permanently.

To recover: run `reset_corpus.py` to delete the partial state, then re-index.
Or use `pipeline.remove_document(doc_id)` followed by `pipeline.add_document(path)`
to force a clean re-extraction of that specific file.

The `--reset` flag on `scripts/index_docs.py index` combines both steps: it wipes the
corpus directory then immediately begins a fresh crawl, without the interactive
confirmation that `reset_corpus.py` requires.

The `pdf_batch_size` config (default 10 pages) controls how often the indexer
checkpoints page progress to disk. A lower value (e.g. 5) reduces the re-work
needed if indexing is interrupted, at a small performance cost.


### Eval script — `scripts/eval_retrieval.py`

Produced during Section 10. Spec:
```bash
python scripts/eval_retrieval.py --db ./arg_db --corpus default --qa eval/qa_pairs.json
```
`eval/qa_pairs.json` — a JSON file you create with hand-written question/answer pairs
from your actual documentation. Format:
```json
[
  {
    "question": "How do I authenticate with the Kraken API?",
    "expected_doc_ids": ["page_a.html"],
    "expected_answer_contains": ["API key", "OAuth"]
  }
]
```
The script runs each question through the full pipeline (retrieval only, LLM mocked)
and reports:
- **Hit rate**: % of questions where `expected_doc_ids` appear in retrieved chunks
- **Enrichment delta**: hit rate with `enrich=True` vs `enrich=False`
- **Mean retrieval latency** (Stage 0 + Stage 1 + Stage 2)
- **`find_document` score distribution**: helps calibrate `enrich_min_score`

Run this after first indexing your real corpus to validate retrieval quality before
trusting answers. Adjust `enrich_min_score`, `top_k_vector`, and `graph_hop_depth`
based on results.
