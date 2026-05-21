# ARG — Archivist RAG Graph
Yarr, the docs give up their secrets to no one but us.

A fully local, knowledge-graph-augmented RAG system for HTML and PDF documentation.
Runs entirely on Apple M1 Max (64GB). Zero cloud calls. Zero telemetry.

## What ARG Does

ARG indexes a directory of HTML, PDF, and plain-text documentation, follows links recursively
to build a knowledge graph of the corpus, and answers questions over it using
hybrid retrieval: dense embeddings (nomic-embed-text via Ollama), sparse BM25
keyword search, and knowledge-graph traversal. Generation uses Llama 3.3 70B
served locally by Ollama. Everything runs on-device — there are no outbound
network calls during operation.

Beyond Q&A, ARG provides DCI ("Direct Corpus Interaction") features for
browsing, summarising, and comparing documents in a corpus. The corpus is live:
a `watchdog`-backed file watcher re-indexes changed files automatically. ARG
exposes both a CLI (`scripts/index_docs.py`) and a local web UI (FastAPI bound
to `127.0.0.1:8000`), and supports multiple independent corpora via a
path-convention namespace (`arg_db/{corpus_name}/`).

## Architecture

Retrieval is a 5-stage pipeline:

- **Stage 0 — DCI enrichment.** BM25 chunk scores aggregated per document find
  the most relevant files; their link neighbourhoods and topic clusters expand
  the candidate pool.
- **Stage 1 — Dense retrieval.** Top-k chunks via nomic-embed-text + ChromaDB.
- **Stage 1.5 — BM25 retrieval.** Sparse exact-term recall via rank_bm25, run in
  parallel with dense.
- **Stage 2 — Graph traversal.** Kuzu-backed link expansion: neighbours of the
  dense/sparse hits are added as additional candidates.
- **Stage 3 — RRF fusion.** Reciprocal Rank Fusion (k=60) merges all ranked lists.
- **Stage 4 — Lost-in-middle reordering.** Top-ranked chunks moved to the head
  and tail of the prompt; mid-ranked chunks placed in the middle.

The fused, reordered context is handed to Llama 3.3 via LlamaIndex's query engine.

## Prerequisites

- Apple M1 Max (or any Apple Silicon with sufficient RAM; 32GB minimum)
- macOS 13+
- Homebrew
- Python 3.11+
- ~50GB free disk space (~38GB for the Llama 3.3 Q4_K_M model, rest for index data)

## Bootstrap (one-time setup)

```bash
bash scripts/bootstrap.sh
```

Bootstrap checks for Ollama and installs it via Homebrew if missing; pulls
`qwen3.6:35b-a3b-q4_K_M` and `nomic-embed-text` only if they are not
already in `ollama list`; installs Tesseract (for pymupdf OCR tessdata);
downloads D3.js v7 once to `arg/static/d3.min.js`; creates a Python 3.11+
virtual environment at `.venv/` using stdlib `python3.11 -m venv`; activates it;
and installs the project via `pip install -e .[dev]`. If a `./vendor/` wheel
cache is present, pip uses it offline (`--no-index --find-links ./vendor/`);
otherwise it falls back to PyPI. **Bootstrap is the only step that touches the
network** — once it completes, ARG runs entirely offline.

## Activating the environment (every subsequent session)

```bash
source .venv/bin/activate
```

## Quick Start

```bash
# 1. Index a documentation directory
python scripts/index_docs.py index --docs /path/to/your/docs --db ./arg_db

# 2. Start the web UI
python scripts/index_docs.py serve --db ./arg_db

# 3. Open http://localhost:8000 in your browser
```

## CLI Reference

```bash
python scripts/index_docs.py index  --docs PATH --db PATH [--corpus NAME] [--no-watch]
                                     [--subset SUBDIR] [--include PATTERN] [--reset]
python scripts/index_docs.py query  --db PATH [--corpus NAME] [--no-enrich]
python scripts/index_docs.py serve  --db PATH [--corpus NAME] [--port PORT]
python scripts/index_docs.py stats  --db PATH [--corpus NAME]
```

### Partial re-index (fast testing)

Index only a subdirectory or file type without waiting for the full corpus:

```bash
# Re-index just one folder (wipe first)
python scripts/index_docs.py index --docs ~/index --db ./index_db \
  --subset ~/index/Retirement --reset --no-watch

# Re-index only PDFs
python scripts/index_docs.py index --docs ~/index --db ./index_db \
  --include "*.pdf" --reset --no-watch

# --include is repeatable; multiple patterns use OR logic
python scripts/index_docs.py index --docs ~/index --db ./index_db \
  --subset ~/index/Retirement --include "*.html" --include "*.pdf" --no-watch
```

`--reset` deletes the corpus before indexing (no confirmation prompt when the flag is
explicit). Without `--reset` the filtered files are merged into the existing index.

## Multiple Documentation Sets (Multi-corpus)

```bash
python scripts/index_docs.py index --corpus product_a --docs /path/to/product_a --db ./arg_db
python scripts/index_docs.py index --corpus product_b --docs /path/to/product_b --db ./arg_db
python scripts/index_docs.py serve --db ./arg_db   # ?corpus= param selects corpus per request
```

## Resetting a Corpus (after failed index or config change)

```bash
# Interactive reset (asks you to type the corpus name to confirm)
python scripts/reset_corpus.py --db ./arg_db --corpus default

# Non-interactive reset (skip prompt — same as passing --reset to index)
python scripts/reset_corpus.py --db ./arg_db --corpus default --confirm
```

## Evaluating Retrieval Quality

Create `eval/qa_pairs.json` with hand-written question/answer pairs, then:

```bash
python scripts/eval_retrieval.py --db ./arg_db --corpus default --qa eval/qa_pairs.json
```

## Configuration

All tunables live in `.env` (copy from `.env.example`). Key settings:

- `CHUNK_SIZE=1024`          — token window per chunk
- `ENRICH_MIN_SCORE=0.5`     — DCI enrichment threshold (tune with `eval_retrieval.py`)
- `QUERY_REWRITE=true`       — rewrite conversational queries to technical language
- `BM25_ENABLED=true`        — sparse exact-term retrieval alongside dense

## Known Limitations

- **Markdown structure** is not parsed — `.md` and `.markdown` files index
  as plain text. Atx-style headings (`# H1`, `## H2`) are not recognised
  as chunk boundaries. A future feature may add Markdown-aware extraction.
- **iframes** are not followed or indexed — content inside an `<iframe src=...>`
  is invisible to the crawler.
- **JavaScript-rendered content** is not indexed — ARG parses the HTML as
  served; it does not execute scripts.
- **Encrypted PDFs** are skipped (logged with a clear reason); ARG does not
  prompt for passwords.
- **Right-to-left scripts** (Arabic, Hebrew) are not specifically supported —
  text extracts but reading order may be wrong for mixed-direction layouts.
- **Scanned-PDF accuracy** depends on Tesseract's `eng` tessdata; non-English
  scanned PDFs need the matching `tessdata` pack installed via Homebrew.
- **External links** (`http://`, `https://`) are recorded as graph edges but
  never fetched — only `file://` and relative paths are crawled.
- **Authentication / access control** is out of scope; unreadable files are
  logged and skipped rather than retried.

## Running Tests

```bash
pytest tests/unit/          # fast; no Ollama required (LLM mocked)
pytest tests/integration/   # requires indexed corpus; LLM mocked
pytest tests/e2e/           # requires running Ollama with qwen3.6 and nomic-embed-text
```
