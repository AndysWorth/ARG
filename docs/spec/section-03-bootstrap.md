# Section 3: Prerequisites & Bootstrap

> **Prompt to Claude:** "Build Section 3 of ARG: the bootstrap script and pyproject.toml"

### What Claude will produce:
- `scripts/bootstrap.sh` — one-time setup script (see rules below)
- `pyproject.toml` — **extend** the existing file (which already holds `[tool.ruff]`,
  `[tool.mypy]`, `[tool.pytest.ini_options]`) by adding the `[project]` and
  `[build-system]` sections plus a `[project.optional-dependencies]` group for dev
  tools. **Do not overwrite the existing `[tool.*]` sections.**
- `.env.example` — tunables with defaults
- `README.md` — full project documentation with the sections below

### README.md — required content:

```
# ARG — Archivist RAG Graph
Yarr, the docs give up their secrets to no one but us.

A fully local, knowledge-graph-augmented RAG system for HTML and PDF documentation.
Runs entirely on Apple M1 Max (64GB). Zero cloud calls. Zero telemetry.

## What ARG Does
[2-paragraph description covering: indexes local HTML+PDF docs following links recursively,
answers questions using hybrid dense+sparse retrieval + knowledge graph traversal,
DCI features for browsing/summarising/comparing documents, live file watching,
web UI + CLI, multi-corpus support]

## Architecture
[Brief description of the 5-stage retrieval pipeline:
Stage 0 (DCI enrichment) → Stage 1 (dense) → Stage 1.5 (BM25) →
Stage 2 (graph) → Stage 3 (RRF) → Stage 4 (reordering)]

## Prerequisites
- Apple M1 Max (or any Apple Silicon with sufficient RAM; 32GB minimum)
- macOS 13+
- Homebrew
- Python 3.11+
- ~50GB free disk space (38GB for the LLM, rest for index data)

## Bootstrap (one-time setup)
bash scripts/bootstrap.sh
[Explain what bootstrap does: checks for Ollama, pulls models if absent,
installs tesseract, downloads D3.js, creates a Python 3.11+ virtual environment
at `.venv/` using stdlib `python3.11 -m venv`, activates it, installs Python deps
via `pip install -e .` (preferring the `./vendor/` wheel cache if present).
Note: bootstrap is the only step that touches the network.]

## Activating the environment (every subsequent session)
source .venv/bin/activate

## Quick Start
# 1. Index a documentation directory
python scripts/index_docs.py index --docs /path/to/your/docs --db ./arg_db

# 2. Start the web UI
python scripts/index_docs.py serve --db ./arg_db

# 3. Open http://localhost:8000 in your browser

## CLI Reference
python scripts/index_docs.py index  --docs PATH --db PATH [--corpus NAME] [--no-watch]
python scripts/index_docs.py query  --db PATH [--corpus NAME] [--no-enrich]
python scripts/index_docs.py serve  --db PATH [--corpus NAME] [--port PORT]
python scripts/index_docs.py stats  --db PATH [--corpus NAME]

## Multiple Documentation Sets (Multi-corpus)
python scripts/index_docs.py index --corpus product_a --docs /path/to/product_a --db ./arg_db
python scripts/index_docs.py index --corpus product_b --docs /path/to/product_b --db ./arg_db
python scripts/index_docs.py serve --db ./arg_db  # ?corpus= param selects corpus per request

## Resetting a Corpus (after failed index or config change)
python scripts/reset_corpus.py --db ./arg_db --corpus default

## Evaluating Retrieval Quality
# Create eval/qa_pairs.json with hand-written question/answer pairs, then:
python scripts/eval_retrieval.py --db ./arg_db --corpus default --qa eval/qa_pairs.json

## Configuration
All tunables are in .env (copy from .env.example). Key settings:
- CHUNK_SIZE=1024          # token window per chunk
- ENRICH_MIN_SCORE=0.5     # DCI enrichment threshold (tune with eval_retrieval.py)
- QUERY_REWRITE=true       # rewrite conversational queries to technical language
- BM25_ENABLED=true        # sparse exact-term retrieval alongside dense

## Known Limitations
[Copy Known Limitations from Sections 5 HTML and 5 PDF: iframe not indexed,
JS-rendered content not indexed, encrypted PDFs skipped, RTL text not supported, etc.]

## Running Tests
pytest tests/unit/          # fast; no Ollama required (LLM mocked)
pytest tests/integration/   # requires indexed corpus; LLM mocked
pytest tests/e2e/           # requires running Ollama with qwen3.6 and nomic-embed-text
```

### Pre-build verification (do this before Section 3):
Run the following to confirm current compatible versions exist on PyPI before pinning:
```bash
pip index versions llama-index-core
pip index versions llama-index-graph-stores-kuzu   # confirm this package name exists
pip index versions kuzu
pip index versions chromadb
```
If `llama-index-graph-stores-kuzu` does not exist, check `llama-index-graph-stores-kuzugraph`
or the LlamaIndex integrations index at https://llamahub.ai. The build Claude must confirm
the exact package name before writing `pyproject.toml`.

### Bootstrap rules — ALL processing must be local at runtime:
1. **Ollama model check first**: run `ollama list` and skip `ollama pull` entirely if
   `qwen3.6:35b-a3b-q4_K_M` and `nomic-embed-text` are already present.
   Bootstrap is a one-time setup tool; it must never pull models during normal operation.
2. **Offline pip support**: bootstrap must support
   `pip install --no-index --find-links ./vendor/` if a `./vendor/` wheel cache exists,
   falling back to PyPI only when the cache is absent. Document how to pre-download wheels:
   `pip download -r requirements.txt -d ./vendor/`
3. **No network calls at runtime**: once bootstrap completes, ARG must never make
   outbound network requests. Ollama serves models from localhost:11434 only.
4. **Virtual environment**: bootstrap creates `.venv/` via `python3.11 -m venv .venv`
   (stdlib only — no Poetry, Conda, `uv`, etc.) and installs project dependencies into
   it via `pip install -e .`. If `.venv/` already exists, bootstrap reuses it. Bootstrap
   must verify `python3 --version` reports 3.11+ and abort with a clear error otherwise.

### Key dependencies (pyproject.toml):
All LlamaIndex packages MUST be pinned to exact matching versions — do not use `>=` ranges
for any `llama-index-*` package. The build Claude must resolve a consistent set using
`pip install llama-index-core==<version> llama-index-vector-stores-chroma==<version> ...`
and record the exact versions that resolve without conflicts before writing pyproject.toml.

```toml
[project]
name = "arg"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # RAG framework — ALL pinned to exact compatible versions (resolve before writing)
    "llama-index-core==<pin>",
    "llama-index-vector-stores-chroma==<pin>",
    "llama-index-graph-stores-kuzu==<pin>",   # verify package name first
    "llama-index-llms-ollama==<pin>",
    "llama-index-embeddings-ollama==<pin>",
    # Stores — floor pins acceptable; these APIs are stable
    "chromadb>=0.5",
    "kuzu>=0.6",
    # Parsing
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "pdfplumber>=0.11",
    "pymupdf>=1.24",          # includes built-in OCR via tessdata
    # File watching
    "watchdog>=4.0",
    # Web UI
    "fastapi>=0.111",
    "uvicorn>=0.30",
    # Utilities
    "httpx>=0.27",            # HTTP client + FastAPI test client
    "python-dotenv>=1.0",
    # DCI clustering
    "scikit-learn>=1.4",      # k-means; local, no network
    # Sparse retrieval
    "rank-bm25>=0.2.2",       # BM25 keyword search; pure Python; no server
]

[project.optional-dependencies]
dev = [
    "ruff>=0.8",
    "mypy>=1.13",
    "pre-commit>=4.0",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

**Note:** `[tool.ruff]`, `[tool.mypy]`, and `[tool.pytest.ini_options]` already exist
in `pyproject.toml` (added before Section 3). Section 3 only adds `[project]`,
`[project.optional-dependencies]`, and `[build-system]` — it must not remove or
rewrite the existing `[tool.*]` sections.

**System dependency (installed by bootstrap.sh via Homebrew):**
```bash
brew install tesseract   # provides tessdata for pymupdf OCR
```

### Tests (unit):
- `test_bootstrap`: verify Ollama is running, both models are available, ChromaDB
  initializes, Kuzu initializes — all before any ARG code runs.
