# ARG — Archivist RAG Graph
### A fully local, knowledge-graph-augmented RAG for HTML + PDF documentation
#### Optimized for Apple M1 Max · 64GB Unified Memory · Open Source Only

---

## 0. How to Use This File with Claude Code

ARG is built using **Claude Code** — Anthropic's agentic coding tool that runs in
your terminal, reads files directly, executes commands, and writes code to disk.
This file (`CLAUDE.md`) is Claude Code's persistent project specification.

---

### Python environment

ARG runs in a project-local virtual environment at `.venv/`, created with the
Python 3.11+ stdlib `venv` module. No Poetry, Conda, `uv`, or other environment
managers — stdlib `venv` is consistent with the "avoid heavyweight frameworks"
decision in Section 1 and the offline `./vendor/` wheel cache used by bootstrap.

**First-time setup** is handled by `scripts/bootstrap.sh` (created in Section 3),
which runs the equivalent of:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Every subsequent shell session** — activate the venv before running `claude`,
`pytest`, or any project script:

```bash
source .venv/bin/activate
```

`.venv/` is already gitignored. Do not commit it. If the environment gets into a
bad state, delete `.venv/` and re-run `scripts/bootstrap.sh`.

---

### Tooling & quality gates

The repo ships with quality gates wired up before any `arg/` code is written, so
the build runs against a consistent standard from line 1.

| Layer | Tool | Config | Runs in |
|---|---|---|---|
| Lint + format | `ruff` | `pyproject.toml` `[tool.ruff]` | pre-commit + CI |
| Type checking | `mypy` | `pyproject.toml` `[tool.mypy]` | pre-commit + CI (once `arg/` exists) |
| Tests | `pytest` | `pyproject.toml` `[tool.pytest.ini_options]` | CI (once tests exist) |
| Pre-commit | `pre-commit` | `.pre-commit-config.yaml` | every `git commit` + CI |
| CI | GitHub Actions | `.github/workflows/test.yml` | every push + PR (parallel jobs: `pre-commit`, `ci`) |
| Dependency updates | Dependabot | `.github/dependabot.yml` | weekly pip · monthly GH Actions |

**One-time install** (inside the activated `.venv/`):

```bash
pip install ruff mypy pytest pytest-asyncio pre-commit
pre-commit install
pre-commit autoupdate         # bump hook revs to latest
pre-commit run --all-files    # baseline sweep
```

Once installed, hooks run on every `git commit`. **Never bypass with `--no-verify`**
— see "When to intervene" below.

**Branch protection on `main`** is the structural complement to the per-section
branch flow. Configure in GitHub Settings → Branches → Add rule for `main`:
- ✅ Require status checks to pass before merging → select `ci` workflow
- ✅ Require branches to be up to date before merging
- (Solo project: PR review requirement can stay off; per-section approval already gates merges)

---

### Starting the build

From your project root (the directory containing this file), run:

```bash
claude
```

Then say:

```
Read CLAUDE.md completely. Then build ARG section by section, starting with
Section 3. After completing each section run its tests. Do not proceed to the
next section until all tests pass. Use git to commit after each section completes.
```

That is the entire setup. Claude Code reads this file, reads the existing project
files at any point, runs commands, writes code, runs tests, and fixes failures —
all without you needing to paste anything.

---

### What Claude Code does that chat cannot

| Chat (old approach) | Claude Code (correct approach) |
|---|---|
| You paste the section spec | Claude Code reads CLAUDE.md directly |
| You paste prior sections' code | Claude Code reads the actual files |
| You run tests and paste errors | Claude Code runs `pytest` and fixes failures |
| You run the locality check | Claude Code runs `grep` itself |
| You run git commands | Claude Code commits after each section |
| One conversation per section | One continuous session for the whole project |
| You copy-paste produced code to files | Claude Code writes files directly to disk |

---

### Your role during the build

Claude Code handles the mechanics. Your role is:

1. **Answer questions** — Claude Code may ask about ambiguities in the spec. Answer them.
2. **Watch for drift** — If Claude Code's implementation diverges from the spec, say:
   `"Re-read Section N of CLAUDE.md and align the implementation with it."`
3. **Approve before continuing** — After each section completes and tests pass,
   Claude Code will tell you. Review the summary and say `"Continue"` or raise concerns.
4. **Handle LlamaIndex conflicts** — If version resolution fails (see below), you
   may need to run one manual command. Claude Code will tell you what to run.

---

### Section-by-section flow (what Claude Code does automatically)

For each section, Claude Code will:
1. Create a fresh branch off `main`: `git switch main && git pull && git switch -c section-N-<short-name>` (e.g. `section-3-bootstrap`)
2. Re-read the relevant section spec from `docs/spec/`
3. Read any existing files it will import from
4. Write all specified files to disk
5. Run the section's unit tests
6. Fix any test failures (iterating until tests pass). Never bypass pre-commit hooks or signing with `--no-verify` / `--no-gpg-sign` — investigate failures and fix the root cause.
7. Run the locality check: `grep -rn "requests\.\|httpx\.get\|http://" arg/ --include="*.py" | grep -v "localhost\|127.0.0.1\|11434\|test_"`
8. Stage **explicit paths only** (never `git add -A` / `git add .`), commit with **Conventional Commits** format, and push the branch:
   ```bash
   # Stage only the paths this section touched — pick from these as relevant:
   git add arg/ tests/ scripts/ docs/ pyproject.toml .env.example README.md
   # type ∈ {feat, fix, test, chore, docs, refactor}; scope = section package (crawler, indexer, retriever, ...)
   git commit -m "feat(section-N): <short summary of what landed>"
   git push -u origin section-N-<short-name>
   ```
9. Report the branch name, test summary, and commit SHA, then wait for your `"Continue"`.
10. Once you say `"Continue"`, fast-forward merge into `main` and clean up:
    ```bash
    git switch main
    git merge --ff-only section-N-<short-name>
    git push
    git branch -d section-N-<short-name>
    git push origin --delete section-N-<short-name>
    ```

---

### The three sections that need your attention

**Section 5 — PDF extractor:** Tell Claude Code up front:
```
Build Section 5 in two passes: HTML extractor and crawler first, then PDF extractor.
Run HTML tests between passes.
```

**Section 7 — BM25 index ownership:** Claude Code will read both Section 7 and
Section 8, so it will understand the BM25 index is built by the indexer and read
by the retriever. If it gets this wrong, say:
```
The BM25 index must be written during pipeline.index() by the indexer, not
lazily created by the retriever. Re-read Sections 7 and 8.
```

**Section 10 — large section:** If Claude Code stalls or produces incomplete output,
say: `"Produce pipeline.py first, run its tests, then continue with the next file."`

---

### LlamaIndex version conflicts — the one thing Claude Code cannot fully automate

LlamaIndex packages must all resolve to a compatible set simultaneously. Claude Code
will attempt this during Section 3, but if pip cannot find a compatible set
automatically, you may see a `ResolutionImpossible` error. If this happens:

```bash
# Run this yourself, note the version it installs
pip install llama-index-core

# Then tell Claude Code:
"llama-index-core resolved to X.Y.Z. Find all compatible llama-index-* versions
for this base version and update pyproject.toml."
```

Claude Code can then re-run resolution with the pinned base version.

---

### When to intervene

Intervene if Claude Code:
- **Weakens a test** to make it pass instead of fixing the implementation
  → Say: `"Do not change tests. Fix the implementation."`
- **Skips a spec requirement** because it seems complex
  → Say: `"You missed [requirement]. Re-read Section N and implement it."`
- **Uses a different library** than specified
  → Say: `"We use [specified library] for this. Do not substitute alternatives."`
- **Makes a network call** outside of Ollama
  → Say: `"This violates the locality guarantee. Remove all outbound network calls."`
- **Bypasses pre-commit hooks or signing** with `--no-verify` / `--no-gpg-sign`
  → Say: `"Do not bypass hooks. Fix the underlying issue and create a new commit."`
- **Uses `git add -A` / `git add .` or `git reset --hard`** instead of explicit paths and branch-abandon recovery
  → Say: `"Stage explicit paths only. For recovery, delete the section branch rather than reset --hard."`

---

### If the build goes wrong

Because each section is built on its own branch, recovery is non-destructive — abandon
the section branch and start over from `main`. Avoid `git reset --hard`; it silently
discards uncommitted work and can't be undone.

```bash
# See what the section branch added vs. main
git diff main...HEAD

# (Optional) save in-progress changes you might want to inspect later
git stash push -m "section-N WIP"

# Abandon the section branch and return to a clean main
git switch main
git branch -D section-N-<short-name>                # delete local
git push origin --delete section-N-<short-name>     # delete remote (if pushed)

# Discard any stray uncommitted edits on main (use `git restore`, not `checkout --`)
git restore .

# Rare case: a bad section already landed on main. Revert it, don't reset —
# `git revert` keeps remote history sane.
git revert <bad-commit-sha>
git push
```

Tell Claude Code: `"Abandon the section-N branch and retry Section N from main."`

---

### What "done" looks like

The build is complete when:
```bash
pytest tests/unit/        # all pass (no Ollama needed — LLM is mocked)
pytest tests/integration/ # all pass (requires Ollama running)
pytest tests/e2e/         # all pass (requires Ollama running)
python scripts/index_docs.py --help   # CLI works
python scripts/serve.py              # server starts on :8000
```

Then run the post-build eval against your real corpus:
```bash
python scripts/eval_retrieval.py --db ./arg_db --corpus default --qa eval/qa_pairs.json
```

---

## 1. Stack Decisions (Locked)

| Concern | Choice | Rationale |
|---|---|---|
| **LLM** | Llama 3.3 70B · Q4_K_M via Ollama | Fits in ~38GB; leaves 26GB headroom; best quality at local scale |
| **Embedding** | nomic-embed-text v1.5 via Ollama | 8192-token context; Matryoshka dims; no second runtime |
| **Vector Store** | ChromaDB (persistent, local) | Pure Python, no server, Metal-accelerated via numpy |
| **Knowledge Graph** | Kuzu (embedded) | No daemon, columnar, fast graph traversal, Python-native |
| **RAG Framework** | LlamaIndex | Best-in-class pipeline abstraction; Kuzu + Chroma integrations exist |
| **Sparse Retrieval** | rank-bm25 (BM25Okapi) | Pure Python; exact-term matching; fused with dense via RRF |
| **Query Processing** | QueryProcessor (custom) | Rewrite → decompose → optional HyDE; all local via Ollama |
| **HTML Parsing** | BeautifulSoup4 + lxml | Robust, handles malformed HTML |
| **PDF Parsing** | pdfplumber (primary) + pymupdf (fallback + OCR) | pdfplumber for text/tables; pymupdf for scanned/image PDFs with built-in OCR |
| **OCR** | pymupdf built-in OCR + Tesseract data files | No second runtime; tessdata via `brew install tesseract` |
| **File watching** | watchdog | Live incremental re-indexing on filesystem events |
| **Web UI** | FastAPI + uvicorn + vanilla HTML/JS | Local only (`127.0.0.1`); no build step; no framework |
| **Multi-corpus** | Path-convention namespacing | `arg_db/{corpus_name}/`; zero extra dependencies |
| **Logging** | Python `logging` → JSON + optional LlamaIndex tracing | Rotating local log file; debug traces opt-in via `--debug` |
| **Orchestration** | Python 3.11+ with asyncio | Native M1; avoid heavyweight frameworks |
| **Testing** | pytest + pytest-asyncio | Standard; easy fixture composition |
| **Runtime** | Ollama (Metal backend auto-detected) | GPU acceleration on M1 Max 32-core GPU out of the box |

---

## 1.5. Locality Guarantee

**ARG makes zero outbound network calls during operation.** This is enforced at every layer:

| Layer | How locality is enforced |
|---|---|
| LLM inference | Ollama serves from `localhost:11434`; `OllamaLLM` points there explicitly |
| Embeddings | Ollama serves nomic-embed-text from `localhost:11434` |
| Vector store | ChromaDB writes to local disk; `anonymized_telemetry=False` |
| Knowledge graph | Kuzu is an embedded library; no daemon, no network |
| Document crawling | Crawler follows only `file://` / relative paths; external `http(s)://` links are skipped |
| OCR | pymupdf built-in OCR; tessdata files installed locally via Homebrew |
| File watching | watchdog watches local filesystem only; no network events |
| Web UI | FastAPI bound to `127.0.0.1:8000` only; no external interface |
| Logging | Writes to local rotating file only; no log shipping |
| LlamaIndex | Telemetry disabled via `set_global_handler("none")` + `LLAMA_INDEX_TELEMETRY=False` |
| Topic clustering | scikit-learn k-means runs entirely in-process; no network calls |
| BM25 index | rank-bm25 runs in-process; index persisted to local `bm25_index.pkl` |
| Query processing | QueryProcessor calls Ollama LLM at `localhost:11434`; no external calls |
| Graph visualisation | D3.js downloaded once by bootstrap to `arg/static/d3.min.js`; served by FastAPI; never loaded from CDN at runtime |
| Python packages | Installed once at bootstrap; vendored wheel cache supported for air-gapped use |

**Bootstrap** (`scripts/bootstrap.sh`) is the **only** step that may touch the network, and only
to download models and packages that are not yet present. Once bootstrap completes, ARG runs
entirely offline.

---

## 2. Project Structure
```
arg/
├── CLAUDE.md                  ← this file
├── README.md
├── pyproject.toml
├── .env.example
│
├── docs/spec/                 ← detailed per-section specs (extracted from this file)
│   ├── section-03-bootstrap.md
│   ├── section-04-config.md
│   ├── section-05-crawler.md
│   ├── section-06-knowledge-graph.md
│   ├── section-07-indexer.md
│   ├── section-08-retriever.md
│   ├── section-09-generator.md
│   ├── section-10-pipeline.md
│   ├── section-11-integration.md
│   ├── section-12-e2e.md
│   └── section-19-operations.md
│
├── arg/
│   ├── __init__.py
│   ├── config.py              ← all tunables in one place
│   ├── server.py              ← FastAPI app (local web UI)
│   ├── static/
│   │   ├── index.html         ← single-page UI (vanilla HTML + JS + D3 graph)
│   │   └── d3.min.js          ← D3.js v7 downloaded once by bootstrap; served locally (never CDN)
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── crawler.py         ← recursive HTML+PDF link follower
│   │   ├── extractors.py      ← HTML→text, PDF→text, PDF→OCR
│   │   └── watcher.py         ← watchdog Observer + debounce logic
│   ├── graph/
│   │   ├── __init__.py
│   │   └── knowledge_graph.py ← Kuzu schema + CRUD + link edges + reverse lookup + analytics
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── chunker.py         ← semantic chunking strategy
│   │   └── indexer.py         ← LlamaIndex pipeline → ChromaDB (chunk + doc-level collections)
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── retriever.py       ← HybridRetriever: dense + BM25 + graph + RRF + reordering
│   │   └── bm25_index.py      ← BM25 sparse index (rank_bm25); persisted to bm25_index.pkl
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── generator.py       ← LlamaIndex query engine → Ollama LLM
│   │   └── query_processor.py ← QueryProcessor: rewrite, decompose, HyDE
│   ├── dci/
│   │   ├── __init__.py
│   │   ├── explorer.py        ← CorpusExplorer: list, reverse links, topic clusters
│   │   └── analyst.py         ← CorpusAnalyst: summarise, compare, scoped search, stats
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── json_formatter.py  ← Python logging → JSON lines
│   │   └── tracing.py         ← LlamaIndex CallbackManager (--debug mode)
│   └── pipeline.py            ← top-level: index(), query(), add/remove/update doc, DCI methods
│
├── tests/
│   ├── conftest.py            ← shared fixtures, temp dirs, mock docs
│   ├── fixtures/
│   │   ├── docs/              ← Corpus A: RAG fixture (Section 12)
│   │   │   ├── index.html
│   │   │   ├── page_a.html
│   │   │   ├── page_b.html
│   │   │   ├── subdir/
│   │   │   │   └── page_c.html
│   │   │   ├── manual.pdf          ← native text + table + running footer + /Subject metadata
│   │   │   ├── scanned_notice.pdf  ← image-only page; exercises OCR path
│   │   │   └── encrypted_notice.pdf ← password-protected; exercises skip path
│   │   └── clustering_docs/   ← Corpus B: clustering fixture (15 docs, 3 topics, Section 12)
│   │       ├── index.html
│   │       ├── t1_overview.html  … t1_backup.html    (Triton Database — 5 docs)
│   │       ├── t2_overview.html  … t2_monitoring.html (Poseidon Networking — 5 docs)
│   │       └── t3_overview.html  … t3_logging.html   (Hydra Scheduler — 5 docs)
│   ├── unit/
│   │   ├── test_crawler.py
│   │   ├── test_extractors.py
│   │   ├── test_chunker.py
│   │   ├── test_knowledge_graph.py
│   │   ├── test_indexer.py
│   │   ├── test_retriever.py
│   │   ├── test_generator.py
│   │   ├── test_watcher.py
│   │   ├── test_logging.py
│   │   ├── test_explorer.py
│   │   ├── test_analyst.py
│   │   └── test_server.py     ← FastAPI endpoint routing + response shape tests
│   ├── integration/
│   │   ├── test_crawler_to_graph.py
│   │   ├── test_graph_to_indexer.py
│   │   ├── test_indexer_to_retriever.py
│   │   ├── test_retriever_to_generator.py
│   │   ├── test_multi_corpus.py
│   │   └── test_dci_pipeline.py   ← DCI methods against real fixture corpus
│   └── e2e/
│       ├── test_full_rag.py        ← end-to-end RAG test
│       └── test_dci_e2e.py         ← end-to-end DCI interaction test
│
└── scripts/
    ├── bootstrap.sh           ← installs Ollama, Tesseract, pulls models, sets up venv
    ├── index_docs.py          ← CLI: index, query, serve, stats
    ├── serve.py               ← starts FastAPI + uvicorn on localhost:8000
    ├── reset_corpus.py        ← deletes arg_db/{corpus_name}/ for clean retry
    └── eval_retrieval.py      ← runs QA eval pairs; reports retrieval hit rate + latency
```

---

## 3. Prerequisites & Bootstrap

> **Full spec:** [docs/spec/section-03-bootstrap.md](docs/spec/section-03-bootstrap.md)

---

## 4. Configuration

> **Full spec:** [docs/spec/section-04-config.md](docs/spec/section-04-config.md)

---

## 5. Crawler & Extractors

> **Full spec:** [docs/spec/section-05-crawler.md](docs/spec/section-05-crawler.md)

---

## 6. Knowledge Graph

> **Full spec:** [docs/spec/section-06-knowledge-graph.md](docs/spec/section-06-knowledge-graph.md)

---

## 7. Indexer & Chunker

> **Full spec:** [docs/spec/section-07-indexer.md](docs/spec/section-07-indexer.md)

---

## 8. Retriever

> **Full spec:** [docs/spec/section-08-retriever.md](docs/spec/section-08-retriever.md)

---

## 9. Generator & CorpusAnalyst

> **Full spec:** [docs/spec/section-09-generator.md](docs/spec/section-09-generator.md)

---

## 10. Pipeline, CorpusExplorer, Web UI & Logging

> **Full spec:** [docs/spec/section-10-pipeline.md](docs/spec/section-10-pipeline.md)

---

## 11. Integration Tests

> **Full spec:** [docs/spec/section-11-integration.md](docs/spec/section-11-integration.md)

---

## 12. End-to-End Test & Fixture Corpus

> **Full spec:** [docs/spec/section-12-e2e.md](docs/spec/section-12-e2e.md)

---

## 13. Architecture Decisions (All Locked — Reference Only)

All decisions are locked and fully integrated into Sections 3–12. This section
is a quick reference only; the build Claude should not need to consult it.

| Decision | Choice |
|---|---|
| Re-indexing | watchdog live watching; hash-check on startup; `--no-watch` for CI |
| Query interface | FastAPI + uvicorn; `127.0.0.1:8000`; `?corpus=` parameter |
| Multi-corpus | `arg_db/{corpus_name}/` path convention; `corpus_name="default"` |
| PDF OCR | pdfplumber → pymupdf text → pymupdf OCR; tessdata via Homebrew |
| Logging | JSON lines rotating log + opt-in LlamaIndex debug tracing |
| Auth/access | Out of scope; unreadable files logged + skipped |
| Enrichment | 3-stage: doc search → link expansion → cluster expansion |
| Clustering | k-means; skip below `min_cluster_docs=10`; cache in `cluster_cache.json` |
| Contextual enrichment | `"{title} > {heading_path}: {text}"` prepended to chunk embedding text |
| Sparse retrieval | BM25 via rank_bm25; fused with dense via RRF (k=60) |
| Context reordering | Lost-in-middle U-shape; rank 1 → position 0; rank 2 → position -1 |
| Query processing | Rewrite → decompose → (optional HyDE); raw query used for generation |
| DCI integration | Fully integrated into Sections 6–10; not a post-RAG layer |

## 14. Build Order Checklist

Claude Code works through these in order, running tests and committing after each.
You monitor progress and say `"Continue"` after each section completes.

**To start:** open a terminal in the project root and run `claude`, then say:
> "Read CLAUDE.md completely. Build ARG section by section starting with Section 3.
> Run tests after each section. Do not proceed until tests pass. Commit after each section."

---

- [ ] **Section 3** — Bootstrap, pyproject.toml, .env.example, README.md
  - **Claude Code will:** run `pip index versions` to verify package names, resolve LlamaIndex versions, write all files, run `pytest tests/unit/test_bootstrap.py`
  - **You may need to:** manually run `pip install llama-index-core` if version resolution fails, then tell Claude Code the resolved version
  - **Done when:** bootstrap test passes; `python -c "import arg"` succeeds

- [ ] **Section 4** — `arg/config.py`
  - **Done when:** `pytest tests/unit/test_config.py` passes; derived paths compute correctly

- [ ] **Section 5 (HTML)** — Crawler + HTML extractor
  - **Tell Claude Code:** `"Build Section 5 HTML extractor and crawler only. Do not build the PDF extractor yet. Run tests before continuing."`
  - **Done when:** `pytest tests/unit/test_crawler.py tests/unit/test_extractors.py -k "not pdf"` passes

- [ ] **Section 5 (PDF)** — PDF extractor
  - **Tell Claude Code:** `"Now build the Section 5 PDF extractor. HTML extractor is already on disk."`
  - **Done when:** `pytest tests/unit/test_extractors.py -k "pdf"` passes

- [ ] **Section 5B** — Watcher
  - **Done when:** `pytest tests/unit/test_watcher.py` passes

- [ ] **Section 6** — Knowledge Graph (all methods — complete in one pass)
  - **Done when:** `pytest tests/unit/test_knowledge_graph.py` passes including persistence test

- [ ] **Section 7** — Indexer + Chunker + BM25 index bootstrap
  - **Tell Claude Code:** `"The BM25 index is written by the indexer during pipeline.index(), not by the retriever. The retriever only reads it."`
  - **Done when:** `pytest tests/unit/test_chunker.py tests/unit/test_indexer.py` passes; verify `bm25_index.pkl` written after a test index run

- [ ] **Section 8** — Hybrid Retriever (all 5 stages)
  - **Done when:** `pytest tests/unit/test_retriever.py` passes; Claude Code should verify each stage independently before running the full suite

- [ ] **Section 9** — Generator + QueryProcessor + CorpusAnalyst
  - **Done when:** `pytest tests/unit/test_generator.py tests/unit/test_analyst.py` passes (LLM is mocked; Ollama not needed)

- [ ] **Section 10** — Pipeline + CorpusExplorer + Web UI + Logging + CLI
  - **Tell Claude Code:** `"Build Section 10 file by file: pipeline.py first, test it, then continue with the next file."`
  - **Done when:** all Section 10 unit tests pass; `python scripts/index_docs.py --help` works

- [ ] **Section 11** — Integration Tests + conftest.py
  - **Tell Claude Code:** `"Produce conftest.py first and confirm it, then produce each integration test file."`
  - **Requires:** Ollama running with llama3.3 and nomic-embed-text
  - **Done when:** `pytest tests/integration/ -v` passes

- [ ] **Section 12** — Fixture Corpora + E2E Tests
  - **Tell Claude Code:** `"Produce the fixture HTML and PDF files first. Confirm them, then produce the test files."`
  - **Requires:** Ollama running
  - **Done when:** `pytest tests/e2e/ -v` passes

- [ ] ~~**Section 13**~~ — Reference only ✓

- [ ] **Section 19** — Read before first production index run

- [ ] **Post-build** — `python scripts/eval_retrieval.py --db ./arg_db --corpus default --qa eval/qa_pairs.json`

---

### After the build completes

```bash
# Full test suite
pytest tests/ -v

# Verify locality (should return nothing)
grep -rn "requests\.\|httpx\.get\|http://" arg/ --include="*.py" \
  | grep -v "localhost\|127.0.0.1\|11434\|test_"

# Index your real docs
python scripts/index_docs.py index --docs /path/to/your/docs --db ./arg_db

# Start the web UI
python scripts/index_docs.py serve --db ./arg_db
# Open http://localhost:8000

# Tag the stable starting point (only once post-build eval passes)
git tag -a v0.1.0 -m "Initial release: full RAG + DCI pipeline; all tests and eval passing"
git push origin v0.1.0
```

## 19. Operational Notes

> **Full spec:** [docs/spec/section-19-operations.md](docs/spec/section-19-operations.md)

---

*ARG — Archivist RAG Graph · Yarr, the docs give up their secrets to no one but us.
