# ARG — Archivist RAG Graph
### A fully local, knowledge-graph-augmented RAG for HTML, PDF, and plain-text documentation
#### Optimized for Apple M1 Max · 64GB Unified Memory · Open Source Only

---

## 0. Working with Claude Code

ARG is built using **Claude Code**. This file is Claude Code's persistent project specification.

**Activate the venv before every session:** `source .venv/bin/activate`

Quality gates: `ruff` (lint+format), `mypy` (types), `pytest`, `pre-commit` — all wired up. **Never bypass with `--no-verify`.**

---

### Session start checklist

Before making any changes in a session:

1. `source .venv/bin/activate`
2. `python scripts/check_imports.py` — verify no imports are broken before touching anything
3. Re-read the relevant `.claude/rules/` file(s) for the module being touched
4. Run the module's tests to establish a baseline: `pytest tests/unit/test_<module>.py -x -q`

After all edits:
- `ruff check arg/` then `mypy arg/ --ignore-missing-imports`
- Run the locality check: `grep -rn "requests\.\|httpx\.get\|http://" arg/ --include="*.py" | grep -v "localhost\|127.0.0.1\|11434\|test_"`
- `pytest tests/unit/ -q` — full unit suite must stay green

---

### Test discipline

**If a test must change to accommodate an implementation change, pause and confirm before proceeding. A test change is a contract change.**

Specifically:
- `tests/unit/test_invariants.py` and `tests/unit/test_concurrency.py` are **off-limits** — only add to them, never modify existing tests
- Never add `pytest.mark.skip` or `pytest.mark.xfail` without a tracked reason in the commit message
- Track test counts: `pytest tests/unit/ --co -q 2>/dev/null | tail -1` — the total must not decrease

---

### Branch sizing

Keep branches to ≤ 7 changed files. When a branch needs more, split it. Oversized branches are harder to review and more likely to introduce conflicts.

---

### Section-by-section flow (what Claude Code does automatically)

For each section, Claude Code will:
1. Create a fresh branch off `main`: `git switch main && git pull && git switch -c section-N-<short-name>`
2. Re-read the relevant section spec from `docs/spec/`
3. Read any existing files it will import from
4. Write all specified files to disk
5. Run the section's unit tests; fix failures until they pass
6. Run the locality check: `grep -rn "requests\.\|httpx\.get\|http://" arg/ --include="*.py" | grep -v "localhost\|127.0.0.1\|11434\|test_"`
7. Stage **explicit paths only** (never `git add -A` / `git add .`), commit with **Conventional Commits** format, push branch
8. Report branch name, test summary, commit SHA — wait for `"Continue"`
9. On `"Continue"`: `git switch main && git merge --ff-only section-N-<name> && git push && git branch -d section-N-<name> && git push origin --delete section-N-<name>`

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

If pip cannot find a compatible LlamaIndex set automatically (`ResolutionImpossible`):

```bash
# Run this yourself, note the version it installs
pip install llama-index-core

# Then tell Claude Code:
"llama-index-core resolved to X.Y.Z. Find all compatible llama-index-* versions
for this base version and update pyproject.toml."
```

---

### When to intervene

Intervene if Claude Code:
- **Weakens a test** → `"Do not change tests. Fix the implementation."`
- **Skips a spec requirement** → `"You missed [requirement]. Re-read Section N and implement it."`
- **Uses a different library** → `"We use [specified library] for this. Do not substitute alternatives."`
- **Makes a network call** outside of Ollama → `"This violates the locality guarantee. Remove all outbound network calls."`
- **Bypasses hooks** with `--no-verify` / `--no-gpg-sign` → `"Do not bypass hooks. Fix the underlying issue and create a new commit."`
- **Uses `git add -A` / `git add .` or `git reset --hard`** → `"Stage explicit paths only. For recovery, delete the section branch rather than reset --hard."`

---

### If the build goes wrong

Recovery is non-destructive — abandon the section branch and start over from `main`.
Use `git revert` not `git reset --hard` if a bad section already merged to main.

```bash
git switch main
git branch -D section-N-<short-name>
git push origin --delete section-N-<short-name>
git restore .   # discard stray uncommitted edits
```

Tell Claude Code: `"Abandon the section-N branch and retry Section N from main."`

---

### What "done" looks like

```bash
pytest tests/unit/        # all pass (no Ollama needed — LLM is mocked)
pytest tests/integration/ # all pass (requires Ollama running)
pytest tests/e2e/         # all pass (requires Ollama running)
python scripts/index_docs.py --help   # CLI works
python scripts/serve.py              # server starts on :8000
```

Post-build eval: `python scripts/eval_retrieval.py --db ./arg_db --corpus default --qa eval/qa_pairs.json`

---

## 1. Locality Guarantee

**ARG makes zero outbound network calls during operation.** All Ollama calls go to
`config.ollama_base_url` (localhost:11434). Bootstrap is the only step that may touch
the network. See `.claude/rules/core.md` for the enforced invariant.

### Technology stack (key choices)

| Layer | Library / tool | Notes |
|---|---|---|
| **Embeddings** | nomic-embed-text via Ollama | Local inference; no cloud API |
| **LLM** | qwen3.6:35b-a3b-q4_K_M via Ollama | Local inference; no cloud API |
| **Vector store** | ChromaDB | Two collections per corpus: documents + chunks |
| **Graph DB** | Kuzu (embedded) | LINKS_TO + CONTAINS edges |
| **Sparse Retrieval** | bm25s (Rust-backed BM25) | Replaced rank_bm25; Feature 0003. |
| **PDF Parsing** | pdfplumber (primary) + pymupdf (fallback + OCR) | Single-pass extraction; Feature 0003. |
| **Text Parsing** | Python stdlib `pathlib.Path.read_text()` | UTF-8 with latin-1 fallback; Feature 0001. |
| **Web framework** | FastAPI | Local-only; never exposed to public internet |

---

## 2. Project Structure

```
arg/
├── CLAUDE.md                  ← this file
├── README.md
├── pyproject.toml
├── .env.example
│
├── docs/spec/                 ← detailed per-section specs
├── docs/BUILD_ORDER.md        ← section-by-section checklist with exact pytest commands
│
├── arg/
│   ├── config.py              ← all tunables in one place
│   ├── server.py              ← FastAPI app (local web UI)
│   ├── static/                ← index.html + d3.min.js (served locally, never CDN)
│   ├── crawler/               ← crawler.py, extractors.py, watcher.py
│   ├── graph/                 ← knowledge_graph.py (Kuzu)
│   ├── indexer/               ← chunker.py, indexer.py (ChromaDB + BM25)
│   ├── retriever/             ← retriever.py (5-stage hybrid), bm25_index.py
│   ├── generator/             ← generator.py, query_processor.py
│   ├── dci/                   ← explorer.py, analyst.py
│   ├── logging/               ← json_formatter.py, tracing.py
│   └── pipeline.py            ← top-level façade
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/              ← Corpus A (RAG) + Corpus B (clustering); see section-12 spec
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── scripts/
    ├── bootstrap.sh
    ├── index_docs.py          ← CLI: index [--subset] [--include] [--reset], query, serve, stats
    ├── serve.py
    ├── reset_corpus.py
    ├── eval_retrieval.py
    ├── inspect_doc.py         ← show all indexed data for a given file path
    ├── debug_retrieval.py     ← show results from all 5 retrieval stages for a query
    └── debug_stage0.py        ← show Stage 0 BM25 sub-steps for a query
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

## 13. Architectural Decision Records

Post-build changes with lasting impact. Each row is a decision + rationale so
future Claude sessions can judge whether it's still load-bearing.

| Decision | Summary |
|---|---|
| Plain-text indexing | Feature 0001: `.txt` / `.md` / `.markdown` accepted via `extract_text`; UTF-8 → latin-1 fallback; no markdown heading detection (future feature). |
| Large-corpus hardening | Feature 0003: bm25s replaces rank_bm25; cluster compute async; batched embedding fetch; single-pass PDF extraction; parallel sub-query embedding; correctness fixes for duplicate edges, SKIP/LIMIT, and nested `$and` filters. See `docs/features/0003-large-corpus-hardening.md`. |
| Index quality (Feature 0006) | Warns on 0-chunk docs; `max_chunks_per_doc` cap prevents large manuals from dominating BM25; AcroForm field values extracted via `page.widgets()`; encrypted PDFs produce searchable stubs; `n_clusters` default raised 8→16; OCR quality logged when page yields <25 chars post-OCR. See `docs/features/0006-index-quality-improvements.md`. |

---

## 14. Build Order Checklist

> **Full checklist with per-section pytest commands:** [docs/BUILD_ORDER.md](docs/BUILD_ORDER.md)

---

## 19. Operational Notes

> **Full spec:** [docs/spec/section-19-operations.md](docs/spec/section-19-operations.md)

---

## 20. Post-Build Behaviours (established after initial build)

These behaviours were added after the original section build and are not described in `docs/spec/`. They are enforced by the existing test suite.

### Clustering

- `pipeline.index()` calls `explorer.get_topic_clusters()` at the end, so the cluster cache is always warm when the server starts. Do not remove this call.
- `_compute_clusters()` performs **incremental labeling**: before calling the LLM for a cluster label, it checks whether the cluster's member set (as a `frozenset`) matches any cluster in the old cache. An exact match reuses the existing label with zero LLM calls. Only clusters whose membership changed pay an LLM call.
- Watcher-triggered `add_document` / `update_document` / `remove_document` call `pipeline._recompute_clusters()`, which invalidates then immediately recomputes (not lazily). This keeps the UI current after file changes.

### Link extraction order

- In `arg/crawler/extractors.py`, `_extract_links(soup)` **must be called before** `_strip_invisible_and_boilerplate(soup, config)`. The strip pass removes `<nav>` and sidebar elements; calling it first silently drops links from index pages that use `<nav>` for cross-document navigation.

### Logging

Operational `INFO` log lines were added to aid observability. Do not remove them:

| File | What is logged |
|---|---|
| `pipeline.py` | `pipeline: ready (corpus=…, docs=…, query_rewrite=…, query_decompose=…)` and `pipeline: closed` |
| `pipeline.py` | `watcher: <event> <filename>` on every file-change event |
| `retriever.py` | `retriever: BM25 index reloaded` after every watcher-triggered reload |
| `generator.py` | `generator: query answered in <ms> — "<query>"` after every non-streaming answer |
| `explorer.py` | `explorer: labeling cluster <id> (<n> docs) via LLM` before each LLM label call |
| `analyst.py` | `analyst: summary cache hit for <file>` or `analyst: summarizing <file> via LLM` |

### Web UI — `/file` endpoint

`GET /file?path=<doc_id>&corpus=<name>` serves a file from `docs_root` with path-traversal protection (403 if the resolved path escapes `docs_root`). Source citations in query results link to this endpoint with `target=_blank`.

### Web UI — topic cluster graph

The "Document graph" panel was replaced with a topic cluster visualization. File types are distinguished by shape (circle=PDF, square=HTML, triangle=text/md, diamond=other). The graph supports scroll-to-zoom and drag-to-pan via `d3.zoom()`. Cluster labels are rendered in a reserved 28px strip at the top of each cell; a boundary force prevents dots from entering that strip.

---

*ARG — Archivist RAG Graph · Yarr, the docs give up their secrets to no one but us.*
