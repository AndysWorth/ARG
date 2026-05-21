# ARG Build Order Checklist

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
  - **Requires:** Ollama running with qwen3.6 and nomic-embed-text
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
