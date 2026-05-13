# Section 11: Integration Tests

> **Prompt to Claude:** "Build Section 11 of ARG: all integration tests"

These tests verify that **component boundaries work correctly together**.
Each test spins up real (non-mocked) instances but uses the small fixture corpus.

### `conftest.py` — shared fixtures (must be produced first):
```python
# tests/conftest.py — Claude must produce this before any test file

@pytest.fixture(scope="session")
def corpus_a_path() -> Path:
    """Absolute path to the Kraken API RAG fixture corpus."""

@pytest.fixture(scope="session")
def corpus_b_path() -> Path:
    """Absolute path to the clustering fixture corpus."""

@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Fresh temporary arg_db directory; deleted after each test."""

@pytest.fixture
def base_config(tmp_db, corpus_a_path) -> ARGConfig:
    """ARGConfig pointing at corpus_a and a fresh tmp_db."""

@pytest.fixture
def indexed_pipeline(base_config) -> ARGPipeline:
    """
    Fully indexed ARGPipeline over corpus_a.
    Cluster cache pre-built. Watcher disabled (--no-watch).
    Torn down (pipeline.close()) after each test.
    """

@pytest.fixture
def mock_llm():
    """
    Patch OllamaLLM to return deterministic responses.
    Used by all tests that call the generator to avoid requiring
    a running Ollama instance in CI.
    Response map: {prompt_substring → canned_answer}
    """
```
All integration and E2E tests import from `conftest.py`. The build Claude must
produce this file before any test file in Section 11.

### `test_crawler_to_graph.py`
- Run crawler on fixture corpus → feed output to KnowledgeGraph
- Assert: all fixture documents appear as nodes
- Assert: known links (index.html → page_a.html) appear as edges
- Assert: `get_linked_docs("index.html", depth=1)` returns `["page_a.html", "page_b.html"]`

### `test_graph_to_indexer.py`
- Run crawler → graph → indexer
- Assert: ChromaDB `chunks` collection has correct number of chunks
- Assert: ChromaDB `documents` collection has one entry per document
- Assert: each chunk's `doc_id` metadata matches a graph node
- Assert: graph `chunk_count` field matches ChromaDB count per document
- Assert: BM25 index file (`bm25_index.pkl`) exists after indexing
- Assert: chunk `embedding_text` starts with `"{title} > {heading_path}:"` when `contextual_enrichment=True`

### `test_indexer_to_retriever.py`
- Index fixture corpus → run retriever with a query known to match fixture content
- Assert: at least one chunk is returned from dense Stage 1
- Assert: BM25 Stage 1.5 returns chunks matching exact term "Kraken API"
- Assert: RRF fusion combines both; deduplicated result set
- Assert: graph expansion retrieves chunks from a document linked by the top hit
- Assert: lost-in-middle reordering: highest-scored chunk at position 0
- Assert: `filters={"has_table": True}` returns only table-containing chunks

### `test_retriever_to_generator.py`
- Index fixture corpus → QueryProcessor → retrieve → generate (LLM **mocked**)
- Assert: conversational query rewritten before retrieval; raw query used in LLM prompt
- Assert: compound query decomposed; chunks from both sub-queries present in context
- Assert: generator receives correctly formatted context string
- Assert: source citations in response match retrieved chunk doc_ids
- Assert: empty retrieval → "The documentation does not cover this topic."
