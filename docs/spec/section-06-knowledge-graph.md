# Section 6: Knowledge Graph

> **Prompt to Claude:** "Build Section 6 of ARG: the Kuzu knowledge graph — complete"

### What Claude will produce:
- `arg/graph/knowledge_graph.py` — complete `KnowledgeGraph` class with all methods

### Schema:
```
Node: Document
  - doc_id: STRING (PK) — absolute path
  - title: STRING
  - file_type: STRING   — "html" | "pdf"
  - chunk_count: INT64

Node: Chunk
  - chunk_id: STRING (PK) — "{doc_id}::chunk::{n}"
  - text: STRING
  - token_count: INT64

Relationship: LINKS_TO (Document → Document)
  - anchor_text: STRING  — the link text, if any

Relationship: CONTAINS (Document → Chunk)
  - position: INT64      — chunk index within document
```

### Complete method set (all methods built in this section, not added later):
```python
class KnowledgeGraph:
    # --- Core CRUD ---
    def add_document(doc: Document) -> None
    def add_chunk(chunk: Chunk, doc_id: str, position: int) -> None
    def add_link(source_doc_id: str, target_doc_id: str, anchor_text: str) -> None
    def remove_document(doc_id: str) -> None          # deletes node + edges + chunks
    def get_chunks_for_doc(doc_id: str) -> list[str]
    def get_doc_metadata(doc_id: str) -> dict
    def stats() -> dict                                # node/edge counts for diagnostics

    # --- Traversal (used by RAG retriever) ---
    def get_linked_docs(doc_id: str, depth: int) -> list[str]   # forward edges

    # --- Exploration (used by CorpusExplorer + retriever enrichment) ---
    def get_reverse_links(doc_id: str) -> list[dict]            # inbound edges
    def list_all_documents() -> list[dict]                      # all Document nodes
    def get_graph_json() -> dict                                # {nodes, edges} for D3

    # --- Analytics (used by CorpusAnalyst + /corpus/stats endpoint) ---
    def most_linked_docs(top_n: int = 10) -> list[dict]         # ranked by inbound count
    def orphaned_docs() -> list[str]                            # zero inbound edges
    def docs_by_chunk_count(descending: bool = True) -> list[dict]  # ranked by size
```

### Tests (unit — `test_knowledge_graph.py`):
- Documents and chunks insert without error; `stats()` reflects counts
- `get_linked_docs` returns correct neighbours at depth 1 and 2
- Circular links (A→B→A) do not cause infinite recursion
- `get_reverse_links("page_b.html")` returns docs that link to page_b
- `list_all_documents()` returns all indexed docs
- `get_graph_json()` returns valid `{nodes, edges}`; counts match
- `most_linked_docs()` ranks by incoming edge count correctly
- `orphaned_docs()` returns docs with zero inbound edges only
- `docs_by_chunk_count()` returns all docs in correct order; no truncation
- `remove_document()` deletes node, all edges, and all chunks from Kuzu
- Graph persists to disk and reloads correctly across process restarts
