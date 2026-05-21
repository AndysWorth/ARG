# Section 7: Indexer & Chunker

> **Prompt to Claude:** "Build Section 7 of ARG: the chunker and LlamaIndex indexer — complete"

### What Claude will produce:
- `arg/indexer/chunker.py` — semantic chunking with heading-aware splits
- `arg/indexer/indexer.py` — LlamaIndex ingestion pipeline → both ChromaDB collections

### Chunker strategy:
1. **Heading boundary detection:** The extractor (Section 5) injects `##H1##`, `##H2##`,
   `##H3##` sentinels into both HTML and PDF text. The chunker splits on these sentinels
   to keep sections semantically whole. H4–H6 (HTML only) are plain text and are NOT
   used as chunk boundaries.
2. **Cross-page stitching (PDFs):** The PDF extractor yields pages as a stream. The chunker
   accumulates pages, stitching text across page boundaries so a section heading on page 12
   whose body continues onto page 13 is kept in a single chunk. Page boundaries are noted
   in metadata but do not force chunk splits.
3. Apply token-based sliding window within each section (`chunk_size=1024, overlap=128`)
4. **Contextual enrichment before embedding (Contextual Retrieval):**
   Before each chunk is embedded, prepend a context prefix constructed from its metadata:
   ```
   "{title} > {heading_path}: {chunk_text}"
   ```
   Example: `"Kraken API — Authentication > OAuth Flow > Token Expiry: The access token expires after 3600 seconds..."`)
   This is stored as `embedding_text` — the text actually fed to nomic-embed-text.
   The raw `chunk_text` (without prefix) is stored separately and is what the LLM receives
   as context. The prefix improves embedding precision without polluting the LLM context.
   Configurable via `contextual_enrichment: bool = True` in config (default on).
5. Each chunk carries metadata: `doc_id`, `title`, `page_description`, `heading_path`,
   `position`, `file_type`, `page_number` (first page of chunk for PDFs), `has_table`,
   `has_code`
6. Chunks are written into both ChromaDB and the Kuzu graph
7. Indexer checkpoints to disk every `pdf_batch_size` pages to survive interruptions

### Indexer pipeline — two ChromaDB collections built simultaneously:
```
Document (from crawler)
  │
  ├─► [Doc-level embedding] ──────────────────────────────────────────────┐
  │     Text = page_description + first 512 tokens of body text           │
  │     (page_description prepended so doc vector reflects page summary)  │
  │     → nomic-embed-text → ChromaDB collection: "documents"             │
  │     Metadata: {doc_id, title, file_type, page_description}            │
  │     Used by: corpus_search(), find_document()                         │
  │     (Stage 0 enrichment uses BM25 chunk aggregation instead)          │
  │                                                                        │
  └─► [Chunk-level processing] ◄──────────────────────────────────────────┘
        chunker.py (H1–H3 boundary splits + sliding window)
        → contextual enrichment: embedding_text = "{title} > {heading_path}: {text}"
        → nomic-embed-text embeds embedding_text (dim=256 Matryoshka)
        → ChromaDB collection: "chunks"
           stored fields: chunk_text (raw), embedding_text (prefixed), all metadata
        → KnowledgeGraph.add_chunk() (writes to Kuzu)
        Metadata: {doc_id, title, page_description, heading_path, position,
                   file_type, page_number, has_table, has_code}
        Used by: RAG retriever, scoped search, raw chunk inspection
```

**Both collections are maintained in sync.** When a document is removed,
both its entry in `documents` and all its entries in `chunks` are deleted.
When a document is updated, both collections are refreshed atomically.

- **ChromaDB must be instantiated with telemetry off** (non-negotiable):
  ```python
  import chromadb
  client = chromadb.PersistentClient(
      path=str(config.chroma_path),
      settings=chromadb.Settings(anonymized_telemetry=False)
  )
  ```
- Uses LlamaIndex `IngestionPipeline` with `ChromaVectorStore` for the `chunks` collection
- Doc-level embeddings are written directly (not via LlamaIndex pipeline)
- Supports incremental re-indexing: hash-checks each file; skips unchanged documents
- On first run after `pipeline.index()` completes, writes `config_hash.json` (Section 19.4.6)
- Logs progress: documents processed, chunks created, doc embeddings created, time elapsed

### Tests (unit):
- `test_chunker.py`:
  - H1/H2/H3 sentinel markers from extractor produce correct chunk boundaries
  - A document with no H1–H3 headings produces a single section (no sentinel splits)
  - Chunks do not exceed `chunk_size` tokens
  - Overlap is correctly applied across heading boundaries
  - PDF page numbers propagate to chunk metadata
  - `has_table=True` set on chunks containing pipe-delimited Markdown tables
  - `has_code=True` set on chunks containing code block content
  - `page_description` present in chunk metadata when extractor provides it
  - `heading_path` reflects current section hierarchy correctly
  - With `contextual_enrichment=True`: `embedding_text` starts with `"{title} > {heading_path}:"`
  - With `contextual_enrichment=False`: `embedding_text` equals `chunk_text` (no prefix)
  - LLM context (`chunk_text`) never contains the contextual prefix
- `test_indexer.py`:
  - After indexing, `chunks` collection has correct chunk count
  - After indexing, `documents` collection has one entry per document
  - Doc-level embedding text is `page_description + body[:512_tokens]`,
    not body alone (verify by checking stored metadata)
  - Chunk `embedding_text` stored separately from `chunk_text` in ChromaDB metadata
  - Chunk metadata (doc_id, heading_path, has_table, has_code) retrievable from `chunks`
  - Doc metadata (doc_id, title, file_type, page_description) retrievable from `documents`
  - Re-indexing unchanged docs is a no-op (hash check passes; no writes)
  - Re-indexing a changed doc updates both `chunks` and `documents` entries
  - Removing a document deletes all its chunks from `chunks` and its entry from `documents`
