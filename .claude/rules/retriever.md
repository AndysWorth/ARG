---
globs: ["arg/retriever/**"]
---

The BM25 index is written by the indexer during `pipeline.index()`, never by the retriever.
`arg/retriever/` only reads `bm25_index.pkl` via `BM25Index.load()`. Never call
`BM25Index.build()` or `BM25Index.save()` from within `arg/retriever/`.
