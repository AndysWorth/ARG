# Section 9: Generator & CorpusAnalyst

> **Prompt to Claude:** "Build Section 9 of ARG: the generator and CorpusAnalyst — complete"

### What Claude will produce:
- `arg/generator/generator.py` — LlamaIndex query engine wrapping Ollama LLM,
  including query rewriting and decomposition
- `arg/generator/query_processor.py` — `QueryProcessor` class (rewrite + decompose)
- `arg/dci/analyst.py` — `CorpusAnalyst` class (shares the same LLM + retriever)
- `arg/dci/__init__.py` — re-exports both DCI classes (stub for CorpusExplorer until Section 10)

### Why together:
`CorpusAnalyst` uses the same Ollama LLM and `HybridRetriever` as the generator.
Building both in one session means one LLM client, one retriever reference, and no
dangling imports.

---

### Pre-retrieval: QueryProcessor (`arg/generator/query_processor.py`)

Before any retrieval runs, the raw user query passes through `QueryProcessor`.
This is the first place the LLM is called in a query flow.

**Step 1 — Query rewriting** (`query_rewrite: bool = True`):

Converts conversational or ambiguous queries into precise technical language that
matches documentation phrasing more closely.

LLM prompt:
```
You are a technical documentation assistant. Rewrite the following user question
into precise technical language that would appear in software documentation.
Keep the same meaning. Output only the rewritten question, nothing else.

User question: {raw_query}
```

Examples:
- `"How do I log in?"` → `"Authentication methods and API key configuration"`
- `"It keeps giving me errors"` → `"Error handling and common error codes"`
- `"What's the limit?"` → `"Rate limiting thresholds and request quotas"`

If the query is already technical (detected by presence of technical terms, version
numbers, HTTP status codes, or method names), skip rewriting and use as-is.
Heuristic: if query contains any of `[A-Z_]{3,}`, `\d{3}`, `/v\d`, `()`, skip.

**Step 2 — Query decomposition** (`query_decompose: bool = True`):

Splits compound questions into independent sub-queries, each retrievable separately.

LLM prompt:
```
Does the following question contain multiple independent sub-questions that should
be researched separately? If yes, list each sub-question on its own line.
If no, output the original question unchanged.

Question: {rewritten_query}
```

Examples:
- `"How does auth work and what are the rate limits?"` →
  `["How does authentication work?", "What are the rate limits?"]`
- `"What is the OAuth flow?"` → `["What is the OAuth flow?"]` (single question, no split)

If decomposition returns only one question, proceed with single retrieval.
If multiple questions, retrieve independently for each, union the chunk sets
(deduplicate by chunk_id), then pass the combined set to the generator once.

**QueryProcessor config:**
```python
query_rewrite: bool    = True   # rewrite conversational queries to technical language
query_decompose: bool  = True   # decompose multi-part questions
hyde_enabled: bool     = False  # HyDE: generate hypothetical answer, embed it instead
                                 # (more expensive; off by default; see note below)
```

**HyDE (Hypothetical Document Embeddings — opt-in):**
When `hyde_enabled=True`, instead of embedding the rewritten query, the LLM generates
a hypothetical answer paragraph and that paragraph is embedded. The hypothesis embedding
is typically much closer in semantic space to real documentation chunks than the query
itself. Expensive (one extra LLM call before every retrieval) but significantly improves
recall on complex questions. Enabled via `ARG_HYDE=1` in `.env`.

HyDE prompt:
```
Write a short paragraph (3-5 sentences) that would be a plausible answer to the
following question, as if it came from technical documentation. Be specific and
use technical language. Do not say "I don't know."

Question: {rewritten_query}
```

**Full pre-retrieval flow:**
```
raw_query
  → [rewrite if query_rewrite=True and query looks conversational]
  → [decompose if query_decompose=True]
  → for each sub-query:
      [HyDE: embed hypothetical answer if hyde_enabled=True]
      [else: embed rewritten query directly]
      → HybridRetriever.retrieve(embedding_query, ...)
  → union chunk sets (deduplicate)
  → Generator.generate(raw_query, chunks)  ← raw_query used for generation, not rewritten
```

Note: the LLM always sees the **original raw query** at generation time, not the
rewritten version. Rewriting is only for retrieval precision.

---

### Generator behaviour:
- Uses `RetrieverQueryEngine` with `HybridRetriever` from Section 8
- LLM: `qwen3.6:35b-a3b-q4_K_M` via `OllamaLLM`
- Returns `ARGResult`:
  ```python
  @dataclass
  class ARGResult:
      answer: str
      sources: list[SourceRef]         # [{doc_id, title, chunk_id, heading_path}]
      latency_ms: int
      enriched_doc_ids: list[str]      # doc_ids used in Stage 0 enrichment
      rewritten_query: str | None      # the rewritten query used for retrieval; None if skipped
      sub_queries: list[str] | None    # decomposed sub-queries; None if single query
  ```
- Streaming supported

### System prompt (used for all RAG queries):
```
You are Archivist, an expert assistant that answers questions using only
the provided documentation. Follow these rules for every answer:

SOURCING:
- Base your answer only on the context provided. Do not use outside knowledge.
- If the answer is not in the context, respond exactly:
  "The documentation does not cover this topic."
- Do not speculate, infer, or extrapolate beyond what the documents say.

CITATIONS:
- After each key claim, cite the source document title in parentheses.
  Example: "API keys expire after 90 days (Kraken API — Authentication)."

FORMAT — choose the format that matches the question type:
- Procedure / how-to: numbered steps. Each step on its own line.
- Reference / lookup (error codes, config values, limits): a compact table or
  bulleted list with key → value pairs.
- Concept / explanation: 2–4 sentences of plain prose. No bullet points.
- Comparison: a two-column table (Feature | Doc A | Doc B).
- Code syntax: a fenced code block with the appropriate language tag.

LENGTH:
- Answer only what is asked. No preamble, no closing remarks.
- Procedures: include all steps. Do not truncate.
- Explanations: maximum 4 sentences unless the topic genuinely requires more.

Context:
{context_str}

Question: {query_str}
```

---

### CorpusAnalyst capabilities (all in `arg/dci/analyst.py`):

**1. Document summarisation** — `summarize_document(doc_id: str) -> str`
- Fetches all chunks for doc via `retriever.retrieve(scope_doc_id=doc_id)`
- Prompts LLM: "Summarise this document in 3–5 sentences, preserving key facts and structure."
- For very long docs exceeding LLM context: map-reduce — summarise chunk batches then combine
- Cache: if `summary_cache=True`, writes to `summaries/{doc_id_hash}.json`; invalidated on
  `update_document()`

**2. Key-points extraction** — `extract_key_points(doc_id: str, max_points: int = 10) -> list[str]`
- Same fetch pattern; LLM prompted for a bulleted key-point list
- Returns JSON array of strings

**3. Document comparison** — `compare_documents(doc_id_a: str, doc_id_b: str) -> str`
- Fetches both docs' chunks; passes to LLM with structured comparison prompt
  (topics, overlap, unique content, contradictions)
- If combined text exceeds context: summarise each first, then compare summaries

**4. Scoped vector search** — `scoped_search(query: str, doc_id: str, top_k: int = 5) -> list[NodeWithScore]`
- Calls `retriever.retrieve(query, scope_doc_id=doc_id)` — no enrichment, no graph expansion
- Returns ranked chunks from that document only

**5. Raw chunk inspection** — `get_chunks(doc_id: str) -> list[dict]`
- Returns `[{chunk_id, position, text, token_count, heading_path}]` from KnowledgeGraph +
  ChromaDB metadata for all chunks belonging to the document

**6. Document-level search (internal)** — `find_document(query: str, top_k: int = 5) -> list[dict]`
- Searches the `documents` ChromaDB collection; returns `[{doc_id, title, similarity_score}]`
- **This method is private to ARG internals.** It is called by:
  - `HybridRetriever._find_document()` (Stage 0.1 of enrichment)
  - `CorpusExplorer.corpus_search()` (wraps it with `file_type` filter)
- It is NOT on the `ARGPipeline` public API

---

### `arg/dci/__init__.py` (stub — completed in Section 10):
```python
from arg.dci.analyst import CorpusAnalyst
# CorpusExplorer imported after Section 10 completes
__all__ = ["CorpusAnalyst"]
```

---

### Tests (unit — LLM mocked throughout):
- `test_generator.py`:
  - Generator calls retriever with rewritten query; passes original raw query to LLM
  - `ARGResult.rewritten_query` populated when `query_rewrite=True` and query is conversational
  - `ARGResult.rewritten_query` is `None` when query contains technical heuristic markers
  - `ARGResult.sub_queries` has multiple entries for compound question
  - `ARGResult.sub_queries` is `None` for simple single question
  - Chunks from all sub-queries unioned before generation; no duplicate chunk_ids
  - Source citations present in `ARGResult.sources`
  - Empty retrieval → "The documentation does not cover this topic."
  - Streaming yields tokens incrementally
  - `enrich=False` → `ARGResult.enriched_doc_ids` is empty list
  - `hyde_enabled=True`: hypothetical answer paragraph is embedded, not raw query
  - `query_rewrite=False`: raw query passed directly; `rewritten_query` is `None`
  - `query_decompose=False`: single retrieval; `sub_queries` is `None`
- `test_analyst.py`:
  - `summarize_document()` returns non-empty string; LLM called once (short doc)
  - `summarize_document()` triggers map-reduce for doc exceeding context (mocked)
  - `extract_key_points()` returns list of strings
  - `compare_documents()` returns non-empty string; both docs' chunks passed to LLM
  - `scoped_search()` returns only chunks from specified doc_id
  - `scoped_search()` never returns chunks from a different document
  - `get_chunks()` count matches `chunk_count` on Document node in Kuzu
  - `find_document()` returns ranked doc_ids from `documents` collection
  - Summary cache: second call with `summary_cache=True` reads from disk, not LLM
