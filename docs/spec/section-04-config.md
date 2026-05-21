# Section 4: Configuration

> **Prompt to Claude:** "Build Section 4 of ARG: config.py"

### What Claude will produce:
- `arg/config.py` — a single `ARGConfig` dataclass loaded from `.env` with sensible defaults

### Key tunables:
```python
@dataclass
class ARGConfig:
    # Required — no defaults
    docs_root: Path          # top-level directory containing index.html
    db_path: Path            # root of all ARG persistent storage

    # Derived paths — all computed from db_path / corpus_name; never set manually
    # corpus_name is passed to ARGPipeline(), not stored in ARGConfig
    # The following are computed properties, not fields:
    #   kuzu_path    → db_path / corpus_name / "kuzu/"
    #   chroma_path  → db_path / corpus_name / "chroma/"
    #   log_path     → db_path / corpus_name / "arg.log"
    #   summary_path → db_path / corpus_name / "summaries/"
    #   cluster_cache_path → db_path / corpus_name / "cluster_cache.json"
    #   debug_traces_path  → db_path / corpus_name / "debug_traces/"

    # Models
    llm_model: str           = "qwen3.6:35b-a3b-q4_K_M"
    embed_model: str         = "nomic-embed-text"
    ollama_base_url: str     = "http://localhost:11434"

    # Chunking
    chunk_size: int          = 1024    # tokens — raised from 512; technical docs need
                                       # larger chunks to keep facts in context.
                                       # Lower to 512 only if memory pressure is observed.
    chunk_overlap: int       = 128     # tokens — 12.5% of chunk_size; adjusted proportionally
    embed_dim: int           = 256     # Matryoshka: 64/128/256/512/768

    # Retrieval quality
    contextual_enrichment: bool = True   # prepend "{title} > {heading_path}:" to chunk embedding text
    bm25_enabled: bool          = True   # sparse BM25 retrieval alongside dense vector search
    top_k_vector: int           = 8      # dense vector hits per query
    top_k_graph: int            = 4      # graph-neighbour hits per traversal hop

    # Query processing
    query_rewrite: bool    = True   # rewrite conversational queries to technical language
    query_decompose: bool  = True   # decompose multi-part questions into sub-queries
    hyde_enabled: bool     = False  # HyDE: embed hypothetical answer; off by default
    graph_hop_depth: int     = 2       # how many link-hops to traverse

    # DCI enrichment
    enrich_enabled: bool     = True    # master switch for DCI enrichment
    enrich_min_score: float  = 0.5     # minimum doc similarity to trigger enrichment
    enrich_top_docs: int     = 3       # how many top docs to seed candidate set

    # Watchdog
    watch_enabled: bool      = True    # set False via --no-watch or .env
    watch_debounce_ms: int   = 500     # ignore rapid successive events on same file

    # PDF extraction
    ocr_enabled: bool        = True    # set False to skip OCR entirely
    ocr_char_threshold: int  = 100     # per-page; evaluated after table text counted
    pdf_layout_analysis: bool = True   # global default; override per-doc via .argconfig sidecar
    pdf_overrides: dict      = None    # programmatic per-filename overrides (None = use sidecar only)
    pdf_batch_size: int      = 10      # indexer checkpoints every N pages of large PDFs

    # Logging
    debug_tracing: bool      = False   # enable via --debug or ARG_DEBUG=1

    # DCI
    summary_cache: bool      = False   # write summaries to disk for reuse
    n_clusters: int          = 8       # topic cluster count; clamped to doc_count at runtime
    min_cluster_docs: int    = 10      # minimum docs required before clustering runs;
                                       # below this threshold, returns [{label:"All documents",
                                       # doc_ids:[...]}] without LLM calls or k-means

    # Crawler
    max_file_depth: int      = 10      # directory recursion limit
    # NOTE: No respect_robots field — all docs are local files; robots.txt is irrelevant

    # HTML extraction
    strip_selectors: list    = None    # CSS selectors for div-based nav; see Section 5 defaults
                                       # (None at dataclass level; extractor uses built-in defaults)
    title_separator: str     = " | "   # split pattern for cleaning site-suffixed page titles
    max_code_block_tokens: int = 256   # truncate <pre> blocks longer than this in chunk text
```

### Tests:
- Config loads correctly from `.env`
- Defaults apply when `.env` keys are absent
- Invalid paths raise clear errors at startup, not mid-run

### Mandatory telemetry-off settings (must be in `.env.example` and applied at startup):
```bash
# Disable all outbound telemetry — ARG is fully local
ANONYMIZED_TELEMETRY=False          # ChromaDB
LLAMA_INDEX_TELEMETRY=False         # LlamaIndex (also set via set_global_handler)
DO_NOT_TRACK=1                      # generic opt-out honoured by several libraries
```
`arg/config.py` must apply these at import time:
```python
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("DO_NOT_TRACK", "1")
from llama_index.core import set_global_handler
set_global_handler("none")
```
