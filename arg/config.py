"""ARG configuration.

A single `ARGConfig` dataclass holds every tunable for the pipeline. Values are
loaded from environment variables (typically populated from a project-local
`.env` file via `python-dotenv`). Sensible defaults exist for everything except
the two paths the user must point at: the documentation root and the persistence
root.

Multi-corpus paths are *derived*, not stored. `corpus_name` is passed to
`ARGPipeline()` (see Section 10) and combined with `db_path` to yield the kuzu /
chroma / log / summary / cluster-cache / debug-traces locations. The helpers on
`ARGConfig` (`kuzu_path("default")`, etc.) are the single source of truth for
this layout.

Telemetry hard-off
------------------
The module sets `ANONYMIZED_TELEMETRY`, `LLAMA_INDEX_TELEMETRY`, and `DO_NOT_TRACK`
at import time so any later `import chromadb` / `import llama_index` picks them
up before reading their own environment. `setdefault` is used so an explicit
override from the environment wins — which is what the test suite relies on.

Why no `set_global_handler("none")`: LlamaIndex's `set_global_handler` is a
hook for *evaluation* backends (Weights & Biases, Arize Phoenix, Langfuse,
etc.) — `"none"` is not a registered eval mode, so the call raises
`ValueError` on 0.14.x. The library's default state (`global_handler is None`)
IS the no-telemetry state, so we leave it alone.

# Implements: docs/spec/section-04-config.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Apply telemetry-off env vars BEFORE chromadb / llama_index / etc. are imported
# anywhere in the process. `setdefault` leaves explicit overrides intact.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("LLAMA_INDEX_TELEMETRY", "False")
os.environ.setdefault("DO_NOT_TRACK", "1")

# Load .env into os.environ. python-dotenv is a hard dependency (see pyproject.toml),
# so this import is unconditional.
from dotenv import load_dotenv

load_dotenv(override=False)

DEFAULT_CORPUS = "default"


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"true", "1", "yes", "on"}


@dataclass
class ARGConfig:
    """Single source of truth for all ARG tunables.

    Construct directly (`ARGConfig(docs_root=..., db_path=...)`) for tests, or
    use `ARGConfig.from_env(...)` for normal use — the latter reads tunables
    from environment variables (populated from `.env`).
    """

    # --- Required ---------------------------------------------------------
    docs_root: Path
    db_path: Path

    # --- Models -----------------------------------------------------------
    llm_model: str = "qwen3.6:35b-a3b-q4_K_M"
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = 300.0

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = 1024
    chunk_overlap: int = 128
    embed_dim: int = 256

    # --- Retrieval quality ------------------------------------------------
    contextual_enrichment: bool = True
    bm25_enabled: bool = True
    top_k_vector: int = 8
    top_k_graph: int = 4

    # --- Query processing -------------------------------------------------
    query_rewrite: bool = True
    query_decompose: bool = True
    hyde_enabled: bool = False
    graph_hop_depth: int = 2

    # --- DCI enrichment ---------------------------------------------------
    enrich_enabled: bool = True
    enrich_min_score: float = 0.5
    enrich_top_docs: int = 3

    # --- Watchdog ---------------------------------------------------------
    watch_enabled: bool = True
    watch_debounce_ms: int = 500

    # --- PDF extraction ---------------------------------------------------
    ocr_enabled: bool = True
    ocr_char_threshold: int = 100
    pdf_layout_analysis: bool = True
    pdf_overrides: dict[str, dict[str, Any]] | None = None
    pdf_min_chars_per_page: int = 30  # below this avg → skip pdfplumber (image-dominated)
    pdf_extract_timeout_seconds: int = 300  # 0 = no timeout

    # --- Embedder ---------------------------------------------------------
    embed_num_ctx: int = 2048  # nomic-embed-text context window; keeps KV cache minimal
    embed_batch_size: int = 64  # chunks per Ollama embed call

    # --- Crawler ----------------------------------------------------------
    extraction_workers: int = 1  # 1 = serial (default); 4-6 recommended on M1 Max

    # --- Logging ----------------------------------------------------------
    debug_tracing: bool = False

    # --- DCI --------------------------------------------------------------
    summary_cache: bool = False
    n_clusters: int = 16
    min_cluster_docs: int = 10
    max_chunks_per_doc: int = 0  # 0 = unlimited

    # --- Crawler ----------------------------------------------------------
    max_file_depth: int = 100

    # --- HTML extraction --------------------------------------------------
    strip_selectors: list[str] | None = None
    title_separator: str = " | "
    max_code_block_tokens: int = 256

    # --- Server -----------------------------------------------------------
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Coerce inputs (callers may pass strings from CLI / env)
        self.docs_root = Path(self.docs_root).expanduser().resolve()
        self.db_path = Path(self.db_path).expanduser().resolve()

        if not self.docs_root.exists():
            raise FileNotFoundError(
                f"docs_root does not exist: {self.docs_root}. "
                "Point ARG at a directory containing the documentation to index."
            )
        if not self.docs_root.is_dir():
            raise NotADirectoryError(f"docs_root is not a directory: {self.docs_root}")

        # db_path may not exist yet — but its parent must, so we can mkdir it.
        if not self.db_path.parent.exists():
            raise FileNotFoundError(
                f"db_path parent does not exist: {self.db_path.parent}. "
                "Create the parent directory first."
            )
        self.db_path.mkdir(parents=True, exist_ok=True)

        # --- Numeric sanity ----------------------------------------------
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0 (got {self.chunk_size})")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size "
                f"(got chunk_overlap={self.chunk_overlap}, chunk_size={self.chunk_size})"
            )
        if self.embed_dim not in (64, 128, 256, 512, 768):
            raise ValueError(
                "embed_dim must be one of {64, 128, 256, 512, 768} "
                f"(Matryoshka dims for nomic-embed-text); got {self.embed_dim}"
            )
        if not 0.0 <= self.enrich_min_score <= 1.0:
            raise ValueError(
                f"enrich_min_score must be in [0.0, 1.0] (got {self.enrich_min_score})"
            )
        if self.graph_hop_depth < 0:
            raise ValueError(f"graph_hop_depth must be >= 0 (got {self.graph_hop_depth})")
        if self.server_port < 1 or self.server_port > 65535:
            raise ValueError(f"server_port out of range: {self.server_port}")

        # Locality guardrail: refuse to operate against a non-local Ollama URL.
        if not self._is_local_url(self.ollama_base_url):
            raise ValueError(
                f"ollama_base_url must point at localhost / 127.0.0.1 / 0.0.0.0 / ::1 "
                f"(got {self.ollama_base_url!r}). ARG is local-only by design."
            )

    @staticmethod
    def _is_local_url(url: str) -> bool:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    # ------------------------------------------------------------------
    # Derived paths (per-corpus)
    # ------------------------------------------------------------------
    def corpus_root(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.db_path / corpus_name

    def kuzu_path(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.corpus_root(corpus_name) / "kuzu"

    def chroma_path(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.corpus_root(corpus_name) / "chroma"

    def log_path(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.corpus_root(corpus_name) / "arg.log"

    def summary_path(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.corpus_root(corpus_name) / "summaries"

    def cluster_cache_path(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.corpus_root(corpus_name) / "cluster_cache.json"

    def debug_traces_path(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.corpus_root(corpus_name) / "debug_traces"

    def bm25_index_path(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.corpus_root(corpus_name) / "bm25_index.pkl"

    def config_hash_path(self, corpus_name: str = DEFAULT_CORPUS) -> Path:
        return self.corpus_root(corpus_name) / "config_hash.json"

    # ------------------------------------------------------------------
    # Env loader
    # ------------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        docs_root: str | Path | None = None,
        db_path: str | Path | None = None,
        **overrides: Any,
    ) -> ARGConfig:
        """Build an `ARGConfig` from environment variables.

        Explicit `docs_root` / `db_path` arguments win over the environment;
        `overrides` (keyword args matching dataclass field names) win over both.
        Loading `.env` happened at module import; nothing reloads it here.
        """
        env = os.environ

        resolved_docs = docs_root if docs_root is not None else env.get("ARG_DOCS_PATH")
        resolved_db = db_path if db_path is not None else env.get("ARG_DB_PATH")
        if not resolved_docs:
            raise ValueError("docs_root not provided and ARG_DOCS_PATH not set in environment")
        if not resolved_db:
            raise ValueError("db_path not provided and ARG_DB_PATH not set in environment")

        kwargs: dict[str, Any] = {
            "docs_root": Path(resolved_docs),
            "db_path": Path(resolved_db),
        }

        # Map of env var → (field name, parser).
        env_map: dict[str, tuple[str, Any]] = {
            # Models
            "OLLAMA_BASE_URL": ("ollama_base_url", str),
            "OLLAMA_LLM_MODEL": ("llm_model", str),
            "OLLAMA_EMBED_MODEL": ("embed_model", str),
            "OLLAMA_TIMEOUT": ("ollama_timeout", float),
            # Chunking
            "CHUNK_SIZE": ("chunk_size", int),
            "CHUNK_OVERLAP": ("chunk_overlap", int),
            "EMBED_DIM": ("embed_dim", int),
            # Retrieval
            "CONTEXTUAL_ENRICHMENT": ("contextual_enrichment", _parse_bool),
            "BM25_ENABLED": ("bm25_enabled", _parse_bool),
            "TOP_K_VECTOR": ("top_k_vector", int),
            "TOP_K_DENSE": ("top_k_vector", int),  # accept either name from .env
            "TOP_K_GRAPH": ("top_k_graph", int),
            # Query processing
            "QUERY_REWRITE": ("query_rewrite", _parse_bool),
            "QUERY_DECOMPOSE": ("query_decompose", _parse_bool),
            "HYDE_ENABLED": ("hyde_enabled", _parse_bool),
            "GRAPH_HOP_DEPTH": ("graph_hop_depth", int),
            # DCI enrichment
            "ENRICH_ENABLED": ("enrich_enabled", _parse_bool),
            "ENRICH_MIN_SCORE": ("enrich_min_score", float),
            "ENRICH_TOP_DOCS": ("enrich_top_docs", int),
            # Watchdog
            "WATCH_ENABLED": ("watch_enabled", _parse_bool),
            "WATCH_DEBOUNCE_MS": ("watch_debounce_ms", int),
            # PDF
            "OCR_ENABLED": ("ocr_enabled", _parse_bool),
            "OCR_CHAR_THRESHOLD": ("ocr_char_threshold", int),
            "PDF_LAYOUT_ANALYSIS": ("pdf_layout_analysis", _parse_bool),
            "PDF_MIN_CHARS_PER_PAGE": ("pdf_min_chars_per_page", int),
            "PDF_EXTRACT_TIMEOUT": ("pdf_extract_timeout_seconds", int),
            # Embedder
            "EMBED_NUM_CTX": ("embed_num_ctx", int),
            "EMBED_BATCH_SIZE": ("embed_batch_size", int),
            # Crawler (also extraction_workers)
            "EXTRACTION_WORKERS": ("extraction_workers", int),
            # Logging
            "DEBUG_TRACING": ("debug_tracing", _parse_bool),
            "ARG_DEBUG": ("debug_tracing", _parse_bool),
            # DCI
            "SUMMARY_CACHE": ("summary_cache", _parse_bool),
            "N_CLUSTERS": ("n_clusters", int),
            "MIN_CLUSTER_DOCS": ("min_cluster_docs", int),
            "MAX_CHUNKS_PER_DOC": ("max_chunks_per_doc", int),
            # Crawler
            "MAX_FILE_DEPTH": ("max_file_depth", int),
            # HTML
            "TITLE_SEPARATOR": ("title_separator", str),
            "MAX_CODE_BLOCK_TOKENS": ("max_code_block_tokens", int),
            # Server
            "SERVER_HOST": ("server_host", str),
            "SERVER_PORT": ("server_port", int),
        }
        for env_name, (field_name, parser) in env_map.items():
            raw = env.get(env_name)
            if raw is None or raw == "":
                continue
            try:
                kwargs[field_name] = parser(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid value for env var {env_name}={raw!r}: {exc}") from exc

        # Explicit `overrides` win over env.
        kwargs.update(overrides)
        return cls(**kwargs)
