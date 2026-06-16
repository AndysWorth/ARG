"""Tests for `arg.config`.

Covers:
  * Required fields are required; missing them raises at startup.
  * Defaults apply when env vars are absent.
  * Env vars override defaults and are correctly typed (int, bool, float).
  * Derived paths are computed from `db_path / corpus_name`, with `corpus_name`
    as a method argument (not a stored field).
  * Telemetry-off env vars are applied at module import time.
  * Locality guardrail: non-local Ollama URLs are rejected at construction.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixture: a clean, scratch docs+db pair
# ---------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    db = tmp_path / "arg_db"
    # db itself does not need to pre-exist; only its parent must.
    return docs, db


# ---------------------------------------------------------------------------
# Telemetry-off (import-time side effect)
# ---------------------------------------------------------------------------


def test_telemetry_env_vars_applied_on_import():
    # arg.config sets these via os.environ.setdefault on import.
    import arg.config  # noqa: F401

    assert os.environ.get("ANONYMIZED_TELEMETRY") == "False"
    assert os.environ.get("LLAMA_INDEX_TELEMETRY") == "False"
    assert os.environ.get("DO_NOT_TRACK") == "1"


def test_telemetry_existing_env_not_overwritten(monkeypatch):
    # If the user has explicitly set one of these (say to "True" for a sandbox), the
    # config module must use `setdefault`, not overwrite.
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    import arg.config

    importlib.reload(arg.config)
    assert os.environ["DO_NOT_TRACK"] == "0"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_defaults_applied(paths):
    from arg.config import ARGConfig

    docs, db = paths
    cfg = ARGConfig(docs_root=docs, db_path=db)
    assert cfg.llm_model == "gemma4:26b-mlx"
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.ollama_base_url == "http://localhost:11434"
    assert cfg.chunk_size == 1024
    assert cfg.chunk_overlap == 128
    assert cfg.embed_dim == 256
    assert cfg.bm25_enabled is True
    assert cfg.enrich_min_score == 0.5
    assert cfg.server_host == "127.0.0.1"
    assert cfg.server_port == 8000


def test_paths_resolved_and_db_created(paths):
    from arg.config import ARGConfig

    docs, db = paths
    cfg = ARGConfig(docs_root=docs, db_path=db)
    assert cfg.docs_root == docs.resolve()
    assert cfg.db_path == db.resolve()
    assert cfg.db_path.is_dir(), "db_path should be created if absent"


def test_string_paths_coerced(paths):
    from arg.config import ARGConfig

    docs, db = paths
    # Intentionally pass strings to verify __post_init__ coerces to Path.
    cfg = ARGConfig(docs_root=str(docs), db_path=str(db))  # type: ignore[arg-type]
    assert isinstance(cfg.docs_root, Path)
    assert isinstance(cfg.db_path, Path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_docs_root_raises(tmp_path):
    from arg.config import ARGConfig

    db = tmp_path / "arg_db"
    with pytest.raises(FileNotFoundError, match="docs_root does not exist"):
        ARGConfig(docs_root=tmp_path / "nope", db_path=db)


def test_docs_root_must_be_directory(tmp_path):
    from arg.config import ARGConfig

    file_path = tmp_path / "notadir.txt"
    file_path.write_text("nope")
    with pytest.raises(NotADirectoryError):
        ARGConfig(docs_root=file_path, db_path=tmp_path / "arg_db")


def test_db_parent_must_exist(tmp_path):
    from arg.config import ARGConfig

    docs = tmp_path / "docs"
    docs.mkdir()
    bad_db = tmp_path / "missing_parent" / "arg_db"
    with pytest.raises(FileNotFoundError, match="db_path parent does not exist"):
        ARGConfig(docs_root=docs, db_path=bad_db)


@pytest.mark.parametrize(
    "field,value,exc_match",
    [
        ("chunk_size", 0, "chunk_size must be > 0"),
        ("chunk_overlap", -1, "chunk_overlap must satisfy"),
        ("chunk_overlap", 2048, "chunk_overlap must satisfy"),  # >= chunk_size
        ("embed_dim", 100, "embed_dim must be one of"),
        ("enrich_min_score", 1.5, r"enrich_min_score must be in \[0.0, 1.0\]"),
        ("enrich_min_score", -0.1, r"enrich_min_score must be in \[0.0, 1.0\]"),
        ("graph_hop_depth", -1, "graph_hop_depth must be >= 0"),
        ("server_port", 0, "server_port out of range"),
        ("server_port", 70000, "server_port out of range"),
    ],
)
def test_invalid_numeric_fields_rejected(paths, field, value, exc_match):
    from arg.config import ARGConfig

    docs, db = paths
    with pytest.raises(ValueError, match=exc_match):
        ARGConfig(docs_root=docs, db_path=db, **{field: value})


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "http://10.0.0.5:11434",
        "https://my-remote-llm.cloud",
    ],
)
def test_non_local_ollama_url_rejected(paths, url):
    from arg.config import ARGConfig

    docs, db = paths
    with pytest.raises(ValueError, match="ollama_base_url must point at localhost"):
        ARGConfig(docs_root=docs, db_path=db, ollama_base_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://0.0.0.0:11434",
        "http://[::1]:11434",
    ],
)
def test_local_ollama_urls_accepted(paths, url):
    from arg.config import ARGConfig

    docs, db = paths
    cfg = ARGConfig(docs_root=docs, db_path=db, ollama_base_url=url)
    assert cfg.ollama_base_url == url


# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------


def test_derived_paths_default_corpus(paths):
    from arg.config import ARGConfig

    docs, db = paths
    cfg = ARGConfig(docs_root=docs, db_path=db)
    root = cfg.db_path / "default"
    assert cfg.corpus_root() == root
    assert cfg.kuzu_path() == root / "kuzu"
    assert cfg.chroma_path() == root / "chroma"
    assert cfg.log_path() == root / "arg.log"
    assert cfg.summary_path() == root / "summaries"
    assert cfg.cluster_cache_path() == root / "cluster_cache.json"
    assert cfg.debug_traces_path() == root / "debug_traces"
    assert cfg.bm25_index_path() == root / "bm25_index.pkl"
    assert cfg.config_hash_path() == root / "config_hash.json"


def test_derived_paths_named_corpus(paths):
    from arg.config import ARGConfig

    docs, db = paths
    cfg = ARGConfig(docs_root=docs, db_path=db)
    assert cfg.kuzu_path("product_a") == cfg.db_path / "product_a" / "kuzu"
    assert cfg.chroma_path("product_b") == cfg.db_path / "product_b" / "chroma"


def test_corpus_name_not_a_field(paths):
    """corpus_name lives on ARGPipeline, not ARGConfig (per Section 4 spec)."""
    from arg.config import ARGConfig

    docs, db = paths
    cfg = ARGConfig(docs_root=docs, db_path=db)
    assert not hasattr(cfg, "corpus_name")


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------


def test_from_env_reads_paths_from_environment(paths, monkeypatch):
    from arg.config import ARGConfig

    docs, db = paths
    monkeypatch.setenv("ARG_DOCS_PATH", str(docs))
    monkeypatch.setenv("ARG_DB_PATH", str(db))
    cfg = ARGConfig.from_env()
    assert cfg.docs_root == docs.resolve()
    assert cfg.db_path == db.resolve()


def test_from_env_requires_paths(monkeypatch):
    from arg.config import ARGConfig

    monkeypatch.delenv("ARG_DOCS_PATH", raising=False)
    monkeypatch.delenv("ARG_DB_PATH", raising=False)
    with pytest.raises(ValueError, match="docs_root not provided"):
        ARGConfig.from_env()


def test_from_env_typed_overrides(paths, monkeypatch):
    from arg.config import ARGConfig

    docs, db = paths
    monkeypatch.setenv("CHUNK_SIZE", "2048")
    monkeypatch.setenv("CHUNK_OVERLAP", "256")
    monkeypatch.setenv("BM25_ENABLED", "false")
    monkeypatch.setenv("ENRICH_MIN_SCORE", "0.75")
    monkeypatch.setenv("QUERY_REWRITE", "true")
    monkeypatch.setenv("SERVER_PORT", "9001")
    cfg = ARGConfig.from_env(docs_root=docs, db_path=db)
    assert cfg.chunk_size == 2048
    assert cfg.chunk_overlap == 256
    assert cfg.bm25_enabled is False
    assert cfg.enrich_min_score == 0.75
    assert cfg.query_rewrite is True
    assert cfg.server_port == 9001


def test_from_env_bad_int_reports_var_name(paths, monkeypatch):
    from arg.config import ARGConfig

    docs, db = paths
    monkeypatch.setenv("CHUNK_SIZE", "not_a_number")
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        ARGConfig.from_env(docs_root=docs, db_path=db)


def test_from_env_explicit_overrides_win(paths, monkeypatch):
    from arg.config import ARGConfig

    docs, db = paths
    monkeypatch.setenv("CHUNK_SIZE", "2048")
    cfg = ARGConfig.from_env(docs_root=docs, db_path=db, chunk_size=512)
    assert cfg.chunk_size == 512


def test_from_env_accepts_legacy_top_k_dense(paths, monkeypatch):
    """The .env.example template ships with TOP_K_DENSE; map it onto top_k_vector."""
    from arg.config import ARGConfig

    docs, db = paths
    monkeypatch.setenv("TOP_K_DENSE", "12")
    cfg = ARGConfig.from_env(docs_root=docs, db_path=db)
    assert cfg.top_k_vector == 12
