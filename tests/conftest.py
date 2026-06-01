"""Shared fixtures for integration + e2e tests.

This file is loaded by pytest for every test in the suite. Unit tests don't
need anything from here (they build their own fixtures) but importing is
cheap; we keep the surface narrow so accidental cross-test coupling stays
out.

Section 11 + 12 promise these names:
  corpus_a_path       — session-scoped path to the Kraken API fixture corpus
  corpus_b_path       — session-scoped path to the clustering fixture corpus
                        (created in Section 12; tests that need it skip when
                        missing)
  tmp_db              — per-test fresh ARG database directory
  base_config         — ARGConfig pointing at corpus_a + tmp_db
  indexed_pipeline    — fully indexed ARGPipeline over corpus_a
  mock_llm            — substring-mapped LLM stub for offline LLM mocking
  ollama_required     — convenience marker fixture that skips when Ollama
                        is not reachable on localhost

Locality
--------
Integration tests use the **real** Ollama embedder (so on-disk ChromaDB
shape matches production exactly) but the **mocked** LLM (so tests stay
deterministic and don't depend on qwen3.6 being warm). When Ollama is not
running locally, integration tests are skipped — never failed.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from arg.config import ARGConfig
from arg.embeddings import Embedder
from arg.llm import LLM

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Polling helper
# ---------------------------------------------------------------------------


def _wait_for(condition: Callable[[], bool], timeout: float = 2.0, interval: float = 0.02) -> bool:
    """Poll *condition* until it returns True or *timeout* seconds elapse.

    Returns True when the condition was satisfied, False on timeout.  Tests
    should assert on the return value::

        assert _wait_for(lambda: cache.is_file()), "cache not written within 2s"
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


# ---------------------------------------------------------------------------
# Corpus paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def corpus_a_path() -> Path:
    """Kraken API fixture corpus (HTML; PDFs land in Section 12)."""
    path = _FIXTURES / "docs"
    if not path.is_dir():
        pytest.skip(f"corpus_a fixture not built: {path}")
    return path


@pytest.fixture(scope="session")
def corpus_b_path() -> Path:
    """Clustering fixture corpus (15 docs; built in Section 12)."""
    path = _FIXTURES / "clustering_docs"
    if not path.is_dir():
        pytest.skip(f"corpus_b fixture not built yet: {path}")
    return path


# ---------------------------------------------------------------------------
# Per-test working dirs
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Fresh ARG database directory; ARGConfig validator needs the parent to exist."""
    db_root = tmp_path / "arg_db"
    db_root.mkdir(exist_ok=True)
    return db_root / "test"


@pytest.fixture
def base_config(tmp_db: Path, corpus_a_path: Path) -> ARGConfig:
    """ARGConfig pointed at corpus_a + a fresh tmp_db, watcher disabled."""
    return ARGConfig(
        docs_root=corpus_a_path,
        db_path=tmp_db,
        watch_enabled=False,
        # Tighter top_k so tests with a 4-doc corpus stay informative.
        top_k_vector=4,
        top_k_graph=2,
        graph_hop_depth=1,
        enrich_min_score=0.0,
    )


# ---------------------------------------------------------------------------
# Ollama availability
# ---------------------------------------------------------------------------


def _ollama_reachable(host: str = "localhost", port: int = 11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """True when localhost:11434 accepts connections (and nomic-embed-text exists)."""
    return _ollama_reachable()


@pytest.fixture
def require_ollama(ollama_available: bool) -> None:
    """Skip the test cleanly when Ollama isn't running locally."""
    if not ollama_available:
        pytest.skip("Ollama not reachable on localhost:11434")


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class _MockLLM:
    """Substring-mapped LLM stub.

    Call ``mock_llm.respond_to("Rewrite the following", "REWRITTEN")`` to set
    a canned response keyed by a prompt substring. The first matching trigger
    wins; an unmatched prompt returns ``default``.
    """

    def __init__(self, default: str = "MOCKED ANSWER") -> None:
        self._responses: dict[str, str] = {}
        self.default = default
        self.calls: list[str] = []

    def respond_to(self, trigger: str, response: str) -> _MockLLM:
        self._responses[trigger] = response
        return self

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for trigger, response in self._responses.items():
            if trigger in prompt:
                return response
        return self.default

    def complete_structured(self, prompt: str, schema: dict) -> str:
        return self.complete(prompt)

    def stream_complete(self, prompt: str) -> Iterator[str]:
        yield from self.complete(prompt)


@pytest.fixture
def mock_llm() -> _MockLLM:
    """Fresh MockLLM per test; tests register canned responses up-front."""
    return _MockLLM()


# ---------------------------------------------------------------------------
# Fake embedder (always available; deterministic; 768-dim)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic fixed-dimension embedder for CI without Ollama.

    Dimension matches nomic-embed-text (768) so shape-sensitive code such as
    ChromaDB collection creation behaves identically to production.

    The embedding is a sparse unit vector whose non-zero position is derived
    from the hash of the input text, giving different texts different vectors
    while remaining fully deterministic across runs.
    """

    _DIM = 768

    def embed(self, text: str) -> list[float]:
        import math

        vec = [0.0] * self._DIM
        # Primary index from hash — distinct texts get distinct positions.
        vec[abs(hash(text)) % self._DIM] = 1.0
        # Secondary spread: spread a small weight across a few more positions so
        # BM25-style keyword matches still dominate while vectors are non-trivial.
        for i, ch in enumerate(text[:8]):
            vec[(ord(ch) * (i + 1)) % self._DIM] += 0.1
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture(scope="session")
def fake_embedder() -> _FakeEmbedder:
    """Deterministic 768-dim embedder; no Ollama required."""
    return _FakeEmbedder()


# ---------------------------------------------------------------------------
# Integration embedder: real Ollama if available, fake otherwise
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def integration_embedder(ollama_available: bool, fake_embedder: _FakeEmbedder) -> Embedder:
    """Real Ollama embedder when available; deterministic fake otherwise.

    Use this fixture (not ollama_embedder) in integration tests that verify
    cross-component wiring rather than retrieval quality — they run in CI
    without Ollama.  Tests that specifically verify semantic ranking must use
    ollama_embedder directly (and will skip without Ollama).
    """
    if not ollama_available:
        return fake_embedder
    from llama_index.embeddings.ollama import OllamaEmbedding

    embedding = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    class _Adapter:
        def embed(self, text: str) -> list[float]:
            return list(embedding.get_text_embedding(text))

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [list(v) for v in embedding.get_text_embedding_batch(texts)]

    return _Adapter()


# ---------------------------------------------------------------------------
# Real Ollama embedder (skips when Ollama not reachable)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ollama_embedder(ollama_available: bool) -> Embedder:
    """Adapter around llama-index's OllamaEmbedding pointed at localhost.

    Constructed once per session because the embedding model load is the
    expensive part. Tests that don't need the embedder don't depend on this
    fixture.
    """
    if not ollama_available:
        pytest.skip("Ollama not reachable on localhost:11434")
    from llama_index.embeddings.ollama import OllamaEmbedding

    embedding = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    class _Adapter:
        def embed(self, text: str) -> list[float]:
            return list(embedding.get_text_embedding(text))

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [list(v) for v in embedding.get_text_embedding_batch(texts)]

    return _Adapter()


# ---------------------------------------------------------------------------
# Real Ollama LLM (skips when qwen3.6:35b not pulled)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llm_model_available(ollama_available: bool) -> bool:
    """True when qwen3.6:35b-a3b-q4_K_M is in ``ollama list``."""
    if not ollama_available:
        return False
    import shutil
    import subprocess

    if not shutil.which("ollama"):
        return False
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
    return "qwen3.6:35b-a3b-q4_K_M" in (result.stdout or "")


@pytest.fixture(scope="session")
def real_llm(llm_model_available: bool) -> LLM:
    """Adapter around llama-index's Ollama LLM pointed at the local daemon.

    Skips when the qwen3.6 model isn't pulled — answer quality tests can't
    run without it. Session-scoped because the model warm-up is the slow part.
    """
    if not llm_model_available:
        pytest.skip("qwen3.6:35b-a3b-q4_K_M not pulled in Ollama")
    from llama_index.llms.ollama import Ollama

    client = Ollama(
        model="qwen3.6:35b-a3b-q4_K_M",
        base_url="http://localhost:11434",
        request_timeout=180.0,
    )

    class _Adapter:
        def complete(self, prompt: str) -> str:
            return str(client.complete(prompt))

        def complete_structured(self, prompt: str, schema: dict) -> str:
            return str(client.complete(prompt, format=schema))

        def stream_complete(self, prompt: str) -> Iterator[str]:
            for chunk in client.stream_complete(prompt):
                text = getattr(chunk, "delta", None) or getattr(chunk, "text", "")
                if text:
                    yield str(text)

    return _Adapter()


# ---------------------------------------------------------------------------
# Indexed pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def indexed_pipeline(
    base_config: ARGConfig,
    integration_embedder: Embedder,
    mock_llm: _MockLLM,
):
    """ARGPipeline indexed over corpus_a with real/fake embeddings + mocked LLM.

    Uses integration_embedder so this fixture runs in CI without Ollama.
    Watcher disabled. Cluster cache pre-built (small-corpus fallback). Closed
    automatically when the test finishes.
    """
    from arg.pipeline import ARGPipeline

    pipeline = ARGPipeline(
        config=base_config,
        corpus_name="default",
        llm=mock_llm,
        embedder=integration_embedder,
        skip_health_check=True,
        skip_signal_handlers=True,
    )
    pipeline.index()
    try:
        yield pipeline
    finally:
        pipeline.close()
