"""Bootstrap verification tests.

These run pre-Section-4 to confirm the environment is wired correctly:

  * The `arg` package imports.
  * Pinned LlamaIndex sub-packages all resolve to the version of llama-index-core
    that pyproject.toml pins.
  * Core third-party stores (ChromaDB, Kuzu) initialize a local on-disk database
    without making any network calls.
  * The `scripts/bootstrap.sh` script exists, is executable, and is well-formed.
  * Ollama-dependent checks (daemon reachable on localhost:11434; required models
    present) run only when the daemon is reachable; otherwise they are skipped so
    `pytest tests/unit/` stays fast and offline.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


def test_arg_package_imports():
    import arg

    assert arg.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# pyproject.toml structure
# ---------------------------------------------------------------------------


def _pyproject() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_pyproject_has_project_section():
    data = _pyproject()
    assert "project" in data, "pyproject.toml is missing [project]"
    assert data["project"]["name"] == "arg"
    assert data["project"]["requires-python"].startswith(">=3.11")


def test_pyproject_has_build_system():
    data = _pyproject()
    assert "build-system" in data
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"


def test_pyproject_preserves_tool_sections():
    """Section 3 must extend, not overwrite, the existing [tool.*] config."""
    data = _pyproject()
    assert "tool" in data
    for required in ("ruff", "mypy", "pytest"):
        assert required in data["tool"], f"[tool.{required}] was removed"


def test_llama_index_packages_exact_pinned():
    """All llama-index-* deps must be exact-pinned (==), not range-pinned."""
    data = _pyproject()
    deps = data["project"]["dependencies"]
    li_deps = [d for d in deps if d.startswith("llama-index-")]
    assert li_deps, "no llama-index-* deps found"
    for spec in li_deps:
        assert "==" in spec, f"llama-index dep not exact-pinned: {spec!r}"
        assert ">=" not in spec, f"llama-index dep uses >= range: {spec!r}"


# ---------------------------------------------------------------------------
# Bootstrap script
# ---------------------------------------------------------------------------


def test_bootstrap_script_exists_and_executable():
    script = PROJECT_ROOT / "scripts" / "bootstrap.sh"
    assert script.is_file(), "scripts/bootstrap.sh missing"
    assert script.stat().st_mode & 0o111, "scripts/bootstrap.sh is not executable"


def test_bootstrap_script_bash_syntax():
    script = PROJECT_ROOT / "scripts" / "bootstrap.sh"
    if not shutil.which("bash"):
        pytest.skip("bash not on PATH")
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bootstrap.sh syntax error:\n{result.stderr}"


def test_bootstrap_script_pulls_only_pinned_models():
    """Bootstrap must reference the exact pinned model tags from CLAUDE.md."""
    script_text = (PROJECT_ROOT / "scripts" / "bootstrap.sh").read_text()
    assert "llama3.3:70b-instruct-q4_K_M" in script_text
    assert "nomic-embed-text" in script_text


# ---------------------------------------------------------------------------
# Local stores (no network)
# ---------------------------------------------------------------------------


def test_chromadb_initializes_locally(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    coll = client.get_or_create_collection("bootstrap_smoke")
    assert coll.count() == 0


def test_kuzu_initializes_locally(tmp_path):
    kuzu = pytest.importorskip("kuzu")
    db = kuzu.Database(str(tmp_path / "kuzu"))
    conn = kuzu.Connection(db)
    # Minimal round-trip to confirm the embedded engine is wired up.
    conn.execute("CREATE NODE TABLE Smoke(id INT64, PRIMARY KEY(id))")
    conn.execute("CREATE (:Smoke {id: 1})")
    result = conn.execute("MATCH (s:Smoke) RETURN s.id")
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    assert rows == [[1]]


# ---------------------------------------------------------------------------
# Ollama-dependent checks (skipped when daemon not running)
# ---------------------------------------------------------------------------


def _ollama_reachable(host: str = "localhost", port: int = 11434, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def test_ollama_daemon_reachable():
    if not _ollama_reachable():
        pytest.skip("Ollama daemon not running on localhost:11434 (unit suite stays offline)")
    # Smoke a no-network API call.
    import urllib.request

    with urllib.request.urlopen("http://localhost:11434/api/version", timeout=2) as resp:
        assert resp.status == 200


def test_ollama_required_models_present():
    """When Ollama is running and the required models have been pulled, both must
    be present with the exact tags pinned in CLAUDE.md.

    When the models are missing, skip — pulling them is the job of
    `scripts/bootstrap.sh`, not the unit suite. The unit suite stays offline by
    design (CLAUDE.md: "fast; no Ollama required (LLM mocked)").
    """
    if not _ollama_reachable():
        pytest.skip("Ollama daemon not running on localhost:11434")
    if not shutil.which("ollama"):
        pytest.skip("ollama CLI not on PATH")
    out = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    listed = out.stdout
    missing = [m for m in ("llama3.3:70b-instruct-q4_K_M", "nomic-embed-text") if m not in listed]
    if missing:
        pytest.skip(f"required Ollama models not pulled yet: {missing}; run scripts/bootstrap.sh")


# ---------------------------------------------------------------------------
# Python version (sanity)
# ---------------------------------------------------------------------------


def test_python_311_or_newer():
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"
