"""FastAPI server tests — endpoint routing + response shape + 404 + 2-corpus isolation.

A real ARGPipeline (with a fake embedder + fake LLM, ``skip_health_check=True``)
is built once per test so the endpoints exercise real Chroma / Kuzu /
BM25 wiring.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arg.config import ARGConfig
from arg.pipeline import ARGPipeline
from arg.server import create_app

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


_VEC_DIM = 32


class _TagEmbedder:
    _BASE = ord("A")

    def embed(self, text: str) -> list[float]:
        import math
        import re

        vec = [0.001] * _VEC_DIM
        for m in re.finditer(r"QUERY_([A-Z])", text):
            idx = (ord(m.group(1)) - self._BASE) % _VEC_DIM
            vec[idx] += 1.0
        vec[abs(hash(text)) % _VEC_DIM] += 0.05
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _ScriptedLLM:
    def __init__(self, default: str = "MOCKED ANSWER") -> None:
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.default

    def complete_structured(self, prompt: str, schema: dict) -> str:
        return self.complete(prompt)

    def stream_complete(self, prompt: str) -> Iterator[str]:
        yield from self.complete(prompt)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, name: str = "default") -> ARGConfig:
    docs = tmp_path / "docs" / name
    docs.mkdir(parents=True)
    db_root = tmp_path / "arg_db"
    db_root.mkdir(exist_ok=True)  # parent must exist per ARGConfig validator
    return ARGConfig(
        docs_root=docs,
        db_path=db_root / name,
        top_k_vector=3,
        graph_hop_depth=0,
        enrich_min_score=0.0,
        watch_enabled=False,
    )


def _write_corpus(docs_dir: Path) -> None:
    (docs_dir / "index.html").write_text(
        "<html><head><title>Index | Site</title></head><body>"
        "<h1>Index</h1>"
        "<p>Welcome. QUERY_A authentication is documented here.</p>"
        '<a href="page.html">More</a>'
        "</body></html>",
        encoding="utf-8",
    )
    (docs_dir / "page.html").write_text(
        "<html><head><title>Page | Site</title></head><body>"
        "<h1>Page</h1>"
        "<p>This page covers QUERY_A authentication tokens.</p>"
        "</body></html>",
        encoding="utf-8",
    )


@pytest.fixture
def pipeline(tmp_path: Path):
    cfg = _config(tmp_path, "default")
    _write_corpus(cfg.docs_root)
    p = ARGPipeline(
        config=cfg,
        corpus_name="default",
        llm=_ScriptedLLM(),
        embedder=_TagEmbedder(),
        skip_health_check=True,
        skip_signal_handlers=True,
    )
    p.index()
    yield p
    p.close()


@pytest.fixture
def client(pipeline):
    app = create_app({"default": pipeline})
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["corpus_name"] == "default"
    assert body["doc_count"] >= 1
    assert body["chunk_count"] >= 1


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


def test_query_endpoint(client):
    r = client.post("/query", json={"question": "QUERY_A authentication"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert isinstance(body["sources"], list)
    assert body["sources"], "non-empty corpus should yield sources"
    assert "latency_ms" in body
    assert "enriched_doc_ids" in body
    assert "rewritten_query" in body
    assert "sub_queries" in body


def test_query_endpoint_requires_question(client):
    r = client.post("/query", json={})
    assert r.status_code == 400


def test_query_endpoint_stream_yields_event_stream(client):
    with client.stream("POST", "/query?stream=true", json={"question": "QUERY_A"}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = b"".join(r.iter_bytes())
    assert b"data: " in body


# ---------------------------------------------------------------------------
# 404 on unknown corpus
# ---------------------------------------------------------------------------


def test_unknown_corpus_returns_404(client):
    r = client.get("/health?corpus=nonexistent")
    assert r.status_code == 404


def test_unknown_corpus_query_returns_404(client):
    r = client.post(
        "/query?corpus=nonexistent",
        json={"question": "anything"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Corpus listing + mutation
# ---------------------------------------------------------------------------


def test_corpus_listing(client):
    r = client.get("/corpus")
    assert r.status_code == 200
    docs = r.json()
    assert isinstance(docs, list)
    assert len(docs) >= 2
    for d in docs:
        assert "doc_id" in d
        assert "title" in d


def test_corpus_remove_unknown_returns_404(client):
    r = client.delete("/corpus/" + "no-such-doc")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Graph + reverse links
# ---------------------------------------------------------------------------


def test_corpus_graph(client):
    r = client.get("/corpus/graph")
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body
    assert "edges" in body


def test_linked_by(client, pipeline):
    page_id = str((pipeline.config.docs_root / "page.html").resolve())
    r = client.get(f"/corpus/{page_id}/linked-by")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------


def test_topics_endpoint(client):
    r = client.get("/corpus/topics")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    for entry in body:
        assert "label" in entry
        assert "doc_ids" in entry


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_corpus_search(client):
    r = client.get("/corpus/search?query=QUERY_A&top_k=5")
    assert r.status_code == 200
    hits = r.json()
    assert hits
    for h in hits:
        assert "doc_id" in h
        assert "similarity_score" in h


# ---------------------------------------------------------------------------
# Per-document
# ---------------------------------------------------------------------------


def test_doc_summary(client, pipeline):
    page_id = str((pipeline.config.docs_root / "page.html").resolve())
    r = client.get(f"/corpus/{page_id}/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == page_id
    assert body["summary"]


def test_doc_chunks(client, pipeline):
    page_id = str((pipeline.config.docs_root / "page.html").resolve())
    r = client.get(f"/corpus/{page_id}/chunks")
    assert r.status_code == 200
    chunks = r.json()
    assert chunks
    for c in chunks:
        for key in ("chunk_id", "position", "text", "token_count", "heading_path"):
            assert key in c


def test_doc_detail(client, pipeline):
    page_id = str((pipeline.config.docs_root / "page.html").resolve())
    r = client.get(f"/corpus/{page_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == page_id
    assert "title" in body
    assert "key_points" in body


def test_scoped_search(client, pipeline):
    page_id = str((pipeline.config.docs_root / "page.html").resolve())
    r = client.get(f"/corpus/{page_id}/search?query=QUERY_A&top_k=3")
    assert r.status_code == 200
    hits = r.json()
    assert isinstance(hits, list)
    for h in hits:
        assert "chunk_id" in h
        assert "text" in h
        assert h["metadata"]["doc_id"] == page_id


def test_compare(client, pipeline):
    index_id = str((pipeline.config.docs_root / "index.html").resolve())
    page_id = str((pipeline.config.docs_root / "page.html").resolve())
    r = client.get(f"/corpus/compare?a={index_id}&b={page_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["comparison"]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_corpus_stats(client):
    r = client.get("/corpus/stats")
    assert r.status_code == 200
    body = r.json()
    for key in ("documents", "chunks", "link_edges", "most_linked", "orphaned", "by_size_preview"):
        assert key in body


def test_corpus_stats_by_size_paginated(client):
    r = client.get("/corpus/stats/by-size?page=1&page_size=2&order=desc")
    assert r.status_code == 200
    body = r.json()
    assert body["page"] == 1
    assert body["page_size"] == 2
    assert body["order"] == "desc"
    assert isinstance(body["items"], list)
    assert len(body["items"]) <= 2


# ---------------------------------------------------------------------------
# Two-corpus isolation
# ---------------------------------------------------------------------------


def test_two_corpus_isolation(tmp_path):
    a_cfg = _config(tmp_path, "alpha")
    _write_corpus(a_cfg.docs_root)
    b_cfg = _config(tmp_path, "beta")
    # Beta gets only one doc so its listing is distinguishable.
    (b_cfg.docs_root / "only.html").write_text(
        "<html><head><title>Only</title></head><body><h1>Only</h1>"
        "<p>Only doc in beta corpus.</p></body></html>",
        encoding="utf-8",
    )

    p_a = ARGPipeline(
        config=a_cfg,
        corpus_name="alpha",
        llm=_ScriptedLLM(),
        embedder=_TagEmbedder(),
        skip_health_check=True,
        skip_signal_handlers=True,
    )
    p_b = ARGPipeline(
        config=b_cfg,
        corpus_name="beta",
        llm=_ScriptedLLM(),
        embedder=_TagEmbedder(),
        skip_health_check=True,
        skip_signal_handlers=True,
    )
    p_a.index()
    p_b.index()
    try:
        app = create_app({"alpha": p_a, "beta": p_b})
        with TestClient(app) as client:
            alpha_docs = client.get("/corpus?corpus=alpha").json()
            beta_docs = client.get("/corpus?corpus=beta").json()
            assert len(alpha_docs) >= 2
            assert len(beta_docs) == 1
            # Verifies no cross-corpus bleed.
            alpha_ids = {d["doc_id"] for d in alpha_docs}
            beta_ids = {d["doc_id"] for d in beta_docs}
            assert alpha_ids.isdisjoint(beta_ids)
    finally:
        p_a.close()
        p_b.close()
