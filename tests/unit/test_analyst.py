"""CorpusAnalyst tests — Section 9 ``test_analyst.py`` test points."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from arg.config import ARGConfig
from arg.crawler.extractors import Document
from arg.dci import CorpusAnalyst
from arg.graph import KnowledgeGraph
from arg.indexer import Indexer
from arg.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Fakes — same shape as the generator tests
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
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "FAKE LLM OUTPUT",
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for trigger, response in self.responses.items():
            if trigger in prompt:
                return response
        return self.default

    def stream_complete(self, prompt: str) -> Iterator[str]:
        yield self.complete(prompt)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> ARGConfig:
    docs = tmp_path / "docs"
    docs.mkdir()
    return ARGConfig(
        docs_root=docs,
        db_path=tmp_path / "arg_db",
        top_k_vector=5,
        graph_hop_depth=0,
        enrich_min_score=0.0,
    )


@pytest.fixture
def kg(config: ARGConfig):
    g = KnowledgeGraph(config.kuzu_path("default"))
    yield g
    g.close()


def _make_doc(
    docs_dir: Path,
    name: str,
    content: str,
    *,
    title: str | None = None,
    file_type: str = "html",
) -> Document:
    p = docs_dir / name
    p.touch()
    return Document(
        path=p,
        content=content,
        metadata={
            "title": title or name,
            "page_description": f"Description for {name}",
            "file_type": file_type,
            "links_to": [],
            "code_blocks": [],
        },
    )


def _build_analyst(
    config: ARGConfig,
    kg: KnowledgeGraph,
    *,
    llm: _ScriptedLLM | None = None,
    documents: list[Document] | None = None,
) -> tuple[CorpusAnalyst, _ScriptedLLM, Indexer]:
    embedder = _TagEmbedder()
    indexer = Indexer(config=config, knowledge_graph=kg, embedder=embedder)
    docs = (
        documents
        if documents is not None
        else [
            _make_doc(
                config.docs_root,
                "alpha.html",
                "##H1## Alpha\nAlpha body QUERY_A discusses authentication tokens.",
                title="Alpha",
            ),
            _make_doc(
                config.docs_root,
                "beta.html",
                "##H1## Beta\nBeta body QUERY_B about database migrations and schema design.",
                title="Beta",
            ),
        ]
    )
    indexer.index(docs)
    retriever = HybridRetriever(
        config=config,
        knowledge_graph=kg,
        embedder=embedder,
        chroma_documents_collection=indexer._docs_coll,
        chroma_chunks_collection=indexer._chunks_coll,
        bm25_index_path=config.bm25_index_path("default"),
        cluster_cache_path=config.cluster_cache_path("default"),
    )
    real_llm = llm or _ScriptedLLM()
    analyst = CorpusAnalyst(
        config=config,
        llm=real_llm,
        retriever=retriever,
        knowledge_graph=kg,
        chroma_chunks_collection=indexer._chunks_coll,
        chroma_documents_collection=indexer._docs_coll,
    )
    return analyst, real_llm, indexer


def _doc_id(config: ARGConfig, name: str) -> str:
    return str((config.docs_root / name).resolve())


# ---------------------------------------------------------------------------
# summarize_document
# ---------------------------------------------------------------------------


def test_summarize_document_returns_non_empty_string(config, kg):
    llm = _ScriptedLLM(responses={"Summarise the following document": "A short alpha summary."})
    analyst, _, _ = _build_analyst(config, kg, llm=llm)
    summary = analyst.summarize_document(_doc_id(config, "alpha.html"))
    assert summary == "A short alpha summary."


def test_summarize_document_calls_llm_once_for_short_doc(config, kg):
    llm = _ScriptedLLM(responses={"Summarise the following document": "short summary"})
    analyst, recorded_llm, _ = _build_analyst(config, kg, llm=llm)
    analyst.summarize_document(_doc_id(config, "alpha.html"))
    summary_calls = [c for c in recorded_llm.calls if "Summarise the following" in c]
    assert len(summary_calls) == 1


def test_summarize_document_map_reduce_for_oversized_doc(config, kg, monkeypatch):
    """Force a tiny budget so even a small doc triggers map-reduce."""
    import arg.dci.analyst as analyst_mod

    monkeypatch.setattr(analyst_mod, "_BATCH_TOKEN_BUDGET", 5)

    llm = _ScriptedLLM(
        responses={
            "Combine the following per-section summaries": "FINAL REDUCED",
            "Summarise the following document": "PARTIAL SUMMARY",
        }
    )
    analyst, recorded_llm, _ = _build_analyst(config, kg, llm=llm)
    summary = analyst.summarize_document(_doc_id(config, "alpha.html"))
    assert summary == "FINAL REDUCED"
    # Map-reduce: at least one partial + one reduce call.
    summary_calls = [c for c in recorded_llm.calls if "Summarise the following" in c]
    reduce_calls = [c for c in recorded_llm.calls if "Combine the following" in c]
    assert len(summary_calls) >= 1
    assert len(reduce_calls) == 1


def test_summary_cache_short_circuits_second_call(config, kg, tmp_path):
    """With summary_cache=True, a second call reads from disk — no LLM call."""
    config.summary_cache = True
    llm = _ScriptedLLM(responses={"Summarise the following document": "FIRST CALL SUMMARY"})
    analyst, recorded_llm, _ = _build_analyst(config, kg, llm=llm)
    doc_id = _doc_id(config, "alpha.html")

    first = analyst.summarize_document(doc_id)
    assert first == "FIRST CALL SUMMARY"
    calls_after_first = len(recorded_llm.calls)

    second = analyst.summarize_document(doc_id)
    assert second == "FIRST CALL SUMMARY"
    # No new LLM calls were issued on the second invocation.
    assert len(recorded_llm.calls) == calls_after_first


# ---------------------------------------------------------------------------
# extract_key_points
# ---------------------------------------------------------------------------


def test_extract_key_points_returns_list_of_strings(config, kg):
    llm = _ScriptedLLM(responses={"List the": "First key point\nSecond key point\nThird key point"})
    analyst, _, _ = _build_analyst(config, kg, llm=llm)
    points = analyst.extract_key_points(_doc_id(config, "alpha.html"), max_points=5)
    assert points == ["First key point", "Second key point", "Third key point"]


def test_extract_key_points_respects_max_points(config, kg):
    llm = _ScriptedLLM(responses={"List the": "\n".join(f"Point {i}" for i in range(10))})
    analyst, _, _ = _build_analyst(config, kg, llm=llm)
    points = analyst.extract_key_points(_doc_id(config, "alpha.html"), max_points=3)
    assert len(points) == 3


def test_extract_key_points_strips_bullet_prefixes(config, kg):
    llm = _ScriptedLLM(responses={"List the": "- First\n* Second\n3. Third\n  • Fourth"})
    analyst, _, _ = _build_analyst(config, kg, llm=llm)
    points = analyst.extract_key_points(_doc_id(config, "alpha.html"), max_points=10)
    assert points == ["First", "Second", "Third", "Fourth"]


# ---------------------------------------------------------------------------
# compare_documents
# ---------------------------------------------------------------------------


def test_compare_documents_returns_non_empty_string(config, kg):
    llm = _ScriptedLLM(responses={"Compare the following two documents": "COMPARISON RESULT"})
    analyst, _, _ = _build_analyst(config, kg, llm=llm)
    out = analyst.compare_documents(
        _doc_id(config, "alpha.html"),
        _doc_id(config, "beta.html"),
    )
    assert out == "COMPARISON RESULT"


def test_compare_documents_passes_both_documents_to_llm(config, kg):
    llm = _ScriptedLLM(responses={"Compare the following two documents": "RESULT"})
    analyst, recorded_llm, _ = _build_analyst(config, kg, llm=llm)
    analyst.compare_documents(
        _doc_id(config, "alpha.html"),
        _doc_id(config, "beta.html"),
    )
    compare_call = next(c for c in recorded_llm.calls if "Compare the following" in c)
    # Both source documents must appear inside the prompt body.
    assert "Alpha body" in compare_call
    assert "Beta body" in compare_call


# ---------------------------------------------------------------------------
# scoped_search
# ---------------------------------------------------------------------------


def test_scoped_search_returns_only_specified_doc_chunks(config, kg):
    analyst, _, _ = _build_analyst(config, kg)
    alpha_id = _doc_id(config, "alpha.html")
    results = analyst.scoped_search("QUERY_B database", alpha_id, top_k=5)
    assert results, "scoped search should still return chunks for the scoped doc"
    for r in results:
        assert r.node.metadata.get("doc_id") == alpha_id


def test_scoped_search_does_not_leak_chunks_from_other_docs(config, kg):
    analyst, _, _ = _build_analyst(config, kg)
    alpha_id = _doc_id(config, "alpha.html")
    beta_id = _doc_id(config, "beta.html")
    results = analyst.scoped_search("QUERY_A authentication", alpha_id, top_k=10)
    for r in results:
        assert r.node.metadata.get("doc_id") != beta_id


def test_scoped_search_respects_top_k(config, kg):
    analyst, _, _ = _build_analyst(config, kg)
    alpha_id = _doc_id(config, "alpha.html")
    results = analyst.scoped_search("QUERY_A authentication", alpha_id, top_k=1)
    assert len(results) <= 1


# ---------------------------------------------------------------------------
# get_chunks
# ---------------------------------------------------------------------------


def test_get_chunks_count_matches_kuzu_chunk_count(config, kg):
    analyst, _, _ = _build_analyst(config, kg)
    alpha_id = _doc_id(config, "alpha.html")
    chunks = analyst.get_chunks(alpha_id)
    kg_count = kg.get_doc_metadata(alpha_id)["chunk_count"]
    assert len(chunks) == kg_count


def test_get_chunks_exposes_spec_fields(config, kg):
    analyst, _, _ = _build_analyst(config, kg)
    alpha_id = _doc_id(config, "alpha.html")
    chunks = analyst.get_chunks(alpha_id)
    assert chunks
    for c in chunks:
        for key in ("chunk_id", "position", "text", "token_count", "heading_path"):
            assert key in c, f"missing field {key} on get_chunks row"


def test_get_chunks_for_unknown_doc_returns_empty(config, kg):
    analyst, _, _ = _build_analyst(config, kg)
    assert analyst.get_chunks("/no/such/doc.html") == []


# ---------------------------------------------------------------------------
# find_document
# ---------------------------------------------------------------------------


def test_find_document_returns_ranked_doc_ids(config, kg):
    analyst, _, _ = _build_analyst(config, kg)
    ranked = analyst.find_document("QUERY_A authentication", top_k=2)
    assert ranked
    alpha_id = _doc_id(config, "alpha.html")
    assert ranked[0]["doc_id"] == alpha_id
    scores = [r["similarity_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_find_document_file_type_filter(config, kg):
    analyst, _, _ = _build_analyst(
        config,
        kg,
        documents=[
            _make_doc(
                config.docs_root,
                "html_doc.html",
                "##H1## HTML\nHTML body QUERY_A here.",
                title="HtmlDoc",
            ),
            _make_doc(
                config.docs_root,
                "pdf_doc.pdf",
                "##H1## PDF\nPDF body QUERY_A here.",
                title="PdfDoc",
                file_type="pdf",
            ),
        ],
    )
    pdf_only = analyst.find_document("QUERY_A", top_k=5, file_type="pdf")
    assert pdf_only
    pdf_id = _doc_id(config, "pdf_doc.pdf")
    assert all(r["doc_id"] == pdf_id for r in pdf_only)
