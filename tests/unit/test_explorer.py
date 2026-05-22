"""CorpusExplorer tests — Section 10 ``test_explorer.py`` test points."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from arg.config import ARGConfig
from arg.crawler.extractors import Document
from arg.dci import CorpusAnalyst, CorpusExplorer
from arg.graph import KnowledgeGraph
from arg.indexer import Indexer
from arg.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Fakes (same shape as other Section-9/10 tests)
# ---------------------------------------------------------------------------


_VEC_DIM = 32


class _ClusterEmbedder:
    """Embedder with controllable cluster signal.

    Each document's content can carry a ``CLUSTER_<N>`` tag; documents with
    the same tag get a vector aligned on the same coordinate of a 32-dim
    space. The result is that k-means returns predictable clusters.
    """

    _BASE = ord("0")

    def embed(self, text: str) -> list[float]:
        import math
        import re

        vec = [0.001] * _VEC_DIM
        for m in re.finditer(r"CLUSTER_(\d)", text):
            idx = (ord(m.group(1)) - self._BASE) % _VEC_DIM
            vec[idx] += 1.0
        # Subtle per-doc variation so vectors aren't all identical
        # within a cluster — otherwise k-means produces NaN centroids.
        vec[abs(hash(text)) % _VEC_DIM] += 0.05
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _ScriptedLLM:
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "Cluster label",
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

    def complete_structured(self, prompt: str, schema: dict) -> str:
        return self.complete(prompt)


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
        min_cluster_docs=6,  # tests can stay small but still exercise the threshold
        n_clusters=3,
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
    links_to: list[str] | None = None,
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
            "links_to": list(links_to or []),
            "code_blocks": [],
        },
    )


def _build_explorer(
    config: ARGConfig,
    kg: KnowledgeGraph,
    documents: list[Document],
    *,
    llm: _ScriptedLLM | None = None,
) -> tuple[CorpusExplorer, _ScriptedLLM, Indexer]:
    embedder = _ClusterEmbedder()
    indexer = Indexer(config=config, knowledge_graph=kg, embedder=embedder)
    indexer.index(documents)
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
    explorer = CorpusExplorer(
        config=config,
        knowledge_graph=kg,
        analyst=analyst,
        llm=real_llm,
        chroma_documents_collection=indexer._docs_coll,
    )
    return explorer, real_llm, indexer


def _small_corpus(config: ARGConfig) -> list[Document]:
    """Three docs — below the default min_cluster_docs threshold."""
    return [
        _make_doc(
            config.docs_root,
            f"doc{i}.html",
            f"##H1## Doc {i}\nBody for doc {i} with CLUSTER_{i % 2} tag.",
            title=f"Doc {i}",
        )
        for i in range(3)
    ]


def _clustered_corpus(config: ARGConfig) -> list[Document]:
    """Eight docs split evenly across three CLUSTER_* tags so k-means produces
    meaningful groups."""
    docs: list[Document] = []
    for i in range(8):
        cluster_tag = i % 3
        docs.append(
            _make_doc(
                config.docs_root,
                f"doc{i}.html",
                (f"##H1## Doc {i}\nTopic body content CLUSTER_{cluster_tag} for doc {i}."),
                title=f"Doc {i}",
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Trivial passthrough surfaces
# ---------------------------------------------------------------------------


def test_list_all_documents_returns_kg_listing(config, kg):
    explorer, _, _ = _build_explorer(config, kg, _small_corpus(config))
    listed = explorer.list_all_documents()
    titles = sorted(d["title"] for d in listed)
    assert titles == ["Doc 0", "Doc 1", "Doc 2"]


def test_get_reverse_links(config, kg):
    docs = _small_corpus(config)
    docs[0].metadata["links_to"] = [str(docs[1].path.resolve())]
    explorer, _, _ = _build_explorer(config, kg, docs)
    reverse = explorer.get_reverse_links(str(docs[1].path.resolve()))
    src_ids = [r["doc_id"] for r in reverse]
    assert src_ids == [str(docs[0].path.resolve())]


def test_get_graph_json(config, kg):
    docs = _small_corpus(config)
    docs[0].metadata["links_to"] = [str(docs[1].path.resolve())]
    explorer, _, _ = _build_explorer(config, kg, docs)
    graph = explorer.get_graph_json()
    assert set(graph.keys()) == {"nodes", "edges"}
    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 1


# ---------------------------------------------------------------------------
# Topic clustering
# ---------------------------------------------------------------------------


def test_clustering_small_corpus_fallback(config, kg):
    """corpus < min_cluster_docs → single 'All documents' cluster, no LLM call."""
    llm = _ScriptedLLM()
    explorer, recorded_llm, _ = _build_explorer(config, kg, _small_corpus(config), llm=llm)
    clusters = explorer.get_topic_clusters()
    assert clusters == [
        {"label": "All documents", "doc_ids": clusters[0]["doc_ids"]},
    ]
    assert sorted(clusters[0]["doc_ids"]) == sorted(
        d["doc_id"] for d in explorer.list_all_documents()
    )
    # No LLM call for labelling — the fallback skips k-means entirely.
    assert recorded_llm.calls == []


def test_clustering_caches_result(config, kg):
    llm = _ScriptedLLM()
    explorer, recorded_llm, _ = _build_explorer(config, kg, _small_corpus(config), llm=llm)
    explorer.get_topic_clusters()
    cache_path = config.cluster_cache_path("default")
    assert cache_path.is_file()
    # Second call reads from cache; doesn't recompute.
    calls_before = len(recorded_llm.calls)
    explorer.get_topic_clusters()
    assert len(recorded_llm.calls) == calls_before


def test_clustering_invalidate_cache_deletes_file(config, kg):
    explorer, _, _ = _build_explorer(config, kg, _small_corpus(config))
    explorer.get_topic_clusters()
    cache_path = config.cluster_cache_path("default")
    assert cache_path.is_file()
    explorer.invalidate_cluster_cache()
    assert not cache_path.is_file()


def test_clustering_normal_path_runs_kmeans(config, kg):
    """corpus >= min_cluster_docs → k-means + LLM labels per cluster."""
    llm = _ScriptedLLM(responses={"Given the following document titles": "Topic Label X"})
    explorer, recorded_llm, _ = _build_explorer(config, kg, _clustered_corpus(config), llm=llm)
    clusters = explorer.get_topic_clusters()
    # n_clusters = min(config.n_clusters, n_docs) — here n_clusters=3, 8 docs.
    assert len(clusters) == 3
    for c in clusters:
        assert c["label"] == "Topic Label X"
        assert c["doc_ids"]
    # One LLM call per cluster for label generation.
    label_calls = [c for c in recorded_llm.calls if "Given the following" in c]
    assert len(label_calls) == 3


def test_clustering_cache_shape_on_disk(config, kg):
    llm = _ScriptedLLM(responses={"Given the following document titles": "Topic"})
    explorer, _, _ = _build_explorer(config, kg, _clustered_corpus(config), llm=llm)
    explorer.get_topic_clusters()
    with config.cluster_cache_path("default").open() as fh:
        data = json.load(fh)
    assert set(data.keys()) == {"doc_to_cluster", "cluster_members", "labels"}
    for doc_id, cluster_id in data["doc_to_cluster"].items():
        assert doc_id in data["cluster_members"][cluster_id]


# ---------------------------------------------------------------------------
# Corpus search
# ---------------------------------------------------------------------------


def test_corpus_search_delegates_to_analyst(config, kg):
    explorer, _, _ = _build_explorer(config, kg, _small_corpus(config))
    hits = explorer.corpus_search("CLUSTER_0", top_k=5)
    assert hits
    for h in hits:
        assert "doc_id" in h
        assert "similarity_score" in h


def test_corpus_search_file_type_filter(config, kg):
    docs = [
        _make_doc(
            config.docs_root,
            "html_doc.html",
            "##H1## H\nbody for the HTML doc with CLUSTER_0 tag.",
            title="HtmlDoc",
        ),
        _make_doc(
            config.docs_root,
            "pdf_doc.pdf",
            "##H1## P\nbody for the PDF doc with CLUSTER_0 tag.",
            title="PdfDoc",
            file_type="pdf",
        ),
    ]
    explorer, _, _ = _build_explorer(config, kg, docs)
    pdf_only = explorer.corpus_search("CLUSTER_0", file_type="pdf", top_k=5)
    assert pdf_only
    assert all(h["doc_id"].endswith(".pdf") for h in pdf_only)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def test_most_linked_docs_passthrough(config, kg):
    docs = _small_corpus(config)
    # hub-and-spoke: doc1 + doc2 → doc0
    docs[1].metadata["links_to"] = [str(docs[0].path.resolve())]
    docs[2].metadata["links_to"] = [str(docs[0].path.resolve())]
    explorer, _, _ = _build_explorer(config, kg, docs)
    ranked = explorer.most_linked_docs(top_n=3)
    assert ranked[0]["doc_id"] == str(docs[0].path.resolve())
    assert ranked[0]["inbound"] == 2


def test_orphaned_docs_passthrough(config, kg):
    docs = _small_corpus(config)
    docs[0].metadata["links_to"] = [str(docs[1].path.resolve())]
    explorer, _, _ = _build_explorer(config, kg, docs)
    # doc2 receives no inbound edges; doc0 also has none. doc1 has one.
    orphans = explorer.orphaned_docs()
    assert str(docs[1].path.resolve()) not in orphans
    assert str(docs[0].path.resolve()) in orphans
    assert str(docs[2].path.resolve()) in orphans


def test_docs_by_chunk_count_pagination(config, kg):
    docs = _small_corpus(config)
    explorer, _, _ = _build_explorer(config, kg, docs)
    page1 = explorer.docs_by_chunk_count(page=1, page_size=2)
    page2 = explorer.docs_by_chunk_count(page=2, page_size=2)
    assert page1["total"] == 3
    assert len(page1["items"]) == 2
    assert len(page2["items"]) == 1
    assert page1["total_pages"] == 2
    # Items in page1 must not overlap with items in page2.
    ids1 = {d["doc_id"] for d in page1["items"]}
    ids2 = {d["doc_id"] for d in page2["items"]}
    assert ids1.isdisjoint(ids2)


def test_docs_by_chunk_count_order(config, kg):
    explorer, _, _ = _build_explorer(config, kg, _small_corpus(config))
    asc = explorer.docs_by_chunk_count(page=1, page_size=10, order="asc")
    desc = explorer.docs_by_chunk_count(page=1, page_size=10, order="desc")
    asc_counts = [d["chunk_count"] for d in asc["items"]]
    desc_counts = [d["chunk_count"] for d in desc["items"]]
    assert asc_counts == sorted(asc_counts)
    assert desc_counts == sorted(desc_counts, reverse=True)
