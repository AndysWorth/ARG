"""FastAPI server — local-only web UI for ARG.

Every endpoint accepts ``?corpus=<name>`` (default ``default``). Unknown
corpus names return 404; never 500.

Locality
--------
The server binds to ``config.server_host`` (validated localhost) on
``config.server_port`` by default. No outbound calls; all storage and the
LLM live in-process or on ``localhost:11434``.

Pipeline lifecycle
------------------
The server stores a ``{corpus_name: ARGPipeline}`` map. Tests build the map
manually with mocked pipelines; ``scripts/serve.py`` (or ``index_docs.py
serve``) builds it once at startup from real Ollama-backed pipelines.

Streaming
---------
``POST /query`` accepts ``?stream=true``; when set, the response is
plaintext SSE-style chunks (`data: <token>\\n\\n`). Non-streaming responses
are JSON. Sources + latency are returned only in the non-streaming path —
streaming callers re-fetch via a follow-up call if they need them.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from arg.pipeline import ARGPipeline

logger = logging.getLogger(__name__)


_STATIC_DIR = Path(__file__).parent / "static"


def create_app(pipelines: dict[str, ARGPipeline]) -> FastAPI:
    """Build a FastAPI app over the given ``{corpus_name: pipeline}`` map.

    Production constructs one app per server process. Tests construct one
    per test. The map is shared by reference; mutate it (add / remove
    pipelines) to reconfigure routing without restarting.
    """
    app = FastAPI(title="ARG — Archivist RAG Graph", version="0.1.0")

    def _pipeline(corpus: str) -> ARGPipeline:
        if corpus not in pipelines:
            raise HTTPException(status_code=404, detail=f"unknown corpus: {corpus}")
        return pipelines[corpus]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health")
    def health(corpus: str = Query("default")) -> dict[str, Any]:
        try:
            p = _pipeline(corpus)
        except HTTPException:
            raise
        stats = p.graph.stats()
        return {
            "status": "ok",
            "model": p.config.llm_model,
            "corpus_name": corpus,
            "doc_count": stats["documents"],
            "chunk_count": stats["chunks"],
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @app.post("/query")
    def query(
        payload: Annotated[dict[str, Any], Body(...)],
        corpus: str = Query("default"),
        stream: bool = Query(False),
    ) -> Any:
        p = _pipeline(corpus)
        question = str(payload.get("question") or payload.get("query") or "")
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        enrich = bool(payload.get("enrich", True))
        filters = payload.get("filters") or None

        if stream:

            async def stream_body() -> AsyncIterator[bytes]:
                for token in p.stream_query(question, enrich=enrich, filters=filters):
                    yield f"data: {token}\n\n".encode()

            return StreamingResponse(stream_body(), media_type="text/event-stream")

        result = p.query(question, enrich=enrich, filters=filters)
        return {
            "answer": result.answer,
            "sources": [
                {
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "chunk_id": s.chunk_id,
                    "heading_path": s.heading_path,
                }
                for s in result.sources
            ],
            "latency_ms": result.latency_ms,
            "enriched_doc_ids": result.enriched_doc_ids,
            "rewritten_query": result.rewritten_query,
            "sub_queries": result.sub_queries,
        }

    # ------------------------------------------------------------------
    # Corpus listing + mutation
    # ------------------------------------------------------------------

    @app.get("/corpus")
    def list_corpus(corpus: str = Query("default")) -> list[dict[str, Any]]:
        return _pipeline(corpus).explorer.list_all_documents()

    @app.post("/corpus/add")
    def add_document(
        payload: Annotated[dict[str, Any], Body(...)],
        corpus: str = Query("default"),
    ) -> dict[str, Any]:
        p = _pipeline(corpus)
        raw_path = payload.get("path")
        if not raw_path:
            raise HTTPException(status_code=400, detail="path is required")
        path = Path(str(raw_path))
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no such file: {path}")
        n = p.add_document(path)
        return {"path": str(path.resolve()), "chunks_written": n}

    @app.delete("/corpus/{doc_id:path}")
    def remove_document(doc_id: str, corpus: str = Query("default")) -> dict[str, Any]:
        p = _pipeline(corpus)
        if p.graph.get_doc_metadata(doc_id) == {}:
            raise HTTPException(status_code=404, detail=f"unknown doc_id: {doc_id}")
        p.remove_document(doc_id)
        return {"doc_id": doc_id, "removed": True}

    # ------------------------------------------------------------------
    # Graph + reverse links
    # ------------------------------------------------------------------

    @app.get("/corpus/graph")
    def graph_json(corpus: str = Query("default")) -> dict[str, Any]:
        return _pipeline(corpus).explorer.get_graph_json()

    @app.get("/corpus/{doc_id:path}/linked-by")
    def linked_by(doc_id: str, corpus: str = Query("default")) -> list[dict[str, Any]]:
        return _pipeline(corpus).explorer.get_reverse_links(doc_id)

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    @app.get("/corpus/topics")
    def topics(corpus: str = Query("default")) -> list[dict[str, Any]]:
        return _pipeline(corpus).get_topic_clusters()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @app.get("/corpus/search")
    def corpus_search(
        corpus: str = Query("default"),
        query: str = Query(..., min_length=1),
        file_type: str | None = Query(None),
        top_k: int = Query(10, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        return _pipeline(corpus).corpus_search(query, file_type=file_type, top_k=top_k)

    # ------------------------------------------------------------------
    # Per-document
    # ------------------------------------------------------------------

    @app.get("/corpus/{doc_id:path}/summary")
    def doc_summary(doc_id: str, corpus: str = Query("default")) -> dict[str, Any]:
        p = _pipeline(corpus)
        if p.graph.get_doc_metadata(doc_id) == {}:
            raise HTTPException(status_code=404, detail=f"unknown doc_id: {doc_id}")
        return {"doc_id": doc_id, "summary": p.summarize_document(doc_id)}

    @app.get("/corpus/{doc_id:path}/chunks")
    def doc_chunks(doc_id: str, corpus: str = Query("default")) -> list[dict[str, Any]]:
        p = _pipeline(corpus)
        if p.graph.get_doc_metadata(doc_id) == {}:
            raise HTTPException(status_code=404, detail=f"unknown doc_id: {doc_id}")
        return p.analyst.get_chunks(doc_id)

    @app.get("/corpus/{doc_id:path}/search")
    def scoped_search(
        doc_id: str,
        corpus: str = Query("default"),
        query: str = Query(..., min_length=1),
        top_k: int = Query(5, ge=1, le=50),
    ) -> list[dict[str, Any]]:
        p = _pipeline(corpus)
        if p.graph.get_doc_metadata(doc_id) == {}:
            raise HTTPException(status_code=404, detail=f"unknown doc_id: {doc_id}")
        results = p.analyst.scoped_search(query, doc_id, top_k=top_k)
        return [
            {
                "chunk_id": r.node.id_,
                "text": r.node.get_content(),
                "score": r.score,
                "metadata": dict(r.node.metadata or {}),
            }
            for r in results
        ]

    @app.get("/corpus/compare")
    def compare(
        corpus: str = Query("default"),
        a: str = Query(..., alias="a", min_length=1),
        b: str = Query(..., alias="b", min_length=1),
    ) -> dict[str, Any]:
        p = _pipeline(corpus)
        for doc_id in (a, b):
            if p.graph.get_doc_metadata(doc_id) == {}:
                raise HTTPException(status_code=404, detail=f"unknown doc_id: {doc_id}")
        return {"a": a, "b": b, "comparison": p.compare_documents(a, b)}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    #
    # Registered BEFORE the doc-detail catch-all so /corpus/stats and
    # /corpus/stats/by-size aren't swallowed by /corpus/{doc_id:path}.

    @app.get("/corpus/stats")
    def stats(corpus: str = Query("default")) -> dict[str, Any]:
        return _pipeline(corpus).corpus_stats()

    @app.get("/corpus/stats/by-size")
    def stats_by_size(
        corpus: str = Query("default"),
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=200),
        order: str = Query("desc"),
    ) -> dict[str, Any]:
        return _pipeline(corpus).explorer.docs_by_chunk_count(
            page=page, page_size=page_size, order=order
        )

    # The doc-detail catch-all MUST be the last /corpus/* route; everything
    # else needs to register first or it gets shadowed.
    @app.get("/corpus/{doc_id:path}")
    def doc_detail(doc_id: str, corpus: str = Query("default")) -> dict[str, Any]:
        """Single-doc metadata + key points. Catch-all for /corpus/<doc_id>."""
        p = _pipeline(corpus)
        meta = p.graph.get_doc_metadata(doc_id)
        if not meta:
            raise HTTPException(status_code=404, detail=f"unknown doc_id: {doc_id}")
        return {
            "doc_id": doc_id,
            "title": meta["title"],
            "file_type": meta["file_type"],
            "chunk_count": meta["chunk_count"],
            "key_points": p.analyst.extract_key_points(doc_id, max_points=10),
        }

    # ------------------------------------------------------------------
    # File serving — lets the UI open source documents in a new tab
    # ------------------------------------------------------------------

    @app.get("/file")
    def serve_file(
        path: str = Query(..., min_length=1),
        corpus: str = Query("default"),
    ) -> Any:
        p = _pipeline(corpus)
        requested = Path(path).resolve()
        try:
            requested.relative_to(p.config.docs_root.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="path is outside docs_root") from None
        if not requested.is_file():
            raise HTTPException(status_code=404, detail=f"no such file: {requested}")
        return FileResponse(str(requested))

    # ------------------------------------------------------------------
    # Static UI
    # ------------------------------------------------------------------

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> Any:
            index_html = _STATIC_DIR / "index.html"
            if index_html.is_file():
                return FileResponse(str(index_html))
            return {"status": "ok", "note": "static index.html not yet present"}

    return app


__all__ = ["create_app"]
