"""ARGPipeline — the public façade that wires every component together.

One instance per corpus. Construction order matches the dependency graph:

  1. KnowledgeGraph        (owns the Kuzu directory)
  2. Indexer               (owns ChromaDB collections; needs the KG + Embedder)
  3. HybridRetriever       (reads Chroma + Kuzu + BM25 pickle)
  4. QueryProcessor        (LLM-based pre-retrieval transforms)
  5. Generator             (composes processor + retriever + LLM)
  6. CorpusAnalyst         (DCI document operations; shares LLM + retriever)
  7. CorpusExplorer        (DCI navigation + clustering; shares everything)
  8. DocsWatcher           (optional; only when ``watch_enabled``)

Pre-flight checks (spec Section 10)
-----------------------------------
  * **Ollama health**: an HTTP GET against ``ollama_base_url + /api/version``
    must succeed unless ``skip_health_check=True``. Failure raises
    RuntimeError so the caller doesn't hand stale embeddings to retrieval.
  * **Schema drift**: ``config_hash.json`` records the embedding-dimension,
    chunk-size, embed-model, and OCR threshold that the corpus was indexed
    against. A mismatch means the on-disk Chroma/Kuzu data was produced
    under a different config and using it would yield silently wrong
    results. The pipeline refuses to load and asks the operator to
    ``reset_corpus`` first.

Concurrency
-----------
Write methods (index, add_document, remove_document, update_document) take a
``threading.RLock`` so the watcher's callback can't race the user's CLI
``index`` call. Reads (query, DCI) don't take the lock — Kuzu and Chroma
both serialise their own writes internally, and a read seeing a partial
state is acceptable per Section 19.4.4.

Locality
--------
The default embedder + LLM use ``config.ollama_base_url`` which the
ARGConfig validator already restricts to localhost. ``skip_health_check``
exists for the test path; production code never sets it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import signal
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from arg.config import ARGConfig
from arg.crawler.crawler import crawl
from arg.crawler.extractors import Document
from arg.crawler.watcher import EVENT_CREATED, EVENT_DELETED, EVENT_MODIFIED, DocsWatcher
from arg.dci import CorpusAnalyst, CorpusExplorer
from arg.embeddings import Embedder
from arg.generator import ARGResult, Generator, QueryProcessor
from arg.graph import KnowledgeGraph
from arg.indexer import Indexer
from arg.llm import LLM
from arg.retriever import HybridRetriever

logger = logging.getLogger(__name__)


# Fields whose change should be treated as schema drift. Each one alters the
# on-disk representation of either the embedding vectors or the chunk text.
_HASH_FIELDS: tuple[str, ...] = (
    "embed_model",
    "embed_dim",
    "chunk_size",
    "chunk_overlap",
    "contextual_enrichment",
    "ocr_char_threshold",
)


class ARGPipeline:
    """Top-level façade. One instance per corpus."""

    def __init__(
        self,
        config: ARGConfig,
        corpus_name: str = "default",
        *,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        skip_health_check: bool = False,
        skip_signal_handlers: bool = False,
        skip_watcher: bool = False,
    ) -> None:
        self.config = config
        self.corpus_name = corpus_name
        self._lock = threading.RLock()
        self._closed = False

        # ------- Pre-flight ----------------------------------------------------
        if not skip_health_check:
            self._verify_ollama_health()
        self._check_schema_hash()

        # ------- Embedder + LLM ------------------------------------------------
        # Production wires the Ollama-backed implementations here. Tests inject
        # fakes via the kwargs.
        if embedder is None:
            embedder = self._default_embedder()
        if llm is None:
            llm = self._default_llm()
        self._embedder = embedder
        self._llm = llm

        # ------- Knowledge graph + indexer -------------------------------------
        self.graph = KnowledgeGraph(config.kuzu_path(corpus_name))
        self.indexer = Indexer(
            config=config,
            knowledge_graph=self.graph,
            embedder=embedder,
            corpus_name=corpus_name,
        )

        # ------- Retriever + query processor + generator ------------------------
        self.retriever = HybridRetriever(
            config=config,
            knowledge_graph=self.graph,
            embedder=embedder,
            chroma_documents_collection=self.indexer._docs_coll,
            chroma_chunks_collection=self.indexer._chunks_coll,
            bm25_index_path=config.bm25_index_path(corpus_name),
            cluster_cache_path=config.cluster_cache_path(corpus_name),
        )
        self.query_processor = QueryProcessor(config=config, llm=llm)
        self.generator = Generator(
            config=config,
            llm=llm,
            retriever=self.retriever,
            query_processor=self.query_processor,
        )

        # ------- DCI ----------------------------------------------------------
        self.analyst = CorpusAnalyst(
            config=config,
            llm=llm,
            retriever=self.retriever,
            knowledge_graph=self.graph,
            chroma_chunks_collection=self.indexer._chunks_coll,
            chroma_documents_collection=self.indexer._docs_coll,
            corpus_name=corpus_name,
        )
        self.explorer = CorpusExplorer(
            config=config,
            knowledge_graph=self.graph,
            analyst=self.analyst,
            llm=llm,
            chroma_documents_collection=self.indexer._docs_coll,
            corpus_name=corpus_name,
        )

        # ------- Watcher ------------------------------------------------------
        self.watcher: DocsWatcher | None = None
        if config.watch_enabled and not skip_watcher:
            self.watcher = DocsWatcher(
                docs_root=config.docs_root,
                config=config,
                on_change=self._on_file_change,
            )
            self.watcher.start()

        # ------- Signal handlers -----------------------------------------------
        if not skip_signal_handlers:
            self._register_signal_handlers()

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def _verify_ollama_health(self) -> None:
        """Hit Ollama's ``/api/version`` and raise if it doesn't answer.

        Failure messages tell the user how to fix it ("start ollama via
        ``brew services start ollama``") because a missing local daemon is
        the single most common cause of a confusing import-time exception.
        """
        url = self.config.ollama_base_url.rstrip("/") + "/api/version"
        try:
            with urllib_request.urlopen(url, timeout=2) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama health check on {url} returned HTTP {resp.status}")
        except (urllib_error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Ollama daemon not reachable at {self.config.ollama_base_url} "
                f"({exc}). Start it with `brew services start ollama`."
            ) from exc

    def _check_schema_hash(self) -> None:
        path = self.config.config_hash_path(self.corpus_name)
        current = self._compute_schema_hash()
        if not path.is_file():
            return  # first run; write happens after index()
        try:
            with path.open("r", encoding="utf-8") as fh:
                stored = json.load(fh)
            if not isinstance(stored, dict):
                raise ValueError("config_hash.json is not a dict")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s (%s); will overwrite after next index()", path, exc)
            return
        if stored.get("hash") != current["hash"]:
            raise RuntimeError(
                f"Schema drift detected for corpus '{self.corpus_name}'. "
                "The indexed data was produced under a different ARGConfig "
                f"(chunk_size / embed_model / etc). Stored: {stored.get('config')!r} "
                f"current: {current['config']!r}. Run scripts/reset_corpus.py to "
                "wipe the corpus and re-index, or revert the config change."
            )

    def _compute_schema_hash(self) -> dict[str, Any]:
        payload = {field: getattr(self.config, field, None) for field in _HASH_FIELDS}
        canonical = json.dumps(payload, sort_keys=True)
        return {
            "config": payload,
            "hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    def _write_schema_hash(self) -> None:
        path = self.config.config_hash_path(self.corpus_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._compute_schema_hash(), fh, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # Default backends
    # ------------------------------------------------------------------

    def _default_embedder(self) -> Embedder:
        """Construct an Ollama embedder pointed at ``config.ollama_base_url``."""
        from llama_index.embeddings.ollama import OllamaEmbedding

        # LlamaIndex's OllamaEmbedding exposes get_text_embedding and
        # get_text_embedding_batch. Wrap them to match the Embedder protocol.
        oe = OllamaEmbedding(
            model_name=self.config.embed_model,
            base_url=self.config.ollama_base_url,
            # nomic-embed-text supports 8192 tokens; Ollama's default num_ctx
            # is only 2048, which can be exceeded by enriched chunks.
            ollama_additional_kwargs={"num_ctx": 8192},
        )

        class _OllamaEmbedderAdapter:
            def embed(self_inner, text: str) -> list[float]:
                return list(oe.get_text_embedding(text))

            def embed_batch(self_inner, texts: list[str]) -> list[list[float]]:
                return [list(v) for v in oe.get_text_embedding_batch(texts)]

        return _OllamaEmbedderAdapter()

    def _default_llm(self) -> LLM:
        """Construct an Ollama LLM pointed at ``config.ollama_base_url``."""
        from llama_index.llms.ollama import Ollama

        client = Ollama(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url,
            request_timeout=120.0,
        )

        class _OllamaLLMAdapter:
            def complete(self_inner, prompt: str) -> str:
                return str(client.complete(prompt))

            def stream_complete(self_inner, prompt: str) -> Iterator[str]:
                for chunk in client.stream_complete(prompt):
                    text = getattr(chunk, "delta", None) or getattr(chunk, "text", "")
                    if text:
                        yield str(text)

        return _OllamaLLMAdapter()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self) -> dict[str, Any]:
        """Crawl ``docs_root`` and (re-)index every document.

        The crawl generator is passed directly to the indexer so each file
        is embedded and persisted before the crawler advances to the next
        one. Cluster cache is invalidated before the run starts — any new
        chunk makes the existing cache stale regardless of how the run ends.
        """
        with self._lock:
            logger.info("pipeline.index: starting crawl + index of %s", self.config.docs_root)
            # Invalidate before the loop: the cache is stale the moment any
            # new chunk lands, so don't wait until the run completes.
            self.explorer.invalidate_cluster_cache()
            stats = self.indexer.index(crawl(self.config.docs_root, self.config))
            logger.info(
                "pipeline.index: crawl + index complete "
                "(indexed=%d, skipped=%d, chunks=%d); reloading retriever",
                stats.documents_indexed,
                stats.documents_skipped,
                stats.chunks_written,
            )
            self.retriever.reload()
            self._write_schema_hash()
            return {
                "documents_indexed": stats.documents_indexed,
                "documents_skipped": stats.documents_skipped,
                "chunks_written": stats.chunks_written,
                "links_recorded": stats.links_recorded,
            }

    def add_document(self, path: Path) -> int:
        """Re-index one document (used by the watcher). Returns chunks written."""
        with self._lock:
            doc = self._extract_one(path)
            if doc is None:
                return 0
            n = self.indexer.add_document(doc)
            self.retriever.reload()
            self.explorer.invalidate_cluster_cache()
            self._invalidate_summary(doc.path.resolve())
            return n

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            self.indexer.remove_document(doc_id)
            self.retriever.reload()
            self.explorer.invalidate_cluster_cache()
            self._invalidate_summary(Path(doc_id))

    def update_document(self, path: Path) -> int:
        with self._lock:
            doc = self._extract_one(path)
            if doc is None:
                return 0
            n = self.indexer.update_document(doc)
            self.retriever.reload()
            # Update is per-doc; cluster cache still invalidated since
            # changing one doc can shift its cluster assignment.
            self.explorer.invalidate_cluster_cache()
            self._invalidate_summary(doc.path.resolve())
            return n

    def _extract_one(self, path: Path) -> Document | None:
        """Dispatch to the appropriate extractor. Returns ``None`` for skips
        (encrypted PDFs, non-indexable suffixes)."""
        from arg.crawler.extractors import (
            extract_html,
            extract_pdf_to_document,
            extract_text,
        )

        suffix = path.suffix.lower()
        if suffix in {".html", ".htm"}:
            return extract_html(path, self.config)
        if suffix == ".pdf":
            return extract_pdf_to_document(path, self.config)
        if suffix in {".txt", ".md", ".markdown"}:
            return extract_text(path, self.config)
        return None

    def _invalidate_summary(self, doc_path: Path) -> None:
        """Delete the cached LLM summary for this document, if any."""
        doc_id = str(doc_path.resolve())
        digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16]
        summary_file = self.config.summary_path(self.corpus_name) / f"{digest}.json"
        if summary_file.is_file():
            try:
                summary_file.unlink()
            except OSError as exc:
                logger.warning("Could not delete summary cache %s: %s", summary_file, exc)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        enrich: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> ARGResult:
        return self.generator.generate(question, enrich=enrich, filters=filters)

    def stream_query(
        self,
        question: str,
        *,
        enrich: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        yield from self.generator.stream_generate(question, enrich=enrich, filters=filters)

    # ------------------------------------------------------------------
    # DCI (public)
    # ------------------------------------------------------------------

    def summarize_document(self, doc_id: str) -> str:
        return self.analyst.summarize_document(doc_id)

    def compare_documents(self, doc_id_a: str, doc_id_b: str) -> str:
        return self.analyst.compare_documents(doc_id_a, doc_id_b)

    def corpus_search(
        self, query: str, file_type: str | None = None, top_k: int = 10
    ) -> list[dict[str, Any]]:
        return self.explorer.corpus_search(query, file_type=file_type, top_k=top_k)

    def get_topic_clusters(self) -> list[dict[str, Any]]:
        return self.explorer.get_topic_clusters()

    def corpus_stats(self) -> dict[str, Any]:
        graph_stats = self.graph.stats()
        return {
            "documents": graph_stats["documents"],
            "chunks": graph_stats["chunks"],
            "link_edges": graph_stats["link_edges"],
            "most_linked": self.explorer.most_linked_docs(top_n=5),
            "orphaned": self.explorer.orphaned_docs(),
            "by_size_preview": self.explorer.docs_by_chunk_count(page=1, page_size=10),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush logs, stop watcher, close Kuzu. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.watcher is not None:
                try:
                    self.watcher.stop()
                except Exception:
                    logger.exception("Error stopping watcher")
            try:
                self.graph.close()
            except Exception:
                logger.exception("Error closing knowledge graph")

    def __enter__(self) -> ARGPipeline:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _register_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers that close the pipeline.

        Only installs handlers in the main thread; signal.signal() raises
        ValueError if called from a background thread, which is the case
        when the pipeline is built inside a worker thread (e.g. a future
        FastAPI worker). Failures are logged but never fatal.
        """
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (ValueError, OSError) as exc:  # pragma: no cover
            logger.debug("Could not register signal handlers: %s", exc)

    def _signal_handler(self, signum: int, frame: Any) -> None:  # pragma: no cover
        """SIGTERM / SIGINT → close + raise KeyboardInterrupt.

        The previous version called ``close()`` and returned, which let
        Python resume the interrupted loop and made Ctrl-C effectively a
        no-op during long indexing runs. Raising KeyboardInterrupt unwinds
        the stack so the user actually exits.
        """
        logger.info("Received signal %s — closing pipeline", signum)
        self.close()
        raise KeyboardInterrupt(f"signal {signum}")

    # ------------------------------------------------------------------
    # Watcher callback
    # ------------------------------------------------------------------

    def _on_file_change(self, path: Path, event_kind: str) -> None:
        """Watcher hands us debounced filesystem events."""
        if self._closed:
            return
        try:
            if event_kind in (EVENT_CREATED, EVENT_MODIFIED):
                self._add_or_update_via_path(path)
            elif event_kind == EVENT_DELETED:
                self.remove_document(str(path.resolve()))
        except Exception:
            logger.exception("Error handling watcher event %s for %s", event_kind, path)

    def _add_or_update_via_path(self, path: Path) -> None:
        doc = self._extract_one(path)
        if doc is None:
            return
        with self._lock:
            self.indexer.update_document(doc)
            self.retriever.reload()
            self.explorer.invalidate_cluster_cache()
            self._invalidate_summary(doc.path.resolve())
