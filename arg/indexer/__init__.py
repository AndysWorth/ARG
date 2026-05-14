"""ARG indexer package — chunker + ingestion pipeline."""

from arg.indexer.chunker import ChunkedSection, chunk_document
from arg.indexer.indexer import Embedder, Indexer, IndexStats

__all__ = ["ChunkedSection", "Embedder", "IndexStats", "Indexer", "chunk_document"]
