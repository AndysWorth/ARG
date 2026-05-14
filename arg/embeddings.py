"""Embedder protocol — shared by indexer and retriever.

Lives in its own module to break the circular import that would otherwise
appear between ``arg.indexer.indexer`` (which uses an Embedder to vectorise
chunks at write time) and ``arg.retriever.retriever`` (which uses an
Embedder to vectorise queries at read time).
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Pluggable embedding source.

    Production wires this to Ollama (Section 9). Unit tests inject a
    deterministic fake so the suite stays offline.
    """

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
