"""BM25 sparse index for exact-term retrieval.

The Section 7 spec is explicit: **the BM25 index is written by the indexer
during `pipeline.index()`, not lazily built by the retriever.** This module
provides the persistence layer — :class:`BM25Index` exposes ``build``,
``save``, ``load``, and ``query`` so the indexer owns the write path and the
retriever (Section 8) only ever reads.

Tokeniser
---------
Pure Python, dependency-free: lowercase + ASCII-word split on ``\\W+``. This
matches what users actually type into a search box ("api key", "OAuth2",
"rate-limit") more closely than the heavier nltk tokenisers and keeps the
index portable across the project's offline-first constraint.

Persistence
-----------
The index is pickled. ``rank_bm25.BM25Okapi`` instances pickle cleanly along
with their internal IDF / doc-length tables, so deserialisation is exact.
The corresponding ``chunk_ids`` list is pickled alongside so queries can map
ranked positions back to chunk identifiers.

Locality
--------
``rank_bm25`` is pure Python and runs in-process. The pickle file lives next
to the rest of the per-corpus state (``arg_db/<corpus>/bm25_index.pkl``).
No network involved.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + word-split tokeniser. Dependency-free; cheap."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """Sparse-retrieval index keyed by ``chunk_id``.

    Construct empty with no arguments; call :meth:`build` to populate from a
    chunk corpus, then :meth:`save` to persist. Consumers (the retriever)
    call :meth:`load` to read the on-disk index, then :meth:`query`.
    """

    chunk_ids: list[str] = field(default_factory=list)
    bm25: BM25Okapi | None = field(default=None, repr=False)

    @property
    def is_empty(self) -> bool:
        return self.bm25 is None or not self.chunk_ids

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------

    def build(self, chunks: list[tuple[str, str]]) -> None:
        """Build the index from ``[(chunk_id, chunk_text), ...]``.

        Passing an empty list leaves the index empty (subsequent queries
        return ``[]``).
        """
        if not chunks:
            self.chunk_ids = []
            self.bm25 = None
            return
        self.chunk_ids = [cid for cid, _ in chunks]
        tokenised = [_tokenize(text) for _, text in chunks]
        # rank_bm25's BM25Okapi requires at least one non-empty document.
        if not any(tokenised):
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(tokenised)

    def save(self, path: Path) -> None:
        """Pickle the index to ``path``. Creates parent dirs if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "chunk_ids": self.chunk_ids,
            "bm25": self.bm25,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Read a pickled index. Returns an empty index if the file is absent."""
        path = Path(path)
        if not path.is_file():
            return cls()
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            logger.warning("BM25 index at %s is not a dict; ignoring", path)
            return cls()
        return cls(
            chunk_ids=list(payload.get("chunk_ids", [])),
            bm25=payload.get("bm25"),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, q: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, score), ...]`` ranked by BM25 score, descending."""
        if self.is_empty or top_k <= 0:
            return []
        tokens = _tokenize(q)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)  # type: ignore[union-attr]
        # Pair with chunk_ids and pick top_k by score desc, breaking ties by
        # chunk_id for deterministic results.
        ranked = sorted(
            zip(self.chunk_ids, scores, strict=True),
            key=lambda x: (-float(x[1]), x[0]),
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]
