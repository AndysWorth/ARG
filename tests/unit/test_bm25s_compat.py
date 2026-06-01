"""Compatibility tests for the bm25s library.

Verifies the exact API surface ARG uses so breaking changes are caught when
the dependency is updated. If any of these tests fail after a bm25s upgrade,
the corresponding call sites in arg/retriever/bm25_index.py must be updated.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import bm25s
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOCS = [
    "the quick brown fox",
    "jumps over the lazy dog",
    "hello world foo bar",
]


def _build_index(texts: list[str] = _DOCS) -> bm25s.BM25:
    tokenized = bm25s.tokenize(texts, stopwords=None, show_progress=False)
    idx = bm25s.BM25()
    idx.index(tokenized, show_progress=False)
    return idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tokenize_returns_tokenized_with_ids() -> None:
    """bm25s.tokenize(...) must return an object with an .ids attribute."""
    tok = bm25s.tokenize(_DOCS, stopwords=None, show_progress=False)
    assert hasattr(tok, "ids"), "tokenized result must have .ids"
    assert isinstance(tok.ids, list), ".ids must be a list"
    assert len(tok.ids) == len(_DOCS)


def test_bm25_constructable_without_args() -> None:
    bm25s.BM25()


def test_bm25_index_runs_without_error() -> None:
    tokenized = bm25s.tokenize(_DOCS, stopwords=None, show_progress=False)
    idx = bm25s.BM25()
    idx.index(tokenized, show_progress=False)


def test_get_scores_returns_ndarray_of_correct_shape() -> None:
    """get_scores(token_list) must return a 1-D numpy array of shape (n_docs,)."""
    idx = _build_index()
    tokens = ["fox"]
    scores = idx.get_scores(tokens)
    assert isinstance(scores, np.ndarray), f"expected ndarray, got {type(scores)}"
    assert scores.shape == (len(_DOCS),), f"expected shape ({len(_DOCS)},), got {scores.shape}"


def test_get_scores_higher_for_matching_doc() -> None:
    """A document containing the query token must outscore one that does not."""
    idx = _build_index()
    scores = idx.get_scores(["fox"])
    # "the quick brown fox" is _DOCS[0]; "hello world foo bar" is _DOCS[2]
    assert scores[0] > scores[2], f"doc containing 'fox' should score higher: scores={scores}"


def test_pickle_roundtrip_preserves_scores(tmp_path: Path) -> None:
    """The BM25 index must survive a pickle round-trip with identical query results."""
    idx = _build_index()
    original_scores = idx.get_scores(["quick"])

    pkl_path = tmp_path / "bm25.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with pkl_path.open("rb") as fh:
        idx2 = pickle.load(fh)

    restored_scores = idx2.get_scores(["quick"])
    np.testing.assert_array_equal(
        original_scores,
        restored_scores,
        err_msg="scores changed after pickle round-trip",
    )


def test_empty_corpus_does_not_raise() -> None:
    """tokenize on an empty list must not crash."""
    tok = bm25s.tokenize([], stopwords=None, show_progress=False)
    assert hasattr(tok, "ids")
    assert tok.ids == []
