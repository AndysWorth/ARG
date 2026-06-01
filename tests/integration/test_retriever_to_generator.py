"""Integration: retriever → generator end-to-end.

Real embedder, mocked LLM. Validates QueryProcessor + retrieval + generator
wiring against the real Chroma + Kuzu + BM25 surface.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Rewrite path
# ---------------------------------------------------------------------------


def test_conversational_query_rewritten_and_raw_query_in_llm_prompt(indexed_pipeline, mock_llm):
    """Rewrite happens before retrieval; raw query is what the LLM sees."""
    mock_llm.respond_to(
        "Rewrite the following",
        "Authentication methods and API key configuration",
    )
    mock_llm.respond_to(
        "Does the following question contain",
        "Authentication methods and API key configuration",
    )
    mock_llm.respond_to("You are Archivist", "FAKE_ANSWER_TOKEN")

    result = indexed_pipeline.query("how do I log in to my account")
    assert result.rewritten_query == "Authentication methods and API key configuration"

    # The raw query must appear in the answer prompt; the rewrite must not.
    answer_prompts = [p for p in mock_llm.calls if "You are Archivist" in p]
    assert answer_prompts
    assert any("how do I log in to my account" in p for p in answer_prompts)
    assert all("Authentication methods and API key configuration" not in p for p in answer_prompts)


# ---------------------------------------------------------------------------
# Decompose path
# ---------------------------------------------------------------------------


def test_compound_query_decomposes_and_unions_chunks(indexed_pipeline, mock_llm):
    """Compound question → 2 sub-queries; sources must include chunks from each."""
    mock_llm.respond_to(
        "Rewrite the following",
        "authentication flow and rate limits",
    )
    mock_llm.respond_to(
        "Does the following question contain",
        '{"sub_questions": ["How do I authenticate?", "What are the rate limits?"]}',
    )
    mock_llm.respond_to("You are Archivist", "MOCKED_ANSWER")

    result = indexed_pipeline.query("how do auth and rate limits work")
    assert result.sub_queries == [
        "How do I authenticate?",
        "What are the rate limits?",
    ]
    # Both topical pages must appear in the source list.
    doc_ids = {s.doc_id for s in result.sources}
    assert any("page_a.html" in d for d in doc_ids)
    assert any("page_b.html" in d for d in doc_ids)


# ---------------------------------------------------------------------------
# Source citations
# ---------------------------------------------------------------------------


def test_source_citations_match_retrieved_chunks(indexed_pipeline, mock_llm):
    mock_llm.respond_to("You are Archivist", "FAKE_ANSWER")
    result = indexed_pipeline.query("authentication", enrich=False)
    assert result.sources
    # Each SourceRef.chunk_id must be a real chunk in the chunks collection.
    chunks = indexed_pipeline.indexer._chunks_coll.get(
        ids=[s.chunk_id for s in result.sources], include=["metadatas"]
    )
    assert chunks["ids"], "every cited chunk_id must resolve in ChromaDB"
    assert set(chunks["ids"]) == {s.chunk_id for s in result.sources}


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def test_generator_receives_formatted_context_string(indexed_pipeline, mock_llm):
    """The system prompt must carry the chunk content + ``[Source: ...]`` headers."""
    mock_llm.respond_to("You are Archivist", "FAKE")
    indexed_pipeline.query("rate limit table", enrich=False)
    answer_prompts = [p for p in mock_llm.calls if "You are Archivist" in p]
    assert answer_prompts
    assert "[Source:" in answer_prompts[0]
    assert "Context:" in answer_prompts[0]
    assert "Question:" in answer_prompts[0]


# ---------------------------------------------------------------------------
# Empty-context fallback
# ---------------------------------------------------------------------------


def test_unrelated_query_returns_fallback_without_invoking_llm(
    indexed_pipeline, mock_llm, base_config
):
    """A query with no retrievable content returns the no-context refusal.

    Real Chroma retrieves *some* chunks for any query (closest neighbours by
    cosine distance), so the empty-retrieval contract on a real index is
    tested by **removing all documents first** rather than by hoping a
    semantic query returns nothing.
    """
    for d in indexed_pipeline.graph.list_all_documents():
        indexed_pipeline.remove_document(d["doc_id"])
    # Re-load the retriever's BM25 view since remove_document already did so.
    indexed_pipeline.retriever.reload()
    result = indexed_pipeline.query("anything at all", enrich=False)
    assert result.answer == "The documentation does not cover this topic."
    assert result.sources == []
    # No "You are Archivist" prompt was issued for generation.
    assert all("You are Archivist" not in p for p in mock_llm.calls)
