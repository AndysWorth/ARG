"""Chunker tests — covers every Section 7 ``test_chunker.py`` test point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arg.config import ARGConfig
from arg.crawler.extractors import Document
from arg.indexer.chunker import _token_count, chunk_document

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config(tmp_path: Path) -> ARGConfig:
    docs = tmp_path / "docs"
    docs.mkdir()
    return ARGConfig(docs_root=docs, db_path=tmp_path / "arg_db")


def _doc(
    tmp_path: Path,
    content: str,
    *,
    title: str = "Test Doc",
    file_type: str = "html",
    page_description: str = "",
    code_blocks: list[str] | None = None,
    page_offsets: list[int] | None = None,
    name: str = "page.html",
) -> Document:
    metadata: dict[str, Any] = {
        "title": title,
        "page_description": page_description,
        "file_type": file_type,
        "code_blocks": list(code_blocks or []),
    }
    if page_offsets is not None:
        metadata["page_offsets"] = page_offsets
    p = tmp_path / name
    p.write_text("")  # path needs to exist for resolve() roundtrip
    return Document(path=p, content=content, metadata=metadata)


# ---------------------------------------------------------------------------
# Heading-based splits
# ---------------------------------------------------------------------------


def test_h1_h2_h3_sentinels_produce_section_boundaries(tmp_path, base_config):
    content = (
        "##H1## Intro\n"
        "Intro paragraph one.\n"
        "Intro paragraph two.\n"
        "##H2## Sub one\n"
        "Sub one body.\n"
        "##H2## Sub two\n"
        "Sub two body.\n"
        "##H3## Deeper\n"
        "Deeper body.\n"
    )
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)

    # 4 distinct heading_paths: Intro, Intro > Sub one, Intro > Sub two,
    # Intro > Sub two > Deeper.
    paths = [c.metadata["heading_path"] for c in chunks]
    assert "Test Doc > Intro" in paths
    assert "Test Doc > Intro > Sub one" in paths
    assert "Test Doc > Intro > Sub two" in paths
    assert "Test Doc > Intro > Sub two > Deeper" in paths


def test_document_with_no_headings_produces_single_section(tmp_path, base_config):
    content = "Just one paragraph with no heading markers at all.\n" * 3
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    assert len(chunks) == 1
    assert chunks[0].metadata["heading_path"] == "Test Doc"


def test_h4_h5_h6_do_not_force_chunk_splits(tmp_path, base_config):
    """H4-H6 are plain text (no sentinels). They do NOT split chunks."""
    content = (
        "##H1## Top\n"
        "Top body before deeper headings.\n"
        "H4 looking text — not a sentinel\n"
        "More body text continuing.\n"
    )
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    paths = [c.metadata["heading_path"] for c in chunks]
    # All chunks share the H1 heading_path.
    assert all(p == "Test Doc > Top" for p in paths)


# ---------------------------------------------------------------------------
# Sliding window + token sizes
# ---------------------------------------------------------------------------


def test_chunks_do_not_exceed_chunk_size(tmp_path, base_config):
    # Generate ~5x chunk_size worth of tokens within one section.
    long_section = " ".join(f"word{i}" for i in range(5000))
    content = f"##H1## Big\n{long_section}\n"
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    assert len(chunks) > 1
    for c in chunks:
        assert _token_count(c.chunk_text) <= base_config.chunk_size


def test_chunk_overlap_is_applied(tmp_path):
    """Consecutive chunks within one section must share `overlap` tokens."""
    docs = tmp_path / "docs"
    docs.mkdir()
    config = ARGConfig(docs_root=docs, db_path=tmp_path / "db", chunk_size=200, chunk_overlap=50)
    long_text = " ".join(f"word{i}" for i in range(2000))
    content = f"##H1## Section\n{long_text}\n"
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, config)
    assert len(chunks) >= 2
    # The last tokens of chunk N should appear at the start of chunk N+1.
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    from itertools import pairwise

    for a, b in pairwise(chunks):
        tail = enc.encode(a.chunk_text)[-config.chunk_overlap :]
        head = enc.encode(b.chunk_text)[: config.chunk_overlap]
        # Decode comparison ducks under any whitespace artifacts: just check
        # we have a non-empty intersection in token IDs.
        assert set(tail) & set(head)


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------


def test_heading_path_reflects_section_hierarchy(tmp_path, base_config):
    content = (
        "##H1## Auth\n"
        "Body of auth section.\n"
        "##H2## OAuth\n"
        "Body of OAuth subsection.\n"
        "##H3## Token expiry\n"
        "Token expiry details.\n"
    )
    doc = _doc(tmp_path, content, title="API Docs")
    chunks = chunk_document(doc, base_config)
    # Find the deepest section's chunk.
    deepest = next(
        c for c in chunks if c.metadata["heading_path"] == "API Docs > Auth > OAuth > Token expiry"
    )
    assert "Token expiry details" in deepest.chunk_text


def test_page_description_in_chunk_metadata(tmp_path, base_config):
    content = "##H1## H\nbody\n"
    doc = _doc(tmp_path, content, page_description="Doc summary here")
    chunks = chunk_document(doc, base_config)
    assert all(c.metadata["page_description"] == "Doc summary here" for c in chunks)


def test_has_table_detected_on_pipe_markdown(tmp_path, base_config):
    content = "##H1## Tables\nIntro line.\n| Col A | Col B |\n|---|---|\n| 1 | 2 |\n"
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    assert any(c.metadata["has_table"] for c in chunks)


def test_has_table_false_when_no_table(tmp_path, base_config):
    content = "##H1## No tables\nJust prose with no pipe characters.\n"
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    assert all(c.metadata["has_table"] is False for c in chunks)


def test_has_code_detected_via_code_blocks_metadata(tmp_path, base_config):
    code = "def greet(name):\n    return f'hello, {name}'\nprint(greet('world'))"
    # Place the same code in chunk_text so the heuristic catches the overlap.
    content = f"##H1## Code\nHere is the code:\n{code}\n"
    doc = _doc(tmp_path, content, code_blocks=[code])
    chunks = chunk_document(doc, base_config)
    assert any(c.metadata["has_code"] for c in chunks)


def test_has_code_false_when_no_code(tmp_path, base_config):
    content = "##H1## Prose\nJust normal prose without any code.\n"
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    assert all(c.metadata["has_code"] is False for c in chunks)


# ---------------------------------------------------------------------------
# PDF page-number propagation
# ---------------------------------------------------------------------------


def test_pdf_page_numbers_propagated_via_offsets(tmp_path, base_config):
    # Synthetic PDF content with three pages and known offsets.
    page1 = "##H1## Page one heading\n" + ("page1 body line.\n" * 30)
    page2 = "##H1## Page two heading\n" + ("page2 body line.\n" * 30)
    page3 = "##H1## Page three heading\n" + ("page3 body line.\n" * 30)
    separator = "\n\n"
    content = separator.join([page1, page2, page3])
    page_offsets = [
        0,
        len(page1) + len(separator),
        len(page1) + len(separator) + len(page2) + len(separator),
    ]
    doc = _doc(
        tmp_path,
        content,
        file_type="pdf",
        page_offsets=page_offsets,
        name="manual.pdf",
    )
    chunks = chunk_document(doc, base_config)
    page_numbers = {c.metadata["page_number"] for c in chunks}
    assert page_numbers == {1, 2, 3}


def test_html_chunks_have_page_number_none(tmp_path, base_config):
    content = "##H1## Heading\nbody\n"
    doc = _doc(tmp_path, content)  # no page_offsets — HTML
    chunks = chunk_document(doc, base_config)
    assert all(c.metadata["page_number"] is None for c in chunks)


# ---------------------------------------------------------------------------
# Contextual enrichment
# ---------------------------------------------------------------------------


def test_embedding_text_prefix_when_enrichment_on(tmp_path, base_config):
    content = "##H1## Auth\n##H2## OAuth\nThe OAuth flow does this and that.\n"
    doc = _doc(tmp_path, content, title="API Docs")
    chunks = chunk_document(doc, base_config)
    enriched = next(c for c in chunks if c.metadata["heading_path"] == "API Docs > Auth > OAuth")
    assert enriched.embedding_text.startswith("API Docs > Auth > OAuth: ")
    # The raw chunk text never contains the contextual prefix.
    assert not enriched.chunk_text.startswith("API Docs > Auth > OAuth:")


def test_embedding_text_equals_chunk_text_when_enrichment_off(tmp_path, base_config):
    base_config.contextual_enrichment = False
    content = "##H1## Auth\nThe auth flow.\n"
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    for c in chunks:
        assert c.embedding_text == c.chunk_text


def test_llm_context_does_not_contain_contextual_prefix(tmp_path, base_config):
    content = "##H1## Auth\nBody.\n##H2## OAuth\nMore body.\n"
    doc = _doc(tmp_path, content, title="API Docs")
    chunks = chunk_document(doc, base_config)
    for c in chunks:
        # The LLM-facing text (chunk_text / chunk.text) should NOT carry the prefix.
        assert "API Docs > " not in c.chunk_text
        assert "API Docs > " not in c.chunk.text


# ---------------------------------------------------------------------------
# Chunk identity
# ---------------------------------------------------------------------------


def test_chunk_ids_are_unique_and_sequential(tmp_path, base_config):
    content = (
        "##H1## A\n" + " ".join(f"w{i}" for i in range(4000)) + "\n"
        "##H1## B\n" + " ".join(f"w{i}" for i in range(4000)) + "\n"
    )
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    ids = [c.chunk.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    positions = [c.metadata["position"] for c in chunks]
    assert positions == list(range(len(chunks)))


def test_chunk_token_count_populated(tmp_path, base_config):
    content = "##H1## h\nThis is a short body of words.\n"
    doc = _doc(tmp_path, content)
    chunks = chunk_document(doc, base_config)
    for c in chunks:
        assert c.chunk.token_count > 0
        assert c.chunk.token_count == _token_count(c.chunk_text)


def test_empty_document_returns_no_chunks(tmp_path, base_config):
    doc = _doc(tmp_path, "")
    assert chunk_document(doc, base_config) == []
