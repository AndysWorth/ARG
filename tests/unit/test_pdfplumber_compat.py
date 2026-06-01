"""Compatibility tests for the pdfplumber library.

Verifies the exact API surface ARG uses so breaking changes are caught when
the dependency is updated. If any of these tests fail after a pdfplumber
upgrade, the corresponding call sites in arg/crawler/extractors.py must be
updated (specifically _pdfplumber_extract_page and related helpers).

Uses the existing test fixture PDFs in tests/fixtures/docs/.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures" / "docs"
MANUAL_PDF = FIXTURES / "manual.pdf"


@pytest.fixture(scope="module")
def first_page():
    """Open manual.pdf and yield the first page."""
    with pdfplumber.open(str(MANUAL_PDF)) as pdf:
        yield pdf.pages[0]


@pytest.fixture(scope="module")
def page_with_table():
    """Yield the first page in manual.pdf that has at least one table."""
    with pdfplumber.open(str(MANUAL_PDF)) as pdf:
        for page in pdf.pages:
            if list(page.find_tables()):
                yield page
                return
    pytest.skip("no page with a table found in manual.pdf")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_open_returns_pdf_object() -> None:
    with pdfplumber.open(str(MANUAL_PDF)) as pdf:
        assert hasattr(pdf, "pages")
        assert len(pdf.pages) >= 1


def test_page_bbox_has_four_floats(first_page) -> None:
    """page.bbox must be a 4-tuple of numeric values."""
    bbox = first_page.bbox
    assert len(bbox) == 4, f"expected 4-tuple, got {len(bbox)} elements"
    for v in bbox:
        assert isinstance(v, (int, float)), f"bbox element {v!r} is not numeric"


def test_page_chars_is_list_of_dicts(first_page) -> None:
    """page.chars must be a list of dicts."""
    chars = first_page.chars
    assert isinstance(chars, list)
    if chars:
        assert isinstance(chars[0], dict)


def test_char_dicts_have_required_keys(first_page) -> None:
    """Each char dict must contain x0, top, x1, bottom — the keys ARG reads."""
    chars = first_page.chars
    if not chars:
        pytest.skip("no chars on first page")
    required = {"x0", "top", "x1", "bottom"}
    for key in required:
        assert key in chars[0], f"char dict missing '{key}'; keys={sorted(chars[0])}"


def test_find_tables_returns_list(first_page) -> None:
    tables = list(first_page.find_tables())
    assert isinstance(tables, list)


def test_table_bbox_has_four_elements(page_with_table) -> None:
    tables = list(page_with_table.find_tables())
    assert len(tables) >= 1
    bbox = tables[0].bbox
    assert len(bbox) == 4


def test_table_extract_returns_list_of_lists(page_with_table) -> None:
    """Table.extract() must return a list of rows, each row a list of cells."""
    tables = list(page_with_table.find_tables())
    rows = tables[0].extract()
    assert isinstance(rows, list), f"expected list, got {type(rows)}"
    assert len(rows) >= 1
    assert isinstance(rows[0], list), f"expected list of lists, got list of {type(rows[0])}"


def test_open_with_string_path_and_context_manager() -> None:
    """pdfplumber.open must work with a string path inside a with statement."""
    with pdfplumber.open(str(MANUAL_PDF)) as pdf:
        assert pdf is not None
