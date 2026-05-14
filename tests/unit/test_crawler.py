"""Crawler tests — link normalisation, dedup, dir-walk fallback, depth cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from arg.config import ARGConfig
from arg.crawler.crawler import crawl, normalise_href

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docs_root(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    return root


@pytest.fixture
def config(tmp_path: Path, docs_root: Path) -> ARGConfig:
    return ARGConfig(docs_root=docs_root, db_path=tmp_path / "arg_db")


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def html_with_links(*hrefs: str, body: str = "body") -> str:
    anchors = "".join(f'<a href="{h}">link</a>' for h in hrefs)
    return f"<html><body><h1>page</h1>{anchors}<p>{body}</p></body></html>"


def _paths(docs) -> set[str]:
    return {d.path.name for d in docs}


# ---------------------------------------------------------------------------
# normalise_href unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "href",
    [
        "#section",
        "#",
        "mailto:foo@bar.com",
        "javascript:void(0)",
        "tel:+15555550100",
        "ftp://ftp.example.com/x.html",
        "//cdn.example.com/page.html",
        "http://example.com/x.html",
        "https://example.com/x.html",
        "",
        "   ",
    ],
)
def test_normalise_href_rejects(href, docs_root):
    src = docs_root / "index.html"
    src.touch()
    assert normalise_href(href, src.resolve(), docs_root.resolve()) is None


def test_normalise_href_rejects_path_escape(docs_root, tmp_path):
    # Sibling file outside docs_root.
    sibling = tmp_path / "evil.html"
    sibling.write_text("nope")
    src = docs_root / "index.html"
    src.touch()
    assert normalise_href("../evil.html", src.resolve(), docs_root.resolve()) is None


def test_normalise_href_rejects_unindexable_suffix(docs_root):
    write(docs_root / "image.png", "binary")
    src = docs_root / "index.html"
    src.touch()
    assert normalise_href("image.png", src.resolve(), docs_root.resolve()) is None


def test_normalise_href_accepts_relative_html(docs_root):
    write(docs_root / "page.html", "<html><body>x</body></html>")
    src = docs_root / "index.html"
    src.touch()
    result = normalise_href("page.html", src.resolve(), docs_root.resolve())
    assert result == (docs_root / "page.html").resolve()


def test_normalise_href_strips_fragment(docs_root):
    write(docs_root / "page.html", "<html><body>x</body></html>")
    src = docs_root / "index.html"
    src.touch()
    result = normalise_href("page.html#section-2", src.resolve(), docs_root.resolve())
    assert result == (docs_root / "page.html").resolve()


def test_normalise_href_rejects_dangling_link(docs_root, caplog):
    """A link target inside docs_root but missing on disk must be skipped.

    Real corpora routinely have stale links to deleted pages; the crawler
    must surface a warning and continue rather than crash on the missing
    file later during extraction.
    """
    src = docs_root / "index.html"
    src.touch()
    import logging

    with caplog.at_level(logging.WARNING):
        result = normalise_href("vanished.html", src.resolve(), docs_root.resolve())
    assert result is None
    assert any("dangling link" in rec.message for rec in caplog.records)


def test_normalise_href_url_decodes_percent_escapes(docs_root):
    """Filenames with spaces / apostrophes round-trip through URL-encoded hrefs.

    HTML generators emit ``Aiden%27s%20Schedule.html`` for files actually
    named ``Aiden's Schedule.html``; the crawler must unquote before
    touching the filesystem or it'll FileNotFoundError on the literal
    percent-encoded path.
    """
    target_name = "Aiden's Schedule and Rules.html"
    write(docs_root / target_name, "<html><body>x</body></html>")
    src = docs_root / "index.html"
    src.touch()
    encoded_href = "Aiden%27s%20Schedule%20and%20Rules.html"
    result = normalise_href(encoded_href, src.resolve(), docs_root.resolve())
    assert result == (docs_root / target_name).resolve()


def test_normalise_href_accepts_pdf(docs_root):
    write(docs_root / "manual.pdf", "%PDF")
    src = docs_root / "index.html"
    src.touch()
    result = normalise_href("manual.pdf", src.resolve(), docs_root.resolve())
    assert result is not None
    assert result.suffix == ".pdf"


# ---------------------------------------------------------------------------
# Crawl: link-following
# ---------------------------------------------------------------------------


def test_crawl_finds_linked_documents(docs_root, config):
    write(docs_root / "index.html", html_with_links("page_a.html", "page_b.html"))
    write(docs_root / "page_a.html", html_with_links("page_b.html"))
    write(docs_root / "page_b.html", html_with_links())

    docs = list(crawl(docs_root, config))
    assert _paths(docs) == {"index.html", "page_a.html", "page_b.html"}


def test_crawl_dedupes_circular_links(docs_root, config):
    write(docs_root / "index.html", html_with_links("a.html"))
    write(docs_root / "a.html", html_with_links("b.html"))
    write(docs_root / "b.html", html_with_links("a.html", "index.html"))

    docs = list(crawl(docs_root, config))
    # Three files, each yielded exactly once.
    assert len(docs) == 3
    assert _paths(docs) == {"index.html", "a.html", "b.html"}


def test_crawl_finds_unlinked_files_via_dirwalk(docs_root, config):
    write(docs_root / "index.html", html_with_links())  # no links
    write(docs_root / "orphan.html", "<html><body>orphan</body></html>")

    docs = list(crawl(docs_root, config))
    assert _paths(docs) == {"index.html", "orphan.html"}


def test_crawl_skips_non_html_files_in_dirwalk(docs_root, config):
    write(docs_root / "index.html", html_with_links())
    write(docs_root / "data.csv", "a,b,c")
    write(docs_root / "img.png", "binary")

    docs = list(crawl(docs_root, config))
    assert _paths(docs) == {"index.html"}


# ---------------------------------------------------------------------------
# Crawl: hostile hrefs do not exit docs_root
# ---------------------------------------------------------------------------


def test_crawl_does_not_follow_external_links(docs_root, config):
    write(
        docs_root / "index.html",
        html_with_links(
            "http://example.com/evil.html",
            "https://example.com/evil.html",
            "//cdn.example.com/x.html",
            "mailto:foo@bar.com",
            "javascript:void(0)",
            "tel:+15555550100",
            "#anchor",
        ),
    )
    docs = list(crawl(docs_root, config))
    assert _paths(docs) == {"index.html"}
    # links_to must be empty (all hostile hrefs rejected).
    assert docs[0].metadata["links_to"] == []


def test_crawl_does_not_follow_path_escape(docs_root, tmp_path, config):
    # Put a file outside docs_root that we must NEVER reach.
    outside = tmp_path / "outside.html"
    outside.write_text("<html><body>SECRET_OUTSIDE_DOCS</body></html>")

    write(
        docs_root / "index.html",
        html_with_links("../outside.html"),
    )
    docs = list(crawl(docs_root, config))
    assert all("SECRET_OUTSIDE_DOCS" not in d.content for d in docs)
    assert _paths(docs) == {"index.html"}


# ---------------------------------------------------------------------------
# Crawl: depth cap
# ---------------------------------------------------------------------------


def test_crawl_respects_max_file_depth(docs_root, tmp_path):
    """max_file_depth caps directory recursion; deeper files are skipped."""
    write(docs_root / "index.html", html_with_links("sub/level1.html"))
    write(docs_root / "sub" / "level1.html", html_with_links("../deep/level2.html"))
    write(docs_root / "deep" / "level2.html", html_with_links("nested/level3.html"))
    write(docs_root / "deep" / "nested" / "level3.html", "<html><body>too deep</body></html>")

    capped = ARGConfig(docs_root=docs_root, db_path=tmp_path / "arg_db", max_file_depth=1)
    docs = list(crawl(docs_root, capped))
    names = _paths(docs)
    # level3.html is at depth 2 (deep/nested/level3.html); should be excluded.
    assert "level3.html" not in names
    assert "level1.html" in names
    assert "level2.html" in names


# ---------------------------------------------------------------------------
# Link normalisation in Document.metadata["links_to"]
# ---------------------------------------------------------------------------


def test_links_to_normalised_to_absolute_paths(docs_root, config):
    write(docs_root / "index.html", html_with_links("a.html", "sub/b.html"))
    write(docs_root / "a.html", "<html><body>a</body></html>")
    write(docs_root / "sub" / "b.html", "<html><body>b</body></html>")

    docs = list(crawl(docs_root, config))
    index = next(d for d in docs if d.path.name == "index.html")
    links = set(index.metadata["links_to"])
    assert str((docs_root / "a.html").resolve()) in links
    assert str((docs_root / "sub" / "b.html").resolve()) in links


def test_links_to_records_unreadable_pdf_as_edge_only(docs_root, config):
    """Unreadable PDFs are skipped as Documents but still recorded as edges."""
    write(docs_root / "index.html", html_with_links("broken.pdf"))
    (docs_root / "broken.pdf").write_bytes(b"%PDF-1.4 not a real pdf")
    docs = list(crawl(docs_root, config))
    assert _paths(docs) == {"index.html"}
    index = docs[0]
    assert str((docs_root / "broken.pdf").resolve()) in index.metadata["links_to"]


def test_crawler_yields_pdf_document_for_valid_pdf(docs_root, config):
    """A valid PDF linked from index.html is yielded as a `file_type=pdf` Document."""
    import pymupdf as fitz

    write(docs_root / "index.html", html_with_links("manual.pdf"))
    pdf_path = docs_root / "manual.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "A long enough paragraph of native body text so pdfplumber wins "
        "and OCR is not triggered for this fixture document.",
    )
    doc.set_metadata({"title": "Manual"})
    doc.save(str(pdf_path))
    doc.close()

    docs = list(crawl(docs_root, config))
    pdf_docs = [d for d in docs if d.metadata.get("file_type") == "pdf"]
    assert len(pdf_docs) == 1
    assert pdf_docs[0].metadata["title"] == "Manual"
    assert pdf_docs[0].path == pdf_path.resolve()
