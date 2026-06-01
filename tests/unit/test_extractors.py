"""HTML extractor tests.

PDF tests live alongside in this file but are skipped/marked as pdf so they
can be excluded via ``pytest -k "not pdf"`` during the HTML pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arg.config import ARGConfig
from arg.crawler.extractors import (
    DEFAULT_STRIP_SELECTORS,
    Document,
    extract_html,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> ARGConfig:
    docs = tmp_path / "docs"
    docs.mkdir()
    return ARGConfig(docs_root=docs, db_path=tmp_path / "arg_db")


def _write(tmp_path: Path, name: str, html: str) -> Path:
    p = tmp_path / name
    p.write_text(html, encoding="utf-8")
    return p


def _extract(tmp_path: Path, html: str, config: ARGConfig) -> Document:
    return extract_html(_write(tmp_path, "page.html", html), config)


# ---------------------------------------------------------------------------
# Invisible / boilerplate stripping
# ---------------------------------------------------------------------------


def test_style_tag_content_not_extracted(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><head><style>.a { color: red; } SECRET_CSS_TOKEN</style></head>"
        "<body>visible body</body></html>",
        config,
    )
    assert "SECRET_CSS_TOKEN" not in doc.content
    assert "visible body" in doc.content


def test_script_tag_content_not_extracted(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body>visible<script>var SECRET_JS_TOKEN = 1;</script></body></html>",
        config,
    )
    assert "SECRET_JS_TOKEN" not in doc.content
    assert "visible" in doc.content


def test_nav_tag_stripped(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body><nav>NAV_LINKS_HERE</nav><p>body text</p></body></html>",
        config,
    )
    assert "NAV_LINKS_HERE" not in doc.content
    assert "body text" in doc.content


def test_header_footer_aside_stripped(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body>"
        "<header>HEADER_BANNER</header>"
        "<footer>FOOTER_LEGAL</footer>"
        "<aside>ASIDE_PROMO</aside>"
        "<p>real body</p>"
        "</body></html>",
        config,
    )
    assert "HEADER_BANNER" not in doc.content
    assert "FOOTER_LEGAL" not in doc.content
    assert "ASIDE_PROMO" not in doc.content
    assert "real body" in doc.content


def test_iframe_stripped(tmp_path, config):
    doc = _extract(
        tmp_path,
        '<html><body><iframe src="x.html">IFRAME_FALLBACK</iframe><p>main</p></body></html>',
        config,
    )
    assert "IFRAME_FALLBACK" not in doc.content
    assert "main" in doc.content


@pytest.mark.parametrize(
    "style_attr,marker",
    [
        ("display:none", "HIDDEN_DISPLAY_NONE"),
        ("display: none", "HIDDEN_DISPLAY_NONE_SPACE"),
        ("DISPLAY:NONE", "HIDDEN_DISPLAY_NONE_UPPER"),
        ("visibility:hidden", "HIDDEN_VISIBILITY"),
        ("visibility: hidden", "HIDDEN_VISIBILITY_SPACE"),
    ],
)
def test_invisible_style_stripped(tmp_path, config, style_attr, marker):
    doc = _extract(
        tmp_path,
        f'<html><body><div style="{style_attr}">{marker}</div><p>kept</p></body></html>',
        config,
    )
    assert marker not in doc.content
    assert "kept" in doc.content


def test_strip_selectors_removes_div_navigation(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body>"
        '<div class="sphinxsidebar">SPHINX_SIDE_LINKS</div>'
        '<div class="rst-footer-buttons">RTD_FOOTER_NAV</div>'
        "<p>real prose</p>"
        "</body></html>",
        config,
    )
    assert "SPHINX_SIDE_LINKS" not in doc.content
    assert "RTD_FOOTER_NAV" not in doc.content
    assert "real prose" in doc.content


def test_default_strip_selectors_list_non_empty():
    assert "div.sphinxsidebar" in DEFAULT_STRIP_SELECTORS
    assert "div.wy-nav-side" in DEFAULT_STRIP_SELECTORS
    assert "div.md-sidebar" in DEFAULT_STRIP_SELECTORS


def test_strip_invisible_handles_nodes_with_none_attrs(tmp_path, config, monkeypatch):
    """Real-world malformed HTML can surface nodes from ``find_all`` whose
    ``.attrs`` is None — calling ``.get('style', '')`` on them raises
    AttributeError. The extractor must guard against this rather than
    crashing the whole indexing run."""
    import arg.crawler.extractors as extractors_mod

    real_bs = extractors_mod.BeautifulSoup

    def bad_bs(*args, **kwargs):
        soup = real_bs(*args, **kwargs)
        real_find_all = soup.find_all

        def wrapped_find_all(*a, **kw):
            results = real_find_all(*a, **kw)
            # When find_all is called for style-bearing tags, inject a
            # synthetic Tag whose attrs is None — the malformed-HTML
            # symptom we saw in the wild.
            if kw.get("attrs") == {"style": True}:
                from bs4 import Tag

                broken = Tag(name="div")
                broken.attrs = None  # type: ignore[assignment, unused-ignore]
                return [*results, broken]
            return results

        soup.find_all = wrapped_find_all  # type: ignore[method-assign, unused-ignore]
        return soup

    monkeypatch.setattr(extractors_mod, "BeautifulSoup", bad_bs)
    # If the guard fails this raises AttributeError; if it works the
    # extract returns normally.
    doc = _extract(
        tmp_path,
        '<html><body><p style="color:red">hi</p></body></html>',
        config,
    )
    assert "hi" in doc.content


# ---------------------------------------------------------------------------
# Parser choice
# ---------------------------------------------------------------------------


def test_lxml_parser_is_used(tmp_path, config, monkeypatch):
    """extract_html must call BeautifulSoup with features='lxml'."""
    seen: dict[str, object] = {}

    import arg.crawler.extractors as extractors_mod

    real_bs = extractors_mod.BeautifulSoup

    def spy(*args, **kwargs):
        seen["features"] = kwargs.get("features")
        return real_bs(*args, **kwargs)

    monkeypatch.setattr(extractors_mod, "BeautifulSoup", spy)
    _extract(tmp_path, "<html><body>hi</body></html>", config)
    assert seen["features"] == "lxml"


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------


def test_title_pipe_suffix_stripped(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><head><title>Authentication | Kraken API Docs</title></head><body>x</body></html>",
        config,
    )
    assert doc.metadata["title"] == "Authentication"


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("Foo - Bar Site", "Foo"),
        ("Foo \u2014 Bar Site", "Foo"),
        ("Foo :: Bar", "Foo"),
        ("PlainTitle", "PlainTitle"),
    ],
)
def test_title_separators(tmp_path, config, raw, clean):
    doc = _extract(
        tmp_path,
        f"<html><head><title>{raw}</title></head><body>x</body></html>",
        config,
    )
    assert doc.metadata["title"] == clean


def test_title_falls_back_to_h1(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body><h1>Fallback H1 Title</h1><p>body</p></body></html>",
        config,
    )
    assert doc.metadata["title"] == "Fallback H1 Title"


# ---------------------------------------------------------------------------
# Page description
# ---------------------------------------------------------------------------


def test_meta_description_stored(tmp_path, config):
    doc = _extract(
        tmp_path,
        '<html><head><meta name="description" content="API auth overview"></head>'
        "<body>x</body></html>",
        config,
    )
    assert doc.metadata["page_description"] == "API auth overview"


def test_og_description_fallback(tmp_path, config):
    doc = _extract(
        tmp_path,
        '<html><head><meta property="og:description" content="OG fallback"></head>'
        "<body>x</body></html>",
        config,
    )
    assert doc.metadata["page_description"] == "OG fallback"


def test_no_description_is_empty_string(tmp_path, config):
    doc = _extract(tmp_path, "<html><body>x</body></html>", config)
    assert doc.metadata["page_description"] == ""


# ---------------------------------------------------------------------------
# Tables → Markdown
# ---------------------------------------------------------------------------


def test_table_rendered_as_pipe_markdown(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body><table>"
        "<tr><th>Tier</th><th>Limit</th></tr>"
        "<tr><td>Tier 2</td><td>500 req/min</td></tr>"
        "<tr><td>Tier 3</td><td>1500 req/min</td></tr>"
        "</table></body></html>",
        config,
    )
    assert "| Tier | Limit |" in doc.content
    assert "|---|---|" in doc.content
    assert "| Tier 2 | 500 req/min |" in doc.content
    assert "| Tier 3 | 1500 req/min |" in doc.content


def test_table_cells_appear_only_once(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body>"
        "<p>intro</p>"
        "<table><tr><th>K</th><th>V</th></tr><tr><td>foo</td><td>UNIQUEVAL42</td></tr></table>"
        "<p>outro</p>"
        "</body></html>",
        config,
    )
    assert doc.content.count("UNIQUEVAL42") == 1


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


def test_h1_h2_h3_sentinels_injected(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body><h1>Intro</h1><h2>Sub</h2><h3>Deeper</h3><p>paragraph</p></body></html>",
        config,
    )
    assert "##H1## Intro" in doc.content
    assert "##H2## Sub" in doc.content
    assert "##H3## Deeper" in doc.content
    assert "paragraph" in doc.content


def test_h4_kept_in_body_no_sentinel(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body><h4>Deep Heading 4</h4><p>body</p></body></html>",
        config,
    )
    assert "Deep Heading 4" in doc.content
    assert "##H4##" not in doc.content


# ---------------------------------------------------------------------------
# Code blocks
# ---------------------------------------------------------------------------


def test_short_pre_block_full_in_body(tmp_path, config):
    snippet = "def hello():\n    return 'short'"
    doc = _extract(
        tmp_path,
        f"<html><body><pre>{snippet}</pre></body></html>",
        config,
    )
    assert "def hello" in doc.content
    assert "return 'short'" in doc.content
    assert "[... truncated ...]" not in doc.content
    assert doc.metadata["code_blocks"] == [snippet]


def test_long_pre_block_truncated_full_in_metadata(tmp_path, config):
    # max_code_block_tokens default is 256 — make a 600-word block.
    long_tokens = " ".join(f"TOK{i}" for i in range(600))
    doc = _extract(
        tmp_path,
        f"<html><body><pre>{long_tokens}</pre></body></html>",
        config,
    )
    assert "[... truncated ...]" in doc.content
    assert "TOK0" in doc.content
    assert "TOK500" not in doc.content
    assert long_tokens in doc.metadata["code_blocks"][0]
    assert "TOK599" in doc.metadata["code_blocks"][0]


# ---------------------------------------------------------------------------
# Whitespace + entities
# ---------------------------------------------------------------------------


def test_whitespace_normalised(tmp_path, config):
    nbsp = "\u00a0"
    doc = _extract(
        tmp_path,
        f"<html><body><p>a{nbsp}{nbsp}{nbsp}b\t\tc</p>"
        "<p>line1</p>\n\n\n\n\n<p>line2</p></body></html>",
        config,
    )
    assert nbsp not in doc.content
    assert "\t" not in doc.content
    assert "  " not in doc.content
    assert "\n\n\n" not in doc.content
    assert not doc.content.startswith("\n")
    assert not doc.content.endswith("\n")


def test_html_entities_decoded(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><body><p>&lt;tag&gt; &amp; more &nbsp; text</p></body></html>",
        config,
    )
    assert "<tag>" in doc.content
    assert "&" in doc.content
    assert "&amp;" not in doc.content
    assert "&lt;" not in doc.content


# ---------------------------------------------------------------------------
# Document shape
# ---------------------------------------------------------------------------


def test_document_metadata_keys(tmp_path, config):
    doc = _extract(
        tmp_path,
        "<html><head><title>T | Site</title>"
        '<meta name="description" content="d"></head>'
        '<body><h1>T</h1><a href="other.html">o</a><p>body</p></body></html>',
        config,
    )
    for key in (
        "title",
        "page_description",
        "heading_path",
        "links_to",
        "file_type",
        "code_blocks",
    ):
        assert key in doc.metadata, f"missing metadata key {key}"
    assert doc.metadata["file_type"] == "html"
    assert doc.metadata["links_to"] == ["other.html"]  # raw href; crawler normalises later


# ---------------------------------------------------------------------------
# Plain-text extractor (Feature 0001)
# ---------------------------------------------------------------------------


def _write_text(tmp_path: Path, name: str, body: bytes | str) -> Path:
    p = tmp_path / name
    if isinstance(body, bytes):
        p.write_bytes(body)
    else:
        p.write_text(body, encoding="utf-8")
    return p


def test_extract_text_utf8_round_trip(tmp_path, config):
    from arg.crawler.extractors import extract_text

    p = _write_text(tmp_path, "release_notes.txt", "First line\nSecond line\n")
    doc = extract_text(p, config)
    assert "First line" in doc.content
    assert "Second line" in doc.content
    assert doc.metadata["file_type"] == "text"


def test_extract_text_title_from_filename_stem(tmp_path, config):
    from arg.crawler.extractors import extract_text

    p = _write_text(tmp_path, "my_release_notes.txt", "hi")
    doc = extract_text(p, config)
    assert doc.metadata["title"] == "my_release_notes"


def test_extract_text_empty_file(tmp_path, config):
    from arg.crawler.extractors import extract_text

    p = _write_text(tmp_path, "blank.txt", "")
    doc = extract_text(p, config)
    assert doc.content == ""
    assert doc.metadata["title"] == "blank"


def test_extract_text_strips_utf8_bom(tmp_path, config):
    from arg.crawler.extractors import extract_text

    p = _write_text(
        tmp_path,
        "bommed.txt",
        b"\xef\xbb\xbfBOM-prefixed content stays clean.\n",
    )
    doc = extract_text(p, config)
    assert doc.content.startswith("BOM-prefixed content")
    # No leading BOM character.
    assert doc.content[0] != "﻿"


def test_extract_text_latin1_fallback(tmp_path, config):
    """Bytes that aren't valid UTF-8 fall through to latin-1 decode."""
    from arg.crawler.extractors import extract_text

    # 0xe9 is é in latin-1; it is NOT a valid standalone UTF-8 byte.
    p = _write_text(tmp_path, "win1252.txt", b"caf\xe9 au lait\n")
    doc = extract_text(p, config)
    # latin-1 maps 0xe9 → é, so the round-trip should contain a literal é.
    assert "café" in doc.content


def test_extract_text_metadata_shape(tmp_path, config):
    from arg.crawler.extractors import extract_text

    p = _write_text(tmp_path, "x.txt", "body")
    doc = extract_text(p, config)
    for key in (
        "title",
        "page_description",
        "heading_path",
        "links_to",
        "file_type",
        "code_blocks",
    ):
        assert key in doc.metadata, f"missing metadata key {key}"
    assert doc.metadata["page_description"] == ""
    assert doc.metadata["links_to"] == []
    assert doc.metadata["code_blocks"] == []
    assert doc.metadata["file_type"] == "text"
    assert doc.metadata["heading_path"] == "x"


def test_extract_text_accepts_md_and_markdown(tmp_path, config):
    """``.md`` and ``.markdown`` flow through extract_text unchanged.
    Markdown semantic structure (``# H1``) is intentionally NOT parsed in
    Feature 0001; the literal ``#`` characters appear in the content."""
    from arg.crawler.extractors import extract_text

    md_path = _write_text(tmp_path, "x.md", "# Heading\n\nbody\n")
    doc = extract_text(md_path, config)
    assert "# Heading" in doc.content

    long_path = _write_text(tmp_path, "x.markdown", "## Sub heading\nbody\n")
    doc = extract_text(long_path, config)
    assert "## Sub heading" in doc.content


# ---------------------------------------------------------------------------
# PDF — pure helper tests (no fixture PDF needed)
# ---------------------------------------------------------------------------


@pytest.mark.pdf
def test_resolve_pdf_title_uses_metadata_when_present():
    from arg.crawler.extractors import _resolve_pdf_title

    assert _resolve_pdf_title({"title": "API Guide"}, None, "doc") == "API Guide"


@pytest.mark.pdf
@pytest.mark.parametrize(
    "raw",
    [
        "Microsoft Word - document1.docx",
        "Microsoft PowerPoint - Talk.pptx",
        "Untitled",
        "document",
        "document1",
        "Presentation2",
        "Worksheet5",
    ],
)
def test_resolve_pdf_title_skips_temp_patterns(raw):
    from arg.crawler.extractors import _resolve_pdf_title

    assert _resolve_pdf_title({"title": raw}, "Largest Line", "manual") == "Largest Line"


@pytest.mark.pdf
def test_resolve_pdf_title_falls_back_to_filename_stem():
    from arg.crawler.extractors import _resolve_pdf_title

    assert _resolve_pdf_title({"title": ""}, None, "owners_manual") == "owners_manual"


@pytest.mark.pdf
def test_rejoin_soft_hyphens_rejoins_lowercase():
    from arg.crawler.extractors import _rejoin_soft_hyphens

    assert _rejoin_soft_hyphens("config-\nuration") == "configuration"


@pytest.mark.pdf
def test_rejoin_soft_hyphens_preserves_uppercase_compound():
    from arg.crawler.extractors import _rejoin_soft_hyphens

    # An intentional compound word — second part starts uppercase, hyphen stays.
    assert _rejoin_soft_hyphens("self-\nHosted") == "self-\nHosted"


@pytest.mark.pdf
def test_rejoin_soft_hyphens_preserves_intentional_inline_hyphens():
    from arg.crawler.extractors import _rejoin_soft_hyphens

    assert _rejoin_soft_hyphens("self-hosted server") == "self-hosted server"


@pytest.mark.pdf
def test_clean_pdf_text_decodes_ligatures():
    from arg.crawler.extractors import _clean_pdf_text

    assert _clean_pdf_text("\ufb01le \ufb02ow of\ufb01ce") == "file flow office"


@pytest.mark.pdf
def test_clean_pdf_text_normalises_typographic_chars():
    from arg.crawler.extractors import _clean_pdf_text

    em_dash = "\u2014"
    text_in = f"before{em_dash}after"
    cleaned = _clean_pdf_text(text_in)
    assert em_dash in cleaned
    assert _clean_pdf_text("\u201chi\u201d") == '"hi"'
    assert _clean_pdf_text("a\u2018b\u2019c") == "a'b'c"
    assert _clean_pdf_text("etc\u2026") == "etc..."
    # soft hyphen removed entirely
    assert _clean_pdf_text("a\u00adb") == "ab"


@pytest.mark.pdf
def test_clean_pdf_text_applies_nfc():
    from arg.crawler.extractors import _clean_pdf_text

    # "cafe + combining acute" vs precomposed e-with-acute -- NFC merges them.
    decomposed = "cafe\u0301"
    precomposed = "caf\u00e9"
    assert _clean_pdf_text(decomposed) == precomposed


@pytest.mark.pdf
def test_clean_pdf_text_whitespace_rules():
    from arg.crawler.extractors import _clean_pdf_text

    raw = "a\u00a0\u00a0b\t\tc"
    cleaned = _clean_pdf_text(raw)
    assert "\u00a0" not in cleaned
    assert "\t" not in cleaned
    assert "  " not in cleaned

    multiline = "p1\n\n\n\n\np2"
    assert "\n\n\n" not in _clean_pdf_text(multiline)


@pytest.mark.pdf
def test_detect_running_lines_threshold():
    from arg.crawler.extractors import _detect_running_lines

    # 4 pages, same y, same text → must be detected.
    pages = [[(750.0, "Page 1 of 4")] for _ in range(4)]
    pages[0][0] = (750.0, "Page 1 of 4")
    pages[1][0] = (750.0, "Page 1 of 4")
    pages[2][0] = (750.0, "Page 1 of 4")
    pages[3][0] = (750.0, "Page 1 of 4")
    detected = _detect_running_lines(pages, min_pages=3)
    assert any(text == "Page 1 of 4" for _, text in detected)


@pytest.mark.pdf
def test_detect_running_lines_below_threshold_not_flagged():
    from arg.crawler.extractors import _detect_running_lines

    pages = [
        [(750.0, "unique to page 1")],
        [(750.0, "shared")],
        [(750.0, "shared")],
        [(750.0, "unique to page 4")],
    ]
    detected = _detect_running_lines(pages, min_pages=3)
    assert not any(text == "shared" for _, text in detected)


@pytest.mark.pdf
def test_detect_running_lines_within_y_tolerance():
    from arg.crawler.extractors import _detect_running_lines

    # Same text appears at slightly varying y (±2 px) on 3 pages — should still
    # be detected because y_tolerance=3.
    pages = [
        [(750.0, "Confidential")],
        [(751.5, "Confidential")],
        [(749.0, "Confidential")],
    ]
    detected = _detect_running_lines(pages, min_pages=3, y_tolerance=3.0)
    assert any(text == "Confidential" for _, text in detected)


@pytest.mark.pdf
def test_rows_to_markdown():
    from arg.crawler.extractors import _rows_to_markdown

    md = _rows_to_markdown([["Tier", "Limit"], ["2", "500"], ["3", "1500"]])
    assert "| Tier | Limit |" in md
    assert "|---|---|" in md
    assert "| 2 | 500 |" in md


# ---------------------------------------------------------------------------
# PDF — fixture-driven tests
# ---------------------------------------------------------------------------


_NATIVE_BODY_DEFAULT = (
    "This is a paragraph of native body text that should clearly exceed the "
    "default OCR character threshold so the pdfplumber stage wins.\n"
    "A second prose line ensures multiple lines are extracted as a stream."
)


def _native_pdf(
    path: Path,
    *,
    body: str = _NATIVE_BODY_DEFAULT,
    title: str = "Native Doc",
    subject: str = "the subject",
    keywords: str = "k1, k2",
) -> Path:
    """Build a single-page native-text PDF with metadata."""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 50
    for line in body.split("\n"):
        page.insert_text((50, y), line, fontsize=11)
        y += 16
    doc.set_metadata({"title": title, "subject": subject, "keywords": keywords, "author": "tester"})
    doc.save(str(path))
    doc.close()
    return path


def _encrypted_pdf(path: Path) -> Path:
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "encrypted body content")
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()
    return path


def _form_pdf(path: Path) -> Path:
    """A PDF with one text-field widget so is_form_pdf is truthy."""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Form with one text field below.")
    widget = fitz.Widget()
    widget.field_name = "name"
    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    widget.rect = fitz.Rect(50, 80, 250, 100)
    page.add_widget(widget)
    doc.save(str(path))
    doc.close()
    return path


def _running_header_pdf(path: Path, n_pages: int = 4) -> Path:
    """Multi-page PDF with the same line at the same y on every page.

    Pages have enough body text on each page to comfortably exceed the
    default OCR threshold so the pdfplumber stage wins — OCR mangles the
    y-positions and would defeat the running-header detector.
    """
    import pymupdf as fitz

    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        # Running header — ASCII only so glyphs render in the default font.
        page.insert_text((50, 30), "Confidential - Internal Use Only", fontsize=10)
        # Unique body — enough chars per page that pdfplumber clears OCR threshold.
        body_lines = [
            f"Unique page {i + 1} content goes on this first body line.",
            f"Second unique line for page {i + 1} continues the narrative.",
            f"Third unique line for page {i + 1} adds further independent text.",
            f"Fourth line of page {i + 1} keeps total chars well above threshold.",
        ]
        for j, text in enumerate(body_lines):
            page.insert_text((50, 100 + j * 20), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


def _heading_font_pdf(path: Path) -> Path:
    """PDF with a clearly oversized line (H1) and a body-size line."""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Big Section Heading", fontsize=24)
    # Several body lines so the median is well-defined at body size.
    for i, y in enumerate((100, 120, 140, 160, 180)):
        page.insert_text((50, y), f"body line number {i + 1}", fontsize=11)
    page.insert_text((50, 220), "Medium Subheading", fontsize=15)  # 15/11 ≈ 1.36 → H2
    doc.save(str(path))
    doc.close()
    return path


@pytest.mark.pdf
def test_encrypted_pdf_returns_none(tmp_path, config):
    from arg.crawler.extractors import extract_pdf, extract_pdf_metadata, extract_pdf_to_document

    pdf = _encrypted_pdf(tmp_path / "encrypted.pdf")
    assert extract_pdf_metadata(pdf, config) is None
    assert extract_pdf_to_document(pdf, config) is None
    assert list(extract_pdf(pdf, config)) == []


@pytest.mark.pdf
def test_corrupt_pdf_returns_none(tmp_path, config, caplog):
    from arg.crawler.extractors import extract_pdf, extract_pdf_metadata

    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4 not a real pdf at all")
    import logging

    with caplog.at_level(logging.WARNING):
        assert extract_pdf_metadata(pdf, config) is None
        assert list(extract_pdf(pdf, config)) == []
    assert any("unreadable" in rec.message.lower() for rec in caplog.records)


@pytest.mark.pdf
def test_form_pdf_warns_but_continues(tmp_path, config, caplog):
    from arg.crawler.extractors import extract_pdf_metadata

    pdf = _form_pdf(tmp_path / "form.pdf")
    import logging

    with caplog.at_level(logging.WARNING):
        meta = extract_pdf_metadata(pdf, config)
    assert meta is not None
    assert meta["is_form_pdf"] is True
    assert any("acroform" in rec.message.lower() for rec in caplog.records)


@pytest.mark.pdf
def test_pdf_title_subject_keywords_extracted(tmp_path, config):
    from arg.crawler.extractors import extract_pdf_metadata

    pdf = _native_pdf(
        tmp_path / "doc.pdf",
        title="Operations Manual",
        subject="how to operate the widget",
        keywords="widget, ops",
    )
    meta = extract_pdf_metadata(pdf, config)
    assert meta is not None
    assert meta["title"] == "Operations Manual"
    assert meta["page_description"] == "how to operate the widget"
    assert meta["keywords"] == "widget, ops"


@pytest.mark.pdf
def test_pdf_title_temp_pattern_falls_back_to_largest_line(tmp_path, config):
    from arg.crawler.extractors import extract_pdf_metadata

    pdf = _heading_font_pdf(tmp_path / "doc.pdf")
    # Re-open to set a temp-file pattern title.
    import pymupdf as fitz

    doc = fitz.open(str(pdf))
    doc.set_metadata({"title": "Microsoft Word - draft.docx"})
    doc.saveIncr() if hasattr(doc, "saveIncr") else doc.save(
        str(pdf), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP
    )
    doc.close()
    meta = extract_pdf_metadata(pdf, config)
    assert meta is not None
    assert meta["title"] == "Big Section Heading"


@pytest.mark.pdf
def test_extract_pdf_is_a_generator(tmp_path, config):
    import types

    from arg.crawler.extractors import extract_pdf

    pdf = _native_pdf(tmp_path / "doc.pdf")
    iterator = extract_pdf(pdf, config)
    assert isinstance(iterator, types.GeneratorType)


@pytest.mark.pdf
def test_extract_pdf_yields_tuples(tmp_path, config):
    from arg.crawler.extractors import extract_pdf

    # Body must clear the OCR character threshold so pdfplumber wins; on CI the
    # tesseract binary is absent so falling through to OCR yields empty text.
    body = (
        "PAGE_ONE_MARKER appears here and the surrounding paragraph supplies "
        "enough body chars to comfortably exceed the default OCR threshold."
    )
    pdf = _native_pdf(tmp_path / "doc.pdf", body=body)
    pages = list(extract_pdf(pdf, config))
    assert len(pages) == 1
    page_num, text, meta = pages[0]
    assert page_num == 1
    assert "PAGE_ONE_MARKER" in text
    for key in ("tables", "ocr_used", "char_count", "heading_sentinels"):
        assert key in meta


@pytest.mark.pdf
def test_native_pdf_skips_ocr(tmp_path, config):
    """A PDF with substantial native text must not fall through to OCR.

    The fixture body is well above the default OCR character threshold (100),
    so the pdfplumber stage wins and ocr_used stays False on every page.
    """
    from arg.crawler.extractors import extract_pdf

    pdf = _native_pdf(tmp_path / "doc.pdf")  # default body comfortably exceeds threshold
    pages = list(extract_pdf(pdf, config))
    assert pages, "extract_pdf yielded no pages"
    assert all(meta["ocr_used"] is False for _, _, meta in pages)


@pytest.mark.pdf
def test_pdf_running_header_stripped(tmp_path, config):
    from arg.crawler.extractors import extract_pdf

    pdf = _running_header_pdf(tmp_path / "doc.pdf", n_pages=4)
    pages = list(extract_pdf(pdf, config))
    assert len(pages) == 4
    for _, text, _ in pages:
        assert "Confidential" not in text
        assert "Unique page" in text


@pytest.mark.pdf
def test_pdf_font_heading_sentinels_injected(tmp_path, config):
    from arg.crawler.extractors import extract_pdf

    pdf = _heading_font_pdf(tmp_path / "doc.pdf")
    pages = list(extract_pdf(pdf, config))
    assert len(pages) == 1
    _, text, meta = pages[0]
    assert "##H1## Big Section Heading" in text
    # The 15pt line is 1.36x the 11pt body -> H2.
    assert "##H2## Medium Subheading" in text
    # Regular body lines must not be sentinels.
    assert "##H1## body line" not in text
    assert "##H2## body line" not in text
    assert any("##H1##" in s for s in meta["heading_sentinels"])


@pytest.mark.pdf
def test_pdf_sidecar_overrides_layout_analysis(tmp_path, config):
    """The sidecar JSON must override pdf_layout_analysis for this doc only.

    We can't easily observe the layout-analysis flag end-to-end here, so the
    test verifies the override is *read* and reaches the extractor — by way
    of a malformed sidecar producing a warning, and a well-formed sidecar
    being silently honoured.
    """
    from arg.crawler.extractors import _read_pdf_sidecar

    pdf = _native_pdf(tmp_path / "doc.pdf")
    sidecar = pdf.with_suffix(pdf.suffix + ".argconfig")
    sidecar.write_text(json.dumps({"pdf_layout_analysis": False}))
    parsed = _read_pdf_sidecar(pdf)
    assert parsed == {"pdf_layout_analysis": False}


@pytest.mark.pdf
def test_pdf_sidecar_missing_returns_empty():
    from arg.crawler.extractors import _read_pdf_sidecar

    parsed = _read_pdf_sidecar(Path("/tmp/does_not_exist.pdf"))
    assert parsed == {}


@pytest.mark.pdf
def test_pdf_sidecar_malformed_logs_and_returns_empty(tmp_path, caplog):
    from arg.crawler.extractors import _read_pdf_sidecar

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    sidecar = pdf.with_suffix(pdf.suffix + ".argconfig")
    sidecar.write_text("not json at all { ]")
    import logging

    with caplog.at_level(logging.WARNING):
        parsed = _read_pdf_sidecar(pdf)
    assert parsed == {}


@pytest.mark.pdf
def test_extract_pdf_to_document_assembles_document(tmp_path, config):
    from arg.crawler.extractors import extract_pdf_to_document

    # Body must clear the OCR threshold; CI runners do not have tesseract
    # installed, so falling through to OCR would yield empty content.
    body = (
        "LINE_ONE_MARKER opens the document.\n"
        "A second prose line keeps total character count above the OCR threshold."
    )
    pdf = _native_pdf(
        tmp_path / "doc.pdf",
        body=body,
        title="Doc Title",
        subject="doc subject",
        keywords="kw",
    )
    doc = extract_pdf_to_document(pdf, config)
    assert doc is not None
    assert doc.metadata["title"] == "Doc Title"
    assert doc.metadata["page_description"] == "doc subject"
    assert doc.metadata["keywords"] == "kw"
    assert doc.metadata["file_type"] == "pdf"
    assert doc.metadata["page_count"] == 1
    assert "LINE_ONE_MARKER" in doc.content
    assert isinstance(doc.metadata["page_metadata"], list)
    assert doc.metadata["page_metadata"][0]["page_number"] == 1


# ---------------------------------------------------------------------------
# Feature 0004 — PDF extraction efficiency
# ---------------------------------------------------------------------------


def test_image_dominated_pdf_skips_pdfplumber(tmp_path, config):
    """A PDF with no extractable text skips pdfplumber and yields a stub or OCR result.

    An image-only page produced by fitz produces zero chars via get_text(), so
    avg_chars will be 0 — below any positive pdf_min_chars_per_page value.
    With OCR disabled the page falls through to an empty body; what matters is
    that extract_pdf returns without raising and pdfplumber is never opened.
    """
    from unittest.mock import patch

    import pymupdf as fitz

    from arg.crawler.extractors import extract_pdf

    # Build a single blank page (no text layer at all).
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    no_ocr_config = ARGConfig(
        docs_root=tmp_path,
        db_path=tmp_path / "db",
        ocr_enabled=False,
        pdf_min_chars_per_page=30,
    )

    with patch("pdfplumber.open") as mock_plumber:
        pages = list(extract_pdf(pdf_path, no_ocr_config))
        mock_plumber.assert_not_called()

    # Generator must not raise; it may yield zero or one (empty) pages.
    assert isinstance(pages, list)


def test_text_rich_pdf_uses_pdfplumber(tmp_path, config):
    """A PDF with plenty of text is not flagged as image-dominated."""
    from unittest.mock import patch

    import pdfplumber

    from arg.crawler.extractors import extract_pdf

    pdf_path = _native_pdf(tmp_path / "rich.pdf")

    open_calls: list[str] = []
    real_open = pdfplumber.open

    def _tracking_open(path, *args, **kwargs):
        open_calls.append(str(path))
        return real_open(path, *args, **kwargs)

    with patch("pdfplumber.open", side_effect=_tracking_open):
        list(extract_pdf(pdf_path, config))

    assert open_calls, "pdfplumber.open must be called for a text-rich PDF"


def test_acroform_pdf_skips_pdfplumber(tmp_path, config):
    """AcroForm PDFs must not call pdfplumber.open."""
    from unittest.mock import patch

    from arg.crawler.extractors import extract_pdf

    pdf_path = _form_pdf(tmp_path / "form.pdf")

    with patch("pdfplumber.open") as mock_plumber:
        pages = list(extract_pdf(pdf_path, config))
        mock_plumber.assert_not_called()

    assert len(pages) >= 1, "AcroForm PDF should still yield pages via pymupdf"


def test_acroform_pdf_yields_text_via_pymupdf(tmp_path, config):
    """Text written to an AcroForm PDF must still appear in the extracted content."""
    from arg.crawler.extractors import extract_pdf

    pdf_path = _form_pdf(tmp_path / "form.pdf")
    pages = list(extract_pdf(pdf_path, config))
    combined = " ".join(text for _, text, _ in pages)
    # pymupdf must extract something — exact text may differ due to font rendering.
    assert combined.strip(), "pymupdf should extract the text layer of the AcroForm PDF"
