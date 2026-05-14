"""HTML extractor tests.

PDF tests live alongside in this file but are skipped/marked as pdf so they
can be excluded via ``pytest -k "not pdf"`` during the HTML pass.
"""

from __future__ import annotations

from collections.abc import Iterator
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
        ("Foo — Bar Site", "Foo"),
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
# PDF stub
# ---------------------------------------------------------------------------


@pytest.mark.pdf
def test_pdf_extractor_not_yet_implemented(tmp_path, config):
    """Pass-1 sentinel: extract_pdf must raise NotImplementedError until pass 2 lands."""
    from arg.crawler.extractors import extract_pdf

    fake = tmp_path / "x.pdf"
    fake.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.raises(NotImplementedError):
        # extract_pdf is a generator; force it to start by calling next().
        iterator: Iterator = extract_pdf(fake, config)
        next(iterator)
