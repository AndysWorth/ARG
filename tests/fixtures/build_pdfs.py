"""Build the three fixture PDFs Section 12 expects.

Idempotent — the script overwrites the targets every time it runs, so the
binary diff stays minimal across regenerations. Run this once when fixtures
change:

    .venv/bin/python tests/fixtures/build_pdfs.py

Produces:

  tests/fixtures/docs/manual.pdf
      Native-text PDF (pdfplumber primary path). 3 pages of body text
      covering Kraken API auth / rate limits / errors. Page 2 carries a
      rate-limit table drawn with grid lines so pdfplumber recognises it.
      Every page carries a "Kraken API Docs - Confidential" running footer
      at the same y-coordinate (exercises Step 0e running-line stripping).
      Page 1 has a large-font section heading (exercises font-based H1
      detection).

  tests/fixtures/docs/scanned_notice.pdf
      Image-only PDF. Body text is rendered to a pixmap and embedded as
      an image so the page has no native text layer. PDF /Title metadata
      is deliberately set to "Microsoft Word - document1.docx" to trigger
      the temp-file title-fallback path (extract_pdf_metadata then falls
      back to the filename stem).

  tests/fixtures/docs/encrypted_notice.pdf
      AES-256 password-protected PDF. _open_pdf must refuse it and the
      crawler must skip it silently with a warning log entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf as fitz

DOCS = Path(__file__).parent / "docs"

# ---------------------------------------------------------------------------
# manual.pdf
# ---------------------------------------------------------------------------

_MANUAL_BODY_PAGE_1 = [
    "This manual covers the Kraken API in full. The API authenticates",
    "every request with either an API key or an OAuth bearer token.",
    "API keys are issued from the developer dashboard and must be sent",
    "in the X-Kraken-Key header on every request. Keys expire after 90",
    "days unless rotated.",
    "",
    "Production integrations use OAuth 2.0 with the authorization-code",
    "grant. Access tokens expire after 3600 seconds; refresh tokens last",
    "30 days. Both are issued by POST /v2/oauth/token.",
    "",
    "All endpoints require an authenticated caller. Unauthenticated",
    "requests are rejected with HTTP 401 and the KRK-001 error code.",
]

_MANUAL_BODY_PAGE_2 = [
    "Kraken API accounts are bucketed by tier. Each tier bounds the",
    "number of requests per minute, requests per day, and the burst",
    "capacity. When a caller exceeds its bound the server responds",
    "with HTTP 429 and an X-Rate-Limit-Retry-After header carrying",
    "the number of seconds to wait before retrying.",
    "",
    "Per-tier limits are listed in the table below. The Free tier is",
    "intended for evaluation only; production traffic should run on",
    "Tier 2 or Tier 3.",
]

_MANUAL_BODY_PAGE_3 = [
    "Every Kraken API endpoint returns standard HTTP status codes.",
    "Authentication failures surface as 401, missing permissions as",
    "403, missing resources as 404, and rate-limit exhaustion as 429.",
    "Internal errors surface as 500.",
    "",
    "On any 4xx response, the JSON body includes a Kraken-specific",
    "error_code field. Common codes include KRK-001 (invalid API key",
    "format), KRK-002 (key expired), KRK-003 (OAuth token expired),",
    "KRK-101 (missing required field), and KRK-201 (rate limit",
    "exceeded for this account tier).",
]


def _draw_running_footer(page: fitz.Page) -> None:
    """Same y on every page — exercises Step 0e header/footer detection."""
    page.insert_text((50, 780), "Kraken API Docs - Confidential", fontsize=9)


def _draw_body(page: fitz.Page, lines: list[str], y0: int = 100) -> None:
    for i, line in enumerate(lines):
        page.insert_text((50, y0 + i * 16), line, fontsize=11)


def _draw_rate_limit_table(page: fitz.Page) -> None:
    """Pdfplumber detects tables from rendered grid lines + text positions."""
    table_left = 50
    table_top = 320
    col_widths = [70, 130, 130, 70]
    rows = [
        ["Tier", "Reqs/min", "Reqs/day", "Burst"],
        ["Free", "60", "10000", "120"],
        ["Tier 2", "500", "200000", "1000"],
        ["Tier 3", "1500", "2000000", "3000"],
    ]
    row_height = 22
    # Grid lines.
    for i in range(len(rows) + 1):
        y = table_top + i * row_height
        page.draw_line(
            (table_left, y),
            (table_left + sum(col_widths), y),
        )
    x = table_left
    for w in [*col_widths]:
        page.draw_line((x, table_top), (x, table_top + len(rows) * row_height))
        x += w
    page.draw_line(
        (x, table_top),
        (x, table_top + len(rows) * row_height),
    )
    # Cell text.
    for r_idx, row in enumerate(rows):
        x = table_left + 6
        for c_idx, cell in enumerate(row):
            page.insert_text(
                (x, table_top + r_idx * row_height + 15),
                cell,
                fontsize=10,
            )
            x += col_widths[c_idx]


def build_manual_pdf(target: Path) -> None:
    doc = fitz.open()

    # Page 1 — section heading in large font + auth body.
    p = doc.new_page()
    _draw_running_footer(p)
    p.insert_text((50, 60), "Authentication", fontsize=22)
    _draw_body(p, _MANUAL_BODY_PAGE_1)

    # Page 2 — rate limits + table.
    p = doc.new_page()
    _draw_running_footer(p)
    p.insert_text((50, 60), "Rate Limits", fontsize=22)
    _draw_body(p, _MANUAL_BODY_PAGE_2)
    _draw_rate_limit_table(p)

    # Page 3 — error codes.
    p = doc.new_page()
    _draw_running_footer(p)
    p.insert_text((50, 60), "Error Codes", fontsize=22)
    _draw_body(p, _MANUAL_BODY_PAGE_3)

    doc.set_metadata(
        {
            "title": "Kraken API Full Manual",
            "subject": (
                "Complete reference for the Kraken API including auth, rate limits, and errors"
            ),
            "keywords": "kraken, api, auth, rate limits, errors",
        }
    )
    doc.save(str(target))
    doc.close()


# ---------------------------------------------------------------------------
# scanned_notice.pdf — text rendered to image so OCR is required
# ---------------------------------------------------------------------------


_SCANNED_NOTICE_TEXT = [
    "OPERATIONAL NOTICE",
    "",
    "Scheduled maintenance window for the Kraken API will run from",
    "02:00 to 03:30 UTC on the first Sunday of every month. During",
    "the window, all authenticated endpoints return HTTP 503 with",
    "the maintenance error code KRK-503.",
    "",
    "Account holders on the Free tier may experience longer recovery",
    "times after the window closes; Tier 2 and Tier 3 customers are",
    "prioritised when the API resumes.",
]


def build_scanned_pdf(target: Path) -> None:
    """Render text to an image and embed it as the page content."""
    # First, render text into a pixmap using pymupdf.
    width_pt, height_pt = 612, 792  # US Letter
    # Scratch doc for text-to-image rendering.
    scratch = fitz.open()
    scratch_page = scratch.new_page(width=width_pt, height=height_pt)
    scratch_page.insert_text((40, 80), "OPERATIONAL NOTICE", fontsize=20)
    for i, line in enumerate(_SCANNED_NOTICE_TEXT[2:]):
        scratch_page.insert_text((40, 130 + i * 22), line, fontsize=14)
    # Rasterise at 120 DPI — high enough for OCR; low enough to stay under
    # the pre-commit large-file cap. JPEG keeps the file size in check too.
    pix = scratch_page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
    img_bytes = pix.tobytes("jpeg", jpg_quality=70)
    scratch.close()

    # Real doc: blank page with the image inserted.
    doc = fitz.open()
    page = doc.new_page(width=width_pt, height=height_pt)
    page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), stream=img_bytes)

    doc.set_metadata(
        {
            # Temp-file pattern — extractor should reject and fall back
            # to the filename stem.
            "title": "Microsoft Word - document1.docx",
            "subject": "",
        }
    )
    doc.save(str(target))
    doc.close()


# ---------------------------------------------------------------------------
# encrypted_notice.pdf — AES-256 password-protected
# ---------------------------------------------------------------------------


def build_encrypted_pdf(target: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 80),
        "Encrypted notice content; never reachable without the password.",
        fontsize=12,
    )
    doc.set_metadata({"title": "Encrypted Kraken Notice", "subject": "Confidential."})
    doc.save(
        str(target),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    build_manual_pdf(DOCS / "manual.pdf")
    build_scanned_pdf(DOCS / "scanned_notice.pdf")
    build_encrypted_pdf(DOCS / "encrypted_notice.pdf")
    for name in ("manual.pdf", "scanned_notice.pdf", "encrypted_notice.pdf"):
        size = (DOCS / name).stat().st_size
        print(f"  wrote {name} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
