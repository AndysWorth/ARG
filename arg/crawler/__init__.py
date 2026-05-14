"""ARG crawler package: recursive document crawler + HTML / PDF extractors."""

from arg.crawler.crawler import crawl, normalise_href
from arg.crawler.extractors import (
    DEFAULT_STRIP_SELECTORS,
    Document,
    extract_html,
    extract_pdf,
    extract_pdf_metadata,
    extract_pdf_to_document,
)

__all__ = [
    "DEFAULT_STRIP_SELECTORS",
    "Document",
    "crawl",
    "extract_html",
    "extract_pdf",
    "extract_pdf_metadata",
    "extract_pdf_to_document",
    "normalise_href",
]
