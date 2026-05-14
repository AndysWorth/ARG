"""ARG crawler package: recursive document crawler + HTML / PDF extractors."""

from arg.crawler.crawler import crawl, normalise_href
from arg.crawler.extractors import (
    DEFAULT_STRIP_SELECTORS,
    Document,
    extract_html,
)

__all__ = [
    "DEFAULT_STRIP_SELECTORS",
    "Document",
    "crawl",
    "extract_html",
    "normalise_href",
]
