"""Recursive document crawler.

The crawler is the boundary between the file system and the rest of ARG. It
turns a docs root directory into an ordered stream of `Document` objects:

  1. Resolves the root and walks the link graph starting at ``index.html`` (if
     present), then performs a directory walk to pick up any indexable files
     not reachable via links.
  2. Normalises every ``<a href>`` it sees, refusing to follow anchors,
     mailto/javascript/tel/ftp/protocol-relative URLs, http(s) externals, and
     paths that escape ``docs_root`` (path-escape attacks).
  3. De-duplicates files by their resolved absolute path so circular links
     (A → B → A) don't loop.
  4. Caps recursion via ``config.max_file_depth``.

PDF files are also yielded as Documents (since Section 5 pass 2). Encrypted
or unreadable PDFs are skipped silently after a warning is logged — they do
not abort the crawl.

Locality
--------
Every following decision goes through `normalise_href`, which strips http/https
schemes and any path that escapes ``docs_root``. Combined with the
``ARGConfig.ollama_base_url`` localhost check, this is one of the layers that
makes ARG's locality guarantee hold even when the corpus contains links to the
public internet.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

from arg.config import ARGConfig
from arg.crawler.extractors import Document, extract_html, extract_pdf_to_document, extract_text

logger = logging.getLogger(__name__)

_HTML_SUFFIXES: frozenset[str] = frozenset({".html", ".htm"})
_PDF_SUFFIXES: frozenset[str] = frozenset({".pdf"})
# Plain-text suffix set per Feature 0001. Kept separate from HTML so the
# crawler's dispatch makes it obvious that text doesn't go through
# BeautifulSoup. .markdown accepted alongside .md (Markdown structure is
# NOT parsed; the file flows through extract_text as plain text — see
# docs/features/0001-plain-text-indexing.md).
_TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".markdown"})
_INDEXABLE_SUFFIXES: frozenset[str] = _HTML_SUFFIXES | _PDF_SUFFIXES | _TEXT_SUFFIXES
_SKIP_SCHEMES: frozenset[str] = frozenset({"mailto", "javascript", "tel", "ftp", "http", "https"})


def normalise_href(href: str, source_path: Path, docs_root: Path) -> Path | None:
    """Resolve a raw ``<a href>`` to an absolute Path inside ``docs_root``.

    Returns ``None`` if the href should be skipped per the Section 5 spec.
    Both ``source_path`` and ``docs_root`` must already be absolute.
    """
    if not href:
        return None
    href = href.strip()
    if not href:
        return None

    # Anchor-only link.
    if href.startswith("#"):
        return None

    # Protocol-relative (//cdn.example.com/...). Always external; never a local file.
    if href.startswith("//"):
        return None

    # `urlsplit` happily parses bare paths (scheme == "") and absolute file URIs.
    parts = urlsplit(href)
    scheme = parts.scheme.lower()
    if scheme in _SKIP_SCHEMES:
        return None
    # Any other non-empty scheme we don't recognise → external, skip.
    if scheme and scheme != "file":
        return None

    # Fragments are discarded — they don't change the target file.
    # URL-decode percent escapes so on-disk filenames with spaces, apostrophes,
    # or other URL-unsafe characters resolve correctly. HTML generators
    # routinely emit hrefs like ``Aiden%27s%20Schedule.html`` for files
    # actually named ``Aiden's Schedule.html``.
    target = unquote(parts.path)
    if not target:
        return None

    # Resolve relative to the source file's directory; absolute paths are honoured.
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = source_path.parent / candidate
    resolved = candidate.resolve()

    # Path-escape guard: must be inside docs_root.
    try:
        resolved.relative_to(docs_root)
    except ValueError:
        return None

    if resolved.suffix.lower() not in _INDEXABLE_SUFFIXES:
        return None

    # Dangling links: the href resolves to a path inside docs_root but no
    # file exists there. Real corpora routinely carry stale links; treat
    # them as graph edges that simply don't have a target Document.
    if not resolved.is_file():
        logger.warning("crawler: dangling link from %s -> %s", source_path, resolved)
        return None

    return resolved


def _relative_dir_depth(path: Path, docs_root: Path) -> int:
    """Number of directory levels under ``docs_root`` (0 for top-level files)."""
    try:
        rel = path.relative_to(docs_root)
    except ValueError:
        return -1
    return len(rel.parts) - 1


def crawl(docs_root: Path, config: ARGConfig) -> Iterator[Document]:
    """Walk ``docs_root`` and yield `Document` objects.

    Yields HTML documents in BFS order starting from ``index.html`` (if it
    exists), interleaving PDF documents at the point they are first reached
    via a link, then any unreached indexable files discovered by directory
    walk in sorted order. ``links_to`` in each Document's metadata is
    normalised to a list of absolute path strings — non-followable hrefs
    are dropped.

    Encrypted or unreadable PDFs are skipped silently after a warning.
    """
    docs_root = docs_root.resolve()
    if not docs_root.is_dir():
        raise NotADirectoryError(f"docs_root is not a directory: {docs_root}")

    seen: set[Path] = set()
    queue: deque[Path] = deque()

    index = docs_root / "index.html"
    if index.is_file():
        queue.append(index)

    while queue:
        path = queue.popleft()
        if path in seen:
            continue
        seen.add(path)

        if _relative_dir_depth(path, docs_root) > config.max_file_depth:
            logger.debug("crawler: skipping %s (depth exceeds max_file_depth)", path)
            continue

        doc = _extract_for_path(path, config)
        if doc is None:
            continue
        doc.metadata["links_to"] = _resolve_links(
            doc.metadata.get("links_to", []), path, docs_root, seen, queue
        )
        yield doc

    # Directory walk for files unreachable from index.html.
    for path in sorted(docs_root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in _INDEXABLE_SUFFIXES:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        if _relative_dir_depth(resolved, docs_root) > config.max_file_depth:
            continue
        seen.add(resolved)
        doc = _extract_for_path(resolved, config)
        if doc is None:
            continue
        doc.metadata["links_to"] = _resolve_links(
            doc.metadata.get("links_to", []), resolved, docs_root, seen, queue
        )
        yield doc


def _extract_for_path(path: Path, config: ARGConfig) -> Document | None:
    suffix = path.suffix.lower()
    try:
        if suffix in _HTML_SUFFIXES:
            return extract_html(path, config)
        if suffix in _PDF_SUFFIXES:
            return extract_pdf_to_document(path, config)
        if suffix in _TEXT_SUFFIXES:
            return extract_text(path, config)
    except FileNotFoundError as exc:
        # A link or directory walk pointed at a path that's no longer there
        # (race with the watcher, broken symlink, etc.).
        logger.warning("crawler: file disappeared during extraction: %s (%s)", path, exc)
    except Exception as exc:
        # Malformed HTML / PDF that the extractor can't handle. Real-world
        # corpora carry these; one bad file must not kill the whole index.
        logger.exception("crawler: extractor failed for %s (%s); skipping", path, exc)
    return None


def _resolve_links(
    raw_hrefs: list[str],
    source_path: Path,
    docs_root: Path,
    seen: set[Path],
    queue: deque[Path],
) -> list[str]:
    """Normalise hrefs and enqueue followable HTML targets for traversal.

    Returns the list of resolved absolute-path strings for the source's
    ``links_to`` metadata (PDFs included — they are link-graph edges even when
    the extractor for them hasn't landed).
    """
    out: list[str] = []
    for href in raw_hrefs:
        resolved = normalise_href(href, source_path, docs_root)
        if resolved is None:
            continue
        out.append(str(resolved))
        if resolved.suffix.lower() in _HTML_SUFFIXES and resolved not in seen:
            queue.append(resolved)
    return out
