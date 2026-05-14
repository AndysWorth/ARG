"""Live filesystem watcher.

`DocsWatcher` runs a `watchdog` `Observer` in a background thread and calls a
single user-supplied callback when an indexable file is created, modified, or
deleted under ``docs_root``. Events are debounced per-path so rapid editor
saves coalesce into one re-index trigger.

Locality
--------
``watchdog`` watches the local filesystem only — no network sockets are opened
or polled. The class never touches anything outside the configured
``docs_root``.

API
---
Two layers are exported:

  * :class:`DocsWatcher` — production entry point. Construct with the project
    config and a callback ``(path, event_kind)``; call ``start()`` to begin
    watching and ``stop()`` to tear down cleanly. ``__enter__``/``__exit__``
    make it usable as a context manager.
  * :class:`_DocsEventHandler` — the underlying ``watchdog`` handler. Exposed
    on the watcher as ``watcher.handler`` so unit tests can drive synthetic
    events without spinning up an Observer thread.

Lifecycle
---------
:meth:`stop` is idempotent — pipelines call it from signal handlers and from
``pipeline.close``, so it is expected to handle "already stopped" gracefully.
Pending debounce timers are cancelled in :meth:`stop` so a callback firing
after ``stop()`` returned won't surprise the caller.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from arg.config import ARGConfig

logger = logging.getLogger(__name__)

# Event kinds emitted to the user callback. Mirrors `watchdog`'s event types
# but reduced to the three the indexer actually cares about.
EVENT_CREATED = "created"
EVENT_MODIFIED = "modified"
EVENT_DELETED = "deleted"

_INDEXABLE_SUFFIXES: frozenset[str] = frozenset({".html", ".htm", ".pdf"})

# Type alias for the user-supplied callback signature.
WatchCallback = Callable[[Path, str], None]


class _DocsEventHandler(FileSystemEventHandler):
    """Internal `watchdog` handler that forwards filtered events to the watcher."""

    def __init__(self, watcher: DocsWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._watcher._schedule(Path(str(event.src_path)), EVENT_CREATED)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._watcher._schedule(Path(str(event.src_path)), EVENT_MODIFIED)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._watcher._schedule(Path(str(event.src_path)), EVENT_DELETED)

    def on_moved(self, event: FileSystemEvent) -> None:
        # A move (rename) translates to delete + create from the indexer's point
        # of view. `watchdog`'s ``FileMovedEvent`` exposes both endpoints; we
        # care about ``src_path`` for the delete side and ``dest_path`` for the
        # create side. Directories are ignored (the constituent files surface
        # as their own events on most platforms).
        if event.is_directory:
            return
        self._watcher._schedule(Path(str(event.src_path)), EVENT_DELETED)
        dest = getattr(event, "dest_path", "")
        if dest:
            self._watcher._schedule(Path(str(dest)), EVENT_CREATED)


class DocsWatcher:
    """Watch ``docs_root`` and fire a callback on indexable-file changes."""

    def __init__(
        self,
        docs_root: Path,
        config: ARGConfig,
        on_change: WatchCallback,
    ) -> None:
        self._docs_root = docs_root.resolve()
        if not self._docs_root.is_dir():
            raise NotADirectoryError(f"DocsWatcher docs_root is not a directory: {self._docs_root}")
        self._debounce_ms = int(config.watch_debounce_ms)
        if self._debounce_ms < 0:
            raise ValueError(f"watch_debounce_ms must be >= 0 (got {config.watch_debounce_ms})")
        self._on_change = on_change
        # Observer() is a factory returning a platform-specific BaseObserver
        # subclass; annotate against the abstract type so mypy is happy.
        self._observer: BaseObserver | None = None
        self._handler = _DocsEventHandler(self)
        self._pending: dict[Path, threading.Timer] = {}
        self._lock = threading.Lock()
        self._stopped = False
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def docs_root(self) -> Path:
        return self._docs_root

    @property
    def handler(self) -> _DocsEventHandler:
        """Underlying `watchdog` handler. Exposed for unit tests."""
        return self._handler

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Spin up the watchdog Observer and begin emitting events.

        Calling ``start`` twice is a no-op after the first call; this keeps
        the pipeline's lifecycle code from caring whether the watcher has
        already been started.
        """
        with self._lock:
            if self._stopped:
                raise RuntimeError("DocsWatcher.start() called after stop()")
            if self._running:
                return
            obs = Observer()
            obs.schedule(self._handler, str(self._docs_root), recursive=True)
            obs.daemon = True
            obs.start()
            self._observer = obs
            self._running = True
            logger.info("DocsWatcher started on %s", self._docs_root)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the observer and cancel any pending debounce timers.

        Idempotent. Safe to call from signal handlers.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._running = False
            for timer in self._pending.values():
                timer.cancel()
            self._pending.clear()
            obs = self._observer
            self._observer = None

        if obs is not None:
            obs.stop()
            obs.join(timeout=timeout)
        logger.info("DocsWatcher stopped on %s", self._docs_root)

    def __enter__(self) -> DocsWatcher:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal: event scheduling
    # ------------------------------------------------------------------

    def _schedule(self, path: Path, event_kind: str) -> None:
        """Filter, debounce, and queue a callback for one filesystem event."""
        # Suffix filter — cheap; do it before any locking.
        if path.suffix.lower() not in _INDEXABLE_SUFFIXES:
            return

        # Path-escape guard: ignore events outside ``docs_root`` even if
        # ``watchdog`` somehow surfaced one (e.g., a symlink target).
        try:
            path.resolve().relative_to(self._docs_root)
        except ValueError:
            return

        with self._lock:
            if self._stopped:
                return
            existing = self._pending.pop(path, None)
            if existing is not None:
                existing.cancel()

            if self._debounce_ms == 0:
                # Fire synchronously; callers (e.g., unit tests) sometimes want
                # zero-debounce immediate dispatch.
                timer = None
            else:
                timer = threading.Timer(
                    self._debounce_ms / 1000.0,
                    self._fire,
                    args=(path, event_kind),
                )
                timer.daemon = True
                self._pending[path] = timer

        if timer is None:
            self._fire(path, event_kind)
        else:
            timer.start()

    def _fire(self, path: Path, event_kind: str) -> None:
        with self._lock:
            self._pending.pop(path, None)
            if self._stopped:
                return
        try:
            self._on_change(path, event_kind)
        except Exception:
            # Don't let a buggy callback take down the observer thread.
            logger.exception("DocsWatcher callback failed for %s (%s)", path, event_kind)
