"""Watcher tests.

Two layers:

  * Handler-level tests drive synthetic ``watchdog`` events through
    :class:`arg.crawler.watcher.DocsWatcher.handler` without a real Observer.
    These are fully deterministic and run in well under a second.
  * One real-Observer test exercises the full path: tmp dir + real file ops +
    debounce timing. Marked ``@pytest.mark.integration`` so CI can opt out of
    filesystem-event timing if it proves flaky on a runner.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from watchdog.events import (
    DirCreatedEvent,
    DirModifiedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from arg.config import ARGConfig
from arg.crawler.watcher import (
    EVENT_CREATED,
    EVENT_DELETED,
    EVENT_MODIFIED,
    DocsWatcher,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def docs_root(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    return root


def make_config(tmp_path: Path, docs_root: Path, debounce_ms: int = 0) -> ARGConfig:
    """Build an `ARGConfig` with a controllable debounce interval.

    Tests default to ``debounce_ms=0`` so callbacks fire synchronously and the
    suite stays deterministic. Tests that specifically exercise debouncing
    pass an explicit value.
    """
    return ARGConfig(
        docs_root=docs_root, db_path=tmp_path / "arg_db", watch_debounce_ms=debounce_ms
    )


class _Recorder:
    """Thread-safe callback that records every dispatched (path, kind) pair."""

    def __init__(self) -> None:
        self.events: list[tuple[Path, str]] = []
        self._lock = threading.Lock()

    def __call__(self, path: Path, kind: str) -> None:
        with self._lock:
            self.events.append((path, kind))

    def kinds_for(self, name: str) -> list[str]:
        return [k for p, k in self.events if p.name == name]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_with_missing_root_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        DocsWatcher(tmp_path / "missing", make_config(tmp_path, tmp_path), _Recorder())


def test_negative_debounce_rejected(tmp_path, docs_root):
    # Use a sentinel ARGConfig that bypasses ARGConfig's own validation by
    # constructing it normally then poking the field.
    cfg = make_config(tmp_path, docs_root, debounce_ms=0)
    cfg.watch_debounce_ms = -1
    with pytest.raises(ValueError, match="watch_debounce_ms must be >= 0"):
        DocsWatcher(docs_root, cfg, _Recorder())


# ---------------------------------------------------------------------------
# Handler-level: filter + dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,kind_event",
    [
        ("a.html", FileCreatedEvent),
        ("b.htm", FileCreatedEvent),
        ("c.pdf", FileCreatedEvent),
    ],
)
def test_indexable_file_created_fires_callback(tmp_path, docs_root, name, kind_event):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    target = docs_root / name
    target.touch()  # must exist so the resolve()/relative_to() check passes
    watcher.handler.on_created(kind_event(str(target)))
    assert rec.events == [(target.resolve(), EVENT_CREATED)]


def test_modified_event_dispatches(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    target = docs_root / "page.html"
    target.touch()
    watcher.handler.on_modified(FileModifiedEvent(str(target)))
    assert rec.events == [(target.resolve(), EVENT_MODIFIED)]


def test_deleted_event_dispatches(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    target = docs_root / "page.html"
    target.touch()
    target.unlink()
    # After deletion, resolve()+relative_to still works on the symbolic path on
    # macOS/Linux because we only inspect path strings, not stat the file.
    watcher.handler.on_deleted(FileDeletedEvent(str(target)))
    assert rec.events == [(target.resolve(), EVENT_DELETED)]


def test_moved_event_dispatches_delete_then_create(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    src = docs_root / "old.html"
    dst = docs_root / "new.html"
    src.touch()
    dst.touch()
    watcher.handler.on_moved(FileMovedEvent(str(src), str(dst)))
    assert (src.resolve(), EVENT_DELETED) in rec.events
    assert (dst.resolve(), EVENT_CREATED) in rec.events
    assert len(rec.events) == 2


@pytest.mark.parametrize(
    "name",
    [
        "page.txt",
        "data.csv",
        "image.png",
        "binary.bin",
        "Makefile",
        "README",
        "x.HTML.bak",
    ],
)
def test_non_indexable_suffix_ignored(tmp_path, docs_root, name):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    target = docs_root / name
    target.touch()
    watcher.handler.on_modified(FileModifiedEvent(str(target)))
    assert rec.events == []


def test_directory_events_ignored(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    subdir = docs_root / "sub"
    subdir.mkdir()
    watcher.handler.on_created(DirCreatedEvent(str(subdir)))
    watcher.handler.on_modified(DirModifiedEvent(str(subdir)))
    assert rec.events == []


def test_indexable_suffix_case_insensitive(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    target = docs_root / "page.HTML"
    target.touch()
    watcher.handler.on_modified(FileModifiedEvent(str(target)))
    assert rec.events == [(target.resolve(), EVENT_MODIFIED)]


def test_path_outside_docs_root_ignored(tmp_path, docs_root):
    """Symlink/path-escape: events for files outside docs_root must not fire."""
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    outside = tmp_path / "outside.html"
    outside.touch()
    watcher.handler.on_modified(FileModifiedEvent(str(outside)))
    assert rec.events == []


# ---------------------------------------------------------------------------
# Debouncing
# ---------------------------------------------------------------------------


def test_rapid_events_coalesce_into_one_callback(tmp_path, docs_root):
    """Multiple modifications inside the debounce window collapse to one call."""
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root, debounce_ms=50), rec)
    target = docs_root / "page.html"
    target.touch()
    for _ in range(5):
        watcher.handler.on_modified(FileModifiedEvent(str(target)))

    # Within the debounce window: no callback yet.
    assert rec.events == []
    # Wait past the debounce; pending timer should fire exactly once.
    time.sleep(0.2)
    assert rec.kinds_for("page.html") == [EVENT_MODIFIED]


def test_debounce_per_path(tmp_path, docs_root):
    """Events on different paths debounce independently."""
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root, debounce_ms=50), rec)
    a = docs_root / "a.html"
    b = docs_root / "b.html"
    a.touch()
    b.touch()
    watcher.handler.on_modified(FileModifiedEvent(str(a)))
    watcher.handler.on_modified(FileModifiedEvent(str(b)))
    time.sleep(0.2)
    names = sorted(p.name for p, _ in rec.events)
    assert names == ["a.html", "b.html"]


def test_debounce_window_resets_on_each_event(tmp_path, docs_root):
    """A new event inside the window cancels the previous timer and restarts it."""
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root, debounce_ms=80), rec)
    target = docs_root / "page.html"
    target.touch()
    # Sustained editing — every 30ms, well within the 80ms debounce window.
    for _ in range(4):
        watcher.handler.on_modified(FileModifiedEvent(str(target)))
        time.sleep(0.03)
    # We've been editing for ~120ms but each event resets the timer; no callback yet.
    assert rec.events == []
    # Stop editing; wait one full window plus headroom.
    time.sleep(0.15)
    assert rec.kinds_for("page.html") == [EVENT_MODIFIED]


def test_zero_debounce_fires_synchronously(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root, debounce_ms=0), rec)
    target = docs_root / "page.html"
    target.touch()
    watcher.handler.on_modified(FileModifiedEvent(str(target)))
    assert rec.events == [(target.resolve(), EVENT_MODIFIED)]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_stop_cancels_pending_callbacks(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root, debounce_ms=200), rec)
    target = docs_root / "page.html"
    target.touch()
    watcher.handler.on_modified(FileModifiedEvent(str(target)))
    watcher.stop()
    time.sleep(0.3)
    assert rec.events == []


def test_stop_is_idempotent(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    watcher.stop()
    watcher.stop()  # must not raise


def test_start_after_stop_raises(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    watcher.stop()
    with pytest.raises(RuntimeError, match="after stop"):
        watcher.start()


def test_start_is_idempotent(tmp_path, docs_root):
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec)
    try:
        watcher.start()
        first_running = watcher.is_running
        watcher.start()  # second call is a no-op
        assert first_running is True
        assert watcher.is_running is True
    finally:
        watcher.stop()


def test_callback_exception_does_not_kill_watcher(tmp_path, docs_root, caplog):
    """A buggy callback must not propagate out of the watcher."""

    def boom(path: Path, kind: str) -> None:
        raise RuntimeError("callback failed on purpose")

    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root), boom)
    target = docs_root / "page.html"
    target.touch()
    import logging

    with caplog.at_level(logging.ERROR):
        watcher.handler.on_modified(FileModifiedEvent(str(target)))
    # No exception escaped; failure was logged.
    assert any("callback failed" in rec.message.lower() for rec in caplog.records)


def test_context_manager_starts_and_stops(tmp_path, docs_root):
    rec = _Recorder()
    with DocsWatcher(docs_root, make_config(tmp_path, docs_root), rec) as w:
        assert w.is_running is True
    assert w.is_running is False


# ---------------------------------------------------------------------------
# Real Observer end-to-end (integration-marked)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_observer_detects_create_and_modify(tmp_path, docs_root):
    """A real watchdog Observer surfaces filesystem events to the callback.

    Filesystem-event timing differs by OS so this test allows generous waits.
    Skipped from default CI runs via the `integration` marker until Section 11.
    """
    rec = _Recorder()
    watcher = DocsWatcher(docs_root, make_config(tmp_path, docs_root, debounce_ms=50), rec)
    try:
        watcher.start()
        # Allow the Observer thread a beat to attach watches.
        time.sleep(0.05)
        target = docs_root / "live.html"
        target.write_text("<html></html>")
        time.sleep(0.5)
        target.write_text("<html><body>updated</body></html>")
        time.sleep(0.5)

        kinds = rec.kinds_for("live.html")
        # We expect at least one event for create+modify; the exact count
        # depends on the OS (macOS often emits CREATE+MODIFY pairs).
        assert kinds, f"no events recorded for live.html: {rec.events}"
        assert any(k in (EVENT_CREATED, EVENT_MODIFIED) for k in kinds)
    finally:
        watcher.stop()
