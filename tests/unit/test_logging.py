"""Logging tests — JSON formatter shape + debug traces directory creation."""

from __future__ import annotations

import json
import logging as stdlib_logging
from pathlib import Path

import pytest

from arg.config import ARGConfig
from arg.logging import JsonFormatter, configure_logging, enable_debug_tracing


@pytest.fixture
def config(tmp_path: Path) -> ARGConfig:
    docs = tmp_path / "docs"
    docs.mkdir()
    return ARGConfig(docs_root=docs, db_path=tmp_path / "arg_db")


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


def test_json_formatter_emits_valid_json_line():
    fmt = JsonFormatter()
    record = stdlib_logging.LogRecord(
        name="arg.test",
        level=stdlib_logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="indexed %d docs",
        args=(7,),
        exc_info=None,
    )
    line = fmt.format(record)
    obj = json.loads(line)
    assert obj["level"] == "INFO"
    assert obj["logger"] == "arg.test"
    assert obj["message"] == "indexed 7 docs"
    assert "timestamp" in obj


def test_json_formatter_includes_extras():
    fmt = JsonFormatter()
    record = stdlib_logging.LogRecord(
        name="arg.test",
        level=stdlib_logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="finished",
        args=(),
        exc_info=None,
    )
    record.docs = 42
    record.corpus = "default"
    line = fmt.format(record)
    obj = json.loads(line)
    assert obj["docs"] == 42
    assert obj["corpus"] == "default"


def test_json_formatter_captures_exception():
    fmt = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = stdlib_logging.LogRecord(
            name="arg.test",
            level=stdlib_logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="caught",
            args=(),
            exc_info=sys.exc_info(),
        )
        line = fmt.format(record)
    obj = json.loads(line)
    assert obj["exc_type"] == "ValueError"
    assert obj["exc_message"] == "boom"


def test_json_formatter_falls_back_to_repr_for_non_serialisable():
    """A LogRecord ``extra`` with an unserialisable value lands as repr()."""

    class Unserialisable:
        def __repr__(self) -> str:
            return "<unique-marker-xyz>"

    fmt = JsonFormatter()
    record = stdlib_logging.LogRecord(
        name="arg.test",
        level=stdlib_logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="weird",
        args=(),
        exc_info=None,
    )
    record.gizmo = Unserialisable()
    line = fmt.format(record)
    obj = json.loads(line)
    assert obj["gizmo"] == "<unique-marker-xyz>"


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


def test_configure_logging_installs_rotating_handler(config):
    log_path = configure_logging(config, corpus_name="default", level=stdlib_logging.INFO)
    try:
        logger = stdlib_logging.getLogger("arg.test.configure")
        logger.info("hello from %s", "ARG")
        for handler in stdlib_logging.getLogger().handlers:
            handler.flush()
        text = log_path.read_text(encoding="utf-8")
        assert text.strip(), "log file should be non-empty after a logged message"
        # Every line must be a valid JSON object.
        for line in text.splitlines():
            json.loads(line)
    finally:
        # Tear down so other tests aren't logged to this file.
        root = stdlib_logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_arg_handler_name", None) == "arg-json-rotating":
                root.removeHandler(handler)
                handler.close()


def test_configure_logging_idempotent_replaces_existing_handler(config):
    p1 = configure_logging(config, corpus_name="default")
    p2 = configure_logging(config, corpus_name="default")
    try:
        assert p1 == p2
        # Only one ARG handler attached, no matter how many times we call.
        root = stdlib_logging.getLogger()
        arg_handlers = [
            h for h in root.handlers if getattr(h, "_arg_handler_name", None) == "arg-json-rotating"
        ]
        assert len(arg_handlers) == 1
    finally:
        root = stdlib_logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_arg_handler_name", None) == "arg-json-rotating":
                root.removeHandler(handler)
                handler.close()


def test_configure_logging_overrides_prior_root_level(config):
    """A prior ``logging.basicConfig(level=WARNING)`` (or anything else)
    must not suppress INFO messages once ``configure_logging`` has run.
    Without this contract the indexer's per-doc progress lines stay
    invisible — regression caught in the wild after Feature 0001."""
    root = stdlib_logging.getLogger()
    root.setLevel(stdlib_logging.WARNING)  # simulate a stale basicConfig
    try:
        configure_logging(config, corpus_name="default")
        assert root.level == stdlib_logging.INFO
    finally:
        for handler in list(root.handlers):
            if getattr(handler, "_arg_handler_name", None) == "arg-json-rotating":
                root.removeHandler(handler)
                handler.close()
        root.setLevel(stdlib_logging.NOTSET)


# ---------------------------------------------------------------------------
# enable_debug_tracing
# ---------------------------------------------------------------------------


def test_debug_tracing_disabled_returns_none(config):
    config.debug_tracing = False
    assert enable_debug_tracing(config) is None


def test_debug_tracing_enabled_creates_directory(config):
    config.debug_tracing = True
    result = enable_debug_tracing(config, corpus_name="default")
    assert result is not None
    assert result.is_dir()
    assert result == config.debug_traces_path("default")
