"""JSON-line log formatter + rotating-file handler wiring.

ARG's runtime logs go to a single per-corpus file at
``{db_path}/{corpus_name}/arg.log``. Each line is a self-contained JSON
object so the file can be tailed, grep'd by key, or shipped to a JSON
log viewer without parsing free-form text.

Configuration is opt-in: callers pass an ``ARGConfig`` and a corpus name
to :func:`configure_logging`, which installs a `RotatingFileHandler` on
the root logger using the size cap and backup count from config. Calling
it twice is a no-op so the FastAPI test client and the pipeline can both
invoke it without fear.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import ClassVar

from arg.config import ARGConfig

_HANDLER_NAME = "arg-json-rotating"


class JsonFormatter(stdlib_logging.Formatter):
    """Render LogRecord as a single JSON line.

    Fields always present: ``timestamp``, ``level``, ``logger``,
    ``message``. Exceptions surface as ``exc_type`` + ``exc_message``.
    Any ``extra={...}`` keys passed to the logging call are merged into
    the top-level object so callers can attach structured context
    (``logger.info("indexed", extra={"docs": 42})``).
    """

    _RESERVED: ClassVar[set[str]] = {
        # Standard LogRecord attributes we explicitly project; everything
        # else from `record.__dict__` is treated as caller-supplied extras.
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",  # already projected
        "asctime",
    }

    def format(self, record: stdlib_logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            exc_type, exc_value, _tb = record.exc_info
            if exc_type is not None:
                payload["exc_type"] = exc_type.__name__
                payload["exc_message"] = str(exc_value)
        # Merge extras.
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._RESERVED:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                payload[key] = repr(value)
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    config: ARGConfig,
    corpus_name: str = "default",
    *,
    level: int | None = None,
) -> Path:
    """Install the rotating JSON-line handler on the root logger.

    Idempotent — a second call with the same corpus replaces the existing
    handler so file paths stay correct after a corpus switch. Returns the
    log-file path so callers can surface it in CLI banners.
    """
    log_path = config.log_path(corpus_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = stdlib_logging.getLogger()
    # Strip any previously-installed ARG handler so this call is idempotent.
    for handler in list(root.handlers):
        if getattr(handler, "_arg_handler_name", None) == _HANDLER_NAME:
            root.removeHandler(handler)
            handler.close()

    rotator = RotatingFileHandler(
        str(log_path),
        maxBytes=10_485_760,
        backupCount=5,
        encoding="utf-8",
    )
    rotator.setFormatter(JsonFormatter())
    # Tag the handler so a re-configure call can recognise + remove it.
    rotator._arg_handler_name = _HANDLER_NAME  # type: ignore[attr-defined]
    root.addHandler(rotator)
    # Always set INFO when no explicit level was passed. The previous
    # "only set if unset" behaviour silently inherited any prior
    # basicConfig(level=WARNING) and suppressed indexer progress messages.
    root.setLevel(level if level is not None else stdlib_logging.INFO)
    return log_path
