"""LlamaIndex callback-based debug tracing.

Off by default (CLAUDE.md locality guarantee: no telemetry). When the user
opts in via ``--debug`` or ``ARG_DEBUG=1``, ``enable_debug_tracing()`` writes
LlamaIndex callback events as JSON lines under
``{db_path}/{corpus_name}/debug_traces/``.

Implementation
--------------
LlamaIndex exposes a `CallbackManager` that downstream code reads from
``Settings.callback_manager``. We install a small handler that captures the
events we care about (LLM calls, retrieval, embedding) and writes one JSON
line per event. The handler is local — it never sends data anywhere.

The function is a no-op when ``config.debug_tracing`` is False so callers
don't have to branch.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import time
from pathlib import Path
from typing import Any

from arg.config import ARGConfig

logger = stdlib_logging.getLogger(__name__)


def enable_debug_tracing(config: ARGConfig, corpus_name: str = "default") -> Path | None:
    """Wire up LlamaIndex callback tracing if ``config.debug_tracing`` is True.

    Returns the trace directory on success, ``None`` when tracing is disabled
    or LlamaIndex is not importable (treated as a soft failure since tracing
    is non-essential).
    """
    if not config.debug_tracing:
        return None

    try:
        from llama_index.core import Settings
        from llama_index.core.callbacks import CallbackManager, CBEventType
        from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    except ImportError as exc:  # pragma: no cover - LlamaIndex is a hard dep
        logger.warning("Debug tracing unavailable: %s", exc)
        return None

    trace_dir = config.debug_traces_path(corpus_name)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / f"trace_{int(time.time())}.jsonl"

    class _JsonLineTraceHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
            self._file = trace_file.open("a", encoding="utf-8")

        def start_trace(self, trace_id: str | None = None) -> None:
            self._write({"event": "trace_start", "trace_id": trace_id, "ts": time.time()})

        def end_trace(
            self,
            trace_id: str | None = None,
            trace_map: dict[str, list[str]] | None = None,
        ) -> None:
            self._write({"event": "trace_end", "trace_id": trace_id, "ts": time.time()})

        def on_event_start(
            self,
            event_type: CBEventType,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
        ) -> str:
            self._write(
                {
                    "event": "start",
                    "type": event_type.value,
                    "event_id": event_id,
                    "parent_id": parent_id,
                    "ts": time.time(),
                    "payload": _scrub_payload(payload),
                }
            )
            return event_id

        def on_event_end(
            self,
            event_type: CBEventType,
            payload: dict[str, Any] | None = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            self._write(
                {
                    "event": "end",
                    "type": event_type.value,
                    "event_id": event_id,
                    "ts": time.time(),
                    "payload": _scrub_payload(payload),
                }
            )

        def _write(self, obj: dict[str, Any]) -> None:
            try:
                self._file.write(json.dumps(obj, default=str) + "\n")
                self._file.flush()
            except OSError as write_exc:
                logger.warning("Could not write trace event: %s", write_exc)

    Settings.callback_manager = CallbackManager([_JsonLineTraceHandler()])
    logger.info("LlamaIndex debug tracing enabled → %s", trace_file)
    return trace_dir


def _scrub_payload(payload: dict[Any, Any] | None) -> dict[str, Any]:
    """Convert a LlamaIndex callback payload into JSON-safe primitives.

    Payloads include arbitrary objects (Documents, Nodes, LLM responses);
    we coerce to ``str(value)`` for anything not directly JSON-encodable.
    """
    if not payload:
        return {}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        key_str = getattr(key, "value", str(key))
        try:
            json.dumps(value)
            out[key_str] = value
        except (TypeError, ValueError):
            out[key_str] = str(value)
    return out
