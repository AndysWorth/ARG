#!/usr/bin/env python3
"""Print LLM prompts captured in debug trace files.

Requires ARG_DEBUG=1 (or DEBUG_TRACING=1) to have been set when the query ran.
Trace files are written to {db_path}/{corpus}/debug_traces/trace_*.jsonl.

Usage:
    python scripts/show_llm_prompts.py
    python scripts/show_llm_prompts.py --db ./index_db --corpus default
    python scripts/show_llm_prompts.py --all   # include non-LLM events too
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print LLM prompts from ARG debug traces.")
    parser.add_argument("--db", default="./index_db", help="ARG database directory")
    parser.add_argument("--corpus", default="default", help="Corpus name")
    parser.add_argument(
        "--all", action="store_true", dest="all_events", help="Print all events, not just LLM"
    )
    args = parser.parse_args()

    trace_dir = Path(args.db).expanduser().resolve() / args.corpus / "debug_traces"
    if not trace_dir.exists():
        print(
            f"No debug_traces directory found at {trace_dir}\n"
            "Run with ARG_DEBUG=1 to enable tracing.",
            file=sys.stderr,
        )
        return 1

    files = sorted(trace_dir.glob("trace_*.jsonl"))
    if not files:
        print(f"No trace files in {trace_dir}", file=sys.stderr)
        return 1

    for path in files:
        print(f"=== {path} ===\n")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[malformed line: {exc}]")
                    continue

                is_llm = event.get("type") == "llm" and event.get("event") == "start"
                if not args.all_events and not is_llm:
                    continue

                if args.all_events:
                    print(json.dumps(event, indent=2))
                    print()
                else:
                    payload = event.get("payload", {})
                    msg = payload.get("messages") or payload.get("prompt") or "(no prompt field)"
                    print(msg)
                    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
