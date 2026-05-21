#!/usr/bin/env python3
"""ARG command-line entry point.

Subcommands per Section 10 spec:
  index   — crawl + index a docs root; writes Chroma + Kuzu + BM25
  query   — one-shot RAG query
  serve   — start the FastAPI server on localhost
  stats   — print corpus statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arg.config import ARGConfig
from arg.logging import configure_logging, enable_debug_tracing


def _build_config_for_cli(
    docs_path: str | None,
    db_path: str,
    *,
    no_watch: bool = False,
    debug: bool = False,
) -> ARGConfig:
    """Construct an ARGConfig from CLI arguments.

    ``docs_path`` is optional for query/serve/stats — those don't need to know
    the original docs root because the on-disk corpus is already complete.
    We fall back to db_path for ``docs_root`` in those cases since
    ARGConfig validates the directory exists.
    """
    db = Path(db_path).expanduser().resolve()
    docs = Path(docs_path).expanduser().resolve() if docs_path else db
    cfg = ARGConfig.from_env(docs_root=docs, db_path=db)
    if no_watch:
        cfg.watch_enabled = False
    if debug:
        cfg.debug_tracing = True
    return cfg


def cmd_index(args: argparse.Namespace) -> int:
    from arg.pipeline import ARGPipeline

    cfg = _build_config_for_cli(args.docs, args.db, no_watch=args.no_watch, debug=args.debug)
    log_path = configure_logging(cfg, corpus_name=args.corpus)
    enable_debug_tracing(cfg, corpus_name=args.corpus)
    print(f"[arg] logging → {log_path}")

    with ARGPipeline(config=cfg, corpus_name=args.corpus) as pipeline:
        stats = pipeline.index()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from arg.pipeline import ARGPipeline

    cfg = _build_config_for_cli(args.docs, args.db, debug=args.debug)
    configure_logging(cfg, corpus_name=args.corpus)
    enable_debug_tracing(cfg, corpus_name=args.corpus)

    with ARGPipeline(config=cfg, corpus_name=args.corpus, skip_watcher=True) as pipeline:
        result = pipeline.query(args.question, enrich=not args.no_enrich)
    print(result.answer)
    print()
    print("--- Sources ---")
    for s in result.sources:
        print(f"  • {s.title} ({s.heading_path})  [{s.doc_id}]")
    print(f"\nlatency_ms: {result.latency_ms}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from arg.pipeline import ARGPipeline
    from arg.server import create_app

    cfg = _build_config_for_cli(args.docs, args.db, debug=args.debug)
    configure_logging(cfg, corpus_name=args.corpus)
    enable_debug_tracing(cfg, corpus_name=args.corpus)

    pipeline = ARGPipeline(config=cfg, corpus_name=args.corpus)
    app = create_app({args.corpus: pipeline})
    host = cfg.server_host
    port = args.port or cfg.server_port
    print(f"[arg] serving corpus='{args.corpus}' on http://{host}:{port}")
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        pipeline.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from arg.pipeline import ARGPipeline

    cfg = _build_config_for_cli(args.docs, args.db, debug=args.debug)
    configure_logging(cfg, corpus_name=args.corpus)
    with ARGPipeline(config=cfg, corpus_name=args.corpus, skip_watcher=True) as pipeline:
        stats = pipeline.corpus_stats()
    print(json.dumps(stats, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arg",
        description="ARG — Archivist RAG Graph",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug tracing")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="crawl and index a documentation root")
    p_index.add_argument("--docs", required=True, help="documentation root")
    p_index.add_argument("--db", required=True, help="ARG database directory")
    p_index.add_argument("--corpus", default="default")
    p_index.add_argument("--no-watch", action="store_true", help="disable live watcher")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="one-shot RAG query")
    p_query.add_argument("--db", required=True)
    p_query.add_argument("--docs", default=None)
    p_query.add_argument("--corpus", default="default")
    p_query.add_argument("--no-enrich", action="store_true")
    p_query.add_argument("question", help="question to ask")
    p_query.set_defaults(func=cmd_query)

    p_serve = sub.add_parser("serve", help="start the FastAPI server on localhost")
    p_serve.add_argument("--db", required=True)
    p_serve.add_argument("--docs", default=None)
    p_serve.add_argument("--corpus", default="default")
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_stats = sub.add_parser("stats", help="print corpus statistics")
    p_stats.add_argument("--db", required=True)
    p_stats.add_argument("--docs", default=None)
    p_stats.add_argument("--corpus", default="default")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Each subcommand calls ``configure_logging`` itself which installs the
    # ARG rotating-JSON handler at INFO level. No basicConfig() here — it
    # would just set a WARNING root level that configure_logging would have
    # to fight.
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
