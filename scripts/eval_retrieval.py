#!/usr/bin/env python3
"""Retrieval evaluator.

Reads a JSON file of ``{question, expected_doc_ids[]}`` pairs and reports
whether each question's expected doc(s) appeared in the retrieved chunks'
``doc_id`` metadata. Output: per-question hit/miss plus an aggregate
hit-rate. Latency is reported as p50/p95.

Spec usage:
    python scripts/eval_retrieval.py --db ./arg_db --corpus default --qa eval/qa_pairs.json

QA file shape:
    [
      {"question": "...", "expected_doc_ids": ["/abs/path/a.html", "..."]},
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def evaluate(pipeline: Any, qa_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Run each question through the pipeline; track hit rate + latency."""
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    hits = 0
    for entry in qa_pairs:
        question = str(entry.get("question") or "")
        expected = set(entry.get("expected_doc_ids") or [])
        if not question:
            continue
        t0 = time.perf_counter()
        result = pipeline.query(question, enrich=True)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)
        retrieved = {s.doc_id for s in result.sources}
        hit = bool(expected & retrieved)
        if hit:
            hits += 1
        results.append(
            {
                "question": question,
                "expected": sorted(expected),
                "retrieved": sorted(retrieved),
                "hit": hit,
                "latency_ms": int(latency),
            }
        )

    summary: dict[str, Any] = {
        "n": len(results),
        "hits": hits,
        "hit_rate": (hits / len(results)) if results else 0.0,
        "latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": (
            int(sorted(latencies)[int(len(latencies) * 0.95) - 1])
            if len(latencies) >= 2
            else (int(latencies[0]) if latencies else 0)
        ),
        "results": results,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_retrieval")
    parser.add_argument("--db", required=True)
    parser.add_argument("--corpus", default="default")
    parser.add_argument("--qa", required=True, help="JSON file of QA pairs")
    parser.add_argument("--docs", default=None, help="docs_root (defaults to db)")
    args = parser.parse_args(argv)

    qa_path = Path(args.qa).expanduser().resolve()
    if not qa_path.is_file():
        print(f"[eval] QA file not found: {qa_path}", file=sys.stderr)
        return 1
    with qa_path.open("r", encoding="utf-8") as fh:
        qa_pairs = json.load(fh)
    if not isinstance(qa_pairs, list):
        print("[eval] QA file must contain a JSON array", file=sys.stderr)
        return 1

    from arg.config import ARGConfig
    from arg.pipeline import ARGPipeline

    db = Path(args.db).expanduser().resolve()
    docs = Path(args.docs).expanduser().resolve() if args.docs else db
    cfg = ARGConfig(docs_root=docs, db_path=db)
    with ARGPipeline(config=cfg, corpus_name=args.corpus, skip_watcher=True) as pipeline:
        summary = evaluate(pipeline, qa_pairs)
    print(json.dumps(summary, indent=2, default=str))
    # Eval is a reporting tool — bad recall is not a CLI failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
