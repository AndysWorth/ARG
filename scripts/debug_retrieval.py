#!/usr/bin/env python3
"""Print results from each of the 5 retrieval stages for a given query.

Usage:
    python scripts/debug_retrieval.py "your query here"
    python scripts/debug_retrieval.py "your query here" --db ./index_db --top-k 8
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def _short(doc_id: str) -> str:
    """Trim the docs root prefix for readable display."""
    p = Path(doc_id)
    # Show last 3 path components: .../parent/dir/file.ext
    parts = p.parts
    return str(Path(*parts[-3:])) if len(parts) >= 3 else doc_id


def _snippet(text: str, width: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return textwrap.shorten(text, width=width, placeholder=" …")


def _print_hits(hits: list, *, stage: str, color: str) -> None:
    print(f"\n{BOLD}{color}{'─' * 60}{RESET}")
    print(f"{BOLD}{color}{stage}{RESET}  ({len(hits)} hits)")
    print(f"{color}{'─' * 60}{RESET}")
    if not hits:
        print(f"  {DIM}(none){RESET}")
        return
    for i, h in enumerate(hits):
        doc = _short(h.metadata.get("doc_id", h.chunk_id))
        pos = h.metadata.get("position", "?")
        page = h.metadata.get("page_number")
        page_str = f"  page {page}" if page is not None else ""
        scores = "  ".join(f"{k}={v:.4f}" for k, v in h.stage_scores.items())
        ranks = "  ".join(f"{k}_rank={v}" for k, v in h.stage_ranks.items())
        print(f"  [{i}] {BOLD}{doc}{RESET}  chunk#{pos}{page_str}")
        print(f"       {DIM}{scores}   {ranks}{RESET}")
        print(f"       {_snippet(h.text)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug each retrieval stage for a query.")
    parser.add_argument("query", help="The query to retrieve")
    parser.add_argument("--db", default="./index_db", help="ARG database directory")
    parser.add_argument("--docs", default=None, help="Docs root (default: same as --db)")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k_vector")
    parser.add_argument("--no-enrich", action="store_true", help="Disable Stage 0 enrichment")
    args = parser.parse_args()

    from arg.config import ARGConfig
    from arg.pipeline import ARGPipeline
    from arg.retriever.retriever import _lost_in_middle_reorder, _rrf_fuse

    db = Path(args.db).expanduser().resolve()
    docs = Path(args.docs).expanduser().resolve() if args.docs else db
    cfg = ARGConfig(docs_root=docs, db_path=db)
    cfg.query_rewrite = False
    cfg.query_decompose = False
    if args.top_k:
        cfg.top_k_vector = args.top_k

    p = ARGPipeline(config=cfg)
    r = p.retriever

    query = args.query
    print(f"\n{BOLD}Query:{RESET} {query}")
    print(
        f"{DIM}top_k={cfg.top_k_vector}  enrich={cfg.enrich_enabled and not args.no_enrich}  "
        f"bm25={cfg.bm25_enabled}  graph_hop={cfg.graph_hop_depth}{RESET}"
    )

    # Intentionally duplicates ARGRetriever.retrieve() so each stage is visible.
    # ── Stage 0: Enrichment ──────────────────────────────────────────────
    candidate_doc_ids = None
    if cfg.enrich_enabled and not args.no_enrich:
        candidate_doc_ids = r._stage0_enrichment(query)

    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}Stage 0 — Enrichment{RESET}  (candidate doc filter)")
    print(f"{CYAN}{'─' * 60}{RESET}")
    if not cfg.enrich_enabled or args.no_enrich:
        print(f"  {DIM}(disabled){RESET}")
    elif candidate_doc_ids is None:
        print(
            f"  {DIM}No docs met enrich_min_score={cfg.enrich_min_score} — full corpus used{RESET}"
        )
    else:
        print(f"  {len(candidate_doc_ids)} candidate docs:")
        for doc_id in sorted(candidate_doc_ids):
            print(f"    {_short(doc_id)}")

    # ── Stage 1: Dense ───────────────────────────────────────────────────
    dense_hits = r._stage1_dense(
        query=query,
        candidate_doc_ids=candidate_doc_ids,
        top_k=cfg.top_k_vector,
        chroma_filters=None,
    )
    _print_hits(dense_hits, stage="Stage 1 — Dense (ChromaDB vector search)", color=GREEN)

    # ── Stage 1.5: BM25 ──────────────────────────────────────────────────
    bm25_hits = r._stage1_5_bm25(
        query=query,
        top_k=cfg.top_k_vector,
        candidate_doc_ids=candidate_doc_ids,
        filters=None,
    )
    _print_hits(bm25_hits, stage="Stage 1.5 — BM25 (sparse keyword search)", color=YELLOW)

    # ── Stage 2: Graph expansion ─────────────────────────────────────────
    graph_hits = []
    if cfg.graph_hop_depth > 0:
        graph_hits = r._stage2_graph(
            query=query,
            seed_hits=dense_hits + bm25_hits,
            chroma_filters=None,
        )
    _print_hits(graph_hits, stage="Stage 2 — Graph expansion", color=CYAN)

    # ── Stage 3: RRF fusion ───────────────────────────────────────────────
    fused = _rrf_fuse({"dense": dense_hits, "bm25": bm25_hits, "graph": graph_hits})
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}Stage 3 — RRF Fusion{RESET}  ({len(fused)} unique chunks, sorted by RRF score)")
    print(f"{'─' * 60}")
    for i, h in enumerate(fused):
        doc = _short(h.metadata.get("doc_id", h.chunk_id))
        pos = h.metadata.get("position", "?")
        stages_present = ", ".join(h.stage_scores.keys())
        print(
            f"  [{i}] rrf={h.rrf_score:.5f}  stages=[{stages_present}]  {BOLD}{doc}{RESET}  chunk#{pos}"
        )
        print(f"       {_snippet(h.text)}")

    # ── Stage 4: Lost-in-middle reorder ──────────────────────────────────
    final = _lost_in_middle_reorder(fused, target_n=cfg.top_k_vector)
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(
        f"{BOLD}Stage 4 — Final (lost-in-middle reorder){RESET}  ({len(final)} chunks returned to LLM)"
    )
    print(f"{'─' * 60}")
    for i, n in enumerate(final):
        doc = _short(n.node.metadata.get("doc_id", n.node.id_))
        pos = n.node.metadata.get("position", "?")
        page = n.node.metadata.get("page_number")
        page_str = f"  page {page}" if page is not None else ""
        print(f"  [{i}] score={n.score:.5f}  {BOLD}{doc}{RESET}  chunk#{pos}{page_str}")
        print(f"       {_snippet(n.node.text)}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
