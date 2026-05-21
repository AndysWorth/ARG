#!/usr/bin/env python3
"""Show the results of each sub-step of Stage 0 (enrichment) for a given query.

Stage 0 uses BM25 chunk aggregation to find relevant documents before
searching chunks. Sub-steps:

  0.1  BM25 chunk search → aggregate scores per doc via max → normalise to [0,1]
  0.2  Link expansion: follow outbound + inbound graph edges from seed docs
  0.3  Cluster expansion: add topic-cluster mates of the top seed

Usage:
    python scripts/debug_stage0.py "your query here"
    python scripts/debug_stage0.py "your query here" --db ./index_db --top-docs 10 --min-score 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def _short(doc_id: str) -> str:
    parts = Path(doc_id).parts
    return str(Path(*parts[-3:])) if len(parts) >= 3 else doc_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug Stage 0 enrichment sub-steps.")
    parser.add_argument("query", help="The query to enrich")
    parser.add_argument("--db", default="./index_db", help="ARG database directory")
    parser.add_argument("--docs", default=None, help="Docs root (default: same as --db)")
    parser.add_argument("--top-docs", type=int, default=None, help="Override enrich_top_docs")
    parser.add_argument("--min-score", type=float, default=None, help="Override enrich_min_score")
    parser.add_argument(
        "--show-chunks",
        type=int,
        default=5,
        help="How many top BM25 chunks to show per doc (default: 5, 0=all)",
    )
    args = parser.parse_args()

    from arg.config import ARGConfig
    from arg.pipeline import ARGPipeline

    db = Path(args.db).expanduser().resolve()
    docs = Path(args.docs).expanduser().resolve() if args.docs else db
    cfg = ARGConfig(docs_root=docs, db_path=db)
    cfg.query_rewrite = False
    cfg.query_decompose = False
    if args.top_docs is not None:
        cfg.enrich_top_docs = args.top_docs
    if args.min_score is not None:
        cfg.enrich_min_score = args.min_score

    p = ARGPipeline(config=cfg)
    r = p.retriever

    print(f"\n{BOLD}Query:{RESET} {args.query}")
    print(
        f"{DIM}enrich_top_docs={cfg.enrich_top_docs}  "
        f"enrich_min_score={cfg.enrich_min_score}{RESET}"
    )

    # ── Sub-step 0.1: BM25 chunk search → doc aggregation ───────────────
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}Sub-step 0.1 — BM25 doc-level search{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")
    print(f"{DIM}Searches BM25 chunk index; aggregates to doc level using max score;")
    print(f"normalises scores to [0, 1] relative to top result.{RESET}\n")

    if r._bm25.is_empty:
        print(f"  {RED}BM25 index is empty — has the corpus been indexed?{RESET}")
        return 1

    raw_chunks = r._bm25.score_all(args.query)

    if not raw_chunks:
        print(f"  {RED}No BM25 hits — query terms not found in any chunk.{RESET}")
        print(f"  {DIM}Stage 0 returns None → full corpus searched.{RESET}")
        return 0

    # Aggregate to doc level using max chunk score
    doc_best_chunk: dict[str, tuple[float, str]] = {}  # doc_id → (score, chunk_id)
    doc_all_chunks: dict[str, list[tuple[str, float]]] = {}
    for chunk_id, score in raw_chunks:
        doc_id = chunk_id.split("::chunk::")[0] if "::chunk::" in chunk_id else chunk_id
        doc_all_chunks.setdefault(doc_id, []).append((chunk_id, score))
        if score > doc_best_chunk.get(doc_id, (0.0, ""))[0]:
            doc_best_chunk[doc_id] = (score, chunk_id)

    # Min-max normalise to [0, 1] — handles negative BM25 IDF values in small corpora
    raw_vals = [s for s, _ in doc_best_chunk.values()]
    min_raw, max_raw = min(raw_vals), max(raw_vals)
    if max_raw == min_raw:
        doc_scores_norm = dict.fromkeys(doc_best_chunk, 1.0)
    else:
        doc_scores_norm = {
            did: (raw_score - min_raw) / (max_raw - min_raw)
            for did, (raw_score, _) in doc_best_chunk.items()
        }

    ranked_docs = sorted(doc_scores_norm.items(), key=lambda x: (-x[1], x[0]))

    print(
        f"  BM25 returned {len(raw_chunks)} chunk hits across "
        f"{len(doc_best_chunk)} distinct documents.\n"
    )

    show_n = args.show_chunks
    seeds = []
    for rank, (doc_id, norm_score) in enumerate(ranked_docs[: max(cfg.enrich_top_docs * 3, 10)]):
        raw_score, best_cid = doc_best_chunk[doc_id]
        all_chunks = doc_all_chunks[doc_id]
        passed = norm_score >= cfg.enrich_min_score and rank < cfg.enrich_top_docs

        flag = (
            f"{GREEN}✓ seed{RESET}"
            if passed
            else f"{RED}✗ below threshold{RESET}"
            if norm_score < cfg.enrich_min_score
            else f"{DIM}✗ outside top-{cfg.enrich_top_docs}{RESET}"
        )

        print(f"  [{rank}] norm={norm_score:.4f}  raw={raw_score:.4f}  {flag}")
        print(f"        {BOLD}{_short(doc_id)}{RESET}")
        print(
            f"        {DIM}best chunk: {Path(best_cid).name if '::' not in best_cid else best_cid.split('::chunk::')[1]}{RESET}"
        )

        # Show top matching chunks for this doc
        chunk_limit = len(all_chunks) if show_n == 0 else min(show_n, len(all_chunks))
        for chunk_id, cscore in all_chunks[:chunk_limit]:
            pos = chunk_id.split("::chunk::")[-1] if "::chunk::" in chunk_id else "?"
            print(f"          chunk #{pos}  bm25={cscore:.4f}")
        if len(all_chunks) > chunk_limit:
            print(f"          {DIM}... {len(all_chunks) - chunk_limit} more chunks{RESET}")
        print()

        if passed:
            seeds.append(doc_id)

    if not seeds:
        print(
            f"  {RED}No documents cleared enrich_min_score={cfg.enrich_min_score} "
            f"within top-{cfg.enrich_top_docs}.{RESET}"
        )
        print(f"  {DIM}Stage 0 returns None → full corpus searched.{RESET}")
        print(f"\n  Tip: try --min-score {ranked_docs[0][1] * 0.8:.2f} to include the top result.")
        return 0

    print(f"  {GREEN}{len(seeds)} seed doc(s) selected.{RESET}")

    # ── Sub-step 0.2: Link expansion ─────────────────────────────────────
    print(f"\n{BOLD}{YELLOW}{'─' * 60}{RESET}")
    print(f"{BOLD}{YELLOW}Sub-step 0.2 — Link expansion{RESET}")
    print(f"{YELLOW}{'─' * 60}{RESET}")
    print(f"{DIM}Follows LINKS_TO edges (outbound) and reverse links (inbound) one hop.{RESET}\n")

    candidates: set[str] = set(seeds)

    for seed in seeds:
        print(f"  Seed: {BOLD}{_short(seed)}{RESET}")

        outbound = r.kg.get_linked_docs(seed, depth=1)
        new_out = [d for d in outbound if d not in candidates]
        print(f"    outbound links → {len(outbound)} doc(s), {len(new_out)} new:")
        for d in outbound:
            marker = f"  {GREEN}+added{RESET}" if d not in candidates else ""
            print(f"      {_short(d)}{marker}")
        candidates.update(outbound)

        reverse = r.kg.get_reverse_links(seed)
        rev_ids = [entry["doc_id"] for entry in reverse]
        new_rev = [d for d in rev_ids if d not in candidates]
        print(f"    inbound links  ← {len(rev_ids)} doc(s), {len(new_rev)} new:")
        for d in rev_ids:
            marker = f"  {GREEN}+added{RESET}" if d not in candidates else ""
            print(f"      {_short(d)}{marker}")
        candidates.update(rev_ids)

    print(f"\n  Candidate set after link expansion: {len(candidates)} doc(s)")

    # ── Sub-step 0.3: Cluster expansion ──────────────────────────────────
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}Sub-step 0.3 — Cluster expansion{RESET}")
    print(f"{'─' * 60}")
    print(f"{DIM}Finds the topic cluster of the top seed and adds all cluster members.{RESET}\n")

    cluster_data = r._load_cluster_cache()
    if cluster_data is None:
        print(f"  {DIM}No cluster cache found — skipped.{RESET}")
        print(f"  {DIM}(Run the pipeline with enough docs to trigger clustering.){RESET}")
    else:
        total_docs = len(r.kg.list_all_documents())
        if total_docs < cfg.min_cluster_docs:
            print(
                f"  {DIM}Corpus has {total_docs} docs, below "
                f"min_cluster_docs={cfg.min_cluster_docs} — skipped.{RESET}"
            )
        else:
            top_seed = seeds[0]
            cluster_id = cluster_data["doc_to_cluster"].get(top_seed)
            if cluster_id is None:
                print(f"  {DIM}Top seed not found in cluster map — skipped.{RESET}")
            else:
                members = cluster_data["cluster_members"].get(str(cluster_id), [])
                new_members = [d for d in members if d not in candidates]
                print(
                    f"  Top seed belongs to cluster {BOLD}{cluster_id}{RESET}  "
                    f"({len(members)} members, {len(new_members)} new)"
                )
                for d in members:
                    marker = f"  {GREEN}+added{RESET}" if d not in candidates else ""
                    print(f"    {_short(d)}{marker}")
                candidates.update(members)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}Stage 0 result — final candidate set{RESET}  ({len(candidates)} docs)")
    print(f"{'─' * 60}")
    print(f"{DIM}Stages 1 and 1.5 will search only chunks from these documents.{RESET}\n")

    for doc_id in sorted(candidates):
        origin = []
        if doc_id in seeds:
            origin.append("seed")
        elif any(
            doc_id in r.kg.get_linked_docs(s, depth=1)
            or doc_id in [e["doc_id"] for e in r.kg.get_reverse_links(s)]
            for s in seeds
        ):
            origin.append("link")
        else:
            origin.append("cluster")
        norm = doc_scores_norm.get(doc_id)
        score_str = f"  norm={norm:.4f}" if norm is not None else ""
        print(f"  [{', '.join(origin)}]{score_str}  {_short(doc_id)}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
