#!/usr/bin/env python3
"""Show everything stored in the index for a given document file path.

Usage:
    python scripts/inspect_doc.py /Users/andy/index/path/to/file.pdf
    python scripts/inspect_doc.py /Users/andy/index/path/to/file.pdf --db ./index_db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect all indexed data for a document.")
    parser.add_argument("path", help="Absolute path to the document file")
    parser.add_argument(
        "--db",
        default="./index_db",
        help="Path to the ARG database directory (default: ./index_db)",
    )
    parser.add_argument("--docs", default="~/index/", help="Docs root (default: same as --db)")
    args = parser.parse_args()

    doc_path = Path(args.path).expanduser().resolve()
    if not doc_path.exists():
        print(f"error: file not found: {doc_path}", file=sys.stderr)
        return 1

    from arg.config import ARGConfig
    from arg.pipeline import ARGPipeline

    db = Path(args.db).expanduser().resolve()
    docs = Path(args.docs).expanduser().resolve() if args.docs else db
    cfg = ARGConfig(docs_root=docs, db_path=db)
    cfg.query_rewrite = False
    cfg.query_decompose = False
    p = ARGPipeline(config=cfg)

    doc_id = str(doc_path)

    # --- 1. Knowledge graph ---
    print("=== GRAPH: document node ===")
    meta = p.indexer.kg.get_doc_metadata(doc_id)
    if not meta:
        print("  (not found in graph — has this file been indexed?)")
    else:
        print(json.dumps(meta, indent=2, default=str))

    print("\n=== GRAPH: chunk IDs ===")
    chunk_ids = p.indexer.kg.get_chunks_for_doc(doc_id)
    print(f"  ({len(chunk_ids)} chunks)")
    for cid in chunk_ids:
        print(f"  {cid}")

    print("\n=== GRAPH: outbound links (this doc → others) ===")
    linked = p.indexer.kg.get_linked_docs(doc_id, depth=1)
    if linked:
        for d in linked:
            print(f"  {d}")
    else:
        print("  (none)")

    print("\n=== GRAPH: inbound links (other docs → this one) ===")
    reverse = p.indexer.kg.get_reverse_links(doc_id)
    if reverse:
        for rev in reverse:
            print(f"  {rev}")
    else:
        print("  (none)")

    # --- 2. ChromaDB doc-level record ---
    print("\n=== CHROMA: doc-level record ===")
    docs_result = p.indexer._docs_coll.get(
        ids=[doc_id], include=["metadatas", "embeddings", "documents"]
    )
    if not docs_result["ids"]:
        print("  (not found in chroma docs collection)")
    else:
        print(json.dumps(docs_result, indent=2, default=str))

    # --- 3. ChromaDB chunks ---
    print("\n=== CHROMA: chunks ===")
    chunks_result = p.indexer._chunks_coll.get(
        where={"doc_id": doc_id},
        include=["metadatas", "embeddings", "documents"],
    )
    n = len(chunks_result["ids"])
    print(f"  ({n} chunks)")
    embeddings = (
        chunks_result["embeddings"] if chunks_result["embeddings"] is not None else [None] * n
    )
    for i, (cid, meta, text, emb) in enumerate(
        zip(
            chunks_result["ids"],
            chunks_result["metadatas"],
            chunks_result["documents"],
            embeddings,
            strict=False,
        )
    ):
        print(f"\n--- chunk {i} ---")
        print(f"  id: {cid}")
        print(json.dumps(meta, indent=2, default=str))
        print(f"  text:\n{text}")
        if emb is not None:
            print(f"  embedding: [{emb[0]:.6f}, {emb[1]:.6f}, ...] ({len(emb)} dims)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
