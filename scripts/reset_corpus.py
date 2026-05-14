#!/usr/bin/env python3
"""Delete a corpus's on-disk state.

Deletes ``{db_path}/{corpus_name}/`` entirely — Kuzu, Chroma, BM25 pickle,
hash file, cluster cache, summaries, debug traces, log file. Forces
``--confirm`` to avoid accidental data loss; the per-corpus directory
can hold significant compute (embeddings, summaries) you don't want to
lose to an off-by-one tab-complete.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def reset_corpus(db_path: Path, corpus_name: str) -> Path | None:
    corpus_dir = (db_path / corpus_name).resolve()
    if not corpus_dir.is_dir():
        return None
    shutil.rmtree(corpus_dir)
    return corpus_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reset_corpus",
        description="Delete a corpus's on-disk state. Asks for confirmation.",
    )
    parser.add_argument("--db", required=True, help="ARG database directory")
    parser.add_argument("--corpus", required=True, help="corpus name to wipe")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip the interactive prompt and delete immediately.",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve()
    corpus_dir = db_path / args.corpus
    if not corpus_dir.is_dir():
        print(f"[reset_corpus] {corpus_dir} does not exist; nothing to do.")
        return 0

    if not args.confirm:
        reply = input(
            f"This will permanently delete {corpus_dir}. Type the corpus name to confirm: "
        )
        if reply.strip() != args.corpus:
            print("[reset_corpus] confirmation did not match; aborting.")
            return 1

    removed = reset_corpus(db_path, args.corpus)
    if removed is None:
        print(f"[reset_corpus] {corpus_dir} not found.")
        return 0
    print(f"[reset_corpus] removed {removed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
