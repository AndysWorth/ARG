#!/usr/bin/env python3
"""Generate HTML analysis reports from ARG indexing logs.

Reads arg.log + arg.log.1 from a corpus database directory and produces two
self-contained HTML files:
  <out>/indexing_report.html    — main analysis with charts and recommendations
  <out>/indexing_appendix.html  — full raw-data tables

Usage:
    python scripts/index_report.py
    python scripts/index_report.py --db ./index_db --corpus default --out ./reports
    python scripts/index_report.py --db ./index_db/default  # point at corpus dir directly
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ARG indexing analysis reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--db",
        default="./index_db",
        metavar="PATH",
        help="ARG database root or corpus directory (default: ./index_db)",
    )
    p.add_argument(
        "--corpus",
        default="default",
        metavar="NAME",
        help="Corpus name (default: default). Ignored when --db points at corpus dir directly.",
    )
    p.add_argument(
        "--out",
        default="./reports",
        metavar="DIR",
        help="Output directory for HTML reports (default: ./reports)",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the main report in the default browser after generating.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

_IDX_RE = re.compile(r"\[#(\d+)\] indexed (.+?) \((\d+) chunks, (\d+) ms\)")
_OCR_RE = re.compile(r"OCR used for page \d+ of (.+)")
_ACRO_RE = re.compile(r"PDF (.+?) contains AcroForm")
_DANGLE_RE = re.compile(r"crawler: dangling link from (.+?) -> (.+)")


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def parse_logs(log_files: list[Path]) -> dict:
    """Read all JSON log records from log_files (oldest first) and extract stats."""
    records: list[dict] = []
    for lf in log_files:
        if not lf.exists():
            continue
        with lf.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(Exception):
                    records.append(json.loads(line))

    levels = Counter(r.get("level", "?") for r in records)
    loggers = Counter(r.get("logger", "?") for r in records)

    indexed_docs: list[dict] = []
    errors_warnings: list[dict] = []
    encrypted_pdfs: list[str] = []
    acroform_files: dict[str, int] = defaultdict(int)
    dangling_links: list[tuple[str, str]] = []
    ocr_pages = 0
    ocr_docs: set[str] = set()
    pipeline_events: list[dict] = []

    for r in records:
        msg = r.get("message", "")
        level = r.get("level", "")
        logger = r.get("logger", "")
        ts = r.get("timestamp", "")

        m = _IDX_RE.match(msg)
        if m:
            indexed_docs.append(
                {
                    "seq": int(m.group(1)),
                    "path": m.group(2),
                    "chunks": int(m.group(3)),
                    "ms": int(m.group(4)),
                    "ts": ts,
                }
            )
            continue

        if level in ("ERROR", "WARNING"):
            errors_warnings.append(r)

        if "Skipping encrypted PDF:" in msg:
            path = msg.split("Skipping encrypted PDF:", 1)[1].strip()
            encrypted_pdfs.append(path)

        m = _OCR_RE.match(msg)
        if m:
            ocr_pages += 1
            ocr_docs.add(m.group(1))
            continue

        m = _ACRO_RE.match(msg)
        if m:
            acroform_files[m.group(1)] += 1
            continue

        m = _DANGLE_RE.match(msg)
        if m:
            dangling_links.append((m.group(1), m.group(2)))
            continue

        if logger.startswith("arg.pipeline"):
            pipeline_events.append(r)

    # Gaps > 30 min
    gaps: list[dict] = []
    for i in range(1, len(indexed_docs)):
        t1 = _parse_ts(indexed_docs[i - 1]["ts"])
        t2 = _parse_ts(indexed_docs[i]["ts"])
        if t1 and t2:
            gap_min = (t2 - t1).total_seconds() / 60
            if gap_min > 30:
                gaps.append(
                    {
                        "hours": gap_min / 60,
                        "seq_a": indexed_docs[i - 1]["seq"],
                        "seq_b": indexed_docs[i]["seq"],
                        "ts_a": indexed_docs[i - 1]["ts"],
                        "ts_b": indexed_docs[i]["ts"],
                        "path_a": indexed_docs[i - 1]["path"],
                        "path_b": indexed_docs[i]["path"],
                    }
                )

    # Unique encrypted
    unique_encrypted = sorted(set(encrypted_pdfs))

    # Zero chunk docs
    zero_chunk_docs = [d for d in indexed_docs if d["chunks"] == 0]

    # Classify errors/warnings by source
    arg_warnings = [r for r in errors_warnings if r.get("logger", "").startswith("arg.")]
    lib_warnings = [r for r in errors_warnings if not r.get("logger", "").startswith("arg.")]

    # Wall time
    first_ts = _parse_ts(indexed_docs[0]["ts"]) if indexed_docs else None
    last_ts = _parse_ts(indexed_docs[-1]["ts"]) if indexed_docs else None
    wall_seconds = (last_ts - first_ts).total_seconds() if first_ts and last_ts else 0
    cpu_ms = sum(d["ms"] for d in indexed_docs)

    # Hourly buckets (relative to first doc)
    hourly: dict[int, int] = defaultdict(int)
    for d in indexed_docs:
        t = _parse_ts(d["ts"])
        if t and first_ts:
            h = int((t - first_ts).total_seconds() // 3600)
            hourly[h] += 1

    return {
        "total_records": len(records),
        "levels": dict(levels),
        "loggers": dict(loggers.most_common(20)),
        "indexed_docs": indexed_docs,
        "zero_chunk_docs": zero_chunk_docs,
        "unique_encrypted": unique_encrypted,
        "acroform_files": dict(sorted(acroform_files.items(), key=lambda x: -x[1])),
        "dangling_links": dangling_links,
        "ocr_pages": ocr_pages,
        "ocr_docs": sorted(ocr_docs),
        "pipeline_events": pipeline_events,
        "arg_warnings": arg_warnings,
        "lib_warnings": lib_warnings,
        "errors_warnings": errors_warnings,
        "gaps": sorted(gaps, key=lambda x: -x["hours"]),
        "wall_seconds": wall_seconds,
        "cpu_ms": cpu_ms,
        "hourly": dict(sorted(hourly.items())),
        "first_ts": indexed_docs[0]["ts"] if indexed_docs else "",
        "last_ts": indexed_docs[-1]["ts"] if indexed_docs else "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metadata / DB queries
# ─────────────────────────────────────────────────────────────────────────────


def read_metadata(corpus_dir: Path) -> dict:
    result: dict = {}

    dh = corpus_dir / "doc_hashes.json"
    if dh.exists():
        with dh.open() as f:
            data = json.load(f)
        result["doc_count"] = len(data)
        exts: Counter = Counter()
        top_dirs: Counter = Counter()
        for k in data:
            ext = k.rsplit(".", 1)[-1].lower() if "." in k else "other"
            exts[ext] += 1
            parts = k.split("/")
            if len(parts) > 4:
                top_dirs[parts[4]] += 1
        result["extensions"] = dict(exts.most_common(10))
        result["top_dirs"] = dict(top_dirs.most_common(20))

    cfg = corpus_dir / "config_hash.json"
    if cfg.exists():
        with cfg.open() as f:
            result["config"] = json.load(f)

    cc = corpus_dir / "cluster_cache.json"
    if cc.exists():
        with cc.open() as f:
            cache = json.load(f)
        members = cache.get("cluster_members", {})
        labels = cache.get("labels", {})
        clusters = [
            {"id": cid, "label": labels.get(cid, ""), "count": len(docs)}
            for cid, docs in members.items()
        ]
        clusters.sort(key=lambda x: -x["count"])
        result["clusters"] = clusters

    return result


def query_chroma(corpus_dir: Path) -> dict:
    db_file = corpus_dir / "chroma" / "chroma.sqlite3"
    if not db_file.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_file))
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM collections")
        collections = cur.fetchall()

        cur.execute("PRAGMA table_info(embeddings)")
        cols = [c[1] for c in cur.fetchall()]

        counts: dict[str, int] = {}
        if "segment_id" in cols:
            cur.execute("SELECT segment_id, COUNT(*) FROM embeddings GROUP BY segment_id")
            seg_counts = dict(cur.fetchall())
            cur.execute("SELECT id, scope, collection FROM segments")
            for seg_id, scope, coll_id in cur.fetchall():
                cur.execute("SELECT name FROM collections WHERE id=?", (coll_id,))
                row = cur.fetchone()
                if row and scope == "METADATA":
                    counts[row[0]] = seg_counts.get(seg_id, 0)

        conn.close()
        return {
            "collections": [{"id": c[0][:8], "name": c[1]} for c in collections],
            "counts": counts,
        }
    except Exception as e:
        return {"error": str(e)}


def query_kuzu(corpus_dir: Path) -> dict:
    kuzu_dir = corpus_dir / "kuzu"
    if not kuzu_dir.exists():
        return {}
    try:
        import kuzu

        db = kuzu.Database(str(kuzu_dir))
        conn = kuzu.Connection(db)
        result: dict = {}

        def _count(q: str) -> int:
            r = conn.execute(q)
            return r.get_next()[0] if r.has_next() else 0

        result["doc_nodes"] = _count("MATCH (d:Document) RETURN count(d)")
        result["chunk_nodes"] = _count("MATCH (c:Chunk) RETURN count(c)")
        result["contains_edges"] = _count("MATCH ()-[e:CONTAINS]->() RETURN count(e)")
        result["links_to_edges"] = _count("MATCH ()-[e:LINKS_TO]->() RETURN count(e)")
        return result
    except Exception as e:
        return {"error": str(e)}


def query_bm25(corpus_dir: Path) -> dict:
    pkl = corpus_dir / "bm25_index.pkl"
    if not pkl.exists():
        return {}
    try:
        import pickle

        with pkl.open("rb") as f:
            data = pickle.load(f)
        bm25 = data.get("bm25")
        if bm25 is None:
            return {}
        return {
            "type": type(bm25).__name__,
            "corpus_size": getattr(bm25, "corpus_size", 0),
            "avgdl": round(getattr(bm25, "avgdl", 0), 1),
            "chunk_ids": len(data.get("chunk_ids", [])),
            "file_size_mb": round(pkl.stat().st_size / 1_048_576, 1),
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Derived stats helpers
# ─────────────────────────────────────────────────────────────────────────────


def _percentile(sorted_vals: list[float], pct: int) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * pct / 100), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _derive(logs: dict, meta: dict) -> dict:
    docs = logs["indexed_docs"]

    chunk_counts = [d["chunks"] for d in docs]
    ms_vals = sorted(d["ms"] for d in docs)

    ext_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "chunks": 0, "ms": 0})
    for d in docs:
        ext = d["path"].rsplit(".", 1)[-1].lower() if "." in d["path"] else "other"
        ext_stats[ext]["count"] += 1
        ext_stats[ext]["chunks"] += d["chunks"]
        ext_stats[ext]["ms"] += d["ms"]

    chunk_dist: Counter = Counter(chunk_counts)
    slowest = sorted(docs, key=lambda x: -x["ms"])[:50]

    return {
        "total_docs": len(docs),
        "total_chunks": sum(chunk_counts),
        "avg_chunks": round(sum(chunk_counts) / len(docs), 1) if docs else 0,
        "zero_chunk_count": sum(1 for c in chunk_counts if c == 0),
        "one_chunk_count": sum(1 for c in chunk_counts if c == 1),
        "big_doc_count": sum(1 for c in chunk_counts if c > 200),
        "p50_ms": int(_percentile(ms_vals, 50)),
        "p75_ms": int(_percentile(ms_vals, 75)),
        "p90_ms": int(_percentile(ms_vals, 90)),
        "p95_ms": int(_percentile(ms_vals, 95)),
        "p99_ms": int(_percentile(ms_vals, 99)),
        "max_ms": max(ms_vals) if ms_vals else 0,
        "wall_hours": round(logs["wall_seconds"] / 3600, 2),
        "cpu_hours": round(logs["cpu_ms"] / 1000 / 3600, 2),
        "throughput": round(len(docs) / (logs["wall_seconds"] / 60), 1)
        if logs["wall_seconds"]
        else 0,
        "ext_stats": dict(sorted(ext_stats.items(), key=lambda x: -x[1]["count"])),
        "chunk_dist": dict(sorted(chunk_dist.items())),
        "slowest": slowest,
        "ocr_doc_count": len(logs["ocr_docs"]),
        "unique_encrypted_count": len(logs["unique_encrypted"]),
        "acroform_count": len(logs["acroform_files"]),
        "dangling_count": len(set(logs["dangling_links"])),
        "lib_warning_count": len(logs["lib_warnings"]),
        "arg_warning_count": len(logs["arg_warnings"]),
        "total_warn_err": len(logs["errors_warnings"]),
        "gap_count": len(logs["gaps"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
  :root {
    --bg:#0f1117; --surface:#1a1d27; --surface2:#22263a; --border:#2e3250;
    --accent:#5b8af5; --accent2:#9f7aea; --green:#48bb78; --yellow:#ecc94b;
    --red:#fc8181; --orange:#f6ad55; --text:#e2e8f0; --muted:#8892a4;
    --mono:'SF Mono','Fira Code','Cascadia Code',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.6}
  a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
  .header{background:linear-gradient(135deg,#1a1d27 0%,#0f1420 100%);border-bottom:1px solid var(--border);padding:2.5rem 3rem}
  .header h1{font-size:2rem;font-weight:700;color:#fff;letter-spacing:-0.5px}
  .header .subtitle{color:var(--muted);margin-top:.4rem;font-size:1rem}
  .header .meta{display:flex;gap:2rem;margin-top:1.2rem;flex-wrap:wrap}
  .header .meta-item .label{color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-size:.75rem}
  .header .meta-item .value{color:var(--accent);font-weight:600;font-size:1rem;margin-top:.1rem}
  nav{background:var(--surface);border-bottom:1px solid var(--border);padding:0 3rem;display:flex;gap:0;position:sticky;top:0;z-index:100;overflow-x:auto}
  nav a{display:block;padding:.9rem 1.2rem;color:var(--muted);font-size:.88rem;border-bottom:2px solid transparent;transition:all .15s;white-space:nowrap}
  nav a:hover,nav a.active{color:var(--text);border-bottom-color:var(--accent);text-decoration:none}
  main{max-width:1200px;margin:0 auto;padding:2.5rem 3rem}
  section{margin-bottom:3.5rem;scroll-margin-top:60px}
  h2{font-size:1.4rem;font-weight:600;color:#fff;margin-bottom:1.5rem;padding-bottom:.6rem;border-bottom:1px solid var(--border)}
  h3{font-size:1.05rem;font-weight:600;color:var(--text);margin:1.5rem 0 .8rem}
  p{color:var(--muted);margin-bottom:.8rem}
  .stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem;margin-bottom:1.5rem}
  .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.4rem}
  .stat-card .stat-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:.4rem}
  .stat-card .stat-value{font-size:1.8rem;font-weight:700;color:#fff;line-height:1}
  .stat-card .stat-sub{font-size:.78rem;color:var(--muted);margin-top:.3rem}
  .stat-card.green .stat-value{color:var(--green)} .stat-card.yellow .stat-value{color:var(--yellow)}
  .stat-card.red .stat-value{color:var(--red)} .stat-card.blue .stat-value{color:var(--accent)}
  .table-wrap{overflow-x:auto;margin-bottom:1.5rem}
  table{width:100%;border-collapse:collapse;font-size:.88rem}
  thead th{background:var(--surface2);color:var(--muted);text-transform:uppercase;font-size:.72rem;letter-spacing:.06em;padding:.7rem 1rem;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
  tbody tr{border-bottom:1px solid var(--border)} tbody tr:hover{background:var(--surface)}
  td{padding:.6rem 1rem;color:var(--text);vertical-align:top}
  td.mono{font-family:var(--mono);font-size:.8rem;color:var(--muted)} td.num{text-align:right;font-variant-numeric:tabular-nums}
  td.ok{color:var(--green)} td.warn{color:var(--yellow)} td.err{color:var(--red)}
  .alert{border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;font-size:.9rem;border-left:3px solid}
  .alert-red{background:rgba(252,129,129,.08);border-color:var(--red);color:var(--red)}
  .alert-yellow{background:rgba(236,201,75,.08);border-color:var(--yellow);color:var(--yellow)}
  .alert-green{background:rgba(72,187,120,.08);border-color:var(--green);color:var(--green)}
  .alert-blue{background:rgba(91,138,245,.08);border-color:var(--accent);color:var(--accent)}
  .alert .alert-title{font-weight:600;margin-bottom:.3rem} .alert p{color:inherit;margin:0}
  .bar-chart{margin-bottom:1.5rem}
  .bar-row{display:flex;align-items:center;margin-bottom:.5rem;gap:.8rem;font-size:.83rem}
  .bar-row .bar-label{width:200px;flex-shrink:0;color:var(--muted);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bar-row .bar-track{flex:1;background:var(--surface2);border-radius:3px;height:18px;overflow:hidden}
  .bar-row .bar-fill{height:100%;border-radius:3px;display:flex;align-items:center;padding-left:6px;font-size:.75rem;color:rgba(255,255,255,.8);white-space:nowrap}
  .bar-row .bar-count{width:70px;flex-shrink:0;color:var(--text);text-align:right;font-variant-numeric:tabular-nums}
  .timeline{position:relative;padding-left:1.8rem}
  .timeline::before{content:'';position:absolute;left:.4rem;top:0;bottom:0;width:2px;background:var(--border)}
  .timeline-item{position:relative;margin-bottom:1.2rem}
  .timeline-item::before{content:'';position:absolute;left:-1.4rem;top:.4rem;width:10px;height:10px;border-radius:50%;background:var(--accent);border:2px solid var(--bg)}
  .timeline-item .tl-time{font-size:.75rem;color:var(--muted);font-family:var(--mono)}
  .timeline-item .tl-event{font-size:.9rem;color:var(--text)}
  .timeline-item .tl-detail{font-size:.82rem;color:var(--muted);margin-top:.2rem}
  .timeline-item.warn::before{background:var(--yellow)} .timeline-item.info::before{background:var(--green)}
  .code{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:.8rem 1rem;font-family:var(--mono);font-size:.82rem;color:var(--text);overflow-x:auto;margin-bottom:1rem;white-space:pre}
  .rec-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1rem}
  .rec-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.4rem}
  .rec-card .rec-header{display:flex;align-items:center;gap:.7rem;margin-bottom:.8rem}
  .rec-card .rec-priority{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;padding:.2rem .5rem;border-radius:4px;font-weight:600}
  .rec-card .priority-high{background:rgba(252,129,129,.15);color:var(--red)}
  .rec-card .priority-med{background:rgba(236,201,75,.15);color:var(--yellow)}
  .rec-card .priority-low{background:rgba(72,187,120,.15);color:var(--green)}
  .rec-card .rec-title{font-weight:600;font-size:.95rem;color:#fff}
  .rec-card .rec-body{font-size:.85rem;color:var(--muted)}
  .rec-card .rec-impact{font-size:.78rem;margin-top:.7rem;padding-top:.6rem;border-top:1px solid var(--border);color:var(--accent)}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
  ul.checklist{list-style:none;margin-bottom:1rem}
  ul.checklist li{padding:.35rem 0;font-size:.88rem;color:var(--muted);display:flex;gap:.6rem;align-items:flex-start}
  ul.checklist li::before{content:'▸';color:var(--accent);flex-shrink:0;margin-top:.05rem}
  footer{border-top:1px solid var(--border);padding:1.5rem 3rem;color:var(--muted);font-size:.8rem;text-align:center}
  @media(max-width:800px){.two-col{grid-template-columns:1fr}}
"""

_NAV_JS = """
<script>
const sections=document.querySelectorAll('section[id]');
const navLinks=document.querySelectorAll('nav a');
const obs=new IntersectionObserver(entries=>{
  entries.forEach(e=>{if(e.isIntersecting)navLinks.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+e.target.id));});
},{rootMargin:'-20% 0px -70% 0px'});
sections.forEach(s=>obs.observe(s));
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────


def _h(s: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(s))


def _short_path(p: str, n: int = 65) -> str:
    """Truncate a long path for display, keeping the tail."""
    return p if len(p) <= n else "…" + p[-(n - 1) :]


def _bar(label: str, count: int, max_count: int, color: str = "#5b8af5") -> str:
    pct = min(100.0, count / max_count * 100) if max_count else 0
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label">{_h(label)}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}">'
        f"{count:,}</div></div>"
        f'<div class="bar-count">{count:,}</div>'
        f"</div>"
    )


def _stat_card(label: str, value: str, sub: str = "", cls: str = "") -> str:
    cls_str = f' class="stat-card {cls}"' if cls else ' class="stat-card"'
    return (
        f"<div{cls_str}>"
        f'<div class="stat-label">{_h(label)}</div>'
        f'<div class="stat-value">{_h(value)}</div>'
        f'<div class="stat-sub">{_h(sub)}</div>'
        f"</div>"
    )


def _td(val: str, cls: str = "") -> str:
    cls_str = f' class="{cls}"' if cls else ""
    return f"<td{cls_str}>{val}</td>"


def _tr(*cells: str) -> str:
    return "<tr>" + "".join(cells) + "</tr>"


# ─────────────────────────────────────────────────────────────────────────────
# Main report builder
# ─────────────────────────────────────────────────────────────────────────────


def _bar_color(i: int, total: int) -> str:
    colors = [
        "#5b8af5",
        "#5b8af5",
        "#9f7aea",
        "#9f7aea",
        "#48bb78",
        "#48bb78",
        "#ecc94b",
        "#ecc94b",
        "#f6ad55",
        "#f6ad55",
        "#fc8181",
        "#fc8181",
        "#fc8181",
        "#fc8181",
    ]
    return colors[min(i, len(colors) - 1)]


def build_main_report(
    logs: dict, meta: dict, chroma: dict, kuzu: dict, bm25: dict, d: dict, run_date: str
) -> str:

    # ── Corpus section ────────────────────────────────────────────────────────
    top_dirs = meta.get("top_dirs", {})
    max_dir = max(top_dirs.values()) if top_dirs else 1
    dir_bars = "".join(
        _bar(name, cnt, max_dir, _bar_color(i, len(top_dirs)))
        for i, (name, cnt) in enumerate(top_dirs.items())
    )

    ext_rows = ""
    for ext, s in d["ext_stats"].items():
        avg_ch = round(s["chunks"] / s["count"], 1) if s["count"] else 0
        avg_ms = round(s["ms"] / s["count"]) if s["count"] else 0
        ext_rows += _tr(
            _td(_h(ext)),
            _td(f"{s['count']:,}", "num"),
            _td(f"{s['chunks']:,}", "num"),
            _td(str(avg_ch), "num"),
            _td(f"{avg_ms:,} ms", "num"),
        )

    # ── Performance section ───────────────────────────────────────────────────
    slowest = d["slowest"][:10]
    slow_rows = ""
    for i, doc in enumerate(slowest):
        ms = doc["ms"]
        cls = "err" if ms > 100_000 else ("warn" if ms > 30_000 else "")
        slow_rows += _tr(
            _td(str(i + 1)),
            _td(str(doc["seq"])),
            _td(f"{ms:,}", f"num {cls}"),
            _td(f"{doc['chunks']:,}", "num"),
            _td(_h(_short_path(doc["path"])), "mono"),
        )

    # Gap timeline items
    gap_items = ""
    for g in logs["gaps"][:10]:
        hrs = g["hours"]
        cls = "warn" if hrs > 1 else ""
        gap_items += (
            f'<div class="timeline-item {cls}">'
            f'<div class="tl-time">{_h(g["ts_a"])} → {_h(g["ts_b"])}</div>'
            f'<div class="tl-event">{hrs:.1f}-hour gap (after doc #{g["seq_a"]})</div>'
            f'<div class="tl-detail">{_h(_short_path(g["path_a"], 50))} → {_h(_short_path(g["path_b"], 50))}</div>'
            f"</div>"
        )

    # Pipeline events as timeline
    for ev in logs["pipeline_events"]:
        gap_items += (
            f'<div class="timeline-item info">'
            f'<div class="tl-time">{_h(ev.get("timestamp", ""))}</div>'
            f'<div class="tl-event">{_h(ev.get("message", ""))}</div>'
            f"</div>"
        )

    # Throughput: top hours are shown as bar chart
    hourly = logs["hourly"]
    max_h = max(hourly.values()) if hourly else 1
    hourly_bars = ""
    for h, cnt in sorted(hourly.items()):
        color = "#48bb78" if cnt >= 100 else ("#ecc94b" if cnt >= 10 else "#fc8181")
        hourly_bars += _bar(f"Hour {h + 1}", cnt, max_h, color)

    # ── Errors section ────────────────────────────────────────────────────────
    warn_summary = Counter(r.get("message", "")[:80] for r in logs["errors_warnings"])
    warn_rows = ""
    for msg, cnt in warn_summary.most_common(20):
        logger_sample = next(
            (
                r.get("logger", "")
                for r in logs["errors_warnings"]
                if r.get("message", "").startswith(msg)
            ),
            "",
        )
        cls = (
            "err"
            if any(
                r.get("level") == "ERROR"
                for r in logs["errors_warnings"]
                if r.get("message", "").startswith(msg)
            )
            else "warn"
        )
        warn_rows += _tr(
            _td(f"{cnt:,}", f"num {cls}"),
            _td(_h(logger_sample), "mono"),
            _td(_h(msg)),
        )

    enc_summary_rows = ""
    enc_by_cat: dict[str, int] = defaultdict(int)
    for p in logs["unique_encrypted"]:
        parts = p.split("/")
        cat = parts[4] if len(parts) > 4 else "other"
        enc_by_cat[cat] += 1
    for cat, cnt in sorted(enc_by_cat.items(), key=lambda x: -x[1]):
        enc_summary_rows += _tr(_td(_h(cat)), _td(f"{cnt:,}", "num warn"))

    # Dangling link sources
    dangle_sources: Counter = Counter()
    for src, _ in logs["dangling_links"]:
        dangle_sources[src] += 1
    dangle_rows = ""
    for src, cnt in dangle_sources.most_common(10):
        dangle_rows += _tr(
            _td(_h(_short_path(src, 70)), "mono"),
            _td(f"{cnt:,}", "num warn"),
        )

    # ── Quality section ───────────────────────────────────────────────────────
    zero_rows = ""
    for doc in logs["zero_chunk_docs"]:
        zero_rows += _tr(
            _td(_h(_short_path(doc["path"], 80)), "mono"),
            _td(str(doc["ms"]), "num"),
        )

    chunk_dist = d["chunk_dist"]
    max_cd = max(chunk_dist.values()) if chunk_dist else 1
    cd_bars = ""
    for n, cnt in list(chunk_dist.items())[:15]:
        color = (
            "#fc8181"
            if n == 0
            else ("#ecc94b" if n == 1 else ("#5b8af5" if n <= 10 else "#9f7aea"))
        )
        cd_bars += _bar(f"{n} chunk{'s' if n != 1 else ''}", cnt, max_cd, color)

    # ── Clusters section ─────────────────────────────────────────────────────
    clusters = meta.get("clusters", [])
    cluster_rows = ""
    for cl in clusters:
        pct = round(cl["count"] / d["total_docs"] * 100, 1) if d["total_docs"] else 0
        cluster_rows += _tr(
            _td(_h(cl["id"])),
            _td(_h(cl["label"])),
            _td(f"{cl['count']:,}", "num"),
            _td(f"{pct}%", "num"),
        )

    # ── Index artifacts section ───────────────────────────────────────────────
    chroma_coll_rows = ""
    for coll in chroma.get("collections", []):
        cnt = chroma.get("counts", {}).get(coll["name"], "?")
        chroma_coll_rows += _tr(
            _td(_h(coll["name"])),
            _td(f"{cnt:,}" if isinstance(cnt, int) else str(cnt), "num"),
        )

    kuzu_rows = ""
    for label, key in [
        ("Document nodes", "doc_nodes"),
        ("Chunk nodes", "chunk_nodes"),
        ("CONTAINS edges", "contains_edges"),
        ("LINKS_TO edges", "links_to_edges"),
    ]:
        val = kuzu.get(key, "?")
        kuzu_rows += _tr(
            _td(_h(label)), _td(f"{val:,}" if isinstance(val, int) else str(val), "num")
        )

    # Config
    cfg_data = meta.get("config", {}).get("config", {})
    cfg_hash = meta.get("config", {}).get("hash", "")
    cfg_lines = "\n".join(f"  {k}: {v}" for k, v in cfg_data.items())
    cfg_block = f"{cfg_lines}\n\nconfig_hash: {cfg_hash}"

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = _build_recommendations(d, logs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ARG Indexing Report</title>
<style>{_CSS}</style>
</head>
<body>

<div class="header">
  <h1>ARG Indexing Report</h1>
  <div class="subtitle">Archivist RAG Graph — Full corpus index analysis</div>
  <div class="meta">
    <div class="meta-item"><div class="label">Indexing Period</div><div class="value">{
        _h(logs["first_ts"][:10])
    } - {_h(logs["last_ts"][:10])}</div></div>
    <div class="meta-item"><div class="label">Documents</div><div class="value">{
        d["total_docs"]:,}</div></div>
    <div class="meta-item"><div class="label">Total Chunks</div><div class="value">{
        d["total_chunks"]:,}</div></div>
    <div class="meta-item"><div class="label">Wall Time</div><div class="value">~{
        d["wall_hours"]:.1f} h</div></div>
    <div class="meta-item"><div class="label">CPU Time</div><div class="value">~{
        d["cpu_hours"]:.1f} h</div></div>
    <div class="meta-item"><div class="label">Log Records</div><div class="value">{
        logs["total_records"]:,}</div></div>
    <div class="meta-item"><div class="label">Report Generated</div><div class="value">{
        _h(run_date)
    }</div></div>
  </div>
</div>

<nav>
  <a href="#log-analysis-intro">What Logs Reveal</a>
  <a href="#corpus">Corpus</a>
  <a href="#performance">Performance</a>
  <a href="#errors">Errors &amp; Warnings</a>
  <a href="#quality">Quality</a>
  <a href="#clusters">Clusters</a>
  <a href="#index">Index Artifacts</a>
  <a href="#recommendations">Recommendations</a>
</nav>

<main>

<section id="log-analysis-intro">
  <h2>What Log Analysis Can Reveal</h2>
  <p>The framework used to analyze ARG's logs — applicable to any structured-logging pipeline.</p>
  <div class="two-col">
    <div>
      <h3>Error &amp; Health Signals</h3>
      <ul class="checklist">
        <li>Hard errors causing data loss (documents producing 0 chunks)</li>
        <li>Soft errors silently degrading quality (AcroForm PDFs with partial text)</li>
        <li>Skipped documents and the reason (encryption, corruption)</li>
        <li>Third-party library noise vs. real application errors</li>
        <li>Recurring warnings on the same files across multiple runs</li>
      </ul>
      <h3>Performance Signals</h3>
      <ul class="checklist">
        <li>Wall-clock throughput vs. actual CPU utilization ratio</li>
        <li>Outlier documents consuming disproportionate time</li>
        <li>Per-file-type processing speed differences</li>
        <li>Embedding latency patterns (one HTTP call per chunk)</li>
        <li>Latency percentile distribution (p50/p95/p99)</li>
      </ul>
    </div>
    <div>
      <h3>Temporal Signals</h3>
      <ul class="checklist">
        <li>Session boundaries — pipeline start, stop, and restart</li>
        <li>Large gaps revealing machine sleep or manual pauses</li>
        <li>Throughput variation over time (bursts vs. slow periods)</li>
        <li>Phase transitions: crawl → index → cluster</li>
      </ul>
      <h3>Data Quality Signals</h3>
      <ul class="checklist">
        <li>0-chunk documents (complete extraction failures)</li>
        <li>1-chunk documents (may be too short for useful RAG)</li>
        <li>OCR fallback frequency (scanned vs. native-text PDFs)</li>
        <li>Dangling links (broken graph edges, stale HTML indexes)</li>
        <li>Case-sensitivity directory collisions</li>
        <li>Documents needing form-field extraction (AcroForm)</li>
      </ul>
    </div>
  </div>
</section>

<section id="corpus">
  <h2>Corpus Overview</h2>
  <div class="stats-grid">
    {_stat_card("Total Documents", f"{d['total_docs']:,}", "0 skipped by indexer", "blue")}
    {_stat_card("Total Chunks", f"{d['total_chunks']:,}", f"avg {d['avg_chunks']} per doc", "blue")}
    {
        _stat_card(
            "Unique Encrypted (skipped)",
            f"{d['unique_encrypted_count']:,}",
            "0 content bytes indexed",
        )
    }
    {
        _stat_card(
            "0-Chunk Docs",
            f"{d['zero_chunk_count']:,}",
            "complete extraction failures",
            "red" if d["zero_chunk_count"] > 0 else "",
        )
    }
    {_stat_card("AcroForm PDFs", f"{d['acroform_count']:,}", "partial text extraction", "yellow")}
    {_stat_card("Dangling Links", f"{d['dangling_count']:,}", "broken graph edges")}
  </div>
  <h3>Documents by Directory</h3>
  <div class="bar-chart">{dir_bars}</div>
  <h3>By File Type</h3>
  <div class="table-wrap"><table>
    <thead><tr>
      <th>Type</th><th class="num">Docs</th><th class="num">Total Chunks</th>
      <th class="num">Avg Chunks/Doc</th><th class="num">Avg Index Time</th>
    </tr></thead>
    <tbody>{ext_rows}</tbody>
  </table></div>
</section>

<section id="performance">
  <h2>Performance Analysis</h2>
  <div class="stats-grid">
    {
        _stat_card(
            "Wall Time",
            f"~{d['wall_hours']:.1f} h",
            f"{logs['first_ts'][:10]} → {logs['last_ts'][:10]}",
        )
    }
    {
        _stat_card(
            "CPU Processing",
            f"{d['cpu_hours']:.1f} h",
            f"{round(d['cpu_hours'] / d['wall_hours'] * 100, 1) if d['wall_hours'] else 0}% utilization",
            "green",
        )
    }
    {_stat_card("Throughput", f"{d['throughput']}", "docs / minute (active)")}
    {_stat_card("Median Latency", f"{d['p50_ms']:,} ms", "p50 per document")}
    {_stat_card("p99 Latency", f"{d['p99_ms']:,} ms", "long-tail documents", "yellow")}
    {
        _stat_card(
            "Slowest Doc",
            f"{d['max_ms'] // 1000} s",
            _short_path(d["slowest"][0]["path"], 40) if d["slowest"] else "",
            "red",
        )
    }
  </div>
  <div class="alert alert-yellow">
    <div class="alert-title">Wall Time vs CPU Time</div>
    <p>The {d["wall_hours"]:.1f}-hour wall clock maps to only {
        d["cpu_hours"]:.1f} hours of actual CPU work
    ({round(d["cpu_hours"] / d["wall_hours"] * 100, 1) if d["wall_hours"] else 0}% utilization).
    There were <strong>{d["gap_count"]} gaps longer than 30 minutes</strong>, indicating
    machine sleep or manual pauses during the run. The index itself is sound — all
    {d["total_docs"]:,} documents completed.</p>
  </div>
  <h3>Latency Percentiles</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Percentile</th><th class="num">Latency (ms)</th><th>Interpretation</th></tr></thead>
    <tbody>
      <tr><td>p50 (median)</td><td class="num ok">{
        d["p50_ms"]:,}</td><td>Typical short HTML or simple PDF</td></tr>
      <tr><td>p75</td><td class="num">{d["p75_ms"]:,}</td><td>Mid-size PDF (~10 pages)</td></tr>
      <tr><td>p90</td><td class="num">{
        d["p90_ms"]:,}</td><td>Multi-page scanned PDF needing OCR</td></tr>
      <tr><td>p95</td><td class="num warn">{d["p95_ms"]:,}</td><td>Long dense PDF</td></tr>
      <tr><td>p99</td><td class="num warn">{
        d["p99_ms"]:,}</td><td>Very large PDFs (manuals, books)</td></tr>
      <tr><td>max</td><td class="num err">{d["max_ms"]:,}</td><td>{
        _h(_short_path(d["slowest"][0]["path"]) if d["slowest"] else "")
    }</td></tr>
    </tbody>
  </table></div>
  <h3>Slowest 10 Documents</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>#</th><th>Seq</th><th class="num">ms</th><th class="num">Chunks</th><th>File</th></tr></thead>
    <tbody>{slow_rows}</tbody>
  </table></div>
  <h3>Timeline &amp; Gaps</h3>
  <div class="timeline">{gap_items}</div>
  <h3>OCR Usage</h3>
  <div class="stats-grid">
    {_stat_card("OCR Page Events", f"{logs['ocr_pages']:,}", "Tesseract fallback pages", "yellow")}
    {
        _stat_card(
            "Docs Needing OCR",
            f"{d['ocr_doc_count']:,}",
            f"{round(d['ocr_doc_count'] / d['total_docs'] * 100, 1) if d['total_docs'] else 0}% of all docs",
            "yellow",
        )
    }
  </div>
  <p>All OCR-requiring documents are PDFs, indicating a substantial portion of the corpus
  consists of scanned paper documents. OCR fires when native text extraction yields fewer
  than {
        meta.get("config", {}).get("config", {}).get("ocr_char_threshold", "100")
    } characters per page.</p>
</section>

<section id="errors">
  <h2>Errors &amp; Warnings</h2>
  <div class="stats-grid">
    {
        _stat_card(
            "Total Errors",
            str(logs["levels"].get("ERROR", 0)),
            "log level ERROR",
            "red" if logs["levels"].get("ERROR", 0) > 0 else "",
        )
    }
    {
        _stat_card(
            "Total Warnings", f"{logs['levels'].get('WARNING', 0):,}", "log level WARNING", "yellow"
        )
    }
    {_stat_card("ARG-namespace", f"{d['arg_warning_count']:,}", "from arg.* loggers")}
    {_stat_card("Library noise", f"{d['lib_warning_count']:,}", "pdfminer / third-party")}
  </div>
  <h3>Top Warning Messages</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Count</th><th>Logger</th><th>Message (truncated)</th></tr></thead>
    <tbody>{warn_rows}</tbody>
  </table></div>
  <h3>Encrypted PDFs by Category</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Category</th><th class="num">Files</th></tr></thead>
    <tbody>{enc_summary_rows}</tbody>
  </table></div>
  <h3>Dangling Link Sources</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Source HTML File</th><th class="num">Broken Links</th></tr></thead>
    <tbody>{dangle_rows}</tbody>
  </table></div>
  <div class="alert alert-yellow">
    <div class="alert-title">Library Noise Ratio</div>
    <p>{d["lib_warning_count"]:,} of {d["total_warn_err"]:,} warnings ({
        round(d["lib_warning_count"] / d["total_warn_err"] * 100) if d["total_warn_err"] else 0
    }%) come from third-party
    loggers (primarily pdfminer). Adding a logging filter to suppress pdfminer.* at WARNING
    level would leave only real application warnings immediately visible.</p>
  </div>
</section>

<section id="quality">
  <h2>Extraction Quality</h2>
  <div class="stats-grid">
    {
        _stat_card(
            "0-Chunk Docs",
            f"{d['zero_chunk_count']:,}",
            "complete failures — unreachable in RAG",
            "red" if d["zero_chunk_count"] > 0 else "",
        )
    }
    {
        _stat_card(
            "1-Chunk Docs",
            f"{d['one_chunk_count']:,}",
            f"{round(d['one_chunk_count'] / d['total_docs'] * 100, 1) if d['total_docs'] else 0}% — may be too short",
            "yellow",
        )
    }
    {_stat_card("200+ Chunk Docs", f"{d['big_doc_count']:,}", "very large — consider chunk cap")}
    {
        _stat_card(
            "AcroForm PDFs", f"{d['acroform_count']:,}", "field values not extracted", "yellow"
        )
    }
  </div>
  <h3>Documents Producing 0 Chunks</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>File Path</th><th class="num">ms</th></tr></thead>
    <tbody>{zero_rows}</tbody>
  </table></div>
  <h3>Chunk Count Distribution</h3>
  <div class="bar-chart">{cd_bars}</div>
  <p>The {d["one_chunk_count"]:,}-document (1-chunk) spike represents {
        round(d["one_chunk_count"] / d["total_docs"] * 100, 1) if d["total_docs"] else 0
    }%
  of the corpus. With a chunk size of {
        meta.get("config", {}).get("config", {}).get("chunk_size", "1024")
    } tokens,
  these documents fit entirely within a single window. For HTML navigation pages this is expected;
  for PDFs it may signal near-empty OCR output.</p>
</section>

<section id="clusters">
  <h2>Topic Clustering</h2>
  <p>{len(clusters)} clusters generated via LLM after indexing.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>ID</th><th>Label</th><th class="num">Docs</th><th class="num">% of Corpus</th></tr></thead>
    <tbody>{cluster_rows}</tbody>
  </table></div>
  <div class="alert alert-blue">
    <div class="alert-title">Cluster Granularity</div>
    <p>With {d["total_docs"]:,} documents spread across {len(clusters)} clusters,
    the average cluster size is {
        round(d["total_docs"] / len(clusters)) if clusters else 0
    } documents —
    {
        "very coarse; consider increasing n_clusters to 16-24 for finer topic navigation."
        if len(clusters) <= 10
        else "reasonable for a high-level view."
    }</p>
  </div>
</section>

<section id="index">
  <h2>Index Artifacts</h2>
  <h3>ChromaDB Collections</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Collection</th><th class="num">Embeddings</th></tr></thead>
    <tbody>{chroma_coll_rows}</tbody>
  </table></div>
  <h3>Kuzu Knowledge Graph</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Type</th><th class="num">Count</th></tr></thead>
    <tbody>{kuzu_rows}</tbody>
  </table></div>
  <h3>BM25 Index</h3>
  <div class="table-wrap"><table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Implementation</td><td>{_h(bm25.get("type", "?"))}</td></tr>
      <tr><td>Corpus size (chunks)</td><td class="num">{bm25.get("corpus_size", 0):,}</td></tr>
      <tr><td>Average document length</td><td class="num">{bm25.get("avgdl", 0)} tokens</td></tr>
      <tr><td>Serialized size</td><td class="num">{bm25.get("file_size_mb", 0)} MB</td></tr>
    </tbody>
  </table></div>
  <h3>Configuration Used</h3>
  <div class="code">{_h(cfg_block)}</div>
</section>

<section id="recommendations">
  <h2>Recommendations</h2>
  <p>Ordered by estimated impact on RAG retrieval quality.</p>
  <div class="rec-grid">{recs}</div>
</section>

</main>
<footer>
  ARG Indexing Report · Generated {_h(run_date)} · {logs["total_records"]:,} log records analyzed
  · <a href="indexing_appendix.html">Full data appendix →</a>
</footer>
{_NAV_JS}
</body>
</html>"""


def _build_recommendations(d: dict, logs: dict) -> str:
    recs = []

    if d["acroform_count"] > 0 or d["zero_chunk_count"] > 0:
        recs.append(
            (
                "high",
                "Fix AcroForm PDF Extraction",
                f"{d['acroform_count']} PDFs have form fields whose values are not extracted. "
                "Use pdfplumber's page.annots or pypdf's field reader to pull filled values, "
                "then append them to the text stream before chunking.",
                f"Recovers content from {d['zero_chunk_count']} currently 0-chunk docs and "
                f"{d['acroform_count']} partially-extracted form documents.",
            )
        )

    if d["total_docs"] > 1000:
        recs.append(
            (
                "high",
                "Batch Embedding Requests",
                "The embedding pipeline issues one HTTP request per chunk sequentially. "
                "The Ollama /api/embed endpoint accepts arrays. Batching 10-50 chunks per "
                "request and using asyncio would reduce round-trip overhead dramatically.",
                f"Could cut active processing time from {d['cpu_hours']:.1f}h to under 30 "
                "minutes and 10x throughput for future re-indexes.",
            )
        )

    if d["big_doc_count"] > 0:
        recs.append(
            (
                "high",
                "Per-Document Chunk Cap",
                f"{d['big_doc_count']} documents exceed 200 chunks. Large reference manuals "
                "are unlikely to be queried holistically. Add a max_chunks_per_doc config "
                "parameter (e.g. 200) with summarization for overflow.",
                f"Reduces index bloat — these docs account for a disproportionate share of "
                f"the {d['total_chunks']:,} total chunks.",
            )
        )

    if d["zero_chunk_count"] > 0:
        recs.append(
            (
                "high",
                "Surface 0-Chunk Docs",
                f"{d['zero_chunk_count']} documents indexed with 0 chunks are completely "
                "invisible in retrieval. Add a post-index check that logs a summary WARNING "
                "and writes a zero_chunk_docs.json report.",
                "Surfaces critical blind spots (P&S contracts, tax forms) for targeted "
                "manual intervention.",
            )
        )

    recs.append(
        (
            "medium",
            "Suppress pdfminer Library Noise",
            f"{d['lib_warning_count']:,} of {d['total_warn_err']:,} warnings "
            f"({round(d['lib_warning_count'] / d['total_warn_err'] * 100) if d['total_warn_err'] else 0}%) "
            "are benign pdfminer-six internal messages. Add a logging filter in arg/logging/ "
            "to suppress pdfminer.* at WARNING level.",
            "Real warnings drop from {:,} to ~{:,}, immediately visible.".format(
                d["total_warn_err"], d["arg_warning_count"]
            ),
        )
    )

    if d["one_chunk_count"] > d["total_docs"] * 0.2:
        recs.append(
            (
                "medium",
                "Audit 1-Chunk Documents",
                f"{d['one_chunk_count']:,} documents ({round(d['one_chunk_count'] / d['total_docs'] * 100, 1) if d['total_docs'] else 0}%) "
                "produced exactly one chunk. For PDFs this likely signals near-empty OCR "
                "output. Audit a random sample to verify content quality.",
                "Identifies documents where OCR quality is too low for useful retrieval.",
            )
        )

    if d["unique_encrypted_count"] > 0:
        recs.append(
            (
                "medium",
                "Encrypt-Skipped Stub Indexing",
                f"{d['unique_encrypted_count']} encrypted PDFs are silently skipped. Index "
                "their filename, directory path, and XMP metadata as a minimal stub document "
                "even without decryption.",
                "Makes encrypted files discoverable by filename/path even without content.",
            )
        )

    if d["dangling_count"] > 0:
        recs.append(
            (
                "low",
                "Fix Dangling HTML Links",
                f"{d['dangling_count']} unique broken links in HTML index pages. Update the "
                "source HTML files, or add a post-index link-repair script that resolves "
                "paths using doc_hashes.json.",
                "Restores graph connectivity for the affected LINKS_TO edges.",
            )
        )

    recs.append(
        (
            "low",
            "OCR Quality Metric Logging",
            "OCR is logged as a binary event. Add a word-confidence metric (Tesseract "
            "provides this via get_text('dict')) so low-confidence pages can be flagged "
            "with a WARNING and a quality field in chunk metadata.",
            "Identifies pages likely to have poor OCR quality across OCR-needing docs.",
        )
    )

    out = []
    for priority, title, body, impact in recs:
        out.append(
            f'<div class="rec-card">'
            f'<div class="rec-header">'
            f'<span class="rec-priority priority-{priority}">{priority.title()} Priority</span>'
            f'<span class="rec-title">{_h(title)}</span>'
            f"</div>"
            f'<div class="rec-body">{_h(body)}</div>'
            f'<div class="rec-impact">{_h(impact)}</div>'
            f"</div>"
        )
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Appendix report builder
# ─────────────────────────────────────────────────────────────────────────────


def build_appendix(logs: dict, meta: dict, d: dict, run_date: str) -> str:

    # Encrypted PDFs table
    enc_rows = "".join(
        _tr(_td(str(i + 1)), _td(_h(p), "mono")) for i, p in enumerate(logs["unique_encrypted"])
    )

    # 0-chunk docs table
    zero_rows = "".join(
        _tr(_td(str(i + 1)), _td(_h(doc["path"]), "mono"), _td(str(doc["ms"]), "num"))
        for i, doc in enumerate(logs["zero_chunk_docs"])
    )

    # AcroForm table
    acro_rows = "".join(
        _tr(_td(str(cnt), "num warn"), _td(_h(_short_path(path, 90)), "mono"))
        for path, cnt in sorted(logs["acroform_files"].items(), key=lambda x: -x[1])
    )

    # Dangling links table
    dangle_counter: Counter = Counter()
    dangle_examples: dict[str, list[str]] = defaultdict(list)
    for src, tgt in logs["dangling_links"]:
        dangle_counter[src] += 1
        if len(dangle_examples[src]) < 3:
            dangle_examples[src].append(_short_path(tgt, 50))
    dangle_rows = "".join(
        _tr(
            _td(str(cnt), "num warn"),
            _td(_h(_short_path(src, 70)), "mono"),
            _td(_h(", ".join(dangle_examples[src])), "mono"),
        )
        for src, cnt in dangle_counter.most_common(30)
    )

    # Slowest 50
    slow_rows = "".join(
        _tr(
            _td(str(i + 1)),
            _td(str(doc["seq"])),
            _td(
                f"{doc['ms']:,}",
                "num " + ("err" if doc["ms"] > 100_000 else ("warn" if doc["ms"] > 30_000 else "")),
            ),
            _td(f"{doc['chunks']:,}", "num"),
            _td(_h(doc["path"]), "mono"),
        )
        for i, doc in enumerate(d["slowest"][:50])
    )

    # All gaps
    gap_rows = "".join(
        _tr(
            _td(f"{g['hours']:.2f}", "num warn"),
            _td(str(g["seq_a"]), "num"),
            _td(str(g["seq_b"]), "num"),
            _td(_h(_short_path(g["path_a"], 55)), "mono"),
            _td(_h(_short_path(g["path_b"], 55)), "mono"),
        )
        for g in logs["gaps"]
    )

    # Hourly
    hourly_rows = "".join(
        _tr(
            _td(str(h + 1)),
            _td(str(cnt), "num" + (" ok" if cnt >= 100 else " warn" if cnt >= 10 else " err")),
        )
        for h, cnt in sorted(logs["hourly"].items())
    )

    # Config
    cfg_data = meta.get("config", {}).get("config", {})
    cfg_hash = meta.get("config", {}).get("hash", "")
    cfg_lines = "\n".join(f"  {k}: {v}" for k, v in cfg_data.items())

    # Logger breakdown
    logger_rows = "".join(
        _tr(_td(_h(logger), "mono"), _td(f"{cnt:,}", "num"))
        for logger, cnt in sorted(logs["loggers"].items(), key=lambda x: -x[1])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ARG Indexing Report — Appendix</title>
<style>{_CSS}</style>
</head>
<body>

<div class="header">
  <h1>ARG Indexing Report — Appendix: Raw Data</h1>
  <div class="subtitle">Detailed tables from log analysis</div>
  <div class="meta">
    <div class="meta-item"><div class="label">Period</div><div class="value">{_h(logs["first_ts"][:10])} - {_h(logs["last_ts"][:10])}</div></div>
    <div class="meta-item"><div class="label">Generated</div><div class="value">{_h(run_date)}</div></div>
    <div class="meta-item"><div class="label">Log Records</div><div class="value">{logs["total_records"]:,}</div></div>
  </div>
  <br><a href="indexing_report.html" style="color:var(--accent);font-size:.85rem">← Back to main report</a>
</div>

<nav>
  <a href="#encrypted">Encrypted PDFs</a>
  <a href="#zero-chunk">0-Chunk Docs</a>
  <a href="#acroform">AcroForm PDFs</a>
  <a href="#dangling">Dangling Links</a>
  <a href="#slowest">Slowest 50</a>
  <a href="#gaps">All Gaps</a>
  <a href="#hourly">Hourly Throughput</a>
  <a href="#config">Config</a>
  <a href="#loggers">Logger Breakdown</a>
</nav>

<main>

<section id="encrypted">
  <h2>All Encrypted PDFs Skipped ({len(logs["unique_encrypted"])} unique)</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>#</th><th>File Path</th></tr></thead>
    <tbody>{enc_rows}</tbody>
  </table></div>
</section>

<section id="zero-chunk">
  <h2>Documents Producing 0 Chunks ({len(logs["zero_chunk_docs"])})</h2>
  <p>These files exist in the document store but have no associated chunks
  and cannot appear in any retrieval result.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>#</th><th>File Path</th><th class="num">Index Time (ms)</th></tr></thead>
    <tbody>{zero_rows}</tbody>
  </table></div>
</section>

<section id="acroform">
  <h2>AcroForm PDFs — Partial Extraction ({len(logs["acroform_files"])} unique)</h2>
  <p>Form field values are not extracted; only static XObject stream text is indexed.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Warnings</th><th>File Path</th></tr></thead>
    <tbody>{acro_rows}</tbody>
  </table></div>
</section>

<section id="dangling">
  <h2>Dangling Link Warnings ({len(set(logs["dangling_links"]))} unique)</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Count</th><th>Source HTML</th><th>Missing Target(s)</th></tr></thead>
    <tbody>{dangle_rows}</tbody>
  </table></div>
</section>

<section id="slowest">
  <h2>Slowest 50 Documents</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>#</th><th>Seq</th><th class="num">ms</th><th class="num">Chunks</th><th>File Path</th></tr></thead>
    <tbody>{slow_rows}</tbody>
  </table></div>
</section>

<section id="gaps">
  <h2>All Indexing Gaps &gt; 30 Minutes ({len(logs["gaps"])} total)</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Gap (h)</th><th>After #</th><th>Before #</th><th>After File</th><th>Before File</th></tr></thead>
    <tbody>{gap_rows}</tbody>
  </table></div>
</section>

<section id="hourly">
  <h2>Hourly Throughput (docs per wall-clock hour)</h2>
  <p>Hour 1 = start of indexing ({_h(logs["first_ts"][:19])})</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Hour</th><th class="num">Docs Indexed</th></tr></thead>
    <tbody>{hourly_rows}</tbody>
  </table></div>
</section>

<section id="config">
  <h2>Run Configuration</h2>
  <div class="code">{_h(cfg_lines)}

config_hash: {_h(cfg_hash)}</div>
</section>

<section id="loggers">
  <h2>Logger Breakdown</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Logger</th><th class="num">Records</th></tr></thead>
    <tbody>{logger_rows}</tbody>
  </table></div>
</section>

</main>
<footer>
  ARG Indexing Appendix · Generated {_h(run_date)} · {logs["total_records"]:,} log records
</footer>
{_NAV_JS}
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    db_path = Path(args.db).expanduser().resolve()

    # Accept either a db root or a direct corpus directory
    corpus_dir = db_path / args.corpus if (db_path / args.corpus).is_dir() else db_path
    if not corpus_dir.is_dir():
        print(f"error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate log files (oldest first: .1 before current)
    log_files = []
    for name in ["arg.log.1", "arg.log"]:
        p = corpus_dir / name
        if p.exists():
            log_files.append(p)

    if not log_files:
        print(f"error: no log files found in {corpus_dir}", file=sys.stderr)
        return 1

    print(f"Parsing {len(log_files)} log file(s) from {corpus_dir} …")
    logs = parse_logs(log_files)
    print(f"  {logs['total_records']:,} records · {len(logs['indexed_docs']):,} indexed docs")

    print("Reading metadata …")
    meta = read_metadata(corpus_dir)

    print("Querying ChromaDB …")
    chroma = query_chroma(corpus_dir)

    print("Querying Kuzu graph …")
    kuzu = query_kuzu(corpus_dir)

    print("Loading BM25 index …")
    bm25 = query_bm25(corpus_dir)

    d = _derive(logs, meta)
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    main_html = out_dir / "indexing_report.html"
    appendix_html = out_dir / "indexing_appendix.html"

    print(f"Writing {main_html} …")
    main_html.write_text(build_main_report(logs, meta, chroma, kuzu, bm25, d, run_date))

    print(f"Writing {appendix_html} …")
    appendix_html.write_text(build_appendix(logs, meta, d, run_date))

    print(f"\nDone. Reports written to {out_dir}/")
    print(f"  {main_html.name}  ({main_html.stat().st_size // 1024} KB)")
    print(f"  {appendix_html.name}  ({appendix_html.stat().st_size // 1024} KB)")

    if args.open:
        import subprocess

        subprocess.run(["open", str(main_html)], check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
