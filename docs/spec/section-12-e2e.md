# Section 12: End-to-End Test & Fixture Corpus

> **Prompt to Claude:** "Build Section 12 of ARG: the fixture corpus and full E2E test"

### Corpus A — RAG fixture (existing): `tests/fixtures/docs/`

Purpose: test link-following, graph traversal, multi-hop retrieval, and the full RAG
answer pipeline. All 5 documents are intentionally on the same topic (Kraken API) to
create a dense, interlinked graph.

**`tests/fixtures/docs/index.html`**
- Title: "ARG Test Documentation"
- Contains: overview paragraph about a fictional "Kraken API"
- Links to: `page_a.html`, `page_b.html`, `subdir/page_c.html`

**`tests/fixtures/docs/page_a.html`**
- Title: "Kraken API — Authentication | Kraken API Docs" (tests title suffix stripping)
- Contains: `<meta name="description" content="...">` (tests page_description extraction)
- Contains: detailed fictional auth flow (API keys, OAuth)
- Contains: a `<div class="sphinxsidebar">` nav block with junk text (tests strip_selectors)
- Contains: a `<span style="display:none">hidden text</span>` (tests invisible element stripping)
- Contains: a `<pre>` code block (tests code block handling and `has_code` metadata)
- Links to: `page_b.html`, `manual.pdf`, `scanned_notice.pdf`

**`tests/fixtures/docs/page_b.html`**
- Title: "Kraken API — Rate Limits"
- Contains: rate limit table (requests/minute by tier)
- Links to: `index.html` (back-link, tests circular dedup)

**`tests/fixtures/docs/subdir/page_c.html`**
- Title: "Kraken API — Error Codes"
- Contains: list of error codes and meanings
- No outbound links (tests isolated node retrieval)

**`tests/fixtures/docs/manual.pdf`**
- Title: set via PDF `/Title` metadata as "Kraken API Full Manual"
- `/Subject` metadata: "Complete reference for the Kraken API including auth, rate limits, and errors"
- Contains: 2–3 pages of native text covering all topics above with unique phrasing
- Contains: one table (rate limits by tier) to exercise table extraction and bbox_exclude
- Contains: at least one section heading in large font to exercise font-based H2 detection
- Contains: a running page footer "Kraken API Docs — Confidential" to exercise header/footer stripping
- Generated as a native-text PDF (pdfplumber primary path)

**`tests/fixtures/docs/scanned_notice.pdf`** ← exercises OCR path
- `/Title` metadata deliberately set to "Microsoft Word - document1.docx" to test
  temp-file title filtering (should fall back to font detection or filename)
- Contains: one paragraph of text rendered as an image (no embedded text layer)
- Generated using reportlab to render text to an image, then embed as a page scan
- Linked from `page_a.html` so it is reachable via crawl

**`tests/fixtures/docs/encrypted_notice.pdf`** ← exercises encryption skip
- A password-protected PDF (any content)
- Crawler finds it via directory walk; extractor skips it with a warning log
- Its doc_id must NOT appear in Kuzu or ChromaDB after indexing
- NOT linked from any HTML (found only via directory walk)

### Corpus A E2E test questions & expected behaviour:

| Question | Expected behaviour |
|---|---|
| "How do I authenticate with the Kraken API?" | Returns chunks from `page_a.html`; sources cited |
| "What is the rate limit for tier 2?" | Returns chunks from `page_b.html`; table content present |
| "What does error code 429 mean?" | Returns chunks from `subdir/page_c.html` |
| "Tell me about the Kraken API" | Fuses chunks from `index.html` + graph neighbours |
| "What is the capital of France?" | Returns "The documentation does not cover this topic" |
| "How do I log in?" | Query rewritten to technical form; answer still cites `page_a.html`; `ARGResult.rewritten_query` not None |
| "X-Rate-Limit-Retry-After header" | BM25 Stage 1.5 finds exact-term match in `page_b.html` even if dense search misses it |
| "How do auth and rate limits work?" | Query decomposed into 2 sub-queries; both `page_a.html` and `page_b.html` in `ARGResult.sources` |
| Rate-limit query with `filters={"has_table": True}` | Only table-bearing chunks returned; `page_b.html` table chunk present |

### Corpus A E2E assertions:
- All 7 documents indexed: 4 HTML + 2 PDFs (manual + scanned) + 1 skipped (encrypted)
- `encrypted_notice.pdf` does NOT appear in Kuzu or ChromaDB; warning in `arg.log`
- All link edges present in Kuzu graph (page_a now links to scanned_notice.pdf)
- Each RAG question returns a non-empty answer
- "France" question returns the refusal string
- Source citations reference actual document titles
- `manual.pdf` title extracted from `/Title` metadata correctly
- `scanned_notice.pdf` title falls back to filename stem (temp-file `/Title` rejected)
- Running footer "Kraken API Docs — Confidential" does NOT appear in any chunk
- Query about rate limits returns content from the Markdown table in `manual.pdf`
- Query about content in `scanned_notice.pdf` returns OCR-extracted content
- `/Subject` metadata from `manual.pdf` present in `documents` collection embedding text
- `get_topic_clusters()` returns small-corpus fallback (doc_count < threshold 10)
- Total query latency logged (baseline: < 30s on M1 Max)

### Corpus A E2E — Watcher test (`test_full_rag.py` addition):
```
Start pipeline with watch_enabled=True over corpus_a
  → Drop a new file `new_page.html` into docs_root
  → Wait debounce_ms + 200ms
  → Assert new_page.html appears in list_all_documents()
  → Assert query about new_page.html content returns it as a source
  → Delete new_page.html
  → Wait debounce_ms + 200ms
  → Assert new_page.html no longer in list_all_documents()
``` `tests/fixtures/clustering_docs/`

Purpose: test topic clustering, `GET /corpus/topics`, and the Section 18 cluster-expansion
stage of enrichment. Uses 15 documents across 3 clearly distinct fictional topics so that
k-means produces semantically meaningful groups. Documents are deliberately sparse (no
cross-topic links) to isolate the clustering signal from graph-expansion noise.

**Claude will generate all 15 files during this section. Spec:**

**Topic 1 — "Triton Database" (5 docs, indices `t1_*.html`):**
- `t1_overview.html` — what Triton Database is; links to t1_schema, t1_query
- `t1_schema.html` — table design, column types, constraints
- `t1_query.html` — SELECT syntax, JOINs, WHERE clauses
- `t1_indexing.html` — B-tree vs hash indexes, query planning
- `t1_backup.html` — backup schedules, restore procedures

**Topic 2 — "Poseidon Networking" (5 docs, indices `t2_*.html`):**
- `t2_overview.html` — what Poseidon Networking is; links to t2_routing, t2_firewall
- `t2_routing.html` — BGP, OSPF, static routes
- `t2_firewall.html` — rule chains, port filtering, NAT
- `t2_vpn.html` — WireGuard tunnels, key exchange
- `t2_monitoring.html` — SNMP, packet capture, alerts

**Topic 3 — "Hydra Scheduler" (5 docs, indices `t3_*.html`):**
- `t3_overview.html` — what Hydra Scheduler is; links to t3_jobs, t3_triggers
- `t3_jobs.html` — job definitions, priority queues, retries
- `t3_triggers.html` — cron syntax, event-based triggers
- `t3_workers.html` — worker pools, concurrency limits
- `t3_logging.html` — job history, failure alerts

**`clustering_docs/index.html`** — top-level index linking to all three `*_overview.html`
files only (links to sub-pages exist within each topic group).

### Corpus B clustering assertions (`test_dci_e2e.py`):
- All 15 documents indexed; 15 nodes in Kuzu graph
- `get_topic_clusters()` runs (doc_count 15 ≥ threshold 10)
- Returns exactly 3 clusters (k-means with `n_clusters=3` for this corpus)
- Each cluster contains only documents from one topic group (Triton / Poseidon / Hydra)
- Each cluster has a non-empty LLM-generated label
- `corpus_search("database indexing")` returns only Triton documents
- `corpus_search("VPN tunnels")` returns only Poseidon documents
- `corpus_search(file_type="html")` returns all 15 documents (all are HTML)
- Enrichment: query "backup restore procedures" reaches `t1_backup.html` via
  cluster expansion even without a direct keyword match in chunk retrieval

### Multi-corpus E2E test (`test_dci_e2e.py` addition):
```
Index corpus_a (Kraken API) and corpus_b (Triton/Poseidon/Hydra) as separate corpora
Start FastAPI server with both corpora available
  → POST /query?corpus=corpus_a "How do I authenticate?"
     Assert: answer references Kraken API authentication
  → POST /query?corpus=corpus_b "How do I configure BGP routing?"
     Assert: answer references Poseidon Networking, not Kraken API
  → Assert no cross-contamination: corpus_a query never returns corpus_b docs
```
**Note:** This requires the FastAPI server to accept a `?corpus=` query parameter on
`POST /query` and all `/corpus/*` endpoints. This parameter must be added to the
`server.py` spec in Section 10B. Default is `"default"` when omitted.
