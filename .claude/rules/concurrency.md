---
globs: ["arg/**"]
---

## Threading model

ARG has three concurrency surfaces. Understand all three before modifying any
threading-adjacent code.

### 1. Watcher debounce timers (`arg/crawler/watcher.py`)

- One `threading.Timer` per watched path, stored in `self._pending: dict[Path, Timer]`.
- Protected by `self._lock: threading.Lock`.
- On each filesystem event, the existing timer for that path is cancelled and a new one
  is started (debounce reset).
- `stop()` cancels all pending timers and sets `self._stopped = True`.
- Callbacks fire on the timer's background thread, not the watcher thread.

### 2. BM25 rebuild timer (`arg/pipeline.py`)

- One `threading.Timer` (5-second delay) in `self._bm25_rebuild_timer`.
- Protected by `self._bm25_rebuild_lock: threading.Lock`.
- Triggered by watcher events; coalesces rapid file changes into one BM25 rebuild.
- `close()` cancels the timer under `_bm25_rebuild_lock` before joining.

### 3. Cluster recompute thread (`arg/pipeline.py`)

- One daemon `threading.Thread` in `self._cluster_thread`.
- Protected by `self._cluster_lock: threading.Lock`.
- Single-slot: if a thread is already alive, new recompute requests are dropped
  (the running thread will see the latest state when it calls `invalidate_cluster_cache`).
- `close()` joins with `timeout=5.0` after setting `_closed=True`.
- The thread checks `_closed` **once at entry only**; it runs to completion thereafter.

### 4. Sub-query parallelism (`arg/generator/generator.py`)

- `ThreadPoolExecutor(max_workers=min(len(queries), 4))` for parallel retrieval.
- No shared mutable state inside the executor; each call is independent.
- The pool is torn down (`.shutdown(wait=True)`) at the end of the `with` block.

### 5. Top-level pipeline lock

- `self._lock: threading.RLock` serialises `index()`, `add_document()`,
  `update_document()`, `remove_document()`, and `query()`.
- `close()` acquires `_lock` to set `_closed` and to stop the watcher/close the graph.
- Do not hold `_lock` while waiting on `_cluster_lock` or `_bm25_rebuild_lock`
  (deadlock risk).

## Shutdown sequence

```
close()
  1. Acquire _lock → set _closed=True → release _lock
  2. Acquire _bm25_rebuild_lock → cancel timer → release _bm25_rebuild_lock
  3. Acquire _cluster_lock → read thread ref → release _cluster_lock
  4. Join cluster thread (timeout=5s)          ← outside all locks
  5. Acquire _lock → stop watcher → close graph → log → release _lock
```

## Rule

Background threads check `_closed` only at the **start**. Once a thread has passed its
entry guard, it runs to completion. `close()` is responsible for waiting via `join()`.
