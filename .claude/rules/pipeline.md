---
globs: ["arg/pipeline.py"]
---

## Cluster recompute invariant

`pipeline.index()` **must** schedule a cluster recompute at the end via
`self._recompute_clusters_bg()`. Do not remove or skip this call. The cluster cache
must be warm when the server starts; without it `get_topic_clusters()` triggers a
blocking k-means + LLM call on the first web request.

Watcher-triggered `add_document`, `update_document`, and `remove_document` must also
call `self._recompute_clusters_bg()`. The background thread invalidates the stale cache
and recomputes, keeping the UI current after file changes.

## Cluster thread shutdown invariant

`close()` must join the cluster thread before tearing down shared resources (graph,
watcher). The join has a 5-second timeout. Acquire `_cluster_lock` to read the thread
reference, then join *outside* the lock so the thread can finish without deadlocking.

Inside `_run_cluster_recompute`, check `_closed` only **at the start**. Once the thread
has passed the initial guard and begun work, it runs to completion — do not add a second
`if not self._closed` check before `get_topic_clusters()`. Adding such a guard caused a
race condition (Feature 0003) where `close()` set `_closed=True` between
`invalidate_cluster_cache()` and `get_topic_clusters()`, leaving the cache deleted but
never rewritten.

## Lock acquisition order

To avoid deadlock, always acquire locks in this order:
1. `_cluster_lock` or `_bm25_rebuild_lock` (narrow-scope)
2. `_lock` (top-level RLock)

Never acquire `_lock` first and then attempt `_cluster_lock` or `_bm25_rebuild_lock`.

Invariants verified by `tests/unit/test_invariants.py` and `tests/unit/test_concurrency.py`.
