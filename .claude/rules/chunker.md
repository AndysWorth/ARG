---
globs: ["arg/indexer/chunker.py"]
---

The `position` counter in `chunk_document()` is **global across all sections** in a
document. It must never be reset to 0 between sections. Kuzu stores `position` as the
chunk ordering field; a reset would produce duplicate position values across sections,
breaking Kuzu queries that sort or filter by position.

```python
position = 0  # initialised ONCE before the outer loop over sections
for section in sections:
    for window_text in _sliding_window(section.text, config):
        ...
        position += 1  # incremented per non-empty chunk, never reset
```

The invariant is verified by `tests/unit/test_invariants.py::test_chunk_position_is_global_not_per_section`.
