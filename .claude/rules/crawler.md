---
globs: ["arg/crawler/**"]
---

In `arg/crawler/extractors.py`, `_extract_links(soup)` **must be called before**
`_strip_invisible_and_boilerplate(soup, config)`. The strip pass removes `<nav>` and
sidebar elements; calling it first silently drops links from index pages that use `<nav>`
for cross-document navigation. The correct call order (lines ~153-155) is:

```python
links_to = _extract_links(soup)          # 1. collect links
_strip_invisible_and_boilerplate(soup, config)  # 2. then strip
```

Never swap this order. The invariant is verified by `tests/unit/test_invariants.py::test_extract_links_called_before_strip`.
