---
globs: ["arg/generator/**"]
---

The raw user query (not the rewritten form) is always passed to the LLM for generation.
Rewrites and HyDE paragraphs are retrieval-only signals; never include them in the
generation prompt sent to the LLM.

`complete_structured(prompt, schema)` is used for any LLM call that requires parsed
structured output (currently `_maybe_decompose`). Free-text outputs use `complete()`.
