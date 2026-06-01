---
globs: ["tests/**"]
---

## Test discipline

**Tests define the contract. Implementations must satisfy the contract — not the other
way around.**

### Never weaken a test to make an implementation pass

If an implementation change causes a test to fail, fix the implementation. Do not:
- Change `assert exact_value` to `assert value is not None`
- Add `pytest.mark.skip` or `pytest.mark.xfail` without a tracked reason in the
  commit message
- Move a test from `tests/unit/` to `tests/integration/` to avoid a dependency
- Change a fixture's scope from `function` to `session` to hide side-effect coupling
- Delete assertions from an existing test

If a test genuinely needs to change (the contract changed), pause and state explicitly
in the commit message what contract changed and why.

### test_invariants.py and test_concurrency.py are off-limits

`tests/unit/test_invariants.py` and `tests/unit/test_concurrency.py` verify structural
invariants that are easy to accidentally break. These two files must never be modified
to accommodate implementation changes — only to *add* new invariant tests.

### Track test counts

Before and after any change that touches `tests/`, run:
```
pytest tests/unit/ --co -q 2>/dev/null | tail -1
```
The total must not decrease unless tests were explicitly removed with a justification.

### Fake objects

Unit test fakes (`_ScriptedLLM`, `_TagEmbedder`, etc.) must implement the same
`typing.Protocol` surface as the real objects they replace. Do not add methods to a
fake that the real object does not have — it hides missing implementation.
