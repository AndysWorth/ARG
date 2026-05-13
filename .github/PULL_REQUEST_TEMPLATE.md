## Summary
What this PR changes and why.

## Section
Which section of `CLAUDE.md` this implements (if applicable). Link to `docs/spec/section-NN-*.md`.

## Checklist
- [ ] Unit tests pass locally: `pytest tests/unit/ -v`
- [ ] Locality check passes (no outbound network calls outside Ollama/localhost):
  ```
  grep -rn "requests\.\|httpx\.get\|http://" arg/ --include="*.py" \
    | grep -v "localhost\|127.0.0.1\|11434\|test_"
  ```
- [ ] Commit messages follow Conventional Commits (`feat(scope): …`, `fix(scope): …`, etc.)
- [ ] No `git add -A` or `git add .` was used; only explicit paths staged
- [ ] Pre-commit hooks were not bypassed (`--no-verify`)
- [ ] CHANGELOG.md updated under `[Unreleased]` (for user-visible changes)
