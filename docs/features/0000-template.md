# Feature {NNNN}: {Short name}

**Status:** draft   <!-- draft | in-progress | shipped (commit: `xxxxxxx`) -->
**Created:** YYYY-MM-DD

---

## Motivation

One short paragraph: what problem this solves and why now. Concrete enough
that a reader six months from now can tell whether the feature is still
relevant.

## Scope

**In scope:**
- Bullet list of behaviours this feature adds.
- Be specific — "indexes `.txt` and `.md` files" is better than "supports
  more file types".

**Out of scope:**
- Things a reader might reasonably expect that this feature does NOT do.
- Defer-to-follow-up items; risks of scope creep go here.

## Deliverables

Files created or modified, in the same shape as the section specs:

- `arg/foo/bar.py` — new module: ...
- `arg/foo/baz.py` — change: ...
- `tests/unit/test_bar.py` — new tests covering ...
- `README.md` / `CLAUDE.md` — what to update; see "CLAUDE.md impact" below.

## Design notes

Integration points with existing components. Decisions made. Anything
non-obvious about why this is structured the way it is — leave a trail for
future-you, especially for non-default choices (e.g., why we picked X over
Y).

Locality: confirm no outbound network calls.
mypy: any expected stub-availability quirks (see prior CLAUDE.md sections
for the canonical pattern of combined `# type: ignore[..., unused-ignore]`).

## Test points

Unit:
- ...

Integration (Ollama-dependent):
- ...

E2E (real LLM):
- ...

## Open questions / risks

- Decisions deferred to implementation, with the reasoning behind the
  deferral.
- Known sharp edges or assumptions.

## CLAUDE.md impact

Whether and what to update. Default to "no impact" for small features.
Touch CLAUDE.md when:
- The product's scope description changes (the preamble + README).
- Section 1 stack decisions gain or lose a row.
- Section 1.5 locality guarantee gains a new layer.
- Section 13 architectural-decisions log needs a new entry.

Quote the proposed edits inline so they can be reviewed without grepping.

---

## Implementation plan

A short ordered list of branches/commits, mirroring the section build flow:

1. Branch `feature/NNNN-short-name` off `main`.
2. ...
3. Run unit + integration tests; mypy clean; locality grep clean.
4. Commit with `feat(scope): one-line summary` body referencing this doc.
5. Push, ff-merge to `main`, delete branch.
