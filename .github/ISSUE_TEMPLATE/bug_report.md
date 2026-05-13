---
name: Bug report
about: Report a bug in ARG
title: "[bug] "
labels: bug
---

## Description
A clear description of the bug, what you expected, and how to reproduce.

## Environment
- macOS version:
- Python version: `python --version`
- Ollama version: `ollama --version`
- ARG commit: `git rev-parse HEAD`
- Corpus type (HTML / PDF / mixed):

## Logs
Relevant lines from `arg_db/{corpus}/logs/`. Run with `--debug` for verbose tracing.

## Locality check
- [ ] I verified the bug is reproducible with `arg_db/` deleted and re-indexed
- [ ] The bug does not require any non-local resource to reproduce
