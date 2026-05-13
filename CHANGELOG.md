# Changelog

All notable changes to ARG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Commits in this repo follow [Conventional Commits](https://www.conventionalcommits.org/);
once `v0.1.0` is tagged, this changelog can be regenerated from `git log` via a tool
such as `git-cliff` or maintained by hand.

## [Unreleased]

### Added
- Initial project scaffolding: `CLAUDE.md` spec, per-section spec files under `docs/spec/`.
- Project quality gates: ruff lint+format, mypy type checking, pytest, pre-commit hooks,
  GitHub Actions CI, Dependabot, issue/PR templates, `.editorconfig`, `.python-version`.
- Python environment standard: stdlib `venv` at `.venv/`, Python 3.11+.
- Git workflow standard: branch-per-section, Conventional Commits, explicit-path staging,
  branch-abandon recovery (no `git reset --hard`).
