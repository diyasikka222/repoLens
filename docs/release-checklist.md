# Release Readiness Checklist (v1.0 preparation)

This checklist is the **preparation gate for the v1.0 release** (Phase 25.8 is
the actual release). Run it against the working tree before cutting the
release. It is documentation, not a release.

## Automated verification

- [ ] Full test suite passes: `python -m pytest -q`
- [ ] Documentation tests pass: `python -m pytest tests/test_documentation.py -q`
- [ ] CLI checks pass:
  - [ ] `python -m repolens.mcp --help` (the only package CLI; `python -m repolens` has no `__main__`)
- [ ] Benchmark smoke tests pass:
  - [ ] `python benchmarks/production_benchmark.py .`
  - [ ] `python benchmarks/verify_change_context.py .`
  - [ ] `python benchmarks/opencode_e2e.py . --mode assisted` (exit 0)
  - [ ] `python benchmarks/opencode_e2e.py .` (exit 1 is *expected* in `both` mode — baseline mode intentionally under-covers; see `docs/opencode-e2e.md`)
- [ ] `git diff --check` is clean (no whitespace errors)

## Documentation

- [ ] `README.md` reviewed: accurate install / quick-start / embeddings /
      OpenCode / MCP-tools pointers / limitations
- [ ] All MCP tools documented in `docs/mcp-tools.md` (names match `repolens/mcp/server.py`)
- [ ] OpenCode integration documented in `docs/opencode-e2e.md`
- [ ] Every documented command exists (no references to removed APIs)
- [ ] No stale milestone/workflow claims in user-facing docs
- [ ] No duplicated large sections across documents

## Package metadata

- [ ] `pyproject.toml`: name, description, `requires-python`, and dependencies
      are accurate; version is declared `dynamic` and derives from
      `repolens.__version__`
- [ ] One canonical version: `repolens.__version__` matches the intended
      release version and is the single source used by `pyproject.toml`, the
      MCP server version, and the `repolens-mcp --version` flag
- [ ] Optional-extras (`mcp`, `dev`) documented where installs are shown
- [ ] `readme` / `[project.urls]` present
- [ ] **License: a human decision is required.** No license is stated anywhere
      in README, docs, or `pyproject.toml`, so no `LICENSE` file or license
      metadata is added until the maintainer chooses a license. Do not invent
      one.

## Repository hygiene

- [ ] Working tree reviewed via `git status` / `git diff`
- [ ] `opencode.json` is **not** staged or committed (it is local/untracked)
- [ ] No secrets, API keys, bearer tokens, or private keys in tracked files
- [ ] No user-specific absolute filesystem paths in tracked docs/examples
- [ ] No temporary benchmark artifacts or local cache directories tracked
- [ ] Generated `*.egg-info/` build artifacts removed from version control
      (they are regenerated on install and can go stale)

## Behavior

- [ ] Determinism verified: cold/warm/cache-disabled produce identical
      substantive output (benchmarks above)
- [ ] Reliability checks verified (`tests/test_reliability.py`, the failure
      matrix in `docs/reliability.md`)
- [ ] No new network dependencies introduced

## Release (Phase 25.8 only)

- [ ] Version bump to a definitive `1.0.0` in `repolens/__init__.py` (the
      single canonical source; `pyproject.toml`, the MCP server, and the
      `repolens-mcp --version` flag derive from it)
- [ ] Tag the release commit

---

## Known follow-ups tracked for Phase 25.8

- Choose a license (required before release; none is invented here).
- Decide whether to add a `[project.scripts]` console entry point (e.g.
  `repolens-mcp`) so the MCP server can be launched by name rather than
  `python -m repolens.mcp`.