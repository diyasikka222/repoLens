# OpenCode End-to-End Workflow (Phase 25.6)

This milestone validates RepoLens through the workflow a coding agent actually
runs inside an IDE or terminal tool, using the **real** MCP integration that
`opencode.json` starts (`python -m repolens.mcp --repo <root>`) — not a
simulated pipeline.  The harness, `benchmarks/opencode_e2e.py`, drives the exact
public MCP tool functions (`get_context`, `change_plan`, `change_context`,
`analyze_impact`, `inspect_symbol`, `architecture_candidates`) through the same
factories the MCP launcher builds, executes each task's deterministic workflow,
and reports honest, fingerprint-stable metrics.

No production code is modified, no LLM is consulted, and nothing requires
network access.

## What is evaluated

Eight realistic work requests spanning the workflow types an agent hits:

| task | workflow type | surface the agent must end up with |
| --- | --- | --- |
| `search-ranking-bug` | bug_fix | `repolens/search.py` + `tests/test_search.py` |
| `context-budget-option` | feature | `repolens/context/config.py` + `engine.py` + tests |
| `context-ranking-refactor` | refactor | `repolens/context/engine.py` + `ranking.py` + tests |
| `change-plan-risk-signal` | api_change | `repolens/mcp/change_plan_tool.py` + `change_plan.py` + tests |
| `dependency-expansion-blast-radius` | impact_investigation | `repolens/context/expansion.py` + dependents + tests |
| `architecture-retrieval-ordering` | architecture_change | `repolens/architecture_retrieval.py` + architecture signal |
| `hybrid-rrf-tests` | test_change | `tests/test_retrieval.py` + `repolens/retrieval.py` |
| `ambiguous-request` | ambiguous | graceful, structured plan response |

Each task's ground truth is a pre-declared, independent surface
(`required_files` / `supporting_files` / `expected_tests` / `expected_symbols`
/ `target_files`) — the same surface discipline the P25.2/P25.3 graded
framework uses.  Success is only ever measured against that explicit surface;
it is never assumed, weighted against a hidden score, or weakened to make a
task pass.

### Two workflow modes

The harness compares two ways an agent could step through a change:

- **baseline** — the core `get_context` tool only.  This is the pre-M24
  RepoLens surface.
- **assisted** — the additive M24/M25 tools.  A fixed, documented routing
  policy (`ASSISTED_TOOLS_BY_TYPE`) decides which tools a task needs:
  change tasks use `change_plan` + `change_context`; the investigation task
  uses `analyze_impact` + `inspect_symbol`; the architecture task adds
  `architecture_candidates` (called twice so ordering determinism is checked).
  Tools a task never needed are never invoked, so they are neither measured
  nor penalised.

### Metric contract

Per task per mode:

- required / supporting / test / symbol **recall** against the pre-declared
  surface (symbol recall = the symbol appears as an identifier in a surfaced
  file, or in tool `matched_symbols` metadata);
- **plan signal** — `change_plan` returned a structured `status: ok` result
  with a risk classification;
- **target validity** — the plan's primary target, an affected file, or a
  surfaced test corresponds to the task's declared target file;
- **architecture signal** — candidates were produced and a repeated call
  returned the identical ranked list;
- **impact / inspect signals** — the investigation produced risk + items and
  a resolved call-graph node;
- **budget compliance** and **no invalid files** (every surfaced path exists
  under the repository root);
- **fingerprint** — a sha256 over the deterministic portion of the outputs
  (wall-clock latency excluded, working-root prefix normalised), so two runs of
  the script must produce byte-identical fingerprints.  Wall-clock fields
  embedded in production payloads (`change_plan.diagnostics.build_time` and
  similar `elapsed_ms` / `duration` keys) are stripped before hashing; every
  other payload field is byte-identical across runs and Python hash seeds.

A task **passes** only when its invoked tools ran without error and the surface
is genuinely satisfied.  The ambiguous task passes only on a graceful,
structured, bounded response — never on a fabricated confident target.

## Deterministic, offline measurement

The repository is materialised once into a pristine temp copy that skips the
same local-only debris as `repolens.production_benchmark._SKIP_NAMES`:
virtualenvs, caches, node modules, quickfix directories, and the
`.benchmark_data` external-library download.  This makes measurements
reflect the tracked project tree a fresh checkout would contain, instead of
whatever local benchmark artifacts happen to sit in the working directory.
Tool outputs are repo-relative paths, so the computation is stable across
copies.

## Running the evaluation

```bash
# default: evaluate the full corpus in both modes against the current repo
python benchmarks/opencode_e2e.py .

# compare against the core get_context surface only
python benchmarks/opencode_e2e.py . --mode baseline

# one task, with per-task misses / signals
python benchmarks/opencode_e2e.py . --task search-ranking-bug --diagnostics
```

The script first validates the `opencode.json` RepoLens server entry
(read-only), then materialises the pristine copy, runs every task in the
requested modes, and prints a per-task table, aggregate pass counts and mean
required-recall per mode, the architecture-determinism signal, and the two
mode fingerprints.  It exits `0` when every evaluated task in every requested
mode passed and the config is valid, `1` otherwise.

## Manual smoke workflow

Run the full workflow as a human would, in an OpenCode terminal session with
the RepoLens MCP server configured (see `opencode.json`, untracked):

1. Start the session in the repository root.  Confirm the `repolens` MCP
   server connects.
2. Ask a concrete change request, e.g. *"make checkout reject empty carts"*.
3. Observe `change_plan` / `change_context` return a primary target, an
   affected-file set, and a bounded safe context package without leaving the
   request repository.
4. For an investigation, ask *"what is affected if
   `repolens/context/expansion.py` changes?"* and use `analyze_impact` /
   `inspect_symbol`; the offline call graph resolves the symbol statically.
5. Make a small edit, run the relevant tests
   (`python -m pytest tests/... -q`), and confirm the change passes.
6. Re-run `python benchmarks/opencode_e2e.py .` and confirm the fingerprints
   are unchanged from earlier runs (determinism contract).

## Current honest result (Phase 25.6)

Running the full corpus against this repository's clean, pristine tree:

- **assisted: 8/8 tasks pass**, mean required-file recall **1.000**
- **baseline: 4/8 tasks pass**, mean required-file recall **0.500**
- the additive tools demonstrably raise the required-surface coverage for
  change workflows (change-planning, change-context, impact/inspect,
  architecture candidates);
- no invalid files, no budget violations, no tool errors in either mode;
- fingerprints are stable across consecutive identical runs.

### Resolved findings

Two review fixes landed in production (see ``docs/change-planning.md``): a
module/package-kind primary target now resolves its **defining source file**
into the affected surface and the change-context candidate set (so
`context-budget-option` and `context-ranking-refactor` both reach the file the
requirement is actually about), and the `change_plan` payload now reports
wall-clock time under a clearly-labelled `diagnostics` object instead of the
deterministic `statistics` block, making the advertised `deterministic: true`
cover every substantive field. Both previously-failing tasks now surface the
correct defining file (`repolens/context/config.py` and
`repolens/context/engine.py`).

The baseline (core-only) misses are expected: the additive tools exist
precisely because the raw `get_context` surface under-covers change workflows —
that is the delta this benchmark quantifies.

## Tests

`tests/test_opencode_e2e.py` pins the harness offline and fast:

- corpus integrity (all workflow types present, unique ids, disjoint surfaces,
  all surface files exist, routing policy closed over the tool whitelist);
- path / fingerprint / target helpers and deterministic metrics math;
- read-only `opencode.json` validation (valid, missing, disabled, wrong repo,
  malformed JSON) and the real config file;
- debris-safe pristine-copy materialisation;
- end-to-end wiring through the real MCP tool functions against the small
  offline fixture repository (`tests/fixtures/change_plan_repository`),
  asserting the honest pass contract plus expectation-driven tool selection.