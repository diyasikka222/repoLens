# OpenCode Integration and End-to-End Workflow

This document is the authoritative guide to using RepoLens from
[OpenCode](https://opencode.ai): where the configuration lives, how the MCP
server is wired in, how to verify it, the expected agent workflow, and the
end-to-end benchmark that validates that workflow.

## Setup

### 1. Create the OpenCode configuration

OpenCode reads its configuration from `opencode.json` (or `opencode.jsonc`) in
the **project root** (the directory where you run `opencode`). RepoLens is
configured as a **local** MCP server there.

`opencode.json` is intentionally **local and untracked**: it contains
machine-specific paths (e.g. the absolute path to your virtualenv). Do not
commit it. `.gitignore` does not need an entry because it simply should never
be added to version control.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "repolens": {
      "type": "local",
      "command": [
        "/absolute/path/to/your/venv/bin/python",
        "-m",
        "repolens.mcp",
        "--repo",
        "/absolute/path/to/this/repository",
        "--use-local-embeddings"
      ],
      "cwd": "/absolute/path/to/this/repository",
      "enabled": true
    }
  }
}
```

Replace the two absolute paths with your own:

- the interpreter: `python -m repolens.mcp` must resolve to a Python
  environment where RepoLens is installed (e.g. the project virtualenv
  `…/.venv/bin/python`); using the `.venv/bin/python` directly avoids PATH
  ambiguity;
- the repository: `--repo` must point at the repository root RepoLens should
  index.

### 2. The command OpenCode invokes

The `command` array is executed as-is. It launches the RepoLens MCP server over
**stdio**:

```bash
python -m repolens.mcp --repo /absolute/path/to/this/repository --use-local-embeddings
```

- `--repo` is required and fixes the repository root at launch time (tool
  callers cannot change it).
- `--use-local-embeddings` enables on-device semantic retrieval via
  [FastEmbed](https://github.com/qdrant/fastembed). Leave it out to use the
  deterministic offline fake provider (no model download, keyword/structural
  retrieval only) or set `REPOLENS_LOCAL_EMBEDDING_MODEL` to override the
  default model `BAAI/bge-small-en-v1.5`.

The full launcher surface (defaults shown):

```text
usage: repolens-mcp [-h] --repo REPO [--default-max-tokens DEFAULT_MAX_TOKENS]
                    [--default-dependency-depth DEFAULT_DEPENDENCY_DEPTH]
                    [--use-local-embeddings] [--log-level LOG_LEVEL] [--version]
```

### 3. Local embeddings configuration

By default the MCP server needs **no API key and no model download** (it uses a
deterministic fake embedding provider). To use real on-device semantic
retrieval:

```bash
export REPOLENS_LOCAL_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"   # default; optional
python -m repolens.mcp --repo /path/to/repository --use-local-embeddings
```

The first `--use-local-embeddings` run downloads the ONNX model (from Hugging
Face) into the local model cache; subsequent runs are offline. No OpenAI key is
ever required.

### 4. Verify the MCP connection

1. Start `opencode` in the repository root.
2. Confirm the `repolens` MCP server connects (OpenCode logs a
   successful/absent server connection). With stdio servers, a server that
   fails to validate its root reports a safe error on stderr and exits with
   code 2.
3. Sanity-check the tool list: `get_context`, `analyze_impact`,
   `inspect_symbol`, `inspect_architecture`, `discover_subsystems`,
   `architecture_candidates`, `explain_architecture_match`, `change_plan`,
   `change_context`.
4. Ask a concrete change request (below) and confirm `change_plan` /
   `change_context` return a primary target and a bounded safe context package.

Or verify outside OpenCode against a built server:

```bash
python -m repolens.mcp --repo /path/to/repository --use-local-embeddings
```

and confirm the server registers the nine tools. `benchmarks/opencode_e2e.py`
also validates the config file *read-only* before running (see below).

## Expected workflow

The harness, `benchmarks/opencode_e2e.py`, drives the exact public MCP tool
functions through the same factories the launcher builds. Its per-task workflow
is the intended agent behaviour:

- **Change tasks** (bug fix, feature, refactor, api change, test change) —
  call `change_plan` then `change_context`, then inspect the surfaced files and
  run the suggested tests (`python -m pytest tests/… -q`).
- **Investigation** — call `analyze_impact` / `inspect_symbol` to trace blast
  radius and call relationships.
- **Architecture reasoning** — call `architecture_candidates` /
  `inspect_architecture` / `discover_subsystems` for structural candidates.
- **General questions** — call `get_context` for the default retrieval path.

### Example change request

> *"make checkout reject empty carts"*

Expected flow:

1. `change_plan` returns a primary target, affected files, callers/callees,
   candidate tests, inspection order, and a risk tier.
2. `change_context` returns firewall-cleared context with `selected_files`
   (each with a deterministic explanation), `change_files`, and a rendered safe
   context — respecting the token budget.
3. Make the edit, run the relevant tests, and confirm the change passes.

For an investigation instead:

> *"what is affected if `repolens/context/expansion.py` changes?"*

→ `analyze_impact` + `inspect_symbol`; the offline call graph resolves the
symbol statically.

---

## End-to-end evaluation (Phase 25.6 benchmark)

The benchmark below validates that this workflow actually covers the files a
change touches, deterministically and offline.

### What is evaluated

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
/ `target_files`). Success is only ever measured against that explicit surface.

### Two workflow modes

- **baseline** — the core `get_context` tool only (the pre-M24 surface).
- **assisted** — the additive change-planning / impact / architecture tools,
  routed by a fixed, documented policy (`ASSISTED_TOOLS_BY_TYPE`). Tools a task
  never needed are never invoked.

### Metric contract

Per task per mode: required / supporting / test / symbol **recall** against the
pre-declared surface; **plan signal** (structured `status: ok` with a risk
tier); **target validity**; **architecture determinism** (repeated call returns
identical ranked list); impact/inspect signals for investigation; **budget
compliance** and **no invalid files**; and a **fingerprint** (sha256 over the
deterministic portions, latency and working-root prefix excluded) that must be
byte-identical across runs.

A task **passes** only when its invoked tools ran without error and the surface
is genuinely satisfied. The ambiguous task passes only on a graceful,
structured, bounded response.

### Running the evaluation

```bash
# default: evaluate the full corpus in both modes against the current repo
python benchmarks/opencode_e2e.py .

# compare against the core get_context surface only
python benchmarks/opencode_e2e.py . --mode baseline

# one task, with per-task misses / signals
python benchmarks/opencode_e2e.py . --task search-ranking-bug --diagnostics
```

The script first validates the `opencode.json` RepoLens server entry
(read-only), then materialises a pristine temp copy of the repository, runs
every task in the requested modes, and prints a per-task table, aggregate pass
counts and mean required-recall per mode, and the mode fingerprints.

> **Exit-code semantics (intentional).** The script exits `0` only when every
> evaluated task in **every** requested mode passed. Running the default
> (`--mode both`) therefore exits `1`, because the **baseline** core-only
> surface deliberately under-covers change workflows (it measures the delta the
> additive tools provide). Run `--mode assisted` to assert the assisted
> workflow passes (exit `0`). Do not treat the `both`/`baseline` non-zero exit
> as a failure of assisted mode.

### Current honest result

Running the full corpus against this repository's clean, pristine tree:

- **assisted: 8/8 tasks pass**, mean required-file recall **1.000**
- **baseline: 4/8 tasks pass**, mean required-file recall **0.500**
- no invalid files, no budget violations, no tool errors in either mode;
- fingerprints are stable across consecutive identical runs.

The baseline (core-only) misses are expected: the additive tools exist
precisely because the raw `get_context` surface under-covers change workflows —
that is the delta this benchmark quantifies.

### Resolved findings

Two review fixes landed in the production change-plan layer: a
module/package-kind primary target now resolves its **defining source file**
into the affected surface and the change-context candidate set, and the
`change_plan` payload now reports wall-clock time under a clearly-labelled
`diagnostics` object instead of the deterministic `statistics` block, making
`deterministic: true` cover every substantive field.

## Deterministic, offline measurement

The repository is materialised once into a pristine temp copy that skips the
same local-only debris as `repolens.production_benchmark._SKIP_NAMES`:
virtualenvs, caches, node modules, quickfix directories, and the
`.benchmark_data` external-library download. Tool outputs are repo-relative
paths, so the computation is stable across copies. No production code is
modified, no LLM is consulted, and nothing requires network access.

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