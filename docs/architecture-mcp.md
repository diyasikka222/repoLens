# Architecture MCP Tools (Milestone 23.3)

This document describes the four **additive** architecture-intelligence tools
exposed by the RepoLens MCP server (`repolens/mcp/architecture_tool.py`):

| Tool | Purpose |
| ---- | ------- |
| `inspect_architecture` | Structured overview of a single repository node (file / module / package): its package/module/file identities, subsystem, bounded dependency/dependent lists, bounded transitive neighborhood, and statistics. |
| `discover_subsystems` | The deterministic subsystem discovery result: subsystem ids, full identity projections (packages, modules, files, entry modules), statistics, and inter-subsystem dependency projections. |
| `architecture_candidates` | Architecture-aware candidate generation (independent of full context retrieval): direct matches plus bounded dependency / dependent / neighbor / subsystem-proximity expansions. |
| `explain_architecture_match` | Why a node was (or was not) considered architecturally relevant to a query: matched nodes, signals, rank, direction, subsystem relationship, and bounded dependency paths. |

M23.3 is a **thin adapter** following the established MCP pattern:

1. each tool **validates its arguments strictly**;
2. tools build the **shared architecture state** through an injected factory
   (lazy, built once, reused by every tool call);
3. they call only existing public RepoLens APIs
   (`ArchitectureGraph`, `discover_subsystems`,
   `extract_architecture_signals`, `architecture_candidates`,
   `explain_architecture_match`);
4. they return only **structured, safe, JSON-serializable** output — no
   internal Python object ever reaches the caller.

The core `get_context` tool and the additive `analyze_impact` / `inspect_symbol`
tools are **unchanged**; the four architecture tools are registered only when
an `architecture_factory` is wired into `build_mcp_server`.

---

## Wiring

The launcher builds and caches the shared state (`launcher.py`):

```python
from repolens.mcp.launcher import make_architecture_factory
from repolens.mcp.server import build_mcp_server

architecture_factory = make_architecture_factory(repo_root)
server = build_mcp_server(
    engine_factory,
    firewall,
    architecture_factory=architecture_factory,   # additive, optional
)
```

`make_architecture_factory` validates the root eagerly (fail fast) and then
lazily constructs:

- the **shared incremental index** (same persistent cache as the engine /
  impact / inspect factories → a warm architecture build parses **zero** files);
- the **architecture graph** (a pure projection of that index);
- the **subsystem discovery** result (uses the bounded, deterministic
  subsystem layer, cycle-safe).

All four tools share that single state instance, so calling any tool never
re-scans or re-parses.

```bash
python -m repolens.mcp --help
```

To run the server against a repository:

```bash
python -m repolens.mcp --repo /path/to/repo
```

---

## `inspect_architecture`

Inputs (validated):

- `target` (required, string): repository-relative path or identifier —
  `store/services/checkout.py`, `store.services.checkout`, or `store/services`.
  Rejects absolute paths, `..`, backslashes/UNC/drive prefixes, `:`, control
  characters, and empty values.
- `max_depth` (int, default `2`, capped at `6`): transitive neighborhood depth.
- `include_dependencies` / `include_dependents` / `include_subsystems`
  (bool, default `true`): toggle output sections.

Output highlights:

```json
{
  "status": "ok",
  "target": "store.services.checkout",
  "kind": "module",
  "id": "store.services.checkout",
  "package": "store/services",
  "module": "store.services.checkout",
  "file": "store/services/checkout.py",
  "subsystem": "store",
  "dependencies": [{"id": "store.models.order", "kind": "module"}, "..."],
  "dependents": [{"id": "admin.reports", "kind": "module"}, "..."],
  "neighborhood": {"max_depth": 2, "dependencies": [...], "dependents": [...]},
  "statistics": {
    "direct_dependency_count": 2,
    "direct_dependent_count": 4,
    "transitive_dependency_count": 2,
    "transitive_dependent_count": 4,
    "package_file_count": 3,
    "package_module_count": 3,
    "subsystem_stats": {"packages": 5, "modules": 12, "files": 12, "entry_modules": 2}
  }
}
```

Unknown targets raise a safe `InvalidArgumentsError` ("The target does not
exist in the current repository index.") — never a stack trace.

---

## `discover_subsystems`

Inputs:

- `max_subsystems` (int, default `null` = all, positive, capped at `500`).
- `include_stats` / `include_dependencies` / `include_dependents`
  (bool, default `true`).

Output highlights:

```json
{
  "status": "ok",
  "subsystem_count": 6,
  "subsystems": [
    {
      "id": "store",
      "display_name": "store",
      "packages": ["store/models", "store/api", "store/services", "store/repositories", "store"],
      "modules": ["store.api.catalog", "..."],
      "files": ["store/services/checkout.py", "..."],
      "entry_modules": ["store", "store.services"],
      "statistics": {"packages": 5, "modules": 12, "files": 12, "entry_modules": 2},
      "dependencies": ["billing"],
      "dependents": ["admin", "billing", "tests"]
    }
  ]
}
```

Subsystem ordering is **deterministic** (`id`-sorted); the response is a pure
projection of the deterministic discovery result and is identical on every
call. An empty repository yields `subsystem_count: 0`.

---

## `architecture_candidates`

Inputs:

- `query` (required, non-empty string).
- `limit` (int, default `20`, positive, capped at `200`).
- `max_depth` (int, default `1`, capped at `6`) — maps to
  `max_neighbor_depth` for expansion.
- `architecture` (object, optional) — the four documented retrieval caps:
  `max_package_candidates` (cap 100), `max_module_candidates` (cap 500),
  `max_neighbor_depth` (cap 6), `max_expanded_nodes` (cap 1000). Unknown keys
  are rejected.

Output highlights:

```json
{
  "status": "ok",
  "query": "checkout",
  "candidate_count": 15,
  "candidates": [
    {
      "file": "store/services/checkout.py",
      "module": "store.services.checkout",
      "package": "store/services",
      "score": 1.0,
      "architecture_score": 0,
      "inclusion_reason": "architecture: direct file match",
      "architecture_node_type": "file",
      "architecture_node_id": "store/services/checkout.py",
      "subsystem": "store",
      "dependency_direction": "matched"
    }
  ],
  "signals": [
    {"kind": "module", "value": "store.services.checkout",
     "node_kind": "module", "node_id": "store.services.checkout",
     "reason": "architecture: direct module match"}
  ]
}
```

Ranking follows the existing retrieval model: **direct matches rank 0**,
neighbors rank 1, same-subsystem proximity ranks 2 — strictly ordered
(`architecture_score` increases) and deterministic. `score = round(1 / (rank + 1), 3)`
is provided for convenience. Matching, dependency/dependent expansion, and
expansion caps all live in `architecture_retrieval.py`; the MCP tool only
projects and caps the result.

---

## `explain_architecture_match`

Inputs: `query` (required) and `target` (required, same safety rules as
`inspect_architecture.target`).

A target that **was** architecturally relevant returns:

```json
{
  "status": "ok",
  "query": "checkout",
  "target": "store.models.order",
  "is_architecturally_relevant": true,
  "node": {"id": "store.models.order", "kind": "module"},
  "path": "store/models/order.py",
  "inclusion_reason": "architecture: dependency of matched module",
  "architecture_score": 1,
  "direction": "dependency",
  "package": "store/models",
  "subsystem": "store",
  "subsystem_relationship": {"target_subsystem": "store", "display_name": "store"},
  "signal_count": 2,
  "signals": [{"kind": "file", "value": "store/services/checkout.py", "..."}],
  "matched_node_count": 15,
  "matched_nodes": [{"id": "...", "kind": "module", "reason": "architecture: dependency of matched module", "rank": 1, "direction": "dependency"}],
  "dependency_paths": [["store.services.checkout", "store.models.order"]]
}
```

A target that **was not** relevant returns a **structured**
`is_architecturally_relevant: false` with an `explanation` (and no
`dependency_paths`) — this is a normal response, not an error. Only an
*invalid* `target` (missing, unsafe, or unknown filter shape) raises a safe
`InvalidArgumentsError`.

`dependency_paths` are bounded (a nested-list, each path ≤ the matched-path
cap) and built with a bounded breadth-first search over dependency edges, so
hostile queries cannot produce unbounded output.

---

## Validation & errors

Every tool follows the strict validation model used by the other MCP tools:

- unknown argument keys → `InvalidArgumentsError`;
- wrong types, non-positive ints, out-of-range ints → `InvalidArgumentsError`;
- unsafe `target` values → `InvalidArgumentsError`;
- missing required arguments → `InvalidArgumentsError`;
- factory/construction failures → safe `ArchitectureError`
  (`repolens/mcp/errors.py`), carrying a private diagnostic for logging;
- any unexpected exception at the protocol boundary is converted to a safe,
  generic message — **no stack trace or internal detail ever reaches the
  caller** (all diagnostics go to stderr via `logging.getLogger("repolens.mcp")`).

The MCP-layer caps are intentionally stricter than the retrieval defaults
(`MAX_MAX_DEPTH = 6`, `MAX_CANDIDATE_LIMIT = 200`,
`MAX_SUBSYSTEMS_LIMIT = 500`, `NODE_NEIGHBOR_CAP = 200`), so responses stay
bounded even on very large repositories.

---

## State & caching

The architecture graph and subsystem discovery are **pure projections** of the
shared incremental index:

- **Warm builds parse zero files.** Once the persistent index cache exists for
  the repository, `make_architecture_factory(...)()` reports
  `parsed_file_count == 0` (verified in the benchmark).
- **No duplicate work.** After the first tool call, the index, graph, and
  subsystem result are cached for the lifetime of the state; every subsequent
  tool call reuses them.
- **Incremental behavior.** A fresh state (or a rebuild) re-derives the graph
  from the *current* index snapshot: modified files contribute new edges,
  deleted files disappear, and new files appear — verified in tests.
- **Cache-disabled mode.** `REPOLENS_CACHE_DISABLED=1` (or
  `REPOLENS_CACHE_DIR=""`) builds an ephemeral in-memory index; the tools keep
  working and every fresh state re-parses, with identical output shape.

---

## Verification

- **Tests** — `tests/test_mcp_architecture.py` (47 tests): target-safety
  validation, all four tools (structure, boundedness, determinism, limits,
  unknown/invalid inputs), lazy + shared construction, no duplicate parses
  across tool calls, warm persistent-index reuse (zero parses), cache-disabled
  mode, incremental modification and file deletion, safe error paths, and
  protocol-level registration + calls (including that the core `get_context`
  tool is unchanged and that the four tools are absent when no architecture
  factory is wired in).
- **Benchmark** — `benchmarks/verify_architecture_mcp.py`: fixture-level
  structured/bounded/deterministic assertions, then a real-repository run
  through the launcher-wired factory (cold build timings, warm build with
  `parsed=0`, per-query candidate latencies, protocol-level calls, cache
  disabled mode, and an incremental-modification check on a disposable copy).
- **Regression** — the full suite (`tests/test_mcp*.py`,
  `tests/test_architecture*.py`, `tests/test_context_engine.py`,
  `tests/test_subsystems.py`) passes unchanged.

---

## Limitations

- The four tools are served over the MCP stdio transport only after an
  `architecture_factory` is wired in; without it they are not registered (the
  base server keeps exactly the `get_context` tool).
- Outputs are projections with **MCP-side caps**; on very large repositories
  `inspect_architecture` neighborhoods and `architecture_candidates` lists are
  intentionally truncated (per-side caps and retrieval caps) rather than
  exhaustive.
- Subsystem discovery is deterministic but heuristic (directory-clustering
  based on the subsystem layer); the tools report it as-is.
- The architecture graph reflects the **last built index snapshot** of the
  shared state; a long-lived server does not hot-swap the graph when files
  change on disk. Restarting the server (or building a fresh factory state)
  picks up changes.