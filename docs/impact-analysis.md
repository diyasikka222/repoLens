# Change Impact Analysis (Milestone 21)

This document describes the deterministic, fully-offline change-impact module
(`repolens/impact.py`), the change-aware context integration, the `analyze_impact`
MCP tool, precise risk-scoring rules, the conservative evidence model, and the
guarantees (and limits) of what this analysis can and cannot claim.

The milestone is *additive*, not a redesign: the analyzer is built on the
existing `RepositoryIndex` (incremental, content-hashed), `DependencyGraph`,
`SymbolIndex`, `ContextEngine`, `ContextBudget`, and `ContextFirewall` without
changing their behavior. `get_context` and `ContextEngine.build_context` are
byte-for-byte unchanged.

---

## What it answers

"*If I change `X`, what else is affected, how, and how risky is it?*" — where `X`
can be a file, a dotted module, a symbol, or a symbol inside a specific file.

```bash
python -c "
from repolens.impact import ImpactAnalyzer
result = ImpactAnalyzer('.').analyze('repolens/incremental_index.py')
print(result.risk, result.summary)
for item in result.items:
    print(item.path, item.relationship.value, item.risk.value, item.evidence)
"
```

`analyze(target, *, max_depth=None, limit=None)` returns an `ImpactResult` with:
`target`, `kind`, `target_path`, `symbol`, `module`, `risk`, `max_depth_reached`,
`summary`, and ordered `items` (each with `path`, `relationship`, `risk`,
`reason`, `symbol`, `evidence`, `depth`).

---

## Supported targets

| Form | Example | Resolution |
| --- | --- | --- |
| File | `lib/core.py` | exact repo-relative path |
| Module | `lib.core` | dotted module name → file |
| Symbol | `Engine` | uniquely-defined symbol (ambiguous → error) |
| File + symbol | `lib/core.py::Engine` | symbol defined in that file |

Unknown targets raise `ImpactTargetError`. Ambiguous bare symbols/module leaves
are rejected with a message listing the candidate files and suggesting the
`path/file.py::symbol` form — nothing leaks outside the repository, and no
silent guess is made.

Resolution order is exact and deterministic: `file::symbol`, file path,
dotted module, symbol, then uniquely-named module leaf.

---

## Relationships

One primary relationship per affected file, chosen deterministically:

| Relationship | Meaning | Item risk |
| --- | --- | --- |
| `direct_dependency` | imports the target directly | high |
| `indirect_dependency` | imports the target transitively | medium |
| `reverse_dependency` | the target imports this file | low |
| `test` | a test file linked to the target | medium |
| `configuration` | a config/settings file or declarative config | medium |
| `api_consumer` | references a target symbol | high (public) / medium (private) |

Priority (higher wins): `test` > `configuration` > `api_consumer` >
`direct_dependency` > `indirect_dependency` > `reverse_dependency`. All evidence
tags are preserved on the item regardless of the winning relationship.

> When a Milestone 22 call graph is supplied (`reference_graph`), symbol targets
> additionally yield `direct_caller` / `indirect_caller` / `direct_callee` /
> `indirect_callee` relationships (priority `test > configuration >
> direct_caller > api_consumer > direct_dependency > indirect_caller >
> indirect_dependency > direct_callee > indirect_callee >
> reverse_dependency`), each with `confidence="static"`. See
> [call-graph.md](call-graph.md).

Ordering within a result is deterministic: `(relationship_bucket, depth, path)`.

---

## Reverse traversal

Impact is discovered by walking **reverse dependency edges** (who imports the
target) with a breadth-first search bounded by:

- `ImpactConfig.max_depth` (default 4; `max_depth=0` disables reverse traversal),
- `ImpactConfig.max_nodes` (default 2000) — a hard cap on visited nodes.

`max_depth`/`limit` can also be passed per `analyze()` call. This is *never*
unbounded: a pathological hub cannot exhaust memory or hang. `max_depth_reached`
records the deepest reverse depth observed and feeds the risk model.

Transitively-reachable files are `indirect_dependency`; a direct importer that
also imports it transitively stays at its nearest depth.

---

## Conservative symbol evidence

Symbol-level findings are **evidence, not a call graph**. The analyzer never
claims "file F calls function G". Instead it reports concrete, checkable
signals, each with a machine-readable tag:

| Tag | Signal |
| --- | --- |
| `symbol_import` | `from <module> import <symbol>` (by name) |
| `symbol_text` | qualified `<module/Class>.<symbol>` text reference |
| `base_class` | class inherits the target symbol |
| `package_export` | package `__init__` re-exports the symbol |
| `test_name_match` / `test_import` | test-file link |
| `config_import` | config-stem file imports the target |
| `config_text` | declarative config names the module |
| `dependency_edge` | the underlying repo-internal import edge |

Symbol matching only scans files that already import the target's module (or
reach it via a package re-export) — it never walks the whole repository, and a
bare *file* analysis emits no symbol claims at all.

---

## Risk scoring (deterministic)

Overall risk is computed from the summary counts:

```
points  = min(direct_dependents, 5) * 2
        + min(indirect_dependents, 3) * 1
        + min(tests, 2) * 1
        + 2 if any api consumers
        + 1 if the symbol is exported
        + 1 if max_depth_reached >= 3
high   := points >= 8
medium := points >= 4
low    := otherwise
```

Per-item risk is table-driven (see Relationships). Every rule is a pure function
of the repository snapshot — the same repository and target always produce the
same result and the same risk.

---

## Test and configuration discovery

Tests (`tests/` / `test_*.py` / `*_test.py`) are linked to a target when they
import it, reference its symbol, or when their filename corresponds to the
target module leaf. Configuration files (python stems like `settings`/`config`,
plus declarative files such as `pyproject.toml`, `setup.cfg`, `tox.ini`) are
linked the same way. Both are excluded by default analysis options only when
explicitly disabled.

---

## Change-aware context

`ContextEngine.build_impact_context(target, *, max_depth=None, budget=None)`
returns a `ContextPackage` whose priority is: changed target, direct dependents
(nearest first), tests, indirect dependents (nearest first), api consumers,
configuration, reverse dependencies. The package flows through `ContextBudget`
(`select_within_budget`) and `ContextFirewall` exactly like any other context,
and adds three inclusion reasons: `test`, `api_consumer`, `configuration`
(`repolens.context.candidate`), keeping the existing token names untouched.

`build_change_context(query)` extracts a symbol from a natural-language query
and runs impact analysis on it. Both are new, additive methods;
`build_context(query)` is unchanged.

---

## MCP integration

The `analyze_impact` tool is registered only when an impact factory is wired
in (the launcher passes it by default):

```bash
python -m repolens.mcp --repo /path/to/repository
```

Arguments: `target` (required), `max_depth` (non-negative int), `limit`
(positive int). The response is a JSON-safe dict: `status`, `target`, `kind`,
`target_path`, `symbol`, `module`, `risk`, `max_depth_reached`, `summary`,
`item_count`, `items`. Unknown/ambiguous targets return a safe error message.
Initialization stays lazy: the root is validated at startup, the analyzer
(reusing the engine's persistent index) is built on first call.

---

## Limitations

- M21 itself adds no call-graph; symbol findings are deliberately
  conservative. (Milestone 22 adds a separate, offline *static* call graph with
  its own guarantees — see [call-graph.md](call-graph.md) — and wires it in as
  an *optional* `reference_graph`; without it this behavior is unchanged.)
- Import resolution uses the existing parser's view of the repository;
  dynamic/`exec`-based import patterns are invisible.
- Risk is a heuristic ranked warning, not a guarantee of breakage below a
  threshold and not a claim of safety above it.
- Cross-repository and dependency-package impact is out of scope.