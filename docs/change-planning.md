# Change-Plan Engine (Milestone 24.1)

The change-plan engine (`repolens/change_plan.py`) is RepoLens' deterministic
**change-planning** layer: it converts a natural-language change request into a
bounded, explainable *inspection / change plan* — which files to look at, in
what order, which tests to run, how risky the change is, and *why*.

It is **not** a code generator and it contains **no LLM**. Every signal is
computed by reusing the existing shared repository index, dependency graph,
call graph, architecture graph, subsystem discovery, impact analyzer, and
lexical search. Nothing is reparsed and no second repository scan exists.

M24.1 is engine-only. Integration with the MCP server is Milestone 24.2
(`change_plan` and `change_context` tools, plus the change-aware
`ContextEngine` mode). A small additive helper — `plan_to_context_candidates` —
is provided so the plan layer can be reused without touching `get_context`.

## Quick start

```python
from repolens.change_plan import ChangePlanEngine

engine = ChangePlanEngine("path/to/repository")
plan = engine.plan("Add refund support to checkout")
print(plan.primary_target)       # highest-ranked candidate
print(plan.affected_files)       # impacted files (impact analysis)
print(plan.tests)                # tests that should be run
print(plan.inspection_order)     # what to inspect first, with priorities
print(plan.risk, plan.risk_factors)  # deterministic risk tier + explanation
print(plan.summary)              # one-line human summary
```

An explicit target may be supplied instead of (or in addition to) free text:

```python
plan = engine.plan("change the discount logic", target="app/services/checkout.py")
```

An explicit target resolves in priority order: **file path** → **package path**
→ **dotted module** → **exact symbol** → **architecture package/module**.

## What the plan contains

`ChangePlan` fields:

| Field | Meaning |
| ----- | ------- |
| `targets` | Ranked candidate targets (kind, score, confidence, reasons). |
| `primary_target` | The single highest-ranked candidate (or `None`). |
| `affected_files` | Impact-enrichment results (callers, dependencies, dependents, tests, …). |
| `affected_symbols` | Symbols tied to the affected files. |
| `architecture` | Per-target architecture metadata (package, subsystem, dependencies, dependents, entry module, dependency/dependent packages). |
| `tests` | Candidate tests with the reason each is recommended. |
| `inspection_order` | Deterministically ordered inspection items with `priority` 1..N. |
| `risk` / `risk_factors` | `low` / `medium` / `high` plus the contributing factors. |
| `summary` | Human-readable one-line summary. |
| `stats` | `parsed_files` and `build_time` (run-environment metadata). |

Every item carries `confidence` (`confirmed` / `likely` / `possible`) and a
free-text `reason`, so the plan never claims false precision. Where a signal is
a lexical or architectural heuristic rather than a confirmed static edge, the
reason and confidence make that explicit.

## Pipeline

Work in the pipeline is overt: each stage has a dedicated function and the plan
carries its results through.

1. **Request analysis** — `analyze_request` deterministically extracts the
   action verb (`add`, `remove`, `modify`, `rename`, `refactor`, `replace`,
   `fix`, `extend`, `migrate`), tokenised terms, domain terms (words not in the
   stop-word list), path candidates, dotted module candidates, and CamelCase
   symbol candidates.
2. **Target discovery** — candidate targets are produced and scored:

   | Signal | Base score | Confidence |
   | ------ | ---------- | ---------- |
   | explicit file path | 100 | confirmed |
   | explicit module | 90 | confirmed |
   | explicit symbol | 85 | confirmed |
   | explicit package | 80 | confirmed |
   | module/file path in request | 95 / 80 | confirmed / likely |
   | dotted module in request | 85 | confirmed |
   | symbol in request | 85 | confirmed |
   | lexical search (domain terms) | 60 | likely |
   | architecture signal | 40 (module) / 35 (package) | likely / possible |

   Duplicate targets keep the highest score and union of reasons, then sort by
   `(-score, target)` for stable determinism.
3. **Impact enrichment** — each primary target runs through the existing
   `ImpactAnalyzer`; results map to plan categories (`direct_caller`,
   `indirect_caller`, `direct_dependency`, `indirect_dependency`, `test`,
   `reverse_dependency` → caller/dependency, …). All traversal is bounded by
   `impact_max_depth` / `max_impact_nodes` / `max_affected_files`.
4. **Architecture enrichment** — per-target package, subsystem, dependencies,
   dependents, entry module, and dependency/dependent packages are recorded via
   the existing `ArchitectureGraph` and `discover_subsystems`.
5. **Test discovery** — every `*test*` file is inspected once: tests that
   import an affected module, plus tests whose filename/leaf matches an
   affected module (e.g. `tests/test_checkout.py` ↔ `app.services.checkout`).
6. **Inspection ordering** — items are ordered as: primary target, direct
   callers, direct dependencies, architecture entry points, tests, then
   indirect callers/dependencies/broader neighbors. Priorities are assigned
   1..N and ties are broken by source order.
7. **Risk assessment** — a deterministic score is computed from affected-file
   count, caller/dependent counts, package/subsystem spread, public/API-facing
   changes, test coverage, and unresolved references.
8. **Summary** — a one-line human summary is assembled.

The pipeline never blocks on external services, never mutates the repository,
and produces identical output for identical input (see Determinism).

## Bounds

All traversal is bounded by `ChangePlanConfig` (defaults shown):

| Bounds | Default |
| ------ | ------- |
| `max_target_candidates` | 30 |
| `max_primary_targets` | 5 |
| `max_affected_files` | 50 |
| `max_call_depth` | 2 |
| `max_dependency_depth` | 2 |
| `max_tests` | 15 |
| `max_inspection_items` | 40 |
| `max_subsystems` | 10 |
| `max_total_nodes` | 200 |
| `max_impact_nodes` | 200 |
| `impact_max_depth` | 3 |
| `search_limit` | 20 |
| `arch_max_expanded` | 50 |

```python
from repolens.change_plan import ChangePlanConfig

engine = ChangePlanEngine(
    "path/to/repository",
    config=ChangePlanConfig(max_inspection_items=10, impact_max_depth=1),
)
```

## Determinism

The engine is deterministic in three senses:

- identical input → identical plan for a given engine instance, including
  `stats`;
- identical input → identical *substantive* plan across engine instances
  (reasons, ordering, risk, architecture) — only `stats.parsed_files` /
  `build_time` reflect the run;
- serializable output: all plan fields can be converted with
  `dataclasses.asdict` and JSON-serialised.

## Warm reuse of the shared index

The engine builds (or accepts) the same **shared incremental index** used by
the rest of RepoLens — it never re-parses:

- **cold:** a fresh engine parses the repository and produces a plan
  (`stats.parsed_files` = number of files parsed);
- **warm:** a second engine for the same root re-uses the persistent
  on-disk cache and parses **zero** files while producing an identical
  substantive plan (`stats.parsed_files` = 0);
- **incremental:** a file added after the first snapshot is picked up by a
  fresh engine without re-parsing unchanged files.

The persistent cache lives under the RepoLens cache base (same location and
identity scheme as the MCP launcher) and honours the standard environment
controls: `REPOLENS_CACHE_DIR` and `REPOLENS_CACHE_DISABLED`. When caching is
disabled the engine builds an ephemeral index and still produces correct,
deterministic plans.

## `plan_to_context_candidates`

A small, **additive** helper converts a plan into context-retrieval candidates
for later milestones:

```python
from repolens.change_plan import plan_to_context_candidates

cands = plan_to_context_candidates(plan, limit=8, include_tests=True)
# [{"path": ..., "module": ..., "symbol": ..., "priority": 1,
#   "category": ..., "confidence": ..., "reason": ..., "relationship": ...}, ...]
```

It is inert today — `get_context` does not use it — so adding it changes
nothing about existing retrieval behavior.

## `change_context`

Milestone 24.2 wires plans into the MCP layer. The `change_context` MCP tool
(and the `ContextEngine.build_context(..., change_request=..., ...)` mode it
drives) folds the plan from the change-plan layer into the existing retrieval
pipeline as a clearly-separated, final-ranking candidate tier — see
[`docs/change-context.md`](change-context.md) for the full contract, the
`change_plan` and `change_context` tools, filtering options, and the
`change_context` helper module.

## Worked example

Fixture: `tests/fixtures/change_plan_repository` (an `app/` package with
`api`, `services`, `models`, `repositories`, `validators`, `config` and a
`tests/` package).

Request: **“Add refund support to checkout”**

1. Request analysis: action=`add`, domain terms ≈ `{refund, checkout}`.
2. Target discovery: ranked candidates include `app/api/checkout.py`,
   `app/api/refunds.py`, `app/models/refund.py`,
   `app/repositories/refunds.py`, `app/services/checkout.py` (lexical
   search + architecture signals). Primary target =
   `app/api/checkout.py`.
3. Impact enrichment: 12 affected files — `app/services/checkout.py`
   (direct caller), `app/repositories/refunds.py`,
   `app/models/refund.py`, … — each with a category, reason, and
   confidence.
4. Architecture enrichment: target packages, subsystem `app`,
   dependency/dependent packages.
5. Test discovery: `tests/test_checkout.py`, `tests/test_refunds.py`,
   `tests/test_payments.py` (import affected modules or match module
   names).
6. Inspection order: primary first, then callers/dependencies/tests.
7. Risk: `high` — `['moderate affected files (12)', 'many callers (7)',
   'many dependents (7)', 'cross-package change (11 packages)']`.

```python
plan = ChangePlanEngine(ROOT).plan("Add refund support to checkout")
assert plan.risk == "high"
assert plan.primary_target.target == "app/api/checkout.py"
assert "tests/test_refunds.py" in {t.path for t in plan.tests}
```

## Forward / backward compatibility

- M24.1 adds a new module and a new fixture; **no existing API is modified**.
- `get_context`, the MCP server, and all M23 tools are untouched
  (`get_context` does not call the helper).
- The engine reuses — never re-implements — the index, graph, search, impact,
  architecture, and subsystem layers; those APIs keep their signatures.
- All scoring, ordering, and risk tiers are plain pure functions of the index
  state, so behaviour is stable and testable offline.

## Scope outside M24.1

- MCP `change_plan` / `change_context` tool wiring and change-aware context
  building (M24.2) — see `docs/change-context.md`.
- Editing / patch generation, test-run triggering, and any kind of code
  synthesis (explicitly out of scope: the engine plans *where* and *in what
  order* to act; it never acts).