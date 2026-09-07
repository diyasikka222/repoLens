# Change-Aware Context (Milestone 24.2)

Milestone 24.2 makes the deterministic change-plan engine (M24.1) usable by AI
agents through the MCP layer, and folds its plan into the existing context
retrieval pipeline as a clearly-separated candidate tier. It adds two MCP tools
and one new context-engine mode; **no existing API or tool is modified**.

## What is added

| Piece | Where | Purpose |
| ----- | ----- | ------- |
| `change_plan` MCP tool | `repolens/mcp/change_plan_tool.py` | Produce a bounded, explainable change plan for a request / target. |
| `change_context` MCP tool | `repolens/mcp/change_plan_tool.py` | Change-aware context: the plan folded into ranking + budget, behind the same firewall. |
| Change-aware engine mode | `repolens/context/engine.py` | `ContextEngine.build_context(query, change_request=..., change_target=..., change_plan=..., change_options=...)`. |
| Bridge module | `repolens/change_context.py` | `ChangeContextOptions`, `plan_to_change_candidates`, `merge_change_candidates`, `explain_change_context`, `plan_response_payload`. |
| Shared lazy state | `repolens/mcp/change_plan_tool.py` | `ChangePlanState` (index + default engine) built once and reused per call. |

`get_context`, `inspect_symbol`, `analyze_impact`, `inspect_architecture`,
`discover_subsystems`, `architecture_candidates`, and
`explain_architecture_match` are byte-for-byte unchanged. `build_context(query)`
(with no `change_request`) runs the exact original pipeline.

## The change-plan tier in context

`build_context(query, change_request=...)` computes the standard candidate set
(retrieval primaries → dependency expansion → architecture enrichment) exactly
as before, then converts the change plan into `ContextCandidate` objects and
appends them as a final tier. Joint ranking and budgeting apply:

- change-plan candidates rank in their own bucket with stable keys
  `(1, 4, change_priority, path)` — always **below** every direct
  query match, so a change-plan file never displaces a strongly-retrieved
  primary;
- files are deduplicated on first occurrence — a file that is both a retrieval
  primary and a change-plan candidate keeps its higher-tier retrieval role;
- surviving change-plan files are exposed on `ContextPackage.change_candidates`
  with per-file change metadata (`change_category`, `change_confidence`,
  `change_relationship`, `change_priority`);
- `total_estimated_tokens` still respects the context budget.

## Tools

### `change_plan`

```
change_plan(
  request,            # required, non-empty
  target=None,        # explicit file path, dotted module, package, or symbol
  target_kind=None,   # one of: file, module, package, symbol
  max_targets=None,   # <= 50
  max_files=None,     # <= 500   -> plan max_affected_files / primary targets
  max_tests=None,     # <= 100
  max_depth=None,     # <= 6     -> impact_max_depth
)
```

Returns a JSON-safe plan: `analysis`, `primary_target`, `target_candidates`,
`affected_files`, `affected_symbols`, `callers`, `callees`, `dependencies`,
`dependents`, `architecture`, `tests`, `inspection_order`, `risk`,
`risk_factors`, `confidence`, `summary`, `statistics`, `deterministic: true`.

### `change_context`

```
change_context(
  request, max_tokens=None, target=None, target_kind=None,
  max_targets=None, max_files=None, max_tests=None, max_depth=None,
  include_tests=True, include_dependencies=True, include_callers=True,
  include_callees=True, include_architecture=True,
)
```

Returns a safe, structured response: status, the effective budget,
`total_estimated_tokens`, intent, matched symbols, `selected_files` (each with
path, role, decision, token estimate, selection reason, inclusion reason, and a
structured per-file `explanation`), `blocked_files`, `change_files` (the
surviving plan candidates), a compact `change_plan` summary, a flat
explanations list, firewall findings, and `rendered_safe_context`.

Only firewall-approved content reaches the caller: the plan output is
structured and the change-aware context passes through `ContextFirewall` /
`safe_package` exactly like `get_context`, so blocked/redacted files never
appear in the returned context.

## Category filters

`ChangeContextOptions` (or the tool's `include_*` flags) deterministically
control which plan categories are *introduced* by the change layer:

| Flag | Controls |
| ---- | -------- |
| `include_tests` | `test` and `indirect_test` plan items |
| `include_callers` | `direct_caller` and `indirect_caller` items |
| `include_callees` | items whose relationship is `direct_callee` / `indirect_callee` |
| `include_dependencies` | `direct_dependency` and `indirect_dependency` items |
| `include_architecture` | `architecture_entry`, `subsystem_neighbor`, `broader_neighbor` |

The primary change target is always included. Filtering never changes normal
retrieval behavior: a file removed here is simply not *introduced* by the plan
(if retrieval still surfaces it, it keeps its retrieval role).

## Shared, lazy, warm state

`ChangePlanState` (built lazily by the launcher's `make_change_plan_factory`)
follows the architecture-tools pattern:

- the incremental index is built **once** on first use;
- the default `ChangePlanEngine` — with its dependency graph, symbol index,
  call graph, architecture graph, subsystems, impact analyzer, and searcher —
  is built **once**;
- per-call bounds (`max_files`, `max_tests`, `max_depth`, …) derive a per-call
  engine that **reuses every shared component** (no re-parse, no rebuild);
- a warm state re-uses the persistent index and parses **zero** files;
- `REPOLENS_CACHE_DISABLED` is honored and still produces correct output.

## Explainability

`explain_change_context(plan, package, path, safe_package=...)` answers "why
was this file selected?" deterministically: plan membership, category
explanation, relationship, priority, confidence, evidence, ranking survival,
budget status, inclusion reason, whether the change plan was the source of the
file, and the firewall decision.

## Determinism, boundedness, safety

- Everything is deterministic: identical input produces identical output (see
  `_substantive` comparisons in the benchmark for run-env stats).
- Everything is bounded: `MAX_TARGETS=50`, `MAX_FILES=500`, `MAX_TESTS=100`,
  `MAX_MAX_DEPTH=6` at the MCP layer, plus the engine/plan config bounds.
- Errors are safe: argument validation raises `InvalidArgumentsError`, engine
  failures raise `ChangePlanError` / `ChangeContextError`, and factory
  misconfiguration raises `InternalError` — no tracebacks reach the caller.
- Unsafe targets (path traversal, absolute paths, windows separators) are
  rejected by the same `_target_is_safe` guard the architecture tools use.

## Extending the existing tests / benchmark

- `tests/test_mcp_change_plan.py` — MCP-layer validation, structured output,
  boundedness, determinism, shared/warm state, error safety, protocol-level
  registration.
- `tests/test_context_engine.py` (`Change-aware engine` section) — the
  `build_context(..., change_request=...)` mode: change candidates, tier
  ranking, dedupe, budget, determinism, empty-repository behavior.
- `tests/test_change_plan.py` (`Change-aware bridge` section) — `plan_to_change_candidates`
  filters and `plan_response_payload` / `explain_change_context`.
- `benchmarks/verify_change_context.py` — fixture + real-repo verification
  (cold / warm / cache-disabled / incremental / protocol), without flaky timing
  assertions.