# Offline Call Graph & Reference Extraction (Milestone 22)

This document describes RepoLens' fully-offline, conservative **cross-file
reference extraction** (`repolens/references.py`) and the **static call graph**
built on top of it (`repolens/call_graph.py`), plus how they plug into change
impact analysis (`ImpactAnalyzer(reference_graph=...)`), the change-aware
context engine, and the MCP server (`inspect_symbol`).

M22 is *additive*: every existing behavior is preserved when no call graph is
supplied. `ImpactAnalyzer` without `reference_graph`, `ContextEngine` without
one, and `get_context` are byte-for-byte unchanged.

---

## What it answers

- **"Who calls this?"** — reverse call edges (`callers`).
- **"What does this call?"** — forward call edges (`callees`).
- **"Who references / imports this?"** — `references_to` / `references_from`,
  `importers_of` / `imports_of`, `direct_dependents`.
- **"What can't be resolved?"** — an honest `unresolved_references()` list.

```bash
python -c "
from repolens.call_graph import CallGraphBuilder
from repolens.references import ReferenceIndexBuilder
from repolens.index import SymbolIndexBuilder
from repolens.incremental_index import IncrementalIndexBuilder

root = '.'
index = IncrementalIndexBuilder(root).build()
refs = ReferenceIndexBuilder(root, index=index).build()
symbols = SymbolIndexBuilder(root, index=index).build()
graph = CallGraphBuilder(root, index=index, reference_index=refs,
                         symbol_index=symbols).build()
print(graph.stats().as_dict())
for node in graph.get_nodes():
    if node.name == 'analyze':
        print([(c.file_path, c.name) for c in graph.callers(node)])
"
```

---

## Guarantees

- **Fully offline.** No LLM, no network, no language server. Everything is
  computed from the existing parser, module analysis, symbol index, dependency
  graph, and the new reference index.
- **Conservative.** A name is resolved to a definition *only* through a fixed,
  deterministic chain (see below). Anything else is recorded as an
  `UnresolvedReference` with a stable `UNRESOLVED_*` reason and **never
  guessed** into an edge.
- **Bounded.** Every traversal is bounded by `max_depth` and
  `CallGraphConfig.max_transitive_nodes` (5000). There is no unbounded
  recursion; `max_depth=0` returns empty.
- **Deterministic.** Same repository → same nodes, edges, unresolved list
  (verified by tests and `benchmarks/verify_call_graph.py`).
- **Warm.** The reference index is content-addressed and persisted; a warm
  build performs **zero** AST parses (verified by tests and the benchmark).

---

## Reference extraction (`references.py`)

`ReferenceIndexBuilder` walks each file and extracts `Reference` mentions with
a `ReferenceKind` of `CALL`, `NAME`, or `ATTRIBUTE`. Imports are *not*
re-extracted here — `ModuleAnalysis` (the parser) stays the source of truth for
imports, and the call graph joins the two. Mentions carry their file, symbol
scope, source line, and a value hash.

Two hint systems feed later resolution:

- `AssignmentHint(scope, name, target)` — records `c = CartController()` and
  `self.cart = models.Cart()` so the call graph can resolve receivers.
- `ParameterTypeHint` — records **typed** parameters (`order: Order`), which
  is the only mechanism that resolves unqualified method calls on parameters.

The reference index is a content-hashed, schema-versioned cache
(`REFERENCE_CACHE_SCHEMA_VERSION = 1`) with atomic writes and tolerant reads:
cold scan parses every file, a warm scan parses none, and a one-file edit
re-scans exactly that file. Corrupt/mismatched entries are ignored, never
fatal.

---

## Resolution precedence

Within a call site, the graph resolves the target name by this exact order
(implemented in `repolens/call_graph.py`):

1. **Exact local symbol** — a name bound in the current scope
   (definitions, imports, assignments via hints).
2. **Explicitly imported symbol** — `from .models import Order`, aliases
   `sanitize as clean`, module-qualified `payments.charge_card`.
3. **Resolved relative import** — `from . import models` binds the submodule
   when it exists (module binding), enabling `models.Cart()`.
4. **Resolved module-qualified symbol** — `app.payments.charge_card(...)`.
5. **Known class/method relationship** — receivers: assignment
   (`self.cart = models.Cart()` → `self.cart.add(...)`), `self` attributes,
   and typed parameters (`order: Order` → `order.total()`).
6. **Unresolved** — anything else is recorded with a reason
   (`UNRESOLVED_MODULE`, `UNRESOLVED_DYNAMIC`, …) and no edge.

Safety rails: a bound module is never treated as callable; builtins come from
`dir(builtins)`; self-assignment (`x = x`) and dotted chains are guarded by a
depth budget (no infinite loops).

---

## Graph model

- **Nodes** (`CallNode`): one per file module plus one per class/function
  (`file_path, name, kind, parent_class`). Module nodes have `name=None`.
- **Edges** (`CallEdge`), each with a kind, source/target, line, and evidence:
  - `CALL` / `METHOD_CALL` / `INSTANTIATION` (collectively "call edges"),
  - `IMPORT`,
  - `REFERENCE`.
- **Relationships** (`CallRelationship`): `CALLS`, `CALLED_BY`, `IMPORTS`,
  `IMPORTED_BY`, `REFERENCES`, `REFERENCED_BY` (six views over the raw edges).
- **Unresolved** (`UnresolvedReference`): `(file, source_symbol, name, reason,
  line)`.

### Key queries

| Query | Meaning |
| --- | --- |
| `callers(node, max_depth)` / `callees(node, max_depth)` | direct or transitive call sites |
| `bounded_transitive_callers/callees(node, max_depth)` | BFS, depth-based, deduplicated |
| `references_to(node)` / `references_from(node)` | REFERENCE edges in/out |
| `importers_of(path)` / `imports_of(path)` | file-level module import edges |
| `direct_dependents(node)` | importers of the node's file that also call/reference it |
| `unresolved_references()` / `unresolved_in(path)` | known-unknowns |
| `stats()` | `nodes`, `edges`, `calls`, `references`, `imports`, `unresolved` |

---

## Integration with impact analysis

`ImpactAnalyzer(root, reference_graph=graph)` adds four relationships for
symbol targets:

| Relationship | Meaning | Item risk | Confidence |
| --- | --- | --- | --- |
| `direct_caller` | calls the symbol directly | high | `static` |
| `indirect_caller` | calls it transitively | medium | `static` |
| `direct_callee` | the symbol calls this directly | medium | `static` |
| `indirect_callee` | the symbol calls it transitively | low | `static` |

- Call relationships are discovered with the *same bounded* BFS used by the
  module graph and layered by depth (`ImpactConfig.max_depth`).
- Files are still classified with one winning relationship — priority order is
  `test > configuration > direct_caller > api_consumer > direct_dependency >
  indirect_caller > indirect_dependency > direct_callee > indirect_callee >
  reverse_dependency` — so a test that calls the target stays a `test` item
  while retaining its call evidence.
- Each item carries `confidence="static"` for call relationships and the
  `resolved_ast_call` evidence tag, clearly separating statically-resolved
  findings from the M21 module/import evidence.
- The summary gains `direct_callers`, `indirect_callers`, `direct_callees`,
  `indirect_callees` counters, and per-item risk is table-driven as shown
  above. `_count_summary` is always recomputed from the finalized items.
- Without `reference_graph` the analyzer is exactly M21 (no call
  relationships, `confidence is None` on every item).

## Change-aware context

`ContextEngine(..., reference_graph=graph)` makes `build_impact_context` /
`build_change_context` surface the call relationships too. Candidate ordering
becomes: changed target, direct callers, direct callees, direct dependents,
tests, indirect callers, indirect callees, indirect dependents, api consumers,
configuration, reverse dependencies — all still flowing through
`select_within_budget` and `ContextFirewall`. The mapping is a pure function of
the item's winning relationship, so a missing call graph leaves M21 ordering
intact.

## MCP integration

`inspect_symbol` is an *additive* tool, registered only when an
`inspect_factory` (lazily building a `CallGraph`) is wired in. The launcher
passes it by default:

```bash
python -m repolens.mcp --repo /path/to/repository
```

Arguments: `name` (symbol or dotted module path, e.g. `charge_card`,
`models.Cart`), `max_depth` (default 1). The response is a JSON-safe dict:
`status`, `name`, `max_depth`, `node_count`, `nodes` (each with
`file`/`name`/`kind`/`parent_class`), `caller_count`/`callers`,
`callee_count`/`callees`. Unknown symbols return a safe error message.
The `analyze_impact` response now also reports
`direct_caller`/`indirect_caller`/`direct_callee`/`indirect_callee`
relationships and `confidence` per item. Initialization stays lazy, and the
reference/symbol indexes are shared with the engine and analyzer, so the graph
build is warm.

---

## Limitations

- The graph resolves names using the repository's own symbols and imports;
  `exec`/`eval`-driven code generation, package `__getattr__`, and dynamic
  import tricks are invisible and reported as unresolved rather than guessed.
- Calls that depend on runtime values (untyped receivers, `globals().get(..)`,
  monkey-patching) are *known unknowns* by design — see `unresolved_references`.
- `from . import x` at the *file-import edge* level points at the parent
  package; the submodule binding used for call resolution is tracked
  separately (both are deterministic and tested).
- This is a lightweight, offline call graph for grounding and blast-radius
  triage — not a substitute for a full language-server analysis when
  dynamically-computed calls matter.