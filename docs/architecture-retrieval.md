# Architecture-Aware Context Retrieval (Milestone 23.2)

This document describes RepoLens' **architecture-aware context retrieval**
(`repolens/architecture_retrieval.py` + `repolens/subsystems.py`): a layer
that reads a developer query, turns it into *architecture signals*, and
expands them — deterministically and *boundedly* — across the architecture
graph so a context package carries not just the matched files but their
structural neighbours (dependencies, dependents, package peers, and
same-subsystem modules).

M23.2 is *additive* and **opt-in**. The `ContextEngine` behaves byte-for-byte
as before unless a caller passes `architecture=ArchitectureRetrievalConfig()`.
No file is re-parsed for the architecture itself: signals, expansion, and
subsystem discovery are pure functions of the M23 `ArchitectureGraph`, which
remains a projection of the existing incremental index + dependency graph.

```bash
python -c "
from pathlib import Path
from repolens.context import ContextEngine, ContextBudget
from repolens.architecture_retrieval import ArchitectureRetrievalConfig
from repolens.search import CodeSearcher

root = '.'
engine = ContextEngine(
    root,
    searcher=CodeSearcher(root),
    architecture=ArchitectureRetrievalConfig(),
    budget=ContextBudget(max_tokens=8000),
)
pkg = engine.build_context('how does the checkout flow work')
print(len(pkg.selected_files), 'files selected')

print(engine.discover_subsystems())           # deterministic subsystem ids
cands = engine.architecture_candidates('checkout')
print([c.node.id for c in cands[:5]])
print(engine.explain_architecture_match('checkout', 'store.services.checkout'))
"
```

---

## Signature

```python
extract_architecture_signals(query, graph, *, symbol_matches=(), config=None)
    -> list[ArchitectureSignal]

architecture_candidates(query, graph, *, subsystems=None, config=None)
    -> list[ArchitectureCandidate]

class ArchitectureRetrievalConfig:
    enabled: bool = True
    max_package_candidates: int = 6
    max_module_candidates: int = 24
    max_neighbor_depth: int = 1
    max_expanded_nodes: int = 50
```

`architecture_candidates` accepts an optional precomputed `subsystems`
(discovery result) to avoid recomputing subsystem membership.
`repolens.architecture_retrieval.explain_architecture_match(query, graph,
node_or_id, *, subsystems=None)` returns a dict describing why a node was
matched, or `None` (used by the engine's public explainer).

---

## Signal extraction

A query is decomposed into *architecture signals* — exact node references plus
tagged leaf meanings. Signals carry a reason, a node, and a direction:

1. **Explicit references.** Dotted module names (`store.services.checkout`),
   directory paths (`store/repositories`), and file paths
   (`store/services/checkout.py`) are resolved as exact file→package→module
   node lookups. A plain word that is merely a component of a longer
   dotted/path mention is consumed so it cannot re-fire as a broader signal.
2. **Term tagging.** Words that are not consumed as explicit references are
   matched against node *leaf components*: `checkout` → modules/packages/files
   whose last path/camel component is `checkout` (role terms tolerate a
   trailing `s`). Dotted/path tokens also carry structural hits.
3. **Symbol links.** When the intent layer has already resolved symbols, a
   `SymbolMatch` links the symbol's file to its defining module, producing an
   *architecture: module containing matched symbol* signal so a symbol hit
   reaches its architectural context.

Extraction never uses project-specific name lists and never requires an LLM or
embedding. Output is capped (`MAX_SIGNALS = 24`, at most 4 token-tagged
signals) and deterministic.

---

## Expansion

For every matched module/package/file the graph is walked *boundedly* by rank:

| Rank | Candidates |
| --- | --- |
| 0 | direct architecture matches (file / module / package / symbol) |
| 1 | dependencies and dependents of matched modules, modules in a matched package, package-level dependency/dependent packages, neighbours within `max_neighbor_depth` |
| 2 | remaining same-subsystem modules (architecture proximity) |

Expansion caps (`max_package_candidates`, `max_module_candidates`,
`max_expanded_nodes`, `max_neighbor_depth`) keep a connected repository from
collapsing the whole graph into the context. Counts at each tier are truncated
deterministically. A candidate is emitted exactly once (keyed by node).

Each `ArchitectureCandidate` carries: `node`, `path`, `reason`, `rank`,
`direction`, `package`, `subsystem`. Directions are `matched`, `dependency`,
`dependent`, `package`, `neighbor`, `symbol`, `subsystem`.

---

## Subsystems

`repolens.subsystems.discover_subsystems(graph, *, max_subsystems=None)`
partitions the repository into deterministic *subsystems* — the packages,
modules, and files hanging off each top-level package (the repository root
package is the `root` subsystem). Cross-subsystem module dependency edges are
projected so every subsystem reports:

- `packages` / `modules` / `files` — members, sorted;
- `entry_modules` — modules imported from other subsystems;
- `dependencies` / `dependents` — neighbouring subsystem ids;
- `layers` — intra-subsystem packages stratified by dependency depth
  (foundations at layer 0). Relaxation is capped at one pass per package, which
  keeps acyclic layering exact and terminates deterministically even under
  pathological package cycles;
- `stats()` — deterministic counters.

`subsystem_of(graph, node_or_id, subsystems=None)` resolves a node to its
subsystem. Subsystems are discovery-only metadata: candidates reference
`subsystem`/`package` for explainability, and the rank-2 proximity tier reads
matched-package membership.

---

## Why it does not change default behavior

The `ContextEngine` accepts `architecture: ArchitectureRetrievalConfig | None
= None`:

- **`None` (default)** — nothing changes; candidates, ranking, budget,
  serialization, firewall, and diagnostics are identical to M23.1.
- **Enabled** — a single index build is materialized once at construction when
  no `index` was supplied; `DependencyGraph`, `SymbolIndex`, and the
  `ArchitectureGraph` all share it (one scan/parse for the whole pipeline).
- Architecture candidates join the candidate list as dependency-role
  candidates carrying `inclusion_reason="architecture"`, `architecture_rank`,
  and `architecture_metadata` (node kind/id, package, subsystem, direction).
  The non-primary ranking key becomes
  `(1, min(architecture_rank, 2) if set else 3, distance, role_order, path)` so
  architecture tiers interleave deterministically with dependency candidates.
- Rank-0 architecture direct matches seed dependency expansion together with
  symbol paths (identical expansion when disabled).
- The engine exposes the retrieval surface explicitly for callers/introspection:
  `engine.architecture_candidates(query)`,
  `engine.discover_subsystems()`, and
  `engine.explain_architecture_match(query, node_id)`.

Because primary (search) hits are ranked ahead of all non-primary candidates, a
file that is both a search hit and an architecture match appears once, as a
primary — the architecture metadata is a superset, never a reduction.

---

## Verification

- **Tests**
  - `tests/test_architecture_retrieval.py` (30 tests): signal extraction for
    explicit module/package/file references, term tagging, plural tolerance,
    symbol→module links, caps and determinism; candidate rank tiers,
    dependency/dependent/neighbour/package/subsystem expansion, boundedness,
    deduplication, metadata, ordering, explainability, and empty/single-file
    repositories.
  - `tests/test_subsystems.py` (17 tests): discovery determinism, inventory,
    max-subsystems bound, root/single-subsystem/empty repos, dependency
    projection, entry points, layers, and `subsystem_of`.
  - `tests/test_context_engine.py` (architecture section): enabled vs disabled
    equality, architecture-only additions, budget preservation, the public
    query surface, and the rendered `Architecture:` line.
- **Fixture** — `tests/fixtures/architecture_retrieval_repository/`:
  multi-subsystem `app/store/billing/admin/web` tree plus a nested `tests/`
  package of fixture test files, with intra- and inter-subsystem imports.
- **Benchmark** — `benchmarks/verify_architecture_retrieval.py`: asserts the
  synthetic structure, then runs against RepoLens itself — cold + warm build
  (warm: zero parses), deterministic repeated retrieval, and a baseline-vs-aware
  `build_context` comparison (architecture-aware is always a superset of the
  files baseline selects).

## Limitations

- A query signal is only as good as the graph: modules with no edges expand to
  their package/subsystem only; a repository with a single top-level package
  yields a single subsystem plus whatever the root file set provides.
- Rank-2 same-subsystem candidates are proximity hints, not semantic matches.
- Subsystem layers under package cycles are still deterministic but bounded by
  the per-package-pass cap rather than a true topological depth.
- Architecture context never rewrites search primaries: if neither search nor
  architecture finds a file, it still is not included.