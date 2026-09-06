# Repository Architecture Graph (Milestone 23)

This document describes RepoLens' **repository-level architecture graph**
(`repolens/architecture.py`): a deterministic model of a repository's physical
*files*, logical Python *modules*, and *packages* (directories holding Python
files), connected by containment and import-dependency edges.

M23 is *additive* and deliberately reuses existing data. The architecture
graph is a **pure projection** of the incremental repository index and the
existing dependency graph — no file is re-parsed, no new parser is
introduced, and no MCP / context-engine / impact-analysis behavior changes.

---

## What it answers

- **"What is in this package?"** — `files_in_package`, `modules_in_package`,
  `contains` / `contained_by`.
- **"Which packages depend on which?"** — `package_dependencies` (aggregated
  from module edges).
- **"What does this module/file depend on?"** — `dependencies_of`,
  `module_dependencies`.
- **"Who depends on me?"** — `dependents_of`, `transitive_dependents`.
- **"How big / how coupled is this?"** — `stats()` and bounded transitive
  traversal with `max_depth`.

```bash
python -c "
from repolens.architecture import ArchitectureGraphBuilder

root = '.'
graph = ArchitectureGraphBuilder(root).build()
print(graph.stats().as_dict())
print([p.id for p in graph.packages()])
print([d.id for d in graph.package_dependencies('repolens')])
"
```

---

## Guarantees

- **Deterministic.** Stable node IDs, sorted edge ordering, and stable
  JSON serialization (`serialize()`). Same repository → same graph, verified
  by tests and `benchmarks/verify_architecture.py`.
- **No duplicates.** Node and edge identities are keyed; re-adding is a no-op.
- **Repository-relative.** Files and packages use repo-relative paths
  (`app/api/routes.py`, `app/api`); modules use dotted names (`app.api.routes`).
- **Nested packages.** Any directory holding Python files, including the
  implicit repository root package (`.`) and deeply nested packages.
- **Safe.** Standard-library / third-party / unresolvable imports never
  produce nodes or edges and never crash.
- **Bounded.** Every transitive query requires `max_depth` and is additionally
  capped by `ArchitectureConfig.max_transitive_nodes` (5000). `max_depth=0`
  returns an empty list.
- **Incremental.** Pass an already-built `RepositoryIndex` (and optionally a
  `DependencyGraph`) and a build performs zero scans and zero parses. Because
  the graph is re-derived from the current index snapshot, deleted or modified
  files automatically update nodes and remove stale edges.

---

## Graph model

Three node kinds (`ArchitectureNodeKind`) — a composite identity
`(kind, id)` keeps a package (`app/api`) distinct from its package module
(`app.api`) even when ids collide.

| Node kind | id | Example |
| --- | --- | --- |
| `FILE` | repo-relative file path | `app/api/routes.py` |
| `MODULE` | dotted module name | `app.api.routes`; `app` for `app/__init__.py` |
| `PACKAGE` | repo-relative directory | `app/api`; `.` for the repository root |

Two canonical edge types (`ArchitectureRelationship`) plus their derived
reverse views:

| Relationship | Meaning |
| --- | --- |
| `contains` / `contained_by` | package → file, package → package (nested), file → module |
| `depends_on` / `depended_on_by` | module → module (from the dependency graph); package → package (aggregated) |

### Key queries

| Query | Returns |
| --- | --- |
| `get_file_node(path)` / `get_module_node(name)` / `get_package_node(path)` | node or `None` |
| `files()`, `modules()`, `packages()`, `get_nodes()`, `get_edges()` | sorted node/edge lists |
| `dependencies_of(x)` / `dependents_of(x)` | direct dependency edges (FILE delegates to its module) |
| `package_dependencies(pkg)` / `module_dependencies(mod)` | strict package/module views |
| `files_in_package(pkg)` / `modules_in_package(pkg)` | direct contents |
| `contains(x)` / `contained_by(x)` | containment neighbors |
| `transitive_dependencies(x, max_depth=…)` / `transitive_dependents(x, …)` | bounded BFS, deduplicated |
| `get_edges_by_relationship(rel)` | canonical or derived-reverse edges |
| `stats()` | `nodes`, `files`, `modules`, `packages`, `edges`, `contains_edges`, `depends_on_edges` |

### Identifier resolution

A bare string is resolved as **file path → package path → module name** (so
`app` means the package `app`). Use `get_module_node("app.api")` to address a
package module explicitly. `ArchitectureGraphBuilder(root, *, index=None,
graph=None, config=None)`:

- `index is None` → cold build: `IncrementalIndexBuilder(root,
  persist=False).build()`.
- `graph is None` → build the dependency graph against `index`.
- both supplied → an entirely warm, incremental build.

---

## Dependency construction

Module edges come straight from the existing `DependencyGraph`: every
repository-resolved import becomes a `depends_on` module edge (self-module
edges dropped). Package edges are the *aggregation* of module edges — for each
module dependency, the deepest containing package of each side is paired and a
package-level edge is added (same-package pairs dropped, duplicates removed).
Because aggregation walks that mapping, module and package dependency views
stay consistent with the file-level import graph.

---

## Incremental behavior

The architecture graph is never patched in place; it is **re-derived from the
current index snapshot**. That makes mutation handling trivial and safe:

- **Modified file** → next build includes its new imports (new edges) and
  drops removed ones.
- **Deleted file** → its file/module nodes and every edge referencing them
  vanish; a package that still has Python files keeps its node.
- **Deleted directory** → the package node (and transitively its subtree)
  disappears.
- **Warm rebuilds** → `ArchitectureGraphBuilder(root, index=warm_index)` does
  zero scanning/parsing; the benchmark verifies `files_parsed == 0` on the warm
  index.

---

## Verification

- **Tests** — `tests/test_architecture.py` (27 tests): node creation, package
  and module hierarchies, file/module/package relationships, module & package
  dependency edges, reverse dependencies, relative imports, external +
  unresolvable imports, duplicate prevention, deterministic ordering and
  serialization, bounded traversal, incremental modification, file deletion,
  empty and single-file repositories.
- **Fixture** — `tests/fixtures/architecture_repository/`: layered `app` tree
  (`api` → `services`/`repositories`/`models`) with relative imports, an
  external import (`json`), and a same-package submodule.
- **Benchmark** — `benchmarks/verify_architecture.py`: asserts the exact
  synthetic structure, the incremental modification/deletion story, and runs a
  cold + warm build against RepoLens itself (warm: zero parses, deterministic).

---

## Limitations

- Package membership is **directory-based**: any directory holding Python
  files is a package, even without an `__init__.py`.
- A root `x.py` next to a package `x/` both map to the module `x`; the graph
  keeps them as a single module node (the dependency graph has the same
  property for import resolution).
- A file that is deleted from disk always disappears — the graph reflects the
  *current* index snapshot and holds no historical state.
- Package-level `depends_on` edges are a pure projection of module-level
  edges; they report which packages are reachable, not per-import multiplicity
  (duplicates are intentionally collapsed).