# MCP Tools

This document describes the complete, currently-exposed MCP surface of the
RepoLens server. It is the authoritative reference for what an agent can call;
per-tool deep dives live in [architecture-mcp.md](architecture-mcp.md),
[change-context.md](change-context.md), [impact-analysis.md](impact-analysis.md)
and [call-graph.md](call-graph.md).

The server is launched over **stdio**:

```bash
python -m repolens.mcp --repo /path/to/repository
```

The repository root is fixed at launch time and never supplied by tool callers.
All nine tools are registered by the default launcher. The core `get_context`
tool is always registered; the eight others are additive and registered when
their factories are wired in (the launcher wires all of them).

Every tool:

- validates its arguments strictly (unknown keys, wrong types, out-of-range
  ints and unsafe targets are rejected);
- is fully **offline and deterministic** in its output;
- returns only structured, JSON-safe data — never internal Python objects;
- on failure returns a safe error message with no stack trace.

> Schema note: this document describes the tools as implemented. It does not
> change any schema; tool parameter names, defaults, and caps are as wired in
> `repolens/mcp/server.py`.

## Tool reference

| Tool | Purpose | Deep dive |
| --- | --- | --- |
| `get_context` | Retrieve safe, budgeted context for a query. | this doc / `repolens/mcp/tool.py` |
| `analyze_impact` | Blast radius of changing a repository target. | [impact-analysis.md](impact-analysis.md) |
| `inspect_symbol` | Callers/callees of a symbol via the offline call graph. | [call-graph.md](call-graph.md) |
| `inspect_architecture` | Overview of one repository node (file/module/package). | [architecture-mcp.md](architecture-mcp.md) |
| `discover_subsystems` | Deterministic subsystem inventory of the repository. | [architecture-mcp.md](architecture-mcp.md) |
| `architecture_candidates` | Architecture-aware candidate generation for a query. | [architecture-mcp.md](architecture-mcp.md) |
| `explain_architecture_match` | Why a node was (or was not) architecturally relevant. | [architecture-mcp.md](architecture-mcp.md) |
| `change_plan` | Deterministic change/inspection plan for a change request. | [change-context.md](change-context.md) |
| `change_context` | Change-aware, firewall-cleared context for a change request. | [change-context.md](change-context.md) |

---

## `get_context`

**Purpose.** The primary retrieval tool. Given a developer query, return the
smallest useful package of repository context (selected files with reasons,
token estimate, budget, firewall decisions, and a rendered safe context).

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `query` | string | yes | Non-empty developer query. |
| `max_tokens` | integer | no | Positive context budget in *estimated* tokens. Default: server `--default-max-tokens` (8000). |
| `dependency_depth` | integer | no | Non-negative dependency-graph depth. Default: server `--default-dependency-depth` (1). |

**Output.** `status`, `query`, `budget`, `total_estimated_tokens`, `intent`,
`matched_symbols`, `selected_files` (each with `path`, `role`, `decision`,
`estimated_tokens`, `selection_reason`, `inclusion_reason`), `blocked_files`,
`firewall` (`enabled`, `policy_version`, `findings`), and
`rendered_safe_context`. Content is filtered by the context firewall before it
is returned.

**When an agent should use it.** As the default "find me the code relevant to
this" tool: understanding an unknown area, locating an implementation, or any
question about how a feature works. Use before the more specialised tools below.

---

## `analyze_impact`

**Purpose.** Analyze the blast radius of changing a repository target: resolve
the target, classify its risk (low/medium/high), and group affected files by
relationship (direct/indirect dependency, reverse dependency, test,
configuration, api consumer, direct/indirect caller, direct/indirect callee).

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `target` | string | yes | File path, dotted module, symbol, or `path/to/file.py::symbol`. |
| `max_depth` | integer | no | Non-negative reverse-traversal depth (default 4). |
| `limit` | integer | no | Positive cap on returned items. |

**Output.** `status`, `target`, `kind`, `target_path`, `symbol`, `module`,
`risk`, `max_depth_reached`, `summary`, `item_count`, `items`. Unknown or
ambiguous targets return a safe error.

**When an agent should use it.** Before making a change, to discover what else
the change could break; for investigation-style prompts ("what is affected if I
change X?").

---

## `inspect_symbol`

**Purpose.** Inspect the statically resolved call relationships of a symbol
using the offline call graph — who calls it and what it calls.

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `name` | string | yes | Bare symbol (`charge_card`) or dotted module path (`models.Cart`). |
| `max_depth` | integer | no | Transitive call depth (default 1). |

**Output.** `status`, `name`, `max_depth`, `node_count`, `nodes` (each with
`file`/`name`/`kind`/`parent_class`), `caller_count`/`callers`,
`callee_count`/`callees`. Unresolvable symbols return a safe error.

**When an agent should use it.** To trace callers and callees of a specific
function/method/class when the change touches that symbol.

---

## `inspect_architecture`

**Purpose.** Structured overview of a single repository node: its
package/module/file identities, subsystem, dependency/dependent lists, bounded
transitive neighborhood, and statistics.

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `target` | string | yes | Repo-relative path, dotted module, or package path (`store/services/checkout.py`, `store.services.checkout`, `store/services`). |
| `max_depth` | integer | no | Transitive neighborhood depth (default 2, capped at 6). |
| `include_dependencies` | boolean | no | Include dependency section (default true). |
| `include_dependents` | boolean | no | Include dependent section (default true). |
| `include_subsystems` | boolean | no | Include subsystem stats (default true). |

**Output.** `status`, `target`, resolved node identities, `subsystem`,
`dependencies`, `dependents`, `neighborhood`, `statistics`. Unknown targets
return a safe error.

**When an agent should use it.** To understand where a file/module/package sits
in the repository structure before planning a change around it.

---

## `discover_subsystems`

**Purpose.** List the deterministic architectural subsystems of the repository
(grouped by top-level package tree, no LLM).

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `max_subsystems` | integer | no | Cap on returned subsystems (capped at 500). |
| `include_dependencies` | boolean | no | Include subsystem dependency ids (default true). |
| `include_dependents` | boolean | no | Include subsystem dependent ids (default true). |
| `include_stats` | boolean | no | Include per-subsystem statistics (default true). |

**Output.** `status`, `subsystem_count`, `subsystems` (each with `id`,
`display_name`, `packages`, `modules`, `files`, `entry_modules`, optional
`statistics`, `dependencies`, `dependents`). Ordering is deterministic.

**When an agent should use it.** To get a high-level map of the repository's
major subsystems before deciding where a change belongs.

---

## `architecture_candidates`

**Purpose.** Generate architecture-aware candidates for a query independent of
full context retrieval: direct matches plus bounded dependency/dependent,
neighbor, package, and subsystem-proximity expansions.

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `query` | string | yes | Free-text query. |
| `limit` | integer | no | Candidate cap (default 20, capped at 200). |
| `max_depth` | integer | no | Neighbor expansion depth (default 1, capped at 6). |
| `architecture` | object | no | Optional retrieval caps: `max_package_candidates` (cap 100), `max_module_candidates` (cap 500), `max_neighbor_depth` (cap 6), `max_expanded_nodes` (cap 1000). |

**Output.** `status`, `query`, `candidate_count`, `candidates` (each with
`file`, `module`, `package`, `score`, `architecture_score`, `inclusion_reason`,
`architecture_node_type`, `architecture_node_id`, `subsystem`,
`dependency_direction`) and the query's matched `signals`. Identical inputs
produce identical ranked lists.

**When an agent should use it.** To generate structural candidates for a query
(a cousin of `get_context` for architecture/subsystem reasoning) without paying
the full context build.

---

## `explain_architecture_match`

**Purpose.** Explain why a specific node was (or was not) considered
architecturally relevant to a query.

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `query` | string | yes | Free-text query. |
| `target` | string | yes | Same safety rules as `inspect_architecture.target`. |

**Output.** When relevant: `status`, `query`, `target`, node identity,
`inclusion_reason`, `architecture_score`, `direction`, `package`, `subsystem`,
`subsystem_relationship`, `signals`, `matched_nodes`, bounded
`dependency_paths`. When not relevant: a structured
`is_architecturally_relevant: false` with an `explanation` (this is a normal
response, not an error).

**When an agent should use it.** To audit or understand a candidate/context
result — "why was this file pulled in (or not)?"

---

## `change_plan`

**Purpose.** Produce a deterministic change plan for a change request: request
analysis, ranked target candidates, primary target, affected files/symbols,
callers/callees, dependencies/dependents, architecture context, candidate
tests, a bounded inspection order, a risk tier, and a summary. Never modifies
the repository and contains no LLM.

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `request` | string | yes | Natural-language change request (e.g. "make checkout reject empty carts"). |
| `target` | string | no | Explicit file path, dotted module, package, or symbol. |
| `target_kind` | string | no | One of `file`, `module`, `package`, `symbol`. |
| `max_targets` | integer | no | Target-candidate cap (≤ 50). |
| `max_files` | integer | no | Affected-files cap (≤ 500). |
| `max_tests` | integer | no | Test cap (≤ 100). |
| `max_depth` | integer | no | Impact depth cap (≤ 6). |

**Output.** `status`, `request`, `analysis`, `primary_target`,
`target_candidates`, `affected_files`, `affected_symbols`, `callers`,
`callees`, `dependencies`, `dependents`, `architecture`, `tests`,
`inspection_order`, `risk`, `risk_factors`, `confidence`, `summary`,
`statistics` (deterministic counters), `diagnostics` (wall-clock run metadata),
`deterministic: true`.

**When an agent should use it.** As the first step of a change workflow: decide
which files to look at, in what order, and how risky the change is.

---

## `change_context`

**Purpose.** Build change-aware, firewall-cleared context: fold the change
plan's affected files into the existing ranking and budget pipeline as a final
inspection tier (never outranking direct query matches) and return safe
context.

**Parameters**

| Param | Type | Required | Notes |
| --- | --- | --- | --- |
| `request` | string | yes | Same change request as `change_plan`. |
| `target` / `target_kind` | string | no | Explicit target, as `change_plan`. |
| `max_tokens` | integer | no | Context budget in estimated tokens. |
| `max_targets` / `max_files` / `max_tests` / `max_depth` | integer | no | Plan bounds, as `change_plan`. |
| `include_tests` | boolean | no | Introduce test plan items (default true). |
| `include_dependencies` | boolean | no | Introduce dependency items (default true). |
| `include_callers` | boolean | no | Introduce caller items (default true). |
| `include_callees` | boolean | no | Introduce callee items (default true). |
| `include_architecture` | boolean | no | Introduce architecture items (default true). |

**Output.** `status`, effective `budget`, `total_estimated_tokens`, `intent`,
`matched_symbols`, `selected_files` (with per-file `explanation`), `blocked_files`,
`change_files` (surviving plan candidates), a compact `change_plan` summary,
explanations list, firewall findings, and `rendered_safe_context`. Only
firewall-approved content is returned.

**When an agent should use it.** For a change task, after `change_plan`: get
the actual context (with tests, callers, dependencies) to implement or inspect
the change safely.

---

## Registration and caps

The launcher wires every additive factory, so all nine tools are present by
default. Without a factory, only `get_context` is registered. The MCP layer
imposes deterministic caps to keep responses bounded on very large
repositories:

| Cap | Value |
| --- | --- |
| Architecture `max_depth` / change `max_depth` | 6 |
| `architecture_candidates` limit | 200 |
| `discover_subsystems` count | 500 |
| `change_plan` / `change_context` targets | 50 |
| `change_plan` / `change_context` files | 500 |
| `change_plan` / `change_context` tests | 100 |
| `inspect_architecture` neighborhood entries | 200 |
| Architecture config caps | 100 packages / 500 modules / 1000 expanded |

## Launch flags that affect the tools

| Flag | Effect |
| --- | --- |
| `--repo <path>` | Repository to index (required). |
| `--default-max-tokens <n>` | Default `get_context` / `change_context` budget (8000). |
| `--default-dependency-depth <n>` | Default `get_context` dependency depth (1). |
| `--use-local-embeddings` | Enable on-device semantic retrieval (one-time model download). |
| `--log-level <level>` | Diagnostics verbosity on stderr (default WARNING). |