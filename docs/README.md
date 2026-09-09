# RepoLens Documentation

This index points to the documentation that describes how RepoLens works, what
it can do, and how it is verified. The [top-level README](../README.md) is the
canonical entry point: it answers *what*, *why*, *how to install*, and *how to
run*. These documents go deeper into specific subsystems and the evidence that
they work.

## Quick orientation

| Topic | Document | Covers |
| --- | --- | --- |
| Architecture graph | [architecture.md](architecture.md) | The repository-level model: files, modules, packages, dependency edges. |
| Architecture-aware retrieval | [architecture-retrieval.md](architecture-retrieval.md) | Turning a query into architecture signals and expanding them bounded. |
| Architecture MCP tools | [architecture-mcp.md](architecture-mcp.md) | The four architecture tools exposed over MCP. |
| Call graph & references | [call-graph.md](call-graph.md) | Offline cross-file reference extraction and the static call graph. |
| Change impact analysis | [impact-analysis.md](impact-analysis.md) | Deterministic blast-radius analysis for a change target. |
| Change planning | [change-planning.md](change-planning.md) | The deterministic change-plan engine (which files, what order, how risky). |
| Change-aware context | [change-context.md](change-context.md) | Folding a change plan into ranked, budgeted, firewall-cleared context. |
| MCP surface | [mcp-tools.md](mcp-tools.md) | Every MCP tool, its parameters, output, and when to use it. |
| Production validation | [production-validation.md](production-validation.md) | Correctness of the pipeline against a real repository (M20). |
| Performance & scalability | [performance.md](performance.md) | What `performance_scalability.py` measures and its honest limits. |
| Reliability | [reliability.md](reliability.md) | Failure injection: corruption, provider faults, bad input, concurrency. |
| OpenCode integration | [opencode-e2e.md](opencode-e2e.md) | Setting up the MCP server in OpenCode and the end-to-end workflow benchmark. |
| Release checklist | [release-checklist.md](release-checklist.md) | Pre-release verification checklist for v1.0. |

## Suggested reading order for a new contributor

1. [Top-level README](../README.md) — install, run, quick start.
2. [production-validation.md](production-validation.md) — how the pipeline
   behaves on a real repository.
3. [architecture.md](architecture.md) and
   [architecture-retrieval.md](architecture-retrieval.md) — the structural
   layer that makes retrieval precise.
4. [impact-analysis.md](impact-analysis.md) and
   [call-graph.md](call-graph.md) — blast radius and static call evidence.
5. [change-planning.md](change-planning.md) and
   [change-context.md](change-context.md) — planning and change-aware context.
6. [mcp-tools.md](mcp-tools.md) — the agent-facing surface.
7. [opencode-e2e.md](opencode-e2e.md) — how an agent actually uses it.
8. [performance.md](performance.md) and [reliability.md](reliability.md) —
   scaling and failure behavior.

## Relation to milestone history

The milestone (P25.x) history is preserved inside individual documents where it
helps explain *why* a layer exists, but user-facing guidance is independent of
it. The current feature set is what matters for v1.0.