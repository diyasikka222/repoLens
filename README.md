# RepoLens

RepoLens is an intelligent codebase context engine for AI coding agents. It analyzes a repository, understands its structure and dependencies, and retrieves the most relevant code for a developer query.

## The Problem

AI coding agents are only as good as the context they receive. On real repositories, they face two recurring failures:

- **Too much context**: Naive approaches dump large portions of the codebase into the context window, wasting tokens and burying the signal in noise.
- **Wrong context**: Simple keyword or file-name matching misses the code that actually matters — callers, dependents, types, and configuration that give a snippet its meaning.

There is no lightweight, agent-friendly layer that sits between a raw repository and an AI agent and answers: *"given this query, which parts of this codebase are relevant, and why?"*

## What RepoLens Does

RepoLens builds a structured understanding of a repository — its layout, symbols, and dependency relationships — and uses that understanding to retrieve precise, relevant code context on demand.

## Long-Term Architecture

At a high level, RepoLens will consist of three layers:

1. **Analysis Engine** — Scans a repository to build a structural map: file tree, language breakdown, symbol definitions, and an inter-module dependency graph.
2. **Retrieval Layer** — Given a natural-language or symbolic query, ranks and returns the most relevant code regions using the structural map, so agents get focused context instead of whole-file dumps.
3. **Interfaces** — The engine will be exposed through:
   - a **CLI** for direct developer use,
   - an **MCP server** so AI coding agents can consume RepoLens as a tool,
   - a **web application** for interactive exploration.

> **Status:** Local semantic search and hybrid retrieval are functional. See the sections below for setup and usage.

## Embedding Providers

RepoLens supports multiple embedding providers through a common `EmbeddingProvider` interface:

| Provider | API Key Required | Cost | Offline After Setup | Default Model |
|---|---|---|---|---|
| `FakeEmbeddingProvider` | No | Free | Yes | Hash-based (test double) |
| `LocalEmbeddingProvider` | No | Free | Yes (after first download) | `BAAI/bge-small-en-v1.5` |
| `OpenAIEmbeddingProvider` | Yes | Paid | No | Any OpenAI-compatible model |

### Local Embeddings (Recommended for Development)

Local embeddings use [FastEmbed](https://github.com/qdrant/fastembed) to run ONNX models entirely on-device. No API key is required.

**Setup:**

```bash
pip install -e ".[dev]"
```

**First run downloads the model:**

The default model `BAAI/bge-small-en-v1.5` (~130 MB) is downloaded from Hugging Face on first use. After download, embeddings run locally with no network access.

**Override the model:**

```bash
export REPOLENS_LOCAL_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
```

**Run the local benchmark:**

```bash
python benchmarks/evaluate_local_embeddings.py
```

## Real-World Repository Retrieval Benchmark

In addition to the small synthetic dataset, RepoLens ships a benchmark that
evaluates retrieval against a **real, pinned open-source repository**: the
[`Textualize/rich`](https://github.com/Textualize/rich) terminal-rendering
library. It exercises the four retrieval strategies (lexical, local semantic,
weighted hybrid, RRF) against ~20 manually curated developer queries, so we
can see whether semantic/hybrid retrieval actually helps on realistic code.

### Repository

| Field       | Value |
|---|---|
| Repository  | `Textualize/rich` |
| URL         | https://github.com/Textualize/rich |
| Ref / tag   | `v14.3.4` (release) |
| Commit      | `ee8378c3bbbd7c75abc2f55c6c19e83b218ae81d` |
| Python files (scanned) | 213 (whole repo; 100 in the `rich/` package) |

The benchmark is pinned to the release tag (not a moving branch), so results
are reproducible.

### How to run

```bash
python -m benchmarks.real_repo
```

The first run downloads the pinned GitHub tarball into the gitignored
`.benchmark_data/` directory and downloads the ONNX embedding model
(`BAAI/bge-small-en-v1.5`); subsequent runs are offline and reuse both.

Optional flags:

```bash
# Exploratory weight sweep (clearly labelled; does NOT change defaults)
python -m benchmarks.real_repo --weight-sweep

# Override where the repository is stored
python -m benchmarks.real_repo --repo-dir /path/to/rich
```

If a prerequisite is missing (e.g. `fastembed` not installed), the command
reports it clearly and exits. The benchmark never commits the external
repository and does not require OpenAI.

### How the queries and ground truth were built

- **Queries** (`benchmarks/real_repo/queries.json`) are realistic developer
  questions phrased as natural language — for example *"Where is the progress
  bar rendering implemented?"* or *"How does the console markup tag parser
  work?"* — rather than exact function-name lookups.
- **Ground truth** was written by hand, by reading the source of the pinned
  release to identify the repository-relative files that implement each
  concern. It was **not** derived from search output.

### Metrics

Each strategy reports `Precision@5`, `Recall@5`, and `MRR`, computed by the
existing [`repolens/evaluation.py`](repolens/evaluation.py) framework.

### Latest results (Rich v14.3.4)

| Strategy | Precision@5 | Recall@5 | MRR |
|---|---|---|---|
| Lexical | 0.1900 | 0.4958 | 0.6350 |
| Local Semantic | 0.2800 | 0.6333 | 0.6250 |
| Weighted Hybrid (0.5/0.5) | 0.2900 | 0.7208 | 0.7033 |
| RRF | 0.3100 | 0.7542 | 0.7958 |

On this realistic repository, lexical search alone has low precision because
exact token matching surfaces many files across a large codebase. Both
semantic and hybrid retrieval recover recall, and the RRF hybrid raises
precision, recall **and** MRR above both single-strategy baselines.

### Limitations

- `dev`/`examples`/test files in the repo are scanned like any production file
  (they are part of the real repository), which adds noise.
- Evaluation is file-level only; it does not measure retrieval of specific
  symbols or regions within a file.
- Ground truth is a manual, small set (20 queries) authored by one person and
  is specific to this pinned release.
- Embedding quality reflects the default local model (`bge-small-en-v1.5`) and
  per-file document representation; results are not directly comparable to the
  synthetic benchmark, which has different ground truth.

## Dependency-Aware Context Engine

RepoLens goes beyond retrieving relevant files. The **Context Engine**
(`repolens.context`) determines the smallest useful set of repository context
an AI coding agent needs to understand a task:

```
User/Agent Query
    ↓
Retrieval
    ↓
Candidate Files
    ↓
Dependency Expansion
    ↓
Context Ranking
    ↓
Context Budget
    ↓
Final Context Package
```

It composes the existing retrieval and dependency-graph components; it does
not re-implement or modify them. No agents or MCP are involved in this
milestone.

### Usage

```python
from repolens.context import ContextEngine, ContextBudget, DependencyExpansionConfig

# Retrieval uses the existing RRF hybrid (default weights, RRF k=60).
engine = ContextEngine(
    "path/to/repo",
    budget=ContextBudget(max_tokens=8000),
    dependency=DependencyExpansionConfig(depth=1),
)

package = engine.build_context("Where is authentication handled?")
print(package.to_json())          # serializable package
print(engine.render(package))     # deterministic text for an agent
```

### Configuration

- **Retrieval** (`RetrievalConfig`): chooses the existing strategy (`rrf`,
  `weighted`, `lexical`, `semantic`). Defaults match production (0.5/0.5
  weights, RRF k=60) and are not changed. A pre-built `Searcher` may be
  injected instead.
- **Dependency expansion** (`DependencyExpansionConfig`): `depth` is the number
  of graph hops (0 = retrieved only, 1 = plus direct dependencies/dependents,
  2 = one more hop); `include_dependencies` / `include_dependents` toggle
  forward and reverse edges. Traversal is breadth-first, deterministic, and
  never revisits a file.
- **Budget** (`ContextBudget`): a maximum in *estimated* tokens. `None` =
  unlimited.

### Ranking policy

Deterministic and explainable (no learned ranking, no LLM):

1. Primary (directly retrieved) files rank first: by retrieval rank, then
   retrieval score, then path.
2. Dependency-expanded files: by graph distance (closer first), then
   relationship strength (dependents/callers before dependencies at equal
   distance), then path.

### Token budget & rendering

Token counts use a documented deterministic approximation —
`max(1, ceil(len(text) / 4))` — for budgeting only. This **approximates**
tokens and is explicitly distinct from actual model tokenizer counts. The
package payload is JSON-serializable and `render_context` emits a stable text
representation for an agent.

### Scope note

An engine can be built that uses RRF directly, but the design keeps retrieval
behind the generic `Searcher` protocol, so any existing strategy composes
cleanly and no retrieval code is duplicated.

## Context Firewall

Before a `ContextPackage` is handed to an AI agent, RepoLens inspects it for
potentially sensitive information. The **Context Firewall**
(`repolens.context.firewall`) sits between the context engine and the agent:

```
ContextEngine
    ↓
ContextPackage
    ↓
ContextFirewall
    ↓
SafeContextPackage
```

### Why the firewall exists

Retrieval selects relevant files, but relevance does not guarantee safety. A
`settings.py` might be highly relevant *and* contain an API key. The firewall
is a deterministic, explainable, LLM-independent layer that classifies each
candidate and produces a `SafeContextPackage` — the same metadata you trust,
with only safe content.

### The decision model

Three explicit decisions:

- **ALLOW** — expose the file as-is.
- **REDACT** — the file is relevant, but sensitive portions are replaced
  before exposure.
- **BLOCK** — do not expose the file to the agent at all.

The firewall does **not** block every file that mentions `token`, `password`,
`secret`, or `auth`. A function like `def refresh_token(token): ...` remains
allowed unless there is real evidence of a secret value.

### Path-based detection

Well-known secret-file names and extensions are BLOCKed regardless of content:

- exact filenames such as `.env`, `id_rsa`, `id_dsa`, `id_ecdsa`,
  `id_ed25519`;
- extensions such as `.pem`, `.key`, `.p12`, `.pfx`;
- `secret.*` / `secrets.*` config/data files (`.json`, `.yaml`, `.env`, etc.) —
  *not* ordinary source modules like `secrets.py`.

### Content-based detection

Deterministic, high-confidence patterns for common secret material:

- OpenAI-style API keys (`sk-...`);
- AWS access key IDs (`AKIA...`);
- GitHub personal-access / app tokens;
- private-key blocks (`-----BEGIN ... PRIVATE KEY-----`);
- database URLs with embedded credentials;
- bearer tokens;
- generic high-entropy secret-like assignments.

The philosophy is **high precision > high recall**. False positives are
dangerous because they can strip legitimate code from agent context, so the
firewall only flags signals strong enough to be confident about. It does not
attempt to detect every possible secret.

Findings never contain the matched secret value. A finding is always safe to
send to an agent:

```json
{
  "path": "config.py",
  "line": 42,
  "type": "api_key",
  "severity": "high",
  "decision": "redact",
  "reason": "Potential API credential detected"
}
```

### Fail-closed behavior

If a file cannot be inspected safely (e.g. an unexpected scanning error), the
firewall **fails closed**: the file is BLOCKed rather than silently exposed,
with a safe diagnostic reason and never the file's contents or secrets in the
exception path.

### Usage

```python
from repolens.context import (
    ContextEngine, ContextBudget, ContextFirewall,
)

engine = ContextEngine("path/to/repo", budget=ContextBudget(max_tokens=8000))
firewall = ContextFirewall()

package = engine.build_context("Where is authentication handled?")

result = firewall.inspect(package)      # FirewallResult
safe = firewall.safe_package(package, result)   # SafeContextPackage

print(safe.to_json())                    # secrets safe; metadata preserved
```

### Configuration

`FirewallConfig` controls the policy with secure defaults (security is **ON**
by default):

- `enabled` — master switch (when `False`, everything passes through).
- `blocked_filenames` / `blocked_extensions` — path rules.
- `content_detectors` — which content detectors are active (disable any).
- `redaction_placeholder` — the string replacing detected secrets
  (default `[REDACTED]`).
- `policy_version` — embedded in results for auditability.

### A note on guarantees

The firewall is a **defense-in-depth layer, not a guarantee** that all secrets
are detected. Secret-scanning is not perfect; new formats and obfuscations
appear constantly. RepoLens deliberately prioritizes precision to avoid
falsely dropping legitimate code. No external secret-scanning service, network
call, or LLM is used — it is fully offline and deterministic.

## MCP Server (`get_context`)

RepoLens exposes its context engine to AI coding agents through the **Model
Context Protocol (MCP)**. MCP is currently a **local stdio integration**; no
IDE-specific integration is bundled or claimed.

### What MCP does in RepoLens

The MCP layer is a **thin adapter**. It does not re-implement retrieval,
ranking, budgeting, or security. It simply connects an agent to the existing
public RepoLens APIs and returns firewall-cleared context:

```
Agent
  ↓
MCP (get_context)
  ↓
Context Firewall
  ↓
Context Engine
  ↓
Retrieval / Graph
```

The tool accepts an agent query (and optional `max_tokens` /
`dependency_depth`), builds a `ContextPackage`, passes it through the
`ContextFirewall`, and returns **only** the safe result. The raw
`ContextPackage` is never exposed.

### The `get_context` tool

`get_context(query, max_tokens=..., dependency_depth=...)`

- `query` *(required)* — the developer query. Must be a non-empty string.
- `max_tokens` *(optional)* — a positive context budget in estimated tokens.
- `dependency_depth` *(optional)* — a non-negative dependency graph depth.

The response is structured JSON including the query, selected files with
selection reasons, estimated token count, budget, firewall decisions, and a
rendered safe context.

### Launching the server

```bash
python -m repolens.mcp --repo /path/to/repository
```

The repository root is configured at launch time and is **not** supplied by
tool callers. The server rejects attempts to reach files outside the
configured root; `get_context` is the only tool, and it cannot be asked to
read arbitrary paths.

Supported flags:

- `--repo <path>` *(required)* — the repository to index.
- `--default-max-tokens <n>` — default budget (default `8000`).
- `--default-dependency-depth <n>` — default graph depth (default `1`).
- `--use-local-embeddings` — use the on-device local embedding provider
  (requires a one-time model download on first use; no API key required).
- `--log-level <level>` — diagnostics verbosity on stderr (default
  `WARNING`).

To install the MCP extra: `pip install "repolens[mcp]"`.

### Connecting an MCP-compatible local client

Point your MCP-compatible client at a command such as:

```
python -m repolens.mcp --repo /path/to/repository
```

using the MCP **stdio** transport. Any client that supports launching a
stdio MCP server can connect. RepoLens does not currently provide HTTP/SSE
transports or an authentication layer.

> Note: RepoLens is tested with the official MCP Python SDK via stdio. No
> specific IDE integration has been tested or is claimed.

### How local embeddings work with MCP

By default the MCP server uses RepoLens' deterministic, fully-offline fake
embedding provider, so it works with **no API key and no model download**. To
enable on-device semantic retrieval, pass `--use-local-embeddings` (uses
`LocalEmbeddingProvider`, which downloads a model on first use and then runs
offline). No OpenAI key is ever required.

### Security boundary

MCP is treated as an **untrusted caller boundary**. A caller can request
context but cannot:

- choose arbitrary filesystem paths;
- bypass the context firewall;
- request blocked files directly;
- disable security policy;
- retrieve raw repository files;
- access environment variables, API keys, or files outside the configured
  repository.

Errors returned to the agent are concise and safe — no stack traces,
environment variables, or secret values. Diagnostics go to stderr via the
standard `logging` module; MCP communicates over stdout, so nothing else is
written to stdout while the server runs.

### Current limitations

- stdio transport only (no HTTP/SSE/authentication).
- Single repository per server.
- Single tool (`get_context`); no MCP resources or prompts yet.
- `max_tokens` / `dependency_depth` overrides build a purpose-configured
  engine for that request.
- The server is a defense-in-depth layer, not a guarantee that all secrets
  are detected.

### Example tool request

```json
{ "query": "Where is authentication handled?", "max_tokens": 8000 }
```

### Example response

```json
{
  "status": "ok",
  "query": "Where is authentication handled?",
  "budget": { "max_tokens": 8000 },
  "total_estimated_tokens": 171,
  "selected_files": [
    {
      "path": "auth/passwords.py",
      "role": "primary",
      "decision": "allow",
      "estimated_tokens": 60,
      "selection_reason": "retrieved as primary result at rank 1"
    }
  ],
  "blocked_files": [],
  "firewall": {
    "enabled": true,
    "policy_version": "1.0.0",
    "findings": []
  },
  "rendered_safe_context": "# RepoLens Safe Context\n..."
}
```

### OpenAI Embeddings (Optional)

Requires a valid API key and available credits.

```bash
export REPOLENS_EMBEDDING_API_KEY="sk-..."
export REPOLENS_EMBEDDING_MODEL="text-embedding-3-small"
python benchmarks/evaluate_real_embeddings.py
```

## Running Tests

```bash
# Run all unit tests (no network, no model download)
python -m pytest -v

# Run integration tests only (requires model download)
python -m pytest -m integration -v
```

## Project Structure

```
repolens/
  embeddings.py          # EmbeddingProvider interface + FakeEmbeddingProvider + OpenAIEmbeddingProvider
  local_embeddings.py    # LocalEmbeddingProvider (FastEmbed)
  semantic_search.py     # SemanticSearcher (vector similarity ranking)
  retrieval.py           # HybridSearcher (lexical + semantic)
  search.py              # CodeSearcher (lexical)
  scanner.py             # Repository file discovery
  parser.py              # Python AST parsing
  index.py               # Symbol index
  graph.py               # Dependency graph
  evaluation.py          # Retrieval quality evaluation
  context/               # Dependency-aware context engine (Milestone 12) + firewall (13)
    __init__.py          #   public API
    candidate.py         #   ContextCandidate / ExcludedCandidate / CandidateRole
    config.py            #   RetrievalConfig / DependencyExpansionConfig / ContextBudget
    engine.py            #   ContextEngine (build_context)
    expansion.py         #   dependency expansion over the graph
    ranking.py           #   deterministic context ranking
    budget.py            #   token-budget selection
    package.py           #   ContextPackage + JSON serialization
    render.py            #   render_context (agent text)
    tokens.py            #   estimate_tokens (deterministic approximation)
    firewall/            #   Context firewall (Milestone 13)
      __init__.py        #     public firewall API
      config.py          #     FirewallConfig (policy, safe defaults)
      decision.py        #     FirewallDecision (ALLOW/REDACT/BLOCK) + Severity
      finding.py         #     Finding (safe, no secret values)
      result.py          #     FirewallResult (structured inspection result)
      path_rules.py      #     path-based BLOCK rules
      content_detectors.py  # high-precision content detectors + redaction
      firewall.py        #     ContextFirewall (inspect / safe_package)
      safe_package.py    #     SafeContextPackage + SafeContextCandidate
      render.py          #     render_safe_context
    mcp/                 #   MCP server / tool adapter (Milestone 14)
      __init__.py        #     public MCP API
      __main__.py        #     python -m repolens.mcp
      launcher.py        #     minimal --repo CLI launcher
      server.py          #     build_mcp_server (registers get_context)
      tool.py            #     get_context handler + input validation
      deps.py            #     engine/firewall wiring + repo-root validation
      errors.py          #     safe MCP error types
benchmarks/
  evaluate_local_embeddings.py   # Local embedding evaluation
  evaluate_real_embeddings.py    # OpenAI embedding evaluation
  real_repo/                     # Real-world repository retrieval benchmark
    __main__.py                  #   python -m benchmarks.real_repo
    config.py                    #   pinned repo metadata / constants
    dataset.py                   #   loads & validates the curated queries
    runner.py                    #   download, evaluate, report
    queries.json                 #   20 manually curated developer queries
tests/
  test_local_embeddings.py       # LocalEmbeddingProvider unit tests
  test_real_repo_benchmark.py    # Benchmark config/dataset tests (offline)
  test_context_engine.py         # Context-engine tests (offline)
  test_context_firewall.py       # Firewall security tests (offline)
  test_mcp.py                    # MCP server tests (offline)
  ...
```
