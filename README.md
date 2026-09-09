# RepoLens

RepoLens is an intelligent codebase context engine for AI coding agents. It
analyzes a repository — layout, symbols, imports, call relationships, packages,
and subsystems — and serves that understanding to an AI agent as precise,
budgeted, security-filtered context through the **Model Context Protocol
(MCP)**.

It is designed to be used from OpenCode (and any MCP-compatible terminal
agent). RepoLens itself is **deterministic and offline**: no LLM is consulted,
no code is generated or edited, and the core pipeline never requires the
network.

## What problem does it solve?

AI coding agents are only as good as the context they receive. On real
repositories they face two recurring failures:

- **Too much context** — naive approaches dump large portions of the codebase
  into the context window, wasting tokens and burying the signal in noise.
- **Wrong context** — simple keyword or file-name matching misses the code that
  actually matters: callers, dependents, types, tests, and configuration that
  give a snippet its meaning.

RepoLens sits between a raw repository and an AI agent and answers: *"given
this query (or this planned change), which parts of this codebase are relevant,
and why?"*

## How does it work?

```
repository
  → discovery / indexing
  → parsing / symbol / reference analysis
  → retrieval (lexical / semantic / hybrid)
  → architecture & impact analysis
  → context & change planning
  → MCP
  → OpenCode (or any MCP stdio client)
```

1. **Discovery & indexing** — a scanner discovers the repository's Python files
   and an incremental, content-hashed index persists per-file analysis.
2. **Parsing / symbol / reference analysis** — files are parsed into a symbol
   index, a dependency graph, cross-file references, and a static call graph —
   all offline and conservative.
3. **Retrieval** — a query is served by lexical search, local semantic search
   (embeddings), and hybrid (RRF / weighted) strategies, plus architecture-aware
   candidate expansion.
4. **Architecture & impact analysis** — the repository is modeled as
   files/modules/packages with dependency edges; change-impact analysis computes
   a deterministic blast radius with a risk tier.
5. **Context & change planning** — the context engine assembles the smallest
   useful, budgeted package of context; the change-plan engine converts a change
   request into which-files-what-order-how-risky; change-aware context folds the
   plan into the retrieved context.
6. **MCP → OpenCode** — a security **firewall** inspects every context package
   before it reaches the agent, redacting or blocking secrets, and the result is
   served over stdio as MCP tools.

Every stage is deterministic: identical input yields identical output, with no
LLM and no re-parsing when a warm index exists.

## Major capabilities

- **Context retrieval** (`get_context`) — budgeted, ranked, firewall-cleared
  content for a natural-language query.
- **Architecture intelligence** — `discover_subsystems`,
  `inspect_architecture`, `architecture_candidates`,
  `explain_architecture_match`.
- **Change impact analysis** (`analyze_impact`) — blast radius and deterministic
  risk classification for a target.
- **Static call graph** (`inspect_symbol`) — offline callers/callees of a
  symbol.
- **Change planning** (`change_plan`) — a deterministic, explainable change
  plan: targets, affected files, candidate tests, inspection order, risk.
- **Change-aware context** (`change_context`) — the plan folded into the
  ranking and budget pipeline, behind the same firewall.
- **Context firewall** — deterministic, offline secret redaction/blocking
  before anything reaches an agent.
- **Deterministic evaluation** — offline benchmarks for retrieval quality,
  agent-task coverage, performance, and reliability (see below).
- **Incremental, persistent caching** — warm rebuilds parse zero files; caches
  are atomic, corruption-tolerant, and honor `REPOLENS_CACHE_DIR` /
  `REPOLENS_CACHE_DISABLED`.

RepoLens **never edits code**. Change planning tells an agent *where* and *in
what order* to act; it does not generate patches, run tests, or modify the
repository. Agent/LLM-based performance is **not** evaluated by RepoLens — its
evaluation is deterministic and offline.

## Installation

Requirements: Python **>= 3.11**.

```bash
# 1. Clone the repository
git clone https://github.com/diyasikka222/repoLens.git
cd repoLens

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install the package and development/test dependencies
pip install -e ".[dev]"
```

The `mcp` dependency is included in `dev`. To install only the MCP extra
(no test tools): `pip install "repolens[mcp]"`.

## Quick start

```bash
# 1. Run the test suite (offline, no model download)
python -m pytest -q

# 2. Start the MCP server for your repository.
#    Default: deterministic offline embeddings (no API key, no model download).
python -m repolens.mcp --repo /path/to/repository

# 3. To enable on-device semantic retrieval (one-time model download):
python -m repolens.mcp --repo /path/to/repository --use-local-embeddings
```

The only package CLI is the MCP launcher. There is **no** `python -m repolens`
or general-purpose CLI; RepoLens is consumed through its Python API, its MCP
server, and its benchmark/validation scripts.

### Using RepoLens from a Python script

```python
from repolens.context import ContextEngine, ContextBudget, ContextFirewall

engine = ContextEngine("path/to/repo", budget=ContextBudget(max_tokens=8000))
package = engine.build_context("Where is authentication handled?")
firewall = ContextFirewall()
safe = firewall.safe_package(package, firewall.inspect(package))
print(safe.to_json())          # firewall-cleared, serializable result
```

See [docs/production-validation.md](docs/production-validation.md) and the
docs index below for the full Python surface.

## Embedding configuration

| Provider | API key | Offline | Notes |
| --- | --- | --- | --- |
| `FakeEmbeddingProvider` (default) | No | Fully | Deterministic hashed bag-of-words; used by the MCP server unless overridden. |
| `LocalEmbeddingProvider` | No | After first download | FastEmbed ONNX on-device; default model `BAAI/bge-small-en-v1.5`. |
| `OpenAIEmbeddingProvider` | Yes | No | Optional; used by `evaluate_real_embeddings.py`. |

### Server selection

- The MCP server defaults to `FakeEmbeddingProvider` (zero setup). Pass
  `--use-local-embeddings` to enable on-device semantic retrieval:
  ```bash
  python -m repolens.mcp --repo /path/to/repo --use-local-embeddings
  ```
- Override the local model:
  ```bash
  export REPOLENS_LOCAL_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
  ```
- The optional OpenAI provider is driven by environment variables
  (`REPOLENS_EMBEDDING_API_KEY`, `REPOLENS_EMBEDDING_MODEL`,
  `REPOLENS_EMBEDDING_BASE_URL`, `REPOLENS_EMBEDDING_DIMENSIONS`) and is used
  only by the `evaluate_real_embeddings.py` benchmark — never by the MCP server.

### Embedding & index caches

| Setting | Effect |
| --- | --- |
| `REPOLENS_CACHE_DIR` | Override the cache base (default `~/.cache/repolens`). Empty string disables caching. |
| `REPOLENS_CACHE_DISABLED=1` | Disable both the index and embedding caches (in-memory only). |
| `XDG_CACHE_HOME` | Used as the platform cache base on Linux/macOS when `REPOLENS_CACHE_DIR` is unset. |
| `REPOLENS_DIAGNOSTICS=1` | Emit structured JSON diagnostics on `repolens.diagnostics` (DEBUG, off by default). |

Caches live outside the indexed repository, are keyed by a hash of the
repository root, and are written atomically. See
[docs/production-validation.md](docs/production-validation.md).

## OpenCode integration

1. Point OpenCode at a `opencode.json` in the project root with `repolens`
   configured as a **local** MCP server (see
   [docs/opencode-e2e.md](docs/opencode-e2e.md) for the exact shape):
   ```jsonc
   {
     "mcp": {
       "repolens": {
         "type": "local",
         "command": ["/path/to/.venv/bin/python", "-m", "repolens.mcp",
                     "--repo", "/path/to/repository", "--use-local-embeddings"],
         "cwd": "/path/to/repository",
         "enabled": true
       }
     }
   }
   ```
2. `opencode.json` is **local and untracked** — it contains machine-specific
   absolute paths and is never committed.
3. Once connected, an OpenCode agent can call the RepoLens tools directly for
   context, impact, and change planning.

## MCP tools

The server exposes nine tools. For each tool's parameters, output, and
"when to use", see **[docs/mcp-tools.md](docs/mcp-tools.md)**.

| Tool | What it does |
| --- | --- |
| `get_context` | Retrieve safe, budgeted context for a query. |
| `analyze_impact` | Blast radius + risk for a change target. |
| `inspect_symbol` | Callers/callees of a symbol (offline call graph). |
| `inspect_architecture` | Node overview (file/module/package). |
| `discover_subsystems` | Deterministic subsystem inventory. |
| `architecture_candidates` | Architecture-aware candidates for a query. |
| `explain_architecture_match` | Why a node was (or wasn't) relevant. |
| `change_plan` | Deterministic change/inspection plan. |
| `change_context` | Change-aware, firewall-cleared context. |

Launching the server:

```bash
python -m repolens.mcp --help   # the one and only package CLI
python -m repolens.mcp --repo /path/to/repository
```

The repository root is fixed at launch time; tool callers cannot read arbitrary
paths. All responses are firewall-cleared and JSON-safe.

## Running tests

```bash
# Full offline suite (no network, no model download)
python -m pytest -q

# Integration-only tests (require a one-time model download)
python -m pytest -m integration -q
```

## Running benchmarks

All benchmark/verification scripts ship in `benchmarks/` and are offline and
deterministic unless noted.

| Command | What it verifies |
| --- | --- |
| `python benchmarks/production_benchmark.py .` | Cold/warm/incremental index, caches, retrieval and context latency on a real repo. |
| `python benchmarks/verify_change_context.py .` | `change_plan` / `change_context` fixtures + real-repo cold/warm determinism. |
| `python benchmarks/opencode_e2e.py . --mode assisted` | Full OpenCode-style change workflows (8 tasks); exit 0. |
| `python benchmarks/opencode_e2e.py .` | Both modes; exits **1 by design** because the core-only baseline under-covers change workflows (see [docs/opencode-e2e.md](docs/opencode-e2e.md)). |
| `python benchmarks/verify_architecture.py .` | Architecture graph fixture + real-repo warm (zero parses). |
| `python benchmarks/verify_architecture_retrieval.py .` | Architecture-aware retrieval. |
| `python benchmarks/verify_architecture_mcp.py .` | Four architecture MCP tools. |
| `python benchmarks/verify_call_graph.py .` | Offline call graph. |
| `python benchmarks/verify_impact_analysis.py .` | Impact analysis. |
| `python benchmarks/verify_change_plan.py .` | Change-plan engine. |
| `python benchmarks/verify_embedding_cache.py .` | Embedding cache determinism. |
| `python benchmarks/verify_incremental_index.py .` | Incremental index invariants. |
| `python benchmarks/verify_context_quality.py .` | Context quality/firewall invariants. |
| `python benchmarks/agent_evaluation.py .` | Agent-task coverage (P25.2/25.3), deterministic. |
| `python benchmarks/performance_scalability.py` | Synthetic scalability (small/medium/large). |
| `python benchmarks/evaluate_local_embeddings.py` | Local embeddings quality — needs a one-time model download. |
| `python -m benchmarks.real_repo` | Retrieval benchmark against a pinned real repository (`Textualize/rich`) — downloads it on first run. |
| `python benchmarks/evaluate_real_embeddings.py` | OpenAI-provider quality — needs `REPOLENS_EMBEDDING_API_KEY`. |

Latency numbers are measurements, never pass/fail assertions; deterministic
structural checks are.

### Real-repo retrieval benchmark (summary)

`python -m benchmarks.real_repo` evaluates the four retrieval strategies
(lexical, local semantic, weighted hybrid, RRF) against ~20 hand-curated
queries on the pinned `Textualize/rich` release. Latest results (Rich v14.3.4):
RRF hybrid leads with Precision@5 0.31, Recall@5 0.75, MRR 0.80, above both
single-strategy baselines. See [docs/architecture-retrieval.md](docs/architecture-retrieval.md)
and the benchmark's own module for full detail and limitations.

## Important limitations

- **Deterministic, not learned.** Ranking, planning, risk, and security
  decisions are pure functions of the repository snapshot. RepoLens never
  calls an LLM and does not evaluate agent/LLM quality; its evaluations are
  offline and deterministic.
- **Does not edit code.** Change planning produces an inspection/change plan,
  never patches or edits.
- **No general-purpose CLI.** The only CLI is the MCP launcher
  (`python -m repolens.mcp`).
- **MCP is a thin adapter over stdio.** stdio only; no HTTP/SSE transport or
  authentication layer; one repository per server.
- **The firewall is defense-in-depth, not a guarantee.** It is high-precision
  by design and will not detect every secret format. Findings never contain the
  matched secret value.
- **Calls are conservative.** Dynamic/`exec`-based resolution, monkey-patching,
  and runtime-computed receivers are reported as unresolved rather than
  guessed. The call graph is for grounding/blast-radius triage, not a full
  language-server analysis.
- **Python-first analysis.** Parsing, symbols, references, and the call graph
  target Python source; other languages are not analyzed at this stage.
- **Single process, single repository.** Caches are not distributed; a
  long-lived MCP server reflects the index snapshot it was built on (restart to
  pick up on-disk changes).
- **Python >= 3.11 required**; optional local embeddings download an ONNX model
  on first use (requires network for that one-time fetch).

## Deeper documentation

See **[docs/README.md](docs/README.md)** — the documentation index — which
covers, at minimum:

- architecture, architecture-aware retrieval, and the architecture MCP tools
- call graph and change impact analysis
- change planning and change-aware context
- the full MCP tool surface
- production validation, performance & scalability, and reliability
- the OpenCode end-to-end workflow
- the v1.0 release checklist

## Project structure

```
repolens/
  scanner.py               # repository file discovery
  incremental_index.py     # content-hashed persistent repository index
  parser.py                # Python AST parsing (imports, definitions)
  index.py                 # symbol index
  graph.py                 # dependency graph
  references.py            # cross-file reference extraction
  call_graph.py            # offline static call graph
  architecture.py          # file/module/package architecture graph
  subsystems.py            # deterministic subsystem discovery
  architecture_retrieval.py# architecture-aware retrieval
  embeddings.py            # EmbeddingProvider + Fake + OpenAI providers
  local_embeddings.py      # FastEmbed on-device provider
  embedding_cache.py       # persistent embedding vector cache
  semantic_search.py       # semantic search
  search.py                # lexical search
  retrieval.py             # hybrid (RRF / weighted) search
  evaluation.py            # deterministic retrieval-quality evaluation
  impact.py                # change impact analysis
  change_plan.py           # deterministic change-plan engine
  change_context.py        # change-aware context bridge
  agent_evaluation.py      # deterministic agent-task evaluation
  production_benchmark.py  # production benchmark harness
  diagnostics.py           # opt-in structured diagnostics
  atomic_write.py          # atomic, fsynced file writes
  context/                 # context engine + firewall
    engine.py, ranking.py, budget.py, package.py, render.py, tokens.py
    firewall/              # ContextFirewall (path + content rules)
  mcp/                     # MCP server (the only CLI)
    launcher.py, server.py, tool.py, deps.py, errors.py
    impact_tool.py, inspect_tool.py, architecture_tool.py, change_plan_tool.py
benchmarks/
  production_benchmark.py      # M20 real-repo validation
  verify_*.py                  # per-subsystem verification benchmarks
  agent_evaluation.py          # P25.2/25.3 task coverage
  performance_scalability.py   # P25.4 synthetic scalability
  opencode_e2e.py              # P25.6 OpenCode end-to-end workflow
  real_repo/                   # pinned external-repo retrieval benchmark
docs/                          # documentation (index in docs/README.md)
tests/                         # offline test suite + fixtures
```