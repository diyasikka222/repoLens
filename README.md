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
  ...
```
