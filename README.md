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
tests/
  test_local_embeddings.py       # LocalEmbeddingProvider unit tests
  ...
```
