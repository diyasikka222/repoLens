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

> **Status:** Project scaffold only. Core functionality is not yet implemented.
