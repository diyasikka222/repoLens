#!/usr/bin/env python3
"""Production validation for RepoLens against an arbitrary repository (M20).

Runs the production benchmark harness
(:func:`repolens.production_benchmark.run_production_benchmark`) against *any*
local repository and prints a concise, machine-parseable report. It never
downloads repositories and never requires credentials; by default it uses the
deterministic offline :class:`~repolens.embeddings.FakeEmbeddingProvider` so
no embedding API is contacted.

Usage::

    python benchmarks/production_benchmark.py /path/to/repository
    python benchmarks/production_benchmark.py --queries queries.txt /path/to/repo
    python benchmarks/production_benchmark.py --measure-memory /path/to/repo

Reported metrics (see ``docs/production-validation.md`` for definitions):

- Repository and files discovered / Python files
- Cold index: clean build (every file parsed)
- Warm index: unchanged rebuild (zero files parsed; all cache hits)
- Incremental update: cold / warm / one-file-modified / one-file-added /
  one-file-deleted transitions measured on a fresh *temporary copy* of the
  repository (the user's repository is never modified)
- Embedding / cache statistics: cold embed count, warm embed count (must be
  zero), single-file-change re-embed count, cache hits/misses
- Retrieval latency: median latency for lexical, semantic, candidate-semantic,
  RRF hybrid, and weighted hybrid over the supplied queries
- Context generation: median latency, candidate files, selected files, and the
  resulting estimated context size against the configured budget

The harness reports *structural* counts and relative timings, and never
asserts machine-dependent absolute-time pass/fail thresholds.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from repolens.context import ContextBudget  # noqa: E402
from repolens.production_benchmark import (  # noqa: E402
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_REPEATS,
    run_production_benchmark,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="production_benchmark.py",
        description="Validate RepoLens against a local repository.",
    )
    parser.add_argument(
        "repository",
        type=str,
        help="Path to the local repository to benchmark (never downloaded).",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Newline-delimited text file of representative queries "
        "(default: built-in generic development queries).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for index/embedding caches. Default: a fresh temporary "
        "directory, so nothing outside /tmp is written.",
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Instrument peak memory with tracemalloc (slower, perturbing).",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="Candidate limit for candidate-based semantic search (default: 40).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Repeated runs per query for latency medians (default: 3).",
    )
    return parser.parse_args(argv)


def _load_queries(path: str | None) -> list[str]:
    if path is None:
        return []
    queries = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        raise SystemExit(f"No queries found in {path}")
    return queries


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.repository).expanduser().resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    # Read-only measurements may use an explicit cache dir; default to a fresh
    # temp base so the user's caches and repository are never touched.
    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="repolens-prodbench-"))

    queries = _load_queries(args.queries)

    print(f"Running production benchmark against {root}")
    print(f"  default budget: {ContextBudget().max_tokens} estimated tokens")
    print(f"  measure-memory: {args.measure_memory}")
    print()

    report = run_production_benchmark(
        root,
        queries=queries or None,
        cache_dir=cache_dir,
        candidate_limit=args.candidate_limit,
        repeats=args.repeats,
        measure_memory=args.measure_memory,
    )
    print(report.to_text())
    print()
    print("NOTE: these numbers are measurements, not pass/fail assertions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())