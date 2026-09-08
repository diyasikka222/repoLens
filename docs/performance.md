# Performance & Scalability (Phase 25.4)

This document describes the performance and scalability benchmark
(`benchmarks/performance_scalability.py`): what it measures, how it keeps the
measurements honest and deterministic, how to run it, and what it
intentionally does **not** guarantee.

The benchmark exercises every major production path — discovery, indexing
(cold / warm / cache-disabled), incremental updates, the embedding cache,
retrieval, context generation, architecture, impact analysis, change planning,
and change-aware context — through the same public classes the MCP server
uses.  It never asserts machine-dependent latency thresholds; instead it
reports measured latencies plus deterministic structural checks that keep
performance *bounded*.

---

## Running the benchmark

### Against a real repository

```bash
# Default: deterministic offline fake embeddings, fresh temp caches.
python benchmarks/performance_scalability.py /path/to/repository

# Repeated runs per query for median/p95/min/max latency statistics.
python benchmarks/performance_scalability.py --repeats 5 /path/to/repository

# Representative queries from a newline-delimited file.
python benchmarks/performance_scalability.py --queries queries.txt /path/to/repo

# Reuse a cache directory (otherwise a fresh temp dir is created and removed).
python benchmarks/performance_scalability.py --cache-dir ~/.cache/repolens /path/to/repo

# Optional peak-memory instrumentation (tracemalloc — slower, perturbing).
python benchmarks/performance_scalability.py --measure-memory /path/to/repo
```

### Against deterministic synthetic corpora

With no positional argument the benchmark generates a deterministic synthetic
repository for each requested scale, runs the full harness, and prints a
summary table.  Corpora are written outside the repository (temp dir by
default, or `--corpora-dir`) and cleaned up unless `--keep` is passed.

```bash
python benchmarks/performance_scalability.py                 # all scales
python benchmarks/performance_scalability.py --scale small   # one scale
python benchmarks/performance_scalability.py --scale medium,large --repeats 5
python benchmarks/performance_scalability.py --seed 42 --corpora-dir ./corpora --keep
```

The same harness is a reusable Python API:

```python
from pathlib import Path
from benchmarks.performance_scalability import (
    generate_synthetic_repository,
    run_scalability_benchmark,
)

spec = generate_synthetic_repository(Path("/tmp/corpus"), "small", seed=7)
report = run_scalability_benchmark(spec.root, queries=list(spec.queries), repeats=3)
print(report.to_text())
```

---

## What is measured

Every operation reports wall-clock time (`time.perf_counter`, rounded to 3
decimal ms) plus operation-specific counts.  Latency-sensitive retrieval and
planning stages run each query `--repeats` times and report `median_ms`,
`p95_ms`, `min_ms`, and `max_ms`.  All stages that produce a result derive a
**deterministic fingerprint** (SHA-256 of the canonical JSON payload, first 16
hex chars) so result *content* is compared across cache regimes and runs.

| Group | Stages | Key metrics |
| --- | --- | --- |
| Discovery and indexing (A–D) | `discovery`, `cold_index`, `warm_index`, `cache_disabled` (x2) | `files_discovered`, `files_parsed`, `cache_hits`, `cache_misses`, `files_removed`, `fingerprint` |
| Incremental updates (E–G) | `cold`, `warm`, `modified`, `added`, `deleted` | `files_parsed`, `cache_hits`, `files_removed` |
| Embedding cache statistics | `cold_embedding`, `warm_embedding`, `changed_embedding` | `embedded_documents`, `embedded_queries`, `cache_hits`, `cache_misses` |
| Retrieval latency (H–L) | `lexical`, `candidate-semantic`, `semantic`, `rrf`, `weighted` | `median_ms`, `p95_ms`, `runs`, `results`, `fingerprint`, embedding cache stats |
| Context generation (M) | `context` | `median_ms`, `queries`, `candidates_avg`, `selected_avg`, `context_size_median`, `context_size_max`, `budget`, `fingerprint` |
| Architecture (N) | `architecture_build`, `architecture_query` | `nodes`, `files`, `modules`, `packages`, `edges`, `fingerprint` |
| Impact analysis (O) | `impact` | `median_ms`, `targets`, `affected_avg`, `affected_max`, `fingerprint` |
| Change planning (P) | `plan_engine_build`, `change_plan_request` | `median_ms`, `requests`, `targets_avg`, `affected_avg`, `risks`, `fingerprint` |
| Change-aware context (Q) | `change_context` | `median_ms`, `requests`, `selected_avg`, `change_candidates_avg`, `tokens_median`, `tokens_max`, `budget`, `fingerprint` |

### Semantics of each scene

- **Cold** — a fresh cache; every file is parsed (`files_parsed` = discovered)
  and re-embedded, with zero cache hits.
- **Warm** — an immediate rebuild against the populated cache; zero files
  parsed, zero documents re-embedded, all entries served from the cache
  (`cache_hits` = discovered), and the index fingerprint must match cold.
- **Cache-disabled** — `IncrementalIndexBuilder(..., persist=False)` so every
  file is reparsed on each run and the cache is never consulted (hits = 0).
  The fingerprint must still match cold/warm.
- **Incremental updates (E–G)** — after the warm baseline: `modified` reparses
  exactly one changed file, `added` adds exactly one new file to the index,
  `deleted` removes exactly one cached entry (`files_removed` = 1).
- **Embedding cache** — `cold_embedding` embeds every document once;
  `warm_embedding` re-embeds zero documents (all cache hits);
  `changed_embedding` re-embeds exactly the one file touched by the
  incremental change.

### Retrieval strategies (H–L)

- `lexical` — keyword search (`CodeSearcher`).
- `candidate-semantic` — embedding-based rerank over the lexical candidate set
  only (never embeds the whole repository).
- `semantic` — full semantic search over the persisted embedding cache.
- `rrf` / `weighted` — `HybridSearcher` fusing lexical + semantic with
  reciprocal-rank-fusion and weighted strategies respectively.

All strategies share one embedding cache, so later strategies reuse the
documents the earlier ones embedded — that is intended warm behavior and the
per-strategy embedding cache counters make it visible.

### Change-aware paths (O–Q)

- `impact` — `ImpactAnalyzer` over a shared dependency, symbol, reference, and call-graph snapshot; targets are the first non-package, non-test modules.
- `change_plan_request` — the change-plan engine (built once, reused across requests) plans a short deterministic English request per target and records `targets_avg` / `affected_avg` / risk.
- `change_context` — end-to-end: change plan → `ContextEngine.build_context` with the change plan, options, and target → firewall `inspect` → `safe_package`.  Records the safe package's estimated tokens against the default `ContextBudget`.

---

## Synthetic corpora

`generate_synthetic_repository(root, scale, seed)` writes a deterministic
Python repository: `packages` packages, each with `modules` module files, an
`__init__.py` per package, one shared `shared.py`, and one test file per
package.  Modules exercise real cross-file structure: `import shared as peer`,
`from shared import transform_N as imported_apply`, sibling-module calls, a
class with methods chained to the sibling module, and seeded numeric salts that
make content (but not structure) differ between seeds.

| Scale | packages × modules | functions | classes | methods | tests | total files |
| --- | --- | --- | --- | --- | --- | --- |
| `small` | 2 × 2 | 3 | 1 | 2 | 2 | **9** |
| `medium` | 4 × 4 | 4 | 2 | 3 | 4 | **25** |
| `large` | 6 × 6 | 5 | 3 | 3 | 8 | **51** |

Total files = `packages*modules + packages + 1 + tests`.  The same seed yields
byte-identical trees; different seeds yield different bytes with identical
structure.  Corpus-specific queries are derived deterministically and are used
by the synthetic CLI mode so retrieval actually matches content.

---

## Structural checks (what is asserted)

`report.checks` lists the machine-independent invariants printed as
`[PASS]`/`[FAIL]` under **Sanity checks**.  Latencies are measurements, never
assertions.  The checks are:

1. warm rebuild parses zero files;
2. cache-disabled indexing parses everything with zero hits;
3. incremental modification reparses exactly one file;
4. incremental addition parses exactly one file;
5. incremental deletion removes exactly one entry;
6. warm embedding reuses the cache (no re-embed);
7. lexical retrieval returns results;
8. RRF / weighted hybrid retrieval were measured;
9. context stays within the token budget;
10. architecture graph builds deterministically (fingerprint);
11. impact analysis resolved targets;
12. change plan produced a primary target;
13. change-aware context respected the budget.

The benchmark intentionally does **not** assert that a machine is "fast enough":
latency targets are machine- and load-dependent.  The report is the evidence
for bottleneck-hunting; the checks are the regression guardrails.

---

## Example output

```text
Repository: /tmp/corpus/small_7
Files discovered: 9
Python files: 9
Discovery and indexing (A–D):
  discovery: 0.304 ms  files_discovered=9 python_files=9
  cold_index: 5.175 ms  files_discovered=9 files_parsed=9 cache_hits=0 cache_misses=9 files_removed=0 fingerprint=48d5082f2ec0f888
  warm_index: 1.13 ms  files_discovered=9 files_parsed=0 cache_hits=9 cache_misses=0 files_removed=0 fingerprint=48d5082f2ec0f888
  ...
Retrieval latency (H–L):
  lexical: 0.238 ms  median_ms=0.033 p95_ms=0.04 min_ms=0.027 max_ms=0.041 runs=7 results=63 fingerprint=70446b11f251d984
  ...
Sanity checks:
  [PASS] warm rebuild parses zero files — parsed=0, hits=9
  [PASS] incremental modification reparses exactly one file — parsed=1
  ...

NOTE: these numbers are measurements, not pass/fail assertions.
```

The synthetic scale summary additionally prints `cold_index_ms` and the
`retrieval_p95`, `context_p95`, and `change_plan_p95` latencies per scale, so
scaling behavior across small → medium → large is visible at a glance.