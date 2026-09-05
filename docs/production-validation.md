# Production Validation (Milestone 20)

This document describes how to validate RepoLens against a real repository,
what every measured metric means, how the caches behave, and what these
benchmarks intentionally do **not** guarantee.

The milestone is *hardening and measurement*, not redesign: the harness uses
the existing `RepositoryScanner`, `IncrementalIndexBuilder`,
`FileSystemEmbeddingCache`, retrieval searchers, and `ContextEngine` exactly
as they are used in production, including in the MCP server.

---

## Running the benchmark

The CLI validates any local repository (it never downloads anything and needs
no credentials):

```bash
# Default: deterministic offline fake embeddings, fresh temp caches.
python benchmarks/production_benchmark.py /path/to/repository

# Representative queries from a newline-delimited file.
python benchmarks/production_benchmark.py --queries queries.txt /path/to/repo

# Optional peak-memory instrumentation (tracemalloc — slower, perturbing).
python benchmarks/production_benchmark.py --measure-memory /path/to/repo

# Explicit cache directory (otherwise a fresh temp dir is used).
python benchmarks/production_benchmark.py --cache-dir ~/.cache/repolens /path/to/repo
```

The same harness is a reusable Python API for scripted use:

```python
from repolens.production_benchmark import run_production_benchmark

report = run_production_benchmark("/path/to/repository", repeats=3, measure_memory=False)
print(report.to_text())
```

## What each metric means

| Report item | Meaning |
|---|---|
| `Files discovered` | Number of `.py` files found under the root (ignoring `.git`, `.venv`, `venv`, `__pycache__`, `node_modules`). |
| `Cold index` | First `IncrementalIndexBuilder.build()` against a **fresh** empty cache. `files_parsed` equals the discovered count; `cache_hits` is 0. |
| `Warm index` | Second build against the *same* cache. `files_parsed` must be **0** and `cache_hits` must equal the discovered count. |
| `Incremental update` | Runs on a fresh temporary *copy* of the repo: cold, warm, one-file-modified (reparses exactly 1), one-file-added (parses exactly 1), one-file-deleted (`files_removed` = 1). |
| `Embedding / cache statistics` | On a temp copy with a fresh cache: cold embed count (`> 0`), warm embed count (must be **0** — all cache hits), and after one candidate file changes, exactly that one document is re-embedded. |
| `Retrieval latency` | Median / min / max latency in milliseconds over `repeats` × queries for each strategy: `lexical`, `semantic` (full candidate surface), `candidate-semantic` (default 40-candidate limit), `rrf`, `weighted`. |
| `Context generation` | `ContextEngine.build_context` latency plus `budget` (estimated-token cap), `context_size_median`/`context_size_max`, and the average `candidates` and `selected` file counts. |
| `peak_mem_bytes` | Peak traced memory during the stage, reported only with `--measure-memory`. |

All timings use the monotonic `time.perf_counter()`.

## Cold vs warm behaviour

- **Cold** = first build with an empty cache. Every file is read, hashed, and
  parsed; every analysis is persisted.
- **Warm** = rebuild with the same files and the same cache. Content hashes
  match, so analyses are loaded from the per-file cache instead of being
  re-parsed. A warm build parses zero files.
- A rebuild after **modifying one file** parses only that file (its content
  hash changed); everything else is a cache hit.
- A rebuild after **adding a file** parses only the new file.
- A rebuild after **deleting a file** prunes its stale cache entry
  (`files_removed`).

## Cache locations

Two **separate** caches exist and are intentionally independent concepts:

| Cache | Default location | Environment override |
|---|---|---|
| Incremental index (`AnalysisCache`) | `~/.cache/repolens/index/<repo-hash>/` | `REPOLENS_CACHE_DIR` |
| Embedding vectors (`FileSystemEmbeddingCache`) | `~/.cache/repolens/embeddings/<repo-hash>/` | `REPOLENS_CACHE_DIR` |

`<repo-hash>` is a short SHA-256 of the resolved repository root, so different
repositories never share cache entries. There is one JSON file per repository
file (index) or per document vector (embeddings); entries are keyed by content
hash and schema/embedding identity, so stale vectors/analyses are never
reused.

### Disabling persistence

- `REPOLENS_CACHE_DISABLED=1` disables both caches.
- `REPOLENS_CACHE_DIR=""` (empty string) also disables both caches.
- The MCP dependency wiring (`repolens.mcp.deps`) honours both; with caching
  disabled the incremental index keeps an in-memory cache (`persist=False`)
  and semantic search runs without a persistent embedding cache, so behaviour
  is otherwise unchanged.
- Direct use of `IncrementalIndexBuilder(..., persist=False)` and
  `SemanticSearcher(..., cache=None)` disables persistence per-object.

## Persistence safety

Cache and index writes are **atomic**: new content is written to a temporary
sibling file (`<name>.part-*.tmp`), flushed and fsynced, then moved over the
destination with `os.replace()`. A crash at any point leaves either the old
complete file or a new complete file — never a truncated one. Stale partials
are ignored by readers and swept by `clear()`. Corrupt, truncated, or
incompatible entries always degrade to a cache *miss* (never a crash, and
never silently replayed as current data).

## Expected incremental-index behaviour

- A warm rebuild parses zero files and only reads + deserializes cached
  analyses.
- A changed file (new content hash) is reparsed; its new analysis replaces the
  old entry.
- A deleted file's entry is pruned on the next build.
- Entries violating the schema version or content hash are treated as misses.
- Malformed Python raises the parser's existing `SyntaxError`; a previously
  valid entry for the *old* content is never reused for the new malformed
  content.
- Query vectors are **never** written to the persistent embedding cache — only
  repository document vectors are persisted.

## Observability

Diagnostics are opt-in and off by default. They are emitted as single-line
JSON records on the `repolens.diagnostics` logger at DEBUG level and never
alter the operation's results. Enable with `REPOLENS_DIAGNOSTICS=1` or
`repolens.diagnostics.enable()` at runtime.

Currently instrumented operations:

- `index_build` — repository, elapsed_ms, files_discovered, files_parsed,
  cache_hits, cache_misses, files_removed.
- `context_build` — repository, elapsed_ms, candidates, selected,
  context_size, budget, intent.

Records never contain source-code contents, API keys, secrets, or sensitive
repository contents — only paths, counts, and durations.

## Diagnosing failures

| Symptom | Likely cause | Check |
|---|---|---|
| Cold always re-parses everything | Cache dir missing or disabled | `REPOLENS_CACHE_DIR`, `REPOLENS_CACHE_DISABLED`; look at `files_parsed` vs `cache_hits`. |
| One file keeps re-parsing | Its content hash keeps changing (build artifacts written into the tree) | `files_parsed` on the warm build; scanner ignores only known dirs. |
| Warm build still parses files | Cache schema bumped, or entries corrupt/mismatched | Entries failing validation are logged as *treated as miss* warnings. |
| Embedding warm run re-embeds | Embedding identity changed (model/dimensions) or content changed | `normalize_embedding_identity` is part of the cache key. |
| Missing cache directory | Nothing to fix — caches create their directories lazily. | |
| Half-written JSON | Should not happen with atomic writes; a legacy/naive or interrupted entry is treated as a miss and re-created. | `clear()` sweeps partials. |
| Benchmark shows wildly varying numbers | Machine load | Use `median` over `--repeats`; compare *ratios* (e.g. warm/cold parsed) rather than absolute ms across machines. |

## What these benchmarks intentionally do NOT guarantee

- **No absolute-time thresholds.** Nothing here asserts "must finish in < N
  milliseconds". Timings are reported so *your* machine's baseline can be
  recorded and diffed, and are meaningful relative to each other on the same
  machine.
- **No quality (rank/metric) guarantee in the production benchmark.** Retrieval
  *quality* (P@K / recall / MRR) is covered separately by
  `repolens.evaluation` and `benchmarks/compare_strategies.py`. The production
  benchmark measures operational cost and structural correctness.
- **No detection of new performance regressions.** These are *measurements*;
  set up your own CI baselines (median/max drift) if you want alerts.
- **No guarantee about very large repositories.** The mutation workflows run on
  a temporary copy of the repository, which costs time/disk proportional to
  repository size.
- **Not a proof of "production ready".** Passing these benchmarks demonstrates
  the pipeline is observable, non-destructive, deterministic, and safe on
  failures — it does not certify correctness of retrieval or security of every
  repository it is pointed at.

## Memory / resource notes

Instrumentation uses Python's standard `tracemalloc` when requested. Known
resource characteristics (by design, not defects):

- `RepositoryIndex` holds per-file source text + parsed analysis in memory for
  the snapshot's lifetime (the price of not re-reading the repository).
- `SemanticSearcher._vectors_by_path` keeps one cached document vector per
  distinct candidate file ever searched, bounded by repository size
  (≈ vectors of ~32 KB each at 1024 dimensions).
- Persistent caches store one small JSON file per entry; lookups read only the
  requested entry.
- Dependency expansion is BFS bounded by `DependencyExpansionConfig.depth` and
  `max_expanded`; context selection is bounded by `ContextBudget`.

None of these were found to be defects during the M20 memory investigation; if
you observe unexpected growth, enable `--measure-memory` and capture
`peak_mem_bytes` per stage on the repository in question.