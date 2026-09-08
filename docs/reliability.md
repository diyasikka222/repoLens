# Reliability & Failure Injection (Phase 25.5)

This document describes how RepoLens behaves when things fail and how that
behavior is verified.  It complements `production-validation.md` (correctness
under normal operation) and `performance.md` (scalability); this document is
about the *failure domain*: corruption, interruption, misbehaving providers,
bad input, and concurrency.

The behavior described here is pinned by `tests/test_reliability.py`.  Test
utilities (fault-injecting providers, cache corruptors, atomic-write faults,
and fingerprints) live in `tests/helpers/failure_injection.py`; none of it is
imported by production code.

## Goals

1. **No half-written state survives.** Persistent cache writes are atomic: a
   reader sees either the old complete entry or the new complete entry, never
   a truncated one, and an interrupted write leaves only an ignored stale
   sibling that eager consumers sweep.
2. **Corruption degrades to a miss, never a wrong answer.** A cache entry
   that cannot be read, does not carry the expected schema, or holds malformed
   payload data is treated as absent; the value is recomputed from source.
   Recovery always reproduces what a pristine rebuild would have produced
   (*recovery equivalence*).
3. **Provider failures are typed and non-poisoning.** An embedding provider
   that raises or returns the wrong number of vectors surfaces a clear,
   typed error, never a cryptic `KeyError`, and never leaves a partial set of
   in-memory or persisted vectors behind.
4. **Repository input edge cases are tolerated.** Non-UTF-8, unreadable, or
   vanishing files become empty analyses (indexing continues); a syntax error
   is the one case that propagates, deterministically, so a caller can repair
   and rebuild.
5. **Traversal is always bounded.** Graph and change-plan queries respect
   hard node/depth budgets and never guess about unresolved references.
6. **MCP degradation is safe.** Failures at the tool boundary are wrapped in
   typed `McpError` subclasses with stable diagnostics; a failed call never
   poisons shared state or a later successful call.
7. **Shared state is concurrency-safe.** Lock-protected lazy state builds
   exactly once and is deterministic; unlocked lazy state must never deadlock
   and must yield consistent results.

## Failure domains and the failure matrix

`tests/test_reliability.py` exports `FAILURE_MATRIX`, a per-domain record of
the injected fault and the observed behavior.  The domains:

| Domain | Injected fault | Observed behavior (invariant pinned by tests) |
| --- | --- | --- |
| Parser / repo input | Non-UTF-8 bytes in a `.py` file | `UnicodeDecodeError` → empty analysis; indexing continues |
| Parser / repo input | Unreadable file (`chmod 000`) | Empty analysis; no crash (test skips when running as root) |
| Parser / repo input | File disappears after scan | Empty analysis recorded; no crash |
| Parser / repo input | Syntax error introduced then repaired | `SyntaxError` propagates; repaired build fingerprint == pristine |
| Embedding | Provider raises on first batch | No partial `.json` or `.part-*.tmp` entry; in-memory vectors untouched |
| Embedding | Provider raises transiently | Failed-then-recovered search results == pristine search results |
| Embedding | Provider returns too few vectors | `EmbeddingProviderError` (typed), no partial state |
| Embedding | Provider returns skewed vector dimensions | Deterministic results; no cryptic crash |
| Embedding | Persistent cache write fails (`os.replace` OSError) | Search still succeeds; results stay deterministic |
| Caches | Corrupt incremental-index entries | Treated as miss; rebuilt index fingerprint == pristine |
| Caches | Corrupt reference-cache entries | Treated as miss; recomputed references equal original |
| Caches | Malformed embedding-cache vector | Treated as miss |
| Caches | Stale `*.part-*.tmp` partials | Swept by `clear()` |
| Atomic writes | `fsync` fails | Temporary removed; target byte-for-byte preserved |
| Atomic writes | `os.replace` fails mid-write | No partial file (target absent for a fresh write) |
| Atomic writes | Read-only target directory | Raises `OSError`; existing target preserved (skipped as root) |
| Incremental index | Second/unchanged build | Cache hits == files; zero parses; identical fingerprint |
| Incremental index | One unrelated change | Only the changed file re-parsed; other hits preserved |
| Incremental index | File deleted | Stale entry pruned on rebuild |
| References / call graph | Malformed source | Empty references; extractor never crashes |
| References / call graph | Unknown external module | Recorded as unresolved (`unknown_module`); no fabricated edges |
| References / call graph | Circular imports | Builds and queries terminate; call edges resolve correctly |
| References / call graph | Dense transitive queries | Bounded by `max_transitive_nodes` / `max_depth` |
| Architecture | Circular packages | Bounded BFS; no recursion; dependencies/dependents found |
| Architecture | Module deleted | Rebuilds without stale nodes |
| Architecture | Empty repository | Empty file-node set; no crash |
| Impact / change plan | Empty / missing / unknown target | `ImpactTargetError` |
| Impact / change plan | Plan on empty repository | Valid, deterministic, empty plan |
| Impact / change plan | Non-default bounds | Limits genuinely applied to the plan surface |
| Impact / change plan | Change-context request | Deterministic, firewall-safe response |
| Context budget | `max_tokens=0` | Empty selection |
| Context budget | `max_tokens=1` | Selected package never exceeds budget; no crash |
| Context budget | Query with no matches | Empty, deterministic package |
| Context budget | Root not a directory | `NotADirectoryError` |
| Context budget | Empty repository | Empty package |
| MCP | Engine factory raises | `ContextEngineError`; server usable afterwards |
| MCP | `build_context` raises | Wrapped; diagnostic preserves the original exception type |
| MCP | Invalid/absent arguments | `InvalidArgumentsError` |
| MCP | Non-callable factory / wrong firewall | `InternalError` |
| MCP | One failed `get_context` then success | Result == pristine result (no shared-state poisoning) |
| MCP | Rejected change-plan request | Later identical plan unchanged; shared state intact |
| MCP | Unsafe target / oversized bounds | `InvalidArgumentsError` |
| MCP | Change-plan factory raises | `ChangePlanError` |
| Concurrency | `ChangePlanState` from many threads | Single shared engine; deterministic plans |
| Concurrency | `ArchitectureState` from many threads | No deadlock; consistent subsystem ids |
| Concurrency | Concurrent incremental builds | Identical fingerprints; cache intact; no partials |
| Concurrency | Concurrent `build_context` | Identical packages |

## Injection methodology

Failures are injected at interfaces, not inside the logic under test:

- **Providers** (`FlakyProvider`, `ShortResultProvider`,
  `WrongDimensionProvider`) wrap a real `FakeEmbeddingProvider` and fail on
  call counts or per-document predicates.  The injected exception is
  `tests.helpers.failure_injection.EmbeddingFailure`, so tests assert that
  the *fault* propagated rather than any unrelated error.
- **Filesystem faults** patch `repolens.atomic_write.os.replace` /
  `os.fsync`, so the real atomic writer runs its full temporary-file/flush
  sequence and fails at the commit or durability point — exercising the
  writer's own cleanup, not a stub.
- **Corruption** rewrites cache payloads on disk (`corrupt_all_json`,
  `mutate_json_entries`) or drops stale partials, then re-runs the same
  builder/searcher.

## Recovery equivalence

The central assertion style is *recovery equivalence*:

> 1. Build a **pristine** result from clean inputs (fingerprinted).
> 2. Inject a fault, observe the failure contract.
> 3. Remove the fault, rebuild through the same path.
> 4. Assert the rebuilt result's fingerprint equals the pristine fingerprint.

Fingerprints (`index_fingerprint`, `package_fingerprint`, `plan_fingerprint`)
are sorted, serializable snapshots of the decision surface — never object
identity, and never wall-clock details (e.g. `build_time` is excluded).
Two fingerprints are equal iff the visible behaviour the test cares about is
identical.

## Design decisions and intentionally *not* guaranteed behavior

- **A raising embedding provider propagates.** `HybridSearcher.search` does
  not catch semantic-provider exceptions; the failure contract is a typed,
  explicit propagation.  What tests pin is that a failure (a) leaves no
  partial persistent entry, (b) is recoverable to pristine results, and
  (c) at the MCP boundary becomes a safe `ContextEngineError`.
- **A provider returning the wrong vector count is an error.** This was a
  genuine bug: `SemanticSearcher._ensure_embeddings` silently truncated via
  `zip`, permanently leaving a partial in-memory vector table and raising a
  cryptic `KeyError` from inside `search`.  The fix validates the count
  against the requested documents *before* mutating any state and raises
  `EmbeddingProviderError`.  Demanding exact-many vectors keeps the permanent
  corruption class impossible.  See `tests/test_reliability.py::test_provider_returning_too_few_vectors_raises_typed_error`.
- **`syntax error in a `.py` file propagates`** — this is deliberate: it is
  the one unambiguous user-input signal that cannot be silently degraded, and
  a caller should repair and rebuild.  Everything *else* about a file that
  cannot be read degrades to an empty analysis.
- **Intentionally not guaranteed:** no distributed locking on RepoLens
  caches (single-process by design; atomic replacement is the only guarantee);
  no timing or throughput guarantees; no crash-consistency for files written
  outside `atomic_write_text`.

## Concurrency expectations

- `ChangePlanState.index` and `ChangePlanState.default_engine`
  (`repolens/mcp/change_plan_tool.py`) are lock-protected double-checked
  builds → tests assert **exactly-once** shared construction.
- `ArchitectureState` (`repolens/mcp/architecture_tool.py`) is *not*
  lock-protected → tests assert **no deadlock and consistent deterministic
  results**, never exact-once (that would force a production change).
- Concurrent `IncrementalIndexBuilder.build()` is safe because writes are
  atomic (identical content, last-writer-wins) and stale pruning is
  exception-tolerant.

## Running

```bash
.venv/bin/python -m pytest tests/test_reliability.py -q
```

The reliability tests are offline, deterministic, and fast (<1s on a
development machine).  All repository fixtures are written under `tmp_path`;
the MCP shared-index cache is routed to an isolated directory via the
`REPOLENS_CACHE_DIR` fixture so the tests never touch the host cache base.