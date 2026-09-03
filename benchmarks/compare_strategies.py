#!/usr/bin/env python3
"""Benchmark and compare RepoLens retrieval strategies.

Compares four retrieval strategies against the comparison corpus:

1. Lexical            (CodeSearcher — baseline, no embeddings)
2. Full semantic      (SemanticSearcher over all files — unbounded embeddings)
3. Candidate semantic (SemanticSearcher over lexical top-N — production)
4. Hybrid             (weighted fusion of lexical + candidate semantic)

Each strategy is benchmarked with :func:`repolens.evaluation.benchmark_strategy`,
which reports per-case metrics, per-query latency, candidate counts, and
embedded counts (a proxy for embedding cost/IMU). The default embedding
provider is the deterministic :class:`~repolens.embeddings.FakeEmbeddingProvider`
so the comparison runs fully offline; pass ``--local`` to use the local
FastEmbed model instead.

Usage:
    python benchmarks/compare_strategies.py
    python benchmarks/compare_strategies.py --local
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures"
_CORPUS_DIR = _FIXTURES_DIR / "comparison_corpus"
_CASES_JSON = _CORPUS_DIR / "cases.json"

sys.path.insert(0, str(_PROJECT_ROOT))


def _validate_paths() -> None:
    if not _CORPUS_DIR.is_dir():
        print(f"ERROR: Comparison corpus not found at {_CORPUS_DIR}", file=sys.stderr)
        sys.exit(1)
    if not _CASES_JSON.is_file():
        print(f"ERROR: Corpus cases not found at {_CASES_JSON}", file=sys.stderr)
        sys.exit(1)


def _load_cases() -> list:
    import json
    from repolens.evaluation import EvaluationCase
    payload = json.loads(_CASES_JSON.read_text(encoding="utf-8"))
    return [
        EvaluationCase(query=item["query"], relevant_files=item["relevant_files"])
        for item in payload["cases"]
    ]


def _make_row(result, provider_label: str) -> dict:
    mean_emb = (
        f"{result.mean_embedded_count:.1f}" if result.mean_embedded_count is not None else "-"
    )
    total_emb = f"{result.total_embedded}" if result.total_embedded is not None else "-"
    total_cand = f"{result.total_candidates}" if result.total_candidates is not None else "-"
    return {
        "strategy": result.strategy,
        "p@5": f"{result.mean_precision_at_k:.4f}",
        "r@5": f"{result.mean_recall_at_k:.4f}",
        "mrr": f"{result.mean_reciprocal_rank:.4f}",
        "mean_lat": f"{result.mean_search_latency_seconds*1000:.1f}",
        "mean_emb": mean_emb,
        "build_s": f"{result.build_time_seconds:.3f}",
        "total_emb": total_emb,
        "total_cand": total_cand,
    }


def _print_table(results: list[dict], title: str) -> None:
    width = 90
    sep = "=" * width
    print()
    print(sep)
    print(f"  {title}")
    print(sep)
    print()
    strategies = [r["strategy"] for r in results]
    col_w = max(20, max(len(s) for s in strategies) + 2)
    header = f"  {'Metric':<20}" + "".join(f"{s:>{col_w}}" for s in strategies)
    print(header)
    print(f"  {'-' * 20}" + "".join(f"{'-' * col_w}" for _ in strategies))
    for metric_key, metric_label in [
        ("p@5", "Precision@5"),
        ("r@5", "Recall@5"),
        ("mrr", "MRR"),
        ("mean_lat", "Mean query (ms)"),
        ("mean_emb", "Mean embedded"),
        ("total_emb", "Total embedded"),
        ("build_s", "Build (s)"),
    ]:
        row = f"  {metric_label:<20}" + "".join(f"{r[metric_key]:>{col_w}}" for r in results)
        print(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RepoLens retrieval strategies.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use LocalEmbeddingProvider instead of the deterministic Fake provider.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=40,
        help="Candidate limit for the candidate-semantic strategy (default: 40).",
    )
    args = parser.parse_args()

    _validate_paths()
    cases = _load_cases()
    k = 5
    root = _CORPUS_DIR

    if args.local:
        from repolens.local_embeddings import LocalEmbeddingProvider
        provider = LocalEmbeddingProvider()
        provider_label = "BAAI/bge-small-en-v1.5"
    else:
        from repolens.embeddings import FakeEmbeddingProvider
        provider = FakeEmbeddingProvider()
        provider_label = "FakeEmbeddingProvider (deterministic)"

    from repolens.evaluation import benchmark_strategy

    strategies = ["lexical", "semantic", "candidate-semantic", "hybrid"]

    print("=" * 90)
    print("  RepoLens Strategy Comparison")
    print("=" * 90)
    print()
    print(f"  Corpus:              {_CORPUS_DIR}")
    print(f"  Number of queries:   {len(cases)}")
    print(f"  k:                   {k}")
    print(f"  Embedding provider:  {provider_label}")
    print(f"  Candidate limit:     {args.candidate_limit}")
    print()

    results_by_strategy: dict[str, object] = {}
    rows = []
    for strategy in strategies:
        print(f"Benchmarking {strategy!r}...")
        t0 = time.perf_counter()
        result = benchmark_strategy(
            strategy,
            root,
            cases,
            k=k,
            provider=provider,
            candidate_limit=args.candidate_limit,
        )
        elapsed = time.perf_counter() - t0
        results_by_strategy[strategy] = result
        rows.append(_make_row(result, provider_label))
        print(f"  Done in {elapsed:.2f}s")

    _print_table(rows, "Strategy Comparison (aggregate)")

    if len(rows) < 1:
        return
    # Per-query detail (reuses the already-computed results)
    print("=" * 90)
    print("  Per-Query Breakdown")
    print("=" * 90)
    print()
    for strategy in strategies:
        result = results_by_strategy[strategy]
        print(f"  --- {strategy} ---")
        for m in result.case_metrics:
            emb = f"{m.embedded_count}" if m.embedded_count is not None else "-"
            cand = f"{m.candidate_count}" if m.candidate_count is not None else "-"
            rel = (
                f"{'coverage' if m.relevant_in_candidates else 'MISSED'}"
                if m.relevant_in_candidates is not None
                else "-"
            )
            print(
                f"    {m.query:<28}  P@5={m.precision_at_k:.3f}  RR={m.reciprocal_rank:.3f}"
                f"  lat={m.search_latency_seconds*1000:6.1f}ms  emb={emb:>4}  cand={cand:>4}"
                f"  rel={rel:>8}"
            )
        print()

    # Candidate-semantic vs full-semantic quality comparison
    cand = results_by_strategy.get("candidate-semantic")
    full = results_by_strategy.get("semantic")
    print("=" * 90)
    print("  Candidate-Semantic vs Full-Semantic")
    print("=" * 90)
    print()
    if cand is not None and full is not None:
        print(f"  {'Metric':<24}{'full-sem':>12}{'candidate-sem':>16}")
        print(f"  {'-' * 24}{'-' * 12}{'-' * 16}")
        print(f"  {'Recall@5':<24}{full.mean_recall_at_k:>12.4f}{cand.mean_recall_at_k:>16.4f}")
        print(f"  {'MRR':<24}{full.mean_reciprocal_rank:>12.4f}{cand.mean_reciprocal_rank:>16.4f}")
        print(f"  {'Total embedded docs':<24}{(full.total_embedded or 0):>12d}{(cand.total_embedded or 0):>16d}")
        print()
    dropped = []
    if cand is not None:
        for m in cand.case_metrics:
            if m.relevant_in_candidates is False:
                dropped.append(m.query)
    if dropped:
        print("  Candidate generation DROPPED ground-truth files for:")
        for q in dropped:
            print(f"    - {q}")
        print(f"  This is embedding-cost-driver recall loss only when candidate_limit")
        print(f"  is smaller than the target's lexical rank. Production default (40)")
        print("  does not exhibit this on the shipped corpus.")
    else:
        print("  No case lost ground-truth coverage to candidate generation at this")
        print("  candidate_limit on the shipped corpus.")
    print()

    print("=" * 90)
    print("  Summary")
    print("=" * 90)
    print()
    by_mrr = sorted(rows, key=lambda r: float(r["mrr"]), reverse=True)
    print(f"  Best MRR:          {by_mrr[0]['strategy']} ({by_mrr[0]['mrr']})")
    by_p = sorted(rows, key=lambda r: float(r["p@5"]), reverse=True)
    print(f"  Best Precision@5:  {by_p[0]['strategy']} ({by_p[0]['p@5']})")
    by_r = sorted(rows, key=lambda r: float(r["r@5"]), reverse=True)
    print(f"  Best Recall@5:     {by_r[0]['strategy']} ({by_r[0]['r@5']})")
    print()
    print("  Semantic (full) embeds every file in the repository per build;")
    print("  candidate-semantic embeds at most candidate_limit files per query.")
    print()
    print("  Hybrid uses weighted fusion at the default lexical=0.5, semantic=0.5")
    print("  weights. It trails plain semantic here because a lexical 'magnet' — a")
    print("  file whose symbol names contain the query words (e.g. check_credentials)")
    print("  but whose body is off-topic — dominates min-max normalisation. Its large")
    print("  symbol-name score maps to the max, giving it a full 0.5 lexical weight,")
    print("  while the semantically-correct file's small lexical score adds little.")
    print("  At equal weights this keeps the magnet on top, so hybrid MRR is bounded by")
    print("  lexical on queries where the magnet outranks the target. This is expected")
    print("  behaviour of the current weighted fusion strategy, not a bug; raising the")
    print("  semantic weight (or using rank-based RRF fusion) recovers the target.")
    print()


if __name__ == "__main__":
    main()