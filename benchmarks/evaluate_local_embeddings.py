#!/usr/bin/env python3
"""Evaluate RepoLens retrieval with local FastEmbed embeddings.

Compares three retrieval strategies against the synthetic evaluation dataset:

1. Lexical baseline   (CodeSearcher)
2. Local semantic     (SemanticSearcher + FastEmbed BAAI/bge-small-en-v1.5)
3. Local hybrid       (HybridSearcher combining lexical + local semantic)

Usage
-----
No environment variables are required.  The first run downloads the
BAAI/bge-small-en-v1.5 model (~130 MB); subsequent runs are fully offline.

    python benchmarks/evaluate_local_embeddings.py

This script is intentionally NOT collected by pytest.  It makes real
computation calls and downloads the embedding model on first run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root (one level up from benchmarks/)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures"
_CASES_JSON = _FIXTURES_DIR / "evaluation_cases.json"
_SYNTHETIC_REPO = _FIXTURES_DIR / "synthetic_repository"

# Add project root to sys.path so repolens is importable
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Validate paths
# ---------------------------------------------------------------------------

def _validate_paths() -> None:
    """Ensure the synthetic repo and evaluation cases exist."""
    if not _SYNTHETIC_REPO.is_dir():
        print(f"ERROR: Synthetic repository not found at {_SYNTHETIC_REPO}", file=sys.stderr)
        sys.exit(1)
    if not _CASES_JSON.is_file():
        print(f"ERROR: Evaluation cases not found at {_CASES_JSON}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Load evaluation cases
# ---------------------------------------------------------------------------

def _load_cases() -> list:
    from repolens.evaluation import EvaluationCase

    payload = json.loads(_CASES_JSON.read_text(encoding="utf-8"))
    return [
        EvaluationCase(query=item["query"], relevant_files=item["relevant_files"])
        for item in payload["cases"]
    ]


# ---------------------------------------------------------------------------
# Build searchers
# ---------------------------------------------------------------------------

def _build_lexical(root: Path):
    from repolens.search import CodeSearcher
    return CodeSearcher(root)


def _build_semantic(root: Path, provider) -> tuple:
    from repolens.semantic_search import SemanticSearcher
    return SemanticSearcher(root, provider)


def _build_hybrid(root: Path, lexical_searcher, semantic_searcher, lex_weight: float, sem_weight: float):
    from repolens.retrieval import HybridSearcher
    return HybridSearcher(
        root,
        lexical_searcher=lexical_searcher,
        semantic_searcher=semantic_searcher,
        lexical_weight=lex_weight,
        semantic_weight=sem_weight,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_report_table(report, label: str) -> dict:
    return {
        "label": label,
        "precision@5": report.mean_precision_at_k,
        "recall@5": report.mean_recall_at_k,
        "mrr": report.mean_reciprocal_rank,
        "num_cases": report.num_cases,
        "k": report.k,
    }


def _print_comparison(results: list[dict], config_info: dict) -> None:
    width = 78
    sep = "=" * width
    thin = "-" * width

    print()
    print(sep)
    print("  RepoLens Local Embedding Evaluation")
    print(sep)
    print()

    print("  Configuration")
    print(thin)
    print(f"  {'Embedding model:':<30} {config_info['model']}")
    print(f"  {'Embedding dimensions:':<30} {config_info['dimensions']}")
    print(f"  {'Hybrid lexical weight:':<30} {config_info['lex_weight']}")
    print(f"  {'Hybrid semantic weight:':<30} {config_info['sem_weight']}")
    print(f"  {'Evaluation queries:':<30} {config_info['num_queries']}")
    print(f"  {'k (cutoff):':<30} {config_info['k']}")
    print()

    header = f"  {'Metric':<20} {'LEXICAL BASELINE':>18} {'LOCAL SEMANTIC':>18} {'LOCAL HYBRID':>18}"
    print(header)
    print(f"  {'-' * 20} {'-' * 18} {'-' * 18} {'-' * 18}")

    for metric_key, metric_label in [
        ("precision@5", "Precision@5"),
        ("recall@5", "Recall@5"),
        ("mrr", "MRR"),
    ]:
        vals = [r[metric_key] for r in results]
        print(f"  {metric_label:<20} {vals[0]:>18.4f} {vals[1]:>18.4f} {vals[2]:>18.4f}")

    print()

    for r in results:
        print(f"  {r['label']}")
        print(f"    Queries evaluated: {r['num_cases']}  |  k = {r['k']}")
        print()


# ---------------------------------------------------------------------------
# Per-query detail
# ---------------------------------------------------------------------------

def _print_per_query_detail(
    lexical_evals, semantic_evals, hybrid_evals, cases
) -> None:
    width = 78
    print("-" * width)
    print("  Per-Query Breakdown")
    print("-" * width)
    print()
    print(f"  {'Query':<30} {'Lex P@5':>8} {'Sem P@5':>8} {'Hyb P@5':>8}  {'Lex RR':>7} {'Sem RR':>7} {'Hyb RR':>7}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 7} {'-' * 7} {'-' * 7}")

    for i, case in enumerate(cases):
        q = case.query if len(case.query) <= 28 else case.query[:25] + "..."
        lex = lexical_evals[i]
        sem = semantic_evals[i]
        hyb = hybrid_evals[i]
        print(
            f"  {q:<30}"
            f" {lex.precision_at_k:>8.4f}"
            f" {sem.precision_at_k:>8.4f}"
            f" {hyb.precision_at_k:>8.4f}"
            f"  {lex.reciprocal_rank:>7.4f}"
            f" {sem.reciprocal_rank:>7.4f}"
            f" {hyb.reciprocal_rank:>7.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Validate paths
    _validate_paths()

    # 2. Load evaluation cases
    cases = _load_cases()
    k = 5
    lex_weight = 0.5
    sem_weight = 0.5

    # 3. Build the local embedding provider
    from repolens.local_embeddings import LocalEmbeddingProvider

    model_name = os.environ.get("REPOLENS_LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip() or "BAAI/bge-small-en-v1.5"
    provider = LocalEmbeddingProvider(model=model_name)

    root = _SYNTHETIC_REPO

    # 4. Build searchers
    print(f"Building lexical searcher...")
    t0 = time.perf_counter()
    lexical = _build_lexical(root)
    t_lex = time.perf_counter() - t0
    print(f"  Done in {t_lex:.2f}s")

    print(f"Building local semantic searcher (model: {model_name})...")
    print("  (First run may download the model — this is expected.)")
    t0 = time.perf_counter()
    semantic = _build_semantic(root, provider)
    t_sem = time.perf_counter() - t0
    print(f"  Done in {t_sem:.2f}s")

    print("Building local hybrid searcher...")
    t0 = time.perf_counter()
    hybrid = _build_hybrid(root, lexical, semantic, lex_weight, sem_weight)
    t_hyb = time.perf_counter() - t0
    print(f"  Done in {t_hyb:.2f}s")

    # 5. Evaluate each strategy
    from repolens.evaluation import EvaluationRunner

    print()
    print("Evaluating lexical baseline...")
    lex_runner = EvaluationRunner(root, searcher=lexical)
    lex_report = lex_runner.evaluate(cases, k=k)

    print("Evaluating local semantic...")
    sem_runner = EvaluationRunner(root, searcher=semantic)
    sem_report = sem_runner.evaluate(cases, k=k)

    print("Evaluating local hybrid...")
    hyb_runner = EvaluationRunner(root, searcher=hybrid)
    hyb_report = hyb_runner.evaluate(cases, k=k)

    # 6. Format results
    lex_metrics = _format_report_table(lex_report, "LEXICAL BASELINE")
    sem_metrics = _format_report_table(sem_report, "LOCAL SEMANTIC")
    hyb_metrics = _format_report_table(hyb_report, "LOCAL HYBRID")

    config_info = {
        "model": model_name,
        "dimensions": provider.dimensions,
        "lex_weight": lex_weight,
        "sem_weight": sem_weight,
        "num_queries": len(cases),
        "k": k,
    }

    # 7. Print comparison
    _print_comparison([lex_metrics, sem_metrics, hyb_metrics], config_info)

    # 8. Per-query detail
    _print_per_query_detail(
        lex_report.case_evaluations,
        sem_report.case_evaluations,
        hyb_report.case_evaluations,
        cases,
    )

    # 9. Summary / verdict
    print("=" * 78)
    print("  Summary")
    print("=" * 78)
    print()

    p5_lex = lex_metrics["precision@5"]
    p5_sem = sem_metrics["precision@5"]
    p5_hyb = hyb_metrics["precision@5"]

    r5_lex = lex_metrics["recall@5"]
    r5_sem = sem_metrics["recall@5"]
    r5_hyb = hyb_metrics["recall@5"]

    mrr_lex = lex_metrics["mrr"]
    mrr_sem = sem_metrics["mrr"]
    mrr_hyb = hyb_metrics["mrr"]

    print(f"  Lexical baseline  — P@5: {p5_lex:.4f}  R@5: {r5_lex:.4f}  MRR: {mrr_lex:.4f}")
    print(f"  Local semantic    — P@5: {p5_sem:.4f}  R@5: {r5_sem:.4f}  MRR: {mrr_sem:.4f}")
    print(f"  Local hybrid      — P@5: {p5_hyb:.4f}  R@5: {r5_hyb:.4f}  MRR: {mrr_hyb:.4f}")
    print()

    strategies = [("Lexical", p5_lex, r5_lex, mrr_lex),
                  ("Semantic", p5_sem, r5_sem, mrr_sem),
                  ("Hybrid", p5_hyb, r5_hyb, mrr_hyb)]

    for metric_name, idx in [("Precision@5", 1), ("Recall@5", 2), ("MRR", 3)]:
        best = max(strategies, key=lambda s: s[idx])
        print(f"  Best {metric_name}: {best[0]} ({best[idx]:.4f})")

    print()
    print("=" * 78)
    print("  Evaluation complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
