#!/usr/bin/env python3
"""Evaluate RepoLens retrieval with local FastEmbed embeddings.

Compares four retrieval strategies against the synthetic evaluation dataset:

1. Lexical baseline    (CodeSearcher)
2. Local semantic      (SemanticSearcher + FastEmbed BAAI/bge-small-en-v1.5)
3. Weighted hybrid     (HybridSearcher, strategy="weighted")
4. RRF hybrid          (HybridSearcher, strategy="rrf")

Also runs a weight sweep across weighted-hybrid configurations.

Usage
-----
No environment variables are required.  The first run downloads the
BAAI/bge-small-en-v1.5 model (~130 MB); subsequent runs are fully offline.

    python benchmarks/evaluate_local_embeddings.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures"
_CASES_JSON = _FIXTURES_DIR / "evaluation_cases.json"
_SYNTHETIC_REPO = _FIXTURES_DIR / "synthetic_repository"

sys.path.insert(0, str(_PROJECT_ROOT))


def _validate_paths() -> None:
    if not _SYNTHETIC_REPO.is_dir():
        print(f"ERROR: Synthetic repository not found at {_SYNTHETIC_REPO}", file=sys.stderr)
        sys.exit(1)
    if not _CASES_JSON.is_file():
        print(f"ERROR: Evaluation cases not found at {_CASES_JSON}", file=sys.stderr)
        sys.exit(1)


def _load_cases() -> list:
    from repolens.evaluation import EvaluationCase
    payload = json.loads(_CASES_JSON.read_text(encoding="utf-8"))
    return [
        EvaluationCase(query=item["query"], relevant_files=item["relevant_files"])
        for item in payload["cases"]
    ]


def _format_report(report, label: str) -> dict:
    return {
        "label": label,
        "precision@5": report.mean_precision_at_k,
        "recall@5": report.mean_recall_at_k,
        "mrr": report.mean_reciprocal_rank,
        "num_cases": report.num_cases,
        "k": report.k,
    }


def _print_table(results: list[dict], title: str) -> None:
    width = 90
    sep = "=" * width
    thin = "-" * width

    print()
    print(sep)
    print(f"  {title}")
    print(sep)
    print()

    labels = [r["label"] for r in results]
    col_w = max(18, max(len(l) for l in labels) + 2)
    header = f"  {'Metric':<20}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print(f"  {'-' * 20}" + "".join(f"{'-' * col_w}" for _ in labels))

    for metric_key, metric_label in [
        ("precision@5", "Precision@5"),
        ("recall@5", "Recall@5"),
        ("mrr", "MRR"),
    ]:
        vals = [r[metric_key] for r in results]
        row = f"  {metric_label:<20}" + "".join(f"{v:>{col_w}.4f}" for v in vals)
        print(row)

    print()


def _print_weight_sweep(sweep_results: list[dict]) -> None:
    width = 90
    sep = "=" * width
    thin = "-" * width

    print()
    print(sep)
    print("  Weight Sweep (Weighted Hybrid)")
    print(sep)
    print()
    print(f"  {'Lex Wt':>8} {'Sem Wt':>8} {'Precision@5':>13} {'Recall@5':>10} {'MRR':>10}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 13} {'-' * 10} {'-' * 10}")

    for r in sweep_results:
        print(
            f"  {r['lex_weight']:>8.1f} {r['sem_weight']:>8.1f}"
            f" {r['precision@5']:>13.4f} {r['recall@5']:>10.4f} {r['mrr']:>10.4f}"
        )

    print()


def _print_per_query_detail(all_evals: dict[str, list], cases: list) -> None:
    width = 110
    print("-" * width)
    print("  Per-Query Breakdown")
    print("-" * width)
    print()

    labels = list(all_evals.keys())
    short_labels = {"LEXICAL BASELINE": "Lex", "LOCAL SEMANTIC": "Sem", "WEIGHTED HYBRID": "WHyb", "RRF HYBRID": "RHyb"}

    header = f"  {'Query':<30}"
    for label in labels:
        short = short_labels.get(label, label[:4])
        header += f" {short + ' P@5':>9} {short + ' RR':>8}"
    print(header)
    print(f"  {'-' * 30}" + "".join(f" {'-' * 9} {'-' * 8}" for _ in labels))

    for i, case in enumerate(cases):
        q = case.query if len(case.query) <= 28 else case.query[:25] + "..."
        row = f"  {q:<30}"
        for label in labels:
            ev = all_evals[label][i]
            row += f" {ev.precision_at_k:>9.4f} {ev.reciprocal_rank:>8.4f}"
        print(row)
    print()


def main() -> None:
    _validate_paths()
    cases = _load_cases()
    k = 5
    root = _SYNTHETIC_REPO

    from repolens.local_embeddings import LocalEmbeddingProvider
    from repolens.search import CodeSearcher
    from repolens.semantic_search import SemanticSearcher
    from repolens.retrieval import HybridSearcher, FusionStrategy
    from repolens.evaluation import EvaluationRunner

    model_name = os.environ.get("REPOLENS_LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip() or "BAAI/bge-small-en-v1.5"
    provider = LocalEmbeddingProvider(model=model_name)

    # Build base searchers
    print("Building lexical searcher...")
    t0 = time.perf_counter()
    lexical = CodeSearcher(root)
    print(f"  Done in {time.perf_counter() - t0:.2f}s")

    print(f"Building local semantic searcher (model: {model_name})...")
    print("  (First run may download the model.)")
    t0 = time.perf_counter()
    semantic = SemanticSearcher(root, provider)
    print(f"  Done in {time.perf_counter() - t0:.2f}s")

    # --- 4-way comparison ---

    print()
    print("Evaluating strategies...")

    # Lexical
    lex_report = EvaluationRunner(root, searcher=lexical).evaluate(cases, k=k)

    # Semantic
    sem_report = EvaluationRunner(root, searcher=semantic).evaluate(cases, k=k)

    # Weighted hybrid (default 0.5/0.5)
    weighted_hybrid = HybridSearcher(
        root, lexical_searcher=lexical, semantic_searcher=semantic,
        lexical_weight=0.5, semantic_weight=0.5, strategy=FusionStrategy.WEIGHTED,
    )
    wh_report = EvaluationRunner(root, searcher=weighted_hybrid).evaluate(cases, k=k)

    # RRF hybrid
    rrf_hybrid = HybridSearcher(
        root, lexical_searcher=lexical, semantic_searcher=semantic,
        strategy=FusionStrategy.RRF,
    )
    rrf_report = EvaluationRunner(root, searcher=rrf_hybrid).evaluate(cases, k=k)

    all_results = [
        _format_report(lex_report, "LEXICAL BASELINE"),
        _format_report(sem_report, "LOCAL SEMANTIC"),
        _format_report(wh_report, "WEIGHTED HYBRID"),
        _format_report(rrf_report, "RRF HYBRID"),
    ]

    config_info = {
        "model": model_name,
        "dimensions": provider.dimensions,
        "k": k,
        "num_queries": len(cases),
    }

    # Print main comparison
    print()
    print("=" * 90)
    print("  RepoLens Local Embedding Evaluation")
    print("=" * 90)
    print()
    print("  Configuration")
    print("-" * 90)
    print(f"  {'Embedding model:':<30} {config_info['model']}")
    print(f"  {'Embedding dimensions:':<30} {config_info['dimensions']}")
    print(f"  {'Evaluation queries:':<30} {config_info['num_queries']}")
    print(f"  {'k (cutoff):':<30} {config_info['k']}")
    print(f"  {'Default hybrid weights:':<30} lexical=0.5, semantic=0.5")
    print(f"  {'RRF k constant:':<30} 60")

    _print_table(all_results, "Strategy Comparison")

    # Per-query detail
    all_evals = {
        "LEXICAL BASELINE": lex_report.case_evaluations,
        "LOCAL SEMANTIC": sem_report.case_evaluations,
        "WEIGHTED HYBRID": wh_report.case_evaluations,
        "RRF HYBRID": rrf_report.case_evaluations,
    }
    _print_per_query_detail(all_evals, cases)

    # --- Weight sweep ---

    print("Running weight sweep...")
    sweep_configs = [
        (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5),
        (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9),
    ]
    sweep_results = []
    for lw, sw in sweep_configs:
        h = HybridSearcher(
            root, lexical_searcher=lexical, semantic_searcher=semantic,
            lexical_weight=lw, semantic_weight=sw, strategy=FusionStrategy.WEIGHTED,
        )
        report = EvaluationRunner(root, searcher=h).evaluate(cases, k=k)
        sweep_results.append({
            "lex_weight": lw,
            "sem_weight": sw,
            "precision@5": report.mean_precision_at_k,
            "recall@5": report.mean_recall_at_k,
            "mrr": report.mean_reciprocal_rank,
        })

    _print_weight_sweep(sweep_results)

    # --- Summary ---

    print("=" * 90)
    print("  Summary")
    print("=" * 90)
    print()

    p5 = {r["label"]: r["precision@5"] for r in all_results}
    r5 = {r["label"]: r["recall@5"] for r in all_results}
    mrr = {r["label"]: r["mrr"] for r in all_results}

    for label in ["LEXICAL BASELINE", "LOCAL SEMANTIC", "WEIGHTED HYBRID", "RRF HYBRID"]:
        print(f"  {label:<22} — P@5: {p5[label]:.4f}  R@5: {r5[label]:.4f}  MRR: {mrr[label]:.4f}")

    print()

    strategies = list(all_results)
    for metric_name, key in [("Precision@5", "precision@5"), ("Recall@5", "recall@5"), ("MRR", "mrr")]:
        best = max(strategies, key=lambda s: s[key])
        print(f"  Best {metric_name}: {best['label']} ({best[key]:.4f})")

    print()

    # Best weight sweep config
    best_sweep = max(sweep_results, key=lambda s: s["mrr"])
    print(f"  Best weight-sweep MRR: lex={best_sweep['lex_weight']:.1f} sem={best_sweep['sem_weight']:.1f} (MRR={best_sweep['mrr']:.4f})")
    best_p5_sweep = max(sweep_results, key=lambda s: s["precision@5"])
    print(f"  Best weight-sweep P@5: lex={best_p5_sweep['lex_weight']:.1f} sem={best_p5_sweep['sem_weight']:.1f} (P@5={best_p5_sweep['precision@5']:.4f})")

    print()
    print("=" * 90)
    print("  Evaluation complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
