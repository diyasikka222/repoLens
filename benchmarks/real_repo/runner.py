"""Execution and reporting for the real-world repository benchmark.

This is the only module in ``benchmarks/real_repo`` that touches the network
(making the pinned repository available locally) and the only one that
downloads an embedding model on first use. Importing this module is offline;
network access happens only when :func:`ensure_repository` or the model
provider is exercised.

The benchmark is driven from the command line::

    python -m benchmarks.real_repo

It evaluates four retrieval strategies with identical, unchanged RepoLens
algorithms:

1. Lexical        (:class:`repolens.search.CodeSearcher`)
2. Local semantic (:class:`repolens.semantic_search.SemanticSearcher` + FastEmbed)
3. Weighted hybrid(:class:`repolens.retrieval.HybridSearcher`, strategy="weighted")
4. RRF            (:class:`repolens.retrieval.HybridSearcher`, strategy="rrf")

Default hybrid weights and the RRF constant are left unchanged. An optional
exploratory weight sweep can be enabled with ``--weight-sweep`` and is
clearly labelled as exploratory (it does not tune the default).
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from benchmarks.real_repo.config import (
    ARCHIVE_PATH,
    DATA_DIR,
    MIN_PYTHON_FILES,
    REPO_DIR,
    REPOSITORY_COMMIT,
    REPOSITORY_NAME,
    REPOSITORY_REF,
    REPOSITORY_URL,
    TARBALL_URL,
)
from benchmarks.real_repo.dataset import load_cases

DEFAULT_K = 5
DEFAULT_LEXICAL_WEIGHT = 0.5
DEFAULT_SEMANTIC_WEIGHT = 0.5
DEFAULT_RRF_K = 60

SEARCHER_LABELS = (
    "LEXICAL",
    "LOCAL SEMANTIC",
    "WEIGHTED HYBRID (0.5/0.5)",
    "RRF",
)


def _module_help() -> str:
    return (
        "Real-world repository retrieval benchmark (Milestone 11).\n\n"
        "Evaluates lexical, local-semantic, weighted-hybrid and RRF retrieval\n"
        "against the pinned external repository "
        f"{REPOSITORY_NAME} @ {REPOSITORY_REF}.\n\n"
        "Usage:\n"
        "  python -m benchmarks.real_repo\n"
        "  python -m benchmarks.real_repo --weight-sweep   (exploratory)\n"
        "  python -m benchmarks.real_repo --repo-dir PATH  (override data dir)\n\n"
        "Requirements:\n"
        "  - Python with the 'repolens' package importable and 'fastembed'\n"
        "    installed (pip install -e \".[dev]\")\n"
        "  - Network access on first run to download the pinned repository\n"
        "    tarball and the ONNX embedding model.\n"
    )


def _prereq_check() -> list[str]:
    """Return a list of missing prerequisites (empty if all are present)."""
    missing: list[str] = []
    try:
        import fastembed  # noqa: F401
    except Exception:
        missing.append("fastembed is not installed (pip install -e \".[dev]\")")
    try:
        import repolens  # noqa: F401
    except Exception:
        missing.append("repolens is not importable from the project root")
    return missing


def ensure_repository(repo_dir: Path = REPO_DIR, quiet: bool = False) -> Path:
    """Return the local path to the pinned repository, downloading it if needed.

    The repository is fetched from the GitHub tarball for the pinned ref. If
    it is already present and non-empty, it is reused without re-downloading.
    Extracted content is placed under the ignored data directory.
    """
    from repolens.scanner import RepositoryScanner

    if repo_dir.is_dir() and any(repo_dir.iterdir()):
        _verify_repository(repo_dir)
        return repo_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.is_file():
        if not quiet:
            print(f"Downloading {REPOSITORY_NAME} @ {REPOSITORY_REF} ...")
            print(f"  {TARBALL_URL}")
        _download_tarball(ARCHIVE_PATH)

    extract_dir = DATA_DIR / "extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        archive.extractall(extract_dir, filter="data")

    members = [p for p in extract_dir.iterdir() if p.is_dir()]
    if not members:
        raise RuntimeError(
            f"Archive {ARCHIVE_PATH} did not contain the expected repository"
        )
    # Tarball root contains a single top-level directory named e.g. rich-14.3.4.
    extracted_root = members[0]

    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    shutil.move(str(extracted_root), str(repo_dir))
    _verify_repository(repo_dir)
    return repo_dir


def _download_tarball(destination: Path) -> None:
    """Download the pinned tarball, preferring urllib and falling back to curl.

    On some macOS Python builds the system CA bundle is not wired into
    :mod:`urllib`, which makes HTTPS fail with a certificate error even when
    the certificate is valid. As a fallback the download is retried with the
    ``curl`` binary, which uses the platform's trust store.
    """
    try:
        with urllib.request.urlopen(TARBALL_URL, timeout=120) as response:
            data = response.read()
    except urllib.error.URLError:
        data = None
    except OSError:
        data = None
    if data is None:
        import subprocess

        result = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--location", TARBALL_URL],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to download {TARBALL_URL}: {result.stderr.decode() or 'unknown error'}. "
                f"Network access is required on the first run."
            )
        data = result.stdout
    destination.write_bytes(data)


def _verify_repository(repo_dir: Path) -> None:
    from repolens.scanner import RepositoryScanner

    count = len(RepositoryScanner(repo_dir).discover_python_files())
    if count < MIN_PYTHON_FILES:
        raise RuntimeError(
            f"Unexpected repository at {repo_dir}: found only {count} Python "
            f"files (expected >= {MIN_PYTHON_FILES}). "
            f"Is this really {REPOSITORY_NAME} @ {REPOSITORY_REF}?"
        )


def count_python_files(repo_dir: Path) -> int:
    """Return the number of Python files the scanner discovers in the repo."""
    from repolens.scanner import RepositoryScanner

    return len(RepositoryScanner(repo_dir).discover_python_files())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _build_searchers(repo_dir: Path):
    from repolens.local_embeddings import LocalEmbeddingProvider
    from repolens.retrieval import FusionStrategy, HybridSearcher
    from repolens.search import CodeSearcher
    from repolens.semantic_search import SemanticSearcher

    provider = LocalEmbeddingProvider()

    print(f"Building lexical searcher ...")
    t0 = time.perf_counter()
    lexical = CodeSearcher(repo_dir)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    print(f"Building local semantic searcher (model: {provider.dimensions} dims) ...")
    print("  (first run downloads the embedding model)")
    t0 = time.perf_counter()
    semantic = SemanticSearcher(repo_dir, provider)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    print("Building hybrid searchers ...")
    weighted = HybridSearcher(
        repo_dir,
        lexical_searcher=lexical,
        semantic_searcher=semantic,
        lexical_weight=DEFAULT_LEXICAL_WEIGHT,
        semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
        strategy=FusionStrategy.WEIGHTED,
    )
    rrf = HybridSearcher(
        repo_dir,
        lexical_searcher=lexical,
        semantic_searcher=semantic,
        strategy=FusionStrategy.RRF,
        rrf_k=DEFAULT_RRF_K,
    )
    return provider, lexical, semantic, weighted, rrf


def _evaluate(searcher, cases, repo_dir: Path, k: int):
    from repolens.evaluation import EvaluationRunner

    return EvaluationRunner(repo_dir, searcher=searcher).evaluate(cases, k=k)


def _report_dict(report, label: str, k: int) -> dict:
    return {
        "label": label,
        "precision@5": report.mean_precision_at_k,
        "recall@5": report.mean_recall_at_k,
        "mrr": report.mean_reciprocal_rank,
        "num_cases": report.num_cases,
        "k": k,
    }


def _render_table(results: list[dict]) -> str:
    width = 76
    lines: list[str] = []
    lines.append("=" * width)
    lines.append("  Real-World Repository Retrieval Benchmark (Milestone 11)")
    lines.append("=" * width)
    header = f"  {'Strategy':<24}" + "".join(
        f"{m:>17}" for m in ("Precision@5", "Recall@5", "MRR")
    )
    lines.append(header)
    lines.append("  " + "-" * (width - 2))
    for r in results:
        lines.append(
            f"  {r['label']:<24}"
            f"{r['precision@5']:>17.4f}"
            f"{r['recall@5']:>17.4f}"
            f"{r['mrr']:>17.4f}"
        )
    lines.append("=" * width)
    return "\n".join(lines)


def _print_configuration(repo_dir: Path, num_queries: int, provider) -> None:
    print()
    print("Configuration")
    print("-" * 76)
    print(f"  {'Repository:':<28} {REPOSITORY_NAME}")
    print(f"  {'URL:':<28} {REPOSITORY_URL}")
    print(f"  {'Ref / tag:':<28} {REPOSITORY_REF}")
    print(f"  {'Commit:':<28} {REPOSITORY_COMMIT}")
    print(f"  {'Python files (scanned):':<28} {count_python_files(repo_dir)}")
    print(f"  {'Evaluation queries:':<28} {num_queries}")
    print(f"  {'Embedding provider:':<28} Local (FastEmbed)")
    print(f"  {'Embedding model:':<28} {provider._model}")
    print(f"  {'Embedding dimensions:':<28} {provider.dimensions}")
    print(f"  {'Hybrid weights:':<28} lexical={DEFAULT_LEXICAL_WEIGHT}, semantic={DEFAULT_SEMANTIC_WEIGHT}")
    print(f"  {'RRF constant:':<28} {DEFAULT_RRF_K}")
    print(f"  {'k (cutoff):':<28} {DEFAULT_K}")


def run_benchmark(repo_dir: Path, *, weight_sweep: bool = False) -> int:
    """Run the four-strategy evaluation and print the report.

    Returns a shell exit code (0 on success).
    """
    cases = load_cases()
    print(f"Loaded {len(cases)} evaluation queries.")
    repo_dir = ensure_repository(repo_dir)

    provider, lexical, semantic, weighted, rrf = _build_searchers(repo_dir)

    print()
    print("Evaluating strategies ...")
    t0 = time.perf_counter()
    lex_report = _evaluate(lexical, cases, repo_dir, DEFAULT_K)
    sem_report = _evaluate(semantic, cases, repo_dir, DEFAULT_K)
    wh_report = _evaluate(weighted, cases, repo_dir, DEFAULT_K)
    rrf_report = _evaluate(rrf, cases, repo_dir, DEFAULT_K)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    results = [
        _report_dict(lex_report, "LEXICAL", DEFAULT_K),
        _report_dict(sem_report, "LOCAL SEMANTIC", DEFAULT_K),
        _report_dict(wh_report, "WEIGHTED HYBRID (0.5/0.5)", DEFAULT_K),
        _report_dict(rrf_report, "RRF", DEFAULT_K),
    ]

    _print_configuration(repo_dir, len(cases), provider)
    print()
    print(_render_table(results))

    _print_per_query(lex_report, sem_report, wh_report, rrf_report, cases)
    _print_summary(results)

    if weight_sweep:
        _run_weight_sweep(repo_dir, lexical, semantic, cases, provider)

    return 0


def _print_per_query(lex, sem, wh, rrf, cases) -> None:
    print("-" * 100)
    print("  Per-Query Breakdown")
    print("-" * 100)
    header = f"  {'Query':<44}" + "".join(
        f" {label + ' RR':>11}" for label in ("Lex", "Sem", "WHyb", "RRF")
    )
    print(header)
    print("  " + "-" * 98)
    reports = {"Lex": lex, "Sem": sem, "WHyb": wh, "RRF": rrf}
    for i, case in enumerate(cases):
        q = case.query if len(case.query) <= 42 else case.query[:39] + "..."
        row = f"  {q:<44}"
        for label in ("Lex", "Sem", "WHyb", "RRF"):
            rr = reports[label].case_evaluations[i].reciprocal_rank
            row += f" {rr:>11.4f}"
        print(row)
    print()


def _print_summary(results: list[dict]) -> None:
    print("=" * 76)
    print("  Summary")
    print("=" * 76)
    for r in results:
        print(
            f"  {r['label']:<24} P@5: {r['precision@5']:.4f}"
            f"  R@5: {r['recall@5']:.4f}  MRR: {r['mrr']:.4f}"
        )
    print()
    for metric_key, metric_label in [
        ("precision@5", "Precision@5"),
        ("recall@5", "Recall@5"),
        ("mrr", "MRR"),
    ]:
        best = max(results, key=lambda r: r[metric_key])
        print(f"  Best {metric_label}: {best['label']} ({best[metric_key]:.4f})")
    print()


def _run_weight_sweep(repo_dir, lexical, semantic, cases, provider) -> None:
    from repolens.retrieval import FusionStrategy, HybridSearcher

    print()
    print("=" * 76)
    print("  EXPLORATORY WEIGHT SWEEP (not part of the default configuration)")
    print("=" * 76)
    print(
        "  This sweep tunes weights against this benchmark and is NOT an\n"
        "  unbiased estimate. The default weighted-hybrid stays 0.5/0.5."
    )
    print()
    sweep = [
        (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4),
        (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9),
    ]
    print(f"  {'Lex Wt':>8} {'Sem Wt':>8} {'Precision@5':>12} {'Recall@5':>10} {'MRR':>10}")
    for lw, sw in sweep:
        hybrid = HybridSearcher(
            repo_dir,
            lexical_searcher=lexical,
            semantic_searcher=semantic,
            lexical_weight=lw,
            semantic_weight=sw,
            strategy=FusionStrategy.WEIGHTED,
        )
        report = _evaluate(hybrid, cases, repo_dir, DEFAULT_K)
        print(
            f"  {lw:>8.1f} {sw:>8.1f}"
            f" {report.mean_precision_at_k:>12.4f}"
            f" {report.mean_recall_at_k:>10.4f}"
            f" {report.mean_reciprocal_rank:>10.4f}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if "--help" in argv or "-h" in argv:
        print(_module_help())
        return 0

    weight_sweep = "--weight-sweep" in argv
    repo_dir = REPO_DIR
    if "--repo-dir" in argv:
        idx = argv.index("--repo-dir")
        if idx + 1 >= len(argv):
            print("ERROR: --repo-dir requires a path argument", file=sys.stderr)
            return 2
        repo_dir = Path(argv[idx + 1])

    missing = _prereq_check()
    if missing:
        print("ERROR: missing prerequisites for the real-repo benchmark:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print(
            "Fix the prerequisites, then re-run: python -m benchmarks.real_repo",
            file=sys.stderr,
        )
        return 2

    try:
        return run_benchmark(repo_dir, weight_sweep=weight_sweep)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
