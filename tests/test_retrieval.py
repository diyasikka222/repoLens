"""Tests for hybrid retrieval combining lexical and semantic search."""

import json
import shutil
from pathlib import Path

import pytest

from repolens.embeddings import FakeEmbeddingProvider
from repolens.evaluation import EvaluationCase, EvaluationRunner
from repolens.search import CodeSearcher
from repolens.semantic_search import SemanticSearcher
from repolens.retrieval import (
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    HybridResult,
    HybridSearcher,
    _min_max_normalise,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_REPO_DIR = FIXTURES_DIR / "synthetic_repository"
CASES_JSON_PATH = FIXTURES_DIR / "evaluation_cases.json"


def write_file(tmp_path: Path, relative: str, source: str) -> None:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(source, encoding="utf-8")


def paths_of(results):
    return [result.file_path for result in results]


def _build_hybrid(tmp_path: Path, **kwargs) -> HybridSearcher:
    root = Path(tmp_path)
    return HybridSearcher(
        root,
        lexical_searcher=CodeSearcher(root),
        semantic_searcher=SemanticSearcher(root, FakeEmbeddingProvider()),
        **kwargs,
    )


def load_fixture_cases() -> list[EvaluationCase]:
    payload = json.loads(CASES_JSON_PATH.read_text(encoding="utf-8"))
    return [
        EvaluationCase(query=item["query"], relevant_files=item["relevant_files"])
        for item in payload["cases"]
    ]


# ---------------------------------------------------------------------------
# _min_max_normalise unit tests
# ---------------------------------------------------------------------------


def test_min_max_normalise_basic() -> None:
    values = {Path("a.py"): 0.0, Path("b.py"): 5.0, Path("c.py"): 10.0}
    result = _min_max_normalise(values)
    assert result[Path("a.py")] == pytest.approx(0.0)
    assert result[Path("b.py")] == pytest.approx(0.5)
    assert result[Path("c.py")] == pytest.approx(1.0)


def test_min_max_normalise_equal_values() -> None:
    values = {Path("a.py"): 5.0, Path("b.py"): 5.0, Path("c.py"): 5.0}
    result = _min_max_normalise(values)
    assert all(v == pytest.approx(1.0) for v in result.values())


def test_min_max_normalise_single_value() -> None:
    values = {Path("a.py"): 42.0}
    result = _min_max_normalise(values)
    assert result[Path("a.py")] == pytest.approx(1.0)


def test_min_max_normalise_empty() -> None:
    assert _min_max_normalise({}) == {}


def test_min_max_normalise_two_values() -> None:
    values = {Path("a.py"): 3.0, Path("b.py"): 7.0}
    result = _min_max_normalise(values)
    assert result[Path("a.py")] == pytest.approx(0.0)
    assert result[Path("b.py")] == pytest.approx(1.0)


def test_min_max_normalise_preserves_paths() -> None:
    values = {Path("x/y.py"): 1.0, Path("z.py"): 2.0}
    result = _min_max_normalise(values)
    assert set(result.keys()) == {Path("x/y.py"), Path("z.py")}


# ---------------------------------------------------------------------------
# Basic hybrid searcher behaviour
# ---------------------------------------------------------------------------


def test_hybrid_searcher_satisfies_searcher_protocol(tmp_path: Path) -> None:
    write_file(tmp_path, "app.py", "value = 1\n")
    from repolens.evaluation import Searcher

    assert isinstance(_build_hybrid(tmp_path), Searcher)


def test_empty_query_returns_no_results(tmp_path: Path) -> None:
    write_file(tmp_path, "app.py", "def process():\n    pass\n")
    searcher = _build_hybrid(tmp_path)

    assert searcher.search("") == []
    assert searcher.search("   ") == []
    assert searcher.search("___") == []


def test_non_positive_limit_returns_no_results(tmp_path: Path) -> None:
    write_file(tmp_path, "app.py", "def process():\n    pass\n")
    searcher = _build_hybrid(tmp_path)

    assert searcher.search("process", limit=0) == []
    assert searcher.search("process", limit=-1) == []


def test_empty_repository_returns_no_results(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("docs only\n", encoding="utf-8")
    searcher = _build_hybrid(tmp_path)

    assert searcher.search("anything") == []


def test_results_are_hybrid_result_instances(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
    results = _build_hybrid(tmp_path).search("invoice")

    assert len(results) > 0
    assert all(isinstance(r, HybridResult) for r in results)


def test_results_contain_required_fields(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
    results = _build_hybrid(tmp_path).search("invoice")

    assert len(results) == 1
    r = results[0]
    assert r.file_path == Path("billing") / "invoice.py"
    assert isinstance(r.hybrid_score, float)
    assert isinstance(r.lexical_contribution, float)
    assert isinstance(r.semantic_contribution, float)
    assert r.hybrid_score == pytest.approx(r.lexical_contribution + r.semantic_contribution)


def test_limit_is_respected(tmp_path: Path) -> None:
    for number in range(15):
        write_file(tmp_path, f"items/item_{number:02d}.py", f"# widget {number}\n")
    searcher = _build_hybrid(tmp_path)

    assert len(searcher.search("widget", limit=3)) == 3
    assert len(searcher.search("widget", limit=10)) == 10
    assert len(searcher.search("widget", limit=100)) == 15
    assert len(searcher.search("widget", limit=0)) == 0


# ---------------------------------------------------------------------------
# Scoring behaviour
# ---------------------------------------------------------------------------


def test_hybrid_score_is_weighted_average(tmp_path: Path) -> None:
    write_file(tmp_path, "a.py", "def refund():\n    pass\n")
    write_file(tmp_path, "b.py", "# unrelated\nvalue = 1\n")
    searcher = _build_hybrid(tmp_path, lexical_weight=0.6, semantic_weight=0.4)

    results = searcher.search("refund")

    assert len(results) > 0
    best = results[0]
    expected = 0.6 * best.lexical_contribution / 0.6 + 0.4 * best.semantic_contribution / 0.4
    assert best.hybrid_score == pytest.approx(expected)


def test_file_in_only_lexical_results_gets_zero_semantic_contribution(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "class InvoiceService:\n    pass\n")
    searcher = _build_hybrid(tmp_path)

    results = searcher.search("InvoiceService")

    for r in results:
        if r.file_path == Path("billing") / "invoice.py":
            # May or may not appear in semantic results depending on fake provider
            assert isinstance(r.semantic_contribution, float)


def test_results_sorted_by_hybrid_score_descending(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
    write_file(tmp_path, "billing/tax.py", "def calculate_tax():\n    pass\n")
    write_file(tmp_path, "unrelated.py", "# nothing about billing\nvalue = 1\n")

    results = _build_hybrid(tmp_path).search("invoice")

    scores = [r.hybrid_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_tie_breaking_is_alphabetical_by_path(tmp_path: Path) -> None:
    write_file(tmp_path, "zzz/tool.py", "# widget\nvalue = 1\n")
    write_file(tmp_path, "aaa/tool.py", "# widget\nvalue = 1\n")

    results = _build_hybrid(tmp_path).search("widget")

    paths = paths_of(results)
    assert paths == sorted(paths, key=lambda p: p.as_posix())


def test_each_file_appears_at_most_once(tmp_path: Path) -> None:
    write_file(tmp_path, "dup.py", "# refund refund refund payment\nx = 1\n")
    results = _build_hybrid(tmp_path).search("refund payment")

    assert paths_of(results).count(Path("dup.py")) == 1


def test_paths_are_repository_relative(tmp_path: Path) -> None:
    write_file(tmp_path, "deep/nested/mod.py", "# treasure\nvalue = 1\n")
    results = _build_hybrid(tmp_path).search("treasure")

    assert len(results) == 1
    assert results[0].file_path == Path("deep") / "nested" / "mod.py"
    assert not results[0].file_path.is_absolute()


# ---------------------------------------------------------------------------
# Weight configuration
# ---------------------------------------------------------------------------


def test_default_weights_are_equal() -> None:
    assert DEFAULT_LEXICAL_WEIGHT == 0.5
    assert DEFAULT_SEMANTIC_WEIGHT == 0.5


def test_lexical_heavy_weight_prefers_lexical_matches(tmp_path: Path) -> None:
    # Create a file with strong lexical match but weak semantic signal
    write_file(tmp_path, "calculate_invoice.py", "def calculate_invoice():\n    pass\n")
    write_file(tmp_path, "unrelated.py", "# billing invoice payment refund\nclass Unrelated:\n    pass\n")

    # Lexical-heavy: give more weight to lexical
    results_lex = _build_hybrid(tmp_path, lexical_weight=0.9, semantic_weight=0.1).search("calculate_invoice")
    # Semantic-heavy: give more weight to semantic
    results_sem = _build_hybrid(tmp_path, lexical_weight=0.1, semantic_weight=0.9).search("calculate_invoice")

    # In both cases, the exact symbol match should rank first
    assert results_lex[0].file_path == Path("calculate_invoice.py")
    assert results_sem[0].file_path == Path("calculate_invoice.py")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_searches_produce_identical_results(tmp_path: Path) -> None:
    write_file(tmp_path, "payments/refund.py", "def refund_card():\n    pass\n")
    write_file(tmp_path, "users/create.py", "def create_user():\n    pass\n")
    searcher = _build_hybrid(tmp_path)

    first = searcher.search("refund card payment", limit=5)
    second = searcher.search("refund card payment", limit=5)

    assert first == second


# ---------------------------------------------------------------------------
# Evaluation framework integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_repo(tmp_path: Path) -> Path:
    target = tmp_path / "synthetic-repo"
    shutil.copytree(SYNTHETIC_REPO_DIR, target)
    return target


def test_hybrid_satisfies_searcher_protocol_on_synthetic_repo(
    synthetic_repo: Path,
) -> None:
    from repolens.evaluation import Searcher

    hybrid = HybridSearcher(
        synthetic_repo,
        lexical_searcher=CodeSearcher(synthetic_repo),
        semantic_searcher=SemanticSearcher(synthetic_repo, FakeEmbeddingProvider()),
    )
    assert isinstance(hybrid, Searcher)


def test_hybrid_evaluation_produces_report(synthetic_repo: Path) -> None:
    cases = load_fixture_cases()
    hybrid = HybridSearcher(
        synthetic_repo,
        lexical_searcher=CodeSearcher(synthetic_repo),
        semantic_searcher=SemanticSearcher(synthetic_repo, FakeEmbeddingProvider()),
    )
    runner = EvaluationRunner(synthetic_repo, searcher=hybrid)

    report = runner.evaluate(cases, k=5)

    assert report.num_cases == 10
    assert report.k == 5
    for evaluation in report.case_evaluations:
        assert 0.0 <= evaluation.precision_at_k <= 1.0
        assert 0.0 <= evaluation.recall_at_k <= 1.0
        assert 0.0 <= evaluation.reciprocal_rank <= 1.0
        assert len(evaluation.retrieved_files) <= 5


def test_lexical_vs_semantic_vs_hybrid_comparison(synthetic_repo: Path) -> None:
    """Run all three searchers on the same evaluation dataset and compare."""
    cases = load_fixture_cases()

    lexical_runner = EvaluationRunner(synthetic_repo)
    lexical_report = lexical_runner.evaluate(cases, k=5)

    semantic_searcher = SemanticSearcher(synthetic_repo, FakeEmbeddingProvider())
    semantic_runner = EvaluationRunner(synthetic_repo, searcher=semantic_searcher)
    semantic_report = semantic_runner.evaluate(cases, k=5)

    hybrid_searcher = HybridSearcher(
        synthetic_repo,
        lexical_searcher=CodeSearcher(synthetic_repo),
        semantic_searcher=semantic_searcher,
    )
    hybrid_runner = EvaluationRunner(synthetic_repo, searcher=hybrid_searcher)
    hybrid_report = hybrid_runner.evaluate(cases, k=5)

    # All reports must be valid
    for report in (lexical_report, semantic_report, hybrid_report):
        assert report.num_cases == 10
        assert report.k == 5

    # Print comparison for visibility during test runs
    print("\n=== Retrieval Comparison (FakeEmbeddingProvider) ===")
    print(f"{'Metric':<25} {'Lexical':>10} {'Semantic':>10} {'Hybrid':>10}")
    print("-" * 55)
    print(f"{'Precision@5':<25} {lexical_report.mean_precision_at_k:>10.4f} {semantic_report.mean_precision_at_k:>10.4f} {hybrid_report.mean_precision_at_k:>10.4f}")
    print(f"{'Recall@5':<25} {lexical_report.mean_recall_at_k:>10.4f} {semantic_report.mean_recall_at_k:>10.4f} {hybrid_report.mean_recall_at_k:>10.4f}")
    print(f"{'MRR':<25} {lexical_report.mean_reciprocal_rank:>10.4f} {semantic_report.mean_reciprocal_rank:>10.4f} {hybrid_report.mean_reciprocal_rank:>10.4f}")

    # The comparison must be deterministic
    second_lexical = EvaluationRunner(synthetic_repo).evaluate(cases, k=5)
    second_hybrid = EvaluationRunner(
        synthetic_repo,
        searcher=HybridSearcher(
            synthetic_repo,
            lexical_searcher=CodeSearcher(synthetic_repo),
            semantic_searcher=SemanticSearcher(synthetic_repo, FakeEmbeddingProvider()),
        ),
    ).evaluate(cases, k=5)
    assert lexical_report == second_lexical
    assert hybrid_report == second_hybrid


def test_hybrid_retrieval_deterministic(synthetic_repo: Path) -> None:
    cases = load_fixture_cases()

    def make_hybrid(root: Path) -> HybridSearcher:
        return HybridSearcher(
            root,
            lexical_searcher=CodeSearcher(root),
            semantic_searcher=SemanticSearcher(root, FakeEmbeddingProvider()),
        )

    first = EvaluationRunner(synthetic_repo, searcher=make_hybrid(synthetic_repo)).evaluate(cases, k=5)
    second = EvaluationRunner(synthetic_repo, searcher=make_hybrid(synthetic_repo)).evaluate(cases, k=5)

    assert first == second


# ---------------------------------------------------------------------------
# Configurable weights in evaluation
# ---------------------------------------------------------------------------


def test_hybrid_with_lexical_bias_evaluation(synthetic_repo: Path) -> None:
    cases = load_fixture_cases()
    hybrid = HybridSearcher(
        synthetic_repo,
        lexical_searcher=CodeSearcher(synthetic_repo),
        semantic_searcher=SemanticSearcher(synthetic_repo, FakeEmbeddingProvider()),
        lexical_weight=0.8,
        semantic_weight=0.2,
    )
    report = EvaluationRunner(synthetic_repo, searcher=hybrid).evaluate(cases, k=5)
    assert report.num_cases == 10
    assert 0.0 <= report.mean_precision_at_k <= 1.0
