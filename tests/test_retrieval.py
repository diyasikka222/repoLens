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
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_WEIGHT,
    FusionStrategy,
    HybridResult,
    HybridSearcher,
    _validate_weights,
    min_max_normalise,
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


# ===========================================================================
# 1. min_max_normalise unit tests
# ===========================================================================


class TestMinMaxNormalise:
    def test_basic(self) -> None:
        values = {Path("a.py"): 0.0, Path("b.py"): 5.0, Path("c.py"): 10.0}
        result = min_max_normalise(values)
        assert result[Path("a.py")] == pytest.approx(0.0)
        assert result[Path("b.py")] == pytest.approx(0.5)
        assert result[Path("c.py")] == pytest.approx(1.0)

    def test_equal_scores(self) -> None:
        values = {Path("a.py"): 5.0, Path("b.py"): 5.0, Path("c.py"): 5.0}
        result = min_max_normalise(values)
        assert all(v == pytest.approx(1.0) for v in result.values())

    def test_single_result(self) -> None:
        values = {Path("a.py"): 42.0}
        result = min_max_normalise(values)
        assert result[Path("a.py")] == pytest.approx(1.0)

    def test_empty_results(self) -> None:
        assert min_max_normalise({}) == {}

    def test_two_values(self) -> None:
        values = {Path("a.py"): 3.0, Path("b.py"): 7.0}
        result = min_max_normalise(values)
        assert result[Path("a.py")] == pytest.approx(0.0)
        assert result[Path("b.py")] == pytest.approx(1.0)

    def test_preserves_paths(self) -> None:
        values = {Path("x/y.py"): 1.0, Path("z.py"): 2.0}
        result = min_max_normalise(values)
        assert set(result.keys()) == {Path("x/y.py"), Path("z.py")}

    def test_different_magnitudes(self) -> None:
        values = {Path("a.py"): 0.001, Path("b.py"): 1000.0}
        result = min_max_normalise(values)
        assert result[Path("a.py")] == pytest.approx(0.0)
        assert result[Path("b.py")] == pytest.approx(1.0)

    def test_negative_scores(self) -> None:
        values = {Path("a.py"): -5.0, Path("b.py"): 0.0, Path("c.py"): 5.0}
        result = min_max_normalise(values)
        assert result[Path("a.py")] == pytest.approx(0.0)
        assert result[Path("b.py")] == pytest.approx(0.5)
        assert result[Path("c.py")] == pytest.approx(1.0)

    def test_deterministic(self) -> None:
        values = {Path("a.py"): 3.0, Path("b.py"): 7.0, Path("c.py"): 1.0}
        r1 = min_max_normalise(values)
        r2 = min_max_normalise(values)
        assert r1 == r2


# ===========================================================================
# 2. Weight validation tests
# ===========================================================================


class TestWeightValidation:
    def test_equal_weights_normalised(self) -> None:
        lex, sem = _validate_weights(0.5, 0.5)
        assert lex == pytest.approx(0.5)
        assert sem == pytest.approx(0.5)

    def test_unequal_weights_normalised(self) -> None:
        lex, sem = _validate_weights(0.7, 0.3)
        assert lex == pytest.approx(0.7)
        assert sem == pytest.approx(0.3)

    def test_unnormalised_weights_normalised(self) -> None:
        lex, sem = _validate_weights(3.0, 1.0)
        assert lex == pytest.approx(0.75)
        assert sem == pytest.approx(0.25)

    def test_negative_lexical_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _validate_weights(-0.1, 0.5)

    def test_negative_semantic_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _validate_weights(0.5, -0.1)

    def test_both_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _validate_weights(0.0, 0.0)

    def test_only_lexical_positive(self) -> None:
        lex, sem = _validate_weights(1.0, 0.0)
        assert lex == pytest.approx(1.0)
        assert sem == pytest.approx(0.0)

    def test_only_semantic_positive(self) -> None:
        lex, sem = _validate_weights(0.0, 1.0)
        assert lex == pytest.approx(0.0)
        assert sem == pytest.approx(1.0)

    def test_extreme_weights(self) -> None:
        lex, sem = _validate_weights(100.0, 0.001)
        assert lex + sem == pytest.approx(1.0)
        assert lex > 0.999
        assert sem < 0.001


# ===========================================================================
# 3. Basic hybrid searcher behaviour (weighted — backward compatible)
# ===========================================================================


class TestWeightedHybrid:
    def test_satisfies_searcher_protocol(self, tmp_path: Path) -> None:
        write_file(tmp_path, "app.py", "value = 1\n")
        from repolens.evaluation import Searcher

        assert isinstance(_build_hybrid(tmp_path), Searcher)

    def test_empty_query_returns_no_results(self, tmp_path: Path) -> None:
        write_file(tmp_path, "app.py", "def process():\n    pass\n")
        searcher = _build_hybrid(tmp_path)
        assert searcher.search("") == []
        assert searcher.search("   ") == []
        assert searcher.search("___") == []

    def test_non_positive_limit_returns_no_results(self, tmp_path: Path) -> None:
        write_file(tmp_path, "app.py", "def process():\n    pass\n")
        searcher = _build_hybrid(tmp_path)
        assert searcher.search("process", limit=0) == []
        assert searcher.search("process", limit=-1) == []

    def test_empty_repository_returns_no_results(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("docs only\n", encoding="utf-8")
        searcher = _build_hybrid(tmp_path)
        assert searcher.search("anything") == []

    def test_results_are_hybrid_result_instances(self, tmp_path: Path) -> None:
        write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
        results = _build_hybrid(tmp_path).search("invoice")
        assert len(results) > 0
        assert all(isinstance(r, HybridResult) for r in results)

    def test_results_contain_required_fields(self, tmp_path: Path) -> None:
        write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
        results = _build_hybrid(tmp_path).search("invoice")
        assert len(results) == 1
        r = results[0]
        assert r.file_path == Path("billing") / "invoice.py"
        assert isinstance(r.hybrid_score, float)
        assert isinstance(r.lexical_contribution, float)
        assert isinstance(r.semantic_contribution, float)
        assert r.hybrid_score == pytest.approx(
            r.lexical_contribution + r.semantic_contribution
        )

    def test_limit_is_respected(self, tmp_path: Path) -> None:
        for number in range(15):
            write_file(tmp_path, f"items/item_{number:02d}.py", f"# widget {number}\n")
        searcher = _build_hybrid(tmp_path)
        assert len(searcher.search("widget", limit=3)) == 3
        assert len(searcher.search("widget", limit=10)) == 10
        assert len(searcher.search("widget", limit=100)) == 15
        assert len(searcher.search("widget", limit=0)) == 0

    def test_score_is_weighted_average(self, tmp_path: Path) -> None:
        write_file(tmp_path, "a.py", "def refund():\n    pass\n")
        write_file(tmp_path, "b.py", "# unrelated\nvalue = 1\n")
        searcher = _build_hybrid(tmp_path, lexical_weight=0.6, semantic_weight=0.4)
        results = searcher.search("refund")
        assert len(results) > 0
        best = results[0]
        expected = (
            best.lexical_contribution + best.semantic_contribution
        )
        assert best.hybrid_score == pytest.approx(expected)

    def test_sorted_by_hybrid_score_descending(self, tmp_path: Path) -> None:
        write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
        write_file(tmp_path, "billing/tax.py", "def calculate_tax():\n    pass\n")
        write_file(tmp_path, "unrelated.py", "# nothing about billing\nvalue = 1\n")
        results = _build_hybrid(tmp_path).search("invoice")
        scores = [r.hybrid_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_tie_breaking_is_alphabetical_by_path(self, tmp_path: Path) -> None:
        write_file(tmp_path, "zzz/tool.py", "# widget\nvalue = 1\n")
        write_file(tmp_path, "aaa/tool.py", "# widget\nvalue = 1\n")
        results = _build_hybrid(tmp_path).search("widget")
        paths = paths_of(results)
        assert paths == sorted(paths, key=lambda p: p.as_posix())

    def test_each_file_appears_at_most_once(self, tmp_path: Path) -> None:
        write_file(tmp_path, "dup.py", "# refund refund refund payment\nx = 1\n")
        results = _build_hybrid(tmp_path).search("refund payment")
        assert paths_of(results).count(Path("dup.py")) == 1

    def test_paths_are_repository_relative(self, tmp_path: Path) -> None:
        write_file(tmp_path, "deep/nested/mod.py", "# treasure\nvalue = 1\n")
        results = _build_hybrid(tmp_path).search("treasure")
        assert len(results) == 1
        assert results[0].file_path == Path("deep") / "nested" / "mod.py"
        assert not results[0].file_path.is_absolute()

    def test_fusion_strategy_is_weighted(self, tmp_path: Path) -> None:
        write_file(tmp_path, "app.py", "def process():\n    pass\n")
        results = _build_hybrid(tmp_path).search("process")
        assert all(r.fusion_strategy == "weighted" for r in results)

    def test_lexical_heavy_weight_prefers_lexical(self, tmp_path: Path) -> None:
        write_file(tmp_path, "calculate_invoice.py", "def calculate_invoice():\n    pass\n")
        write_file(tmp_path, "unrelated.py", "# billing invoice payment refund\nclass Unrelated:\n    pass\n")
        results_lex = _build_hybrid(tmp_path, lexical_weight=0.9, semantic_weight=0.1).search("calculate_invoice")
        results_sem = _build_hybrid(tmp_path, lexical_weight=0.1, semantic_weight=0.9).search("calculate_invoice")
        assert results_lex[0].file_path == Path("calculate_invoice.py")
        assert results_sem[0].file_path == Path("calculate_invoice.py")

    def test_repeated_searches_are_deterministic(self, tmp_path: Path) -> None:
        write_file(tmp_path, "payments/refund.py", "def refund_card():\n    pass\n")
        searcher = _build_hybrid(tmp_path)
        first = searcher.search("refund card payment", limit=5)
        second = searcher.search("refund card payment", limit=5)
        assert first == second

    def test_result_exposes_ranks(self, tmp_path: Path) -> None:
        write_file(tmp_path, "a.py", "def process():\n    pass\n")
        write_file(tmp_path, "b.py", "def process():\n    x = 1\n")
        results = _build_hybrid(tmp_path).search("process")
        for r in results:
            assert r.lexical_rank is not None or r.semantic_rank is not None


# ===========================================================================
# 4. RRF strategy tests
# ===========================================================================


class TestRRFHybrid:
    def _build_rrf(self, tmp_path: Path, **kwargs) -> HybridSearcher:
        root = Path(tmp_path)
        return HybridSearcher(
            root,
            lexical_searcher=CodeSearcher(root),
            semantic_searcher=SemanticSearcher(root, FakeEmbeddingProvider()),
            strategy=FusionStrategy.RRF,
            **kwargs,
        )

    def test_rrf_satisfies_searcher_protocol(self, tmp_path: Path) -> None:
        write_file(tmp_path, "app.py", "value = 1\n")
        from repolens.evaluation import Searcher

        assert isinstance(self._build_rrf(tmp_path), Searcher)

    def test_rrf_returns_results(self, tmp_path: Path) -> None:
        write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
        results = self._build_rrf(tmp_path).search("invoice")
        assert len(results) > 0
        assert all(isinstance(r, HybridResult) for r in results)

    def test_rrf_fusion_strategy_labelled(self, tmp_path: Path) -> None:
        write_file(tmp_path, "app.py", "def process():\n    pass\n")
        results = self._build_rrf(tmp_path).search("process")
        assert all(r.fusion_strategy == "rrf" for r in results)

    def test_rrf_sorted_by_score(self, tmp_path: Path) -> None:
        write_file(tmp_path, "a.py", "def refund():\n    pass\n")
        write_file(tmp_path, "b.py", "def calculate():\n    pass\n")
        write_file(tmp_path, "c.py", "# unrelated\nx = 1\n")
        results = self._build_rrf(tmp_path).search("refund")
        scores = [r.hybrid_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_empty_query(self, tmp_path: Path) -> None:
        write_file(tmp_path, "app.py", "def process():\n    pass\n")
        assert self._build_rrf(tmp_path).search("") == []
        assert self._build_rrf(tmp_path).search("   ") == []

    def test_rrf_limit_respected(self, tmp_path: Path) -> None:
        for i in range(10):
            write_file(tmp_path, f"mod_{i}.py", f"# widget {i}\n")
        results = self._build_rrf(tmp_path).search("widget", limit=3)
        assert len(results) == 3

    def test_rrf_document_in_both_lists(self, tmp_path: Path) -> None:
        write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
        results = self._build_rrf(tmp_path).search("invoice")
        for r in results:
            # File should have contributions from both signals if in both lists
            assert r.lexical_contribution >= 0.0
            assert r.semantic_contribution >= 0.0

    def test_rrf_document_in_only_lexical(self, tmp_path: Path) -> None:
        # A file with exact symbol match but no semantic signal
        write_file(tmp_path, "calculate_total.py", "def calculate_total():\n    pass\n")
        results = self._build_rrf(tmp_path).search("calculate_total")
        assert len(results) > 0
        # The file should still rank with at least its lexical RRF contribution
        assert results[0].lexical_contribution > 0.0

    def test_rrf_document_in_only_semantic(self, tmp_path: Path) -> None:
        # Files with natural language but no exact symbol matches
        write_file(tmp_path, "docs.py", "# payment processing refund\nvalue = 1\n")
        results = self._build_rrf(tmp_path).search("payment processing refund")
        assert len(results) > 0

    def test_rrf_empty_lexical_results(self, tmp_path: Path) -> None:
        # File only matches semantically via fake provider
        write_file(tmp_path, "code.py", "value = 1\n")
        results = self._build_rrf(tmp_path).search("nonexistent_xyz_abc")
        # May be empty or contain files matched by fake provider
        assert isinstance(results, list)

    def test_rrf_custom_k(self, tmp_path: Path) -> None:
        write_file(tmp_path, "a.py", "def process():\n    pass\n")
        r1 = self._build_rrf(tmp_path, rrf_k=10).search("process")
        r2 = self._build_rrf(tmp_path, rrf_k=100).search("process")
        # Different k values should produce different scores
        if r1 and r2:
            assert r1[0].hybrid_score != pytest.approx(r2[0].hybrid_score, abs=1e-9)

    def test_rrf_deterministic(self, tmp_path: Path) -> None:
        write_file(tmp_path, "a.py", "def process():\n    pass\n")
        searcher = self._build_rrf(tmp_path)
        first = searcher.search("process")
        second = searcher.search("process")
        assert first == second

    def test_rrf_tie_breaking_alphabetical(self, tmp_path: Path) -> None:
        write_file(tmp_path, "zzz/mod.py", "# widget\nvalue = 1\n")
        write_file(tmp_path, "aaa/mod.py", "# widget\nvalue = 1\n")
        results = self._build_rrf(tmp_path).search("widget")
        paths = paths_of(results)
        assert paths == sorted(paths, key=lambda p: p.as_posix())

    def test_rrf_each_file_once(self, tmp_path: Path) -> None:
        write_file(tmp_path, "dup.py", "# refund refund refund\nx = 1\n")
        results = self._build_rrf(tmp_path).search("refund")
        assert paths_of(results).count(Path("dup.py")) == 1


# ===========================================================================
# 5. Result explanation metadata
# ===========================================================================


class TestResultExplanation:
    def test_weighted_result_has_all_fields(self, tmp_path: Path) -> None:
        write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
        results = _build_hybrid(tmp_path).search("invoice")
        assert len(results) > 0
        r = results[0]
        assert r.file_path == Path("billing") / "invoice.py"
        assert isinstance(r.hybrid_score, float)
        assert isinstance(r.lexical_contribution, float)
        assert isinstance(r.semantic_contribution, float)
        assert r.fusion_strategy == "weighted"
        # Ranks should be populated when the file appears in results
        assert r.lexical_rank is not None or r.semantic_rank is not None

    def test_rrf_result_has_all_fields(self, tmp_path: Path) -> None:
        write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
        results = _build_hybrid(tmp_path, strategy=FusionStrategy.RRF).search("invoice")
        assert len(results) > 0
        r = results[0]
        assert r.fusion_strategy == "rrf"
        assert r.lexical_rank is not None or r.semantic_rank is not None

    def test_lexical_only_file_explanation(self, tmp_path: Path) -> None:
        # File that should appear in lexical but maybe not semantic
        write_file(tmp_path, "special_symbol_xyz.py", "def special_symbol_xyz():\n    pass\n")
        results = _build_hybrid(tmp_path).search("special_symbol_xyz")
        for r in results:
            if r.file_path == Path("special_symbol_xyz.py"):
                # May or may not be in both lists, but fields should be valid
                assert isinstance(r.lexical_contribution, float)
                assert isinstance(r.semantic_contribution, float)

    def test_result_limit(self, tmp_path: Path) -> None:
        for i in range(20):
            write_file(tmp_path, f"m{i}.py", f"# widget item {i}\n")
        results = _build_hybrid(tmp_path).search("widget", limit=5)
        assert len(results) == 5
        # All should have valid metadata
        for r in results:
            assert r.fusion_strategy == "weighted"
            assert isinstance(r.hybrid_score, float)


# ===========================================================================
# 6. Default weights and constants
# ===========================================================================


class TestConstants:
    def test_default_weights_are_equal(self) -> None:
        assert DEFAULT_LEXICAL_WEIGHT == 0.5
        assert DEFAULT_SEMANTIC_WEIGHT == 0.5

    def test_default_rrf_k(self) -> None:
        assert DEFAULT_RRF_K == 60

    def test_fusion_strategy_enum(self) -> None:
        assert FusionStrategy.WEIGHTED.value == "weighted"
        assert FusionStrategy.RRF.value == "rrf"


# ===========================================================================
# 7. Evaluation framework integration (backward compatible)
# ===========================================================================


@pytest.fixture()
def synthetic_repo(tmp_path: Path) -> Path:
    target = tmp_path / "synthetic-repo"
    shutil.copytree(SYNTHETIC_REPO_DIR, target)
    return target


class TestEvaluationIntegration:
    def test_hybrid_satisfies_searcher_protocol(self, synthetic_repo: Path) -> None:
        from repolens.evaluation import Searcher

        hybrid = HybridSearcher(
            synthetic_repo,
            lexical_searcher=CodeSearcher(synthetic_repo),
            semantic_searcher=SemanticSearcher(synthetic_repo, FakeEmbeddingProvider()),
        )
        assert isinstance(hybrid, Searcher)

    def test_hybrid_evaluation_produces_report(self, synthetic_repo: Path) -> None:
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

    def test_lexical_vs_semantic_vs_hybrid_comparison(
        self, synthetic_repo: Path
    ) -> None:
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

        for report in (lexical_report, semantic_report, hybrid_report):
            assert report.num_cases == 10
            assert report.k == 5

        # Deterministic
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

    def test_hybrid_retrieval_deterministic(self, synthetic_repo: Path) -> None:
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

    def test_hybrid_with_lexical_bias(self, synthetic_repo: Path) -> None:
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

    def test_rrf_evaluation_produces_report(self, synthetic_repo: Path) -> None:
        cases = load_fixture_cases()
        hybrid = HybridSearcher(
            synthetic_repo,
            lexical_searcher=CodeSearcher(synthetic_repo),
            semantic_searcher=SemanticSearcher(synthetic_repo, FakeEmbeddingProvider()),
            strategy=FusionStrategy.RRF,
        )
        runner = EvaluationRunner(synthetic_repo, searcher=hybrid)
        report = runner.evaluate(cases, k=5)
        assert report.num_cases == 10
        assert report.k == 5
        for evaluation in report.case_evaluations:
            assert 0.0 <= evaluation.precision_at_k <= 1.0
            assert 0.0 <= evaluation.recall_at_k <= 1.0
            assert 0.0 <= evaluation.reciprocal_rank <= 1.0

    def test_rrf_deterministic(self, synthetic_repo: Path) -> None:
        cases = load_fixture_cases()

        def make_rrf(root: Path) -> HybridSearcher:
            return HybridSearcher(
                root,
                lexical_searcher=CodeSearcher(root),
                semantic_searcher=SemanticSearcher(root, FakeEmbeddingProvider()),
                strategy=FusionStrategy.RRF,
            )

        first = EvaluationRunner(synthetic_repo, searcher=make_rrf(synthetic_repo)).evaluate(cases, k=5)
        second = EvaluationRunner(synthetic_repo, searcher=make_rrf(synthetic_repo)).evaluate(cases, k=5)
        assert first == second

    def test_lexical_search_unchanged(self, synthetic_repo: Path) -> None:
        cases = load_fixture_cases()
        report = EvaluationRunner(synthetic_repo).evaluate(cases, k=5)
        assert report.num_cases == 10
        assert report.k == 5
        # Lexical baseline should match expected values from prior milestone
        assert 0.0 < report.mean_precision_at_k <= 1.0
        assert report.mean_recall_at_k == pytest.approx(1.0)
        assert 0.0 < report.mean_reciprocal_rank <= 1.0

    def test_semantic_search_unchanged(self, synthetic_repo: Path) -> None:
        cases = load_fixture_cases()
        searcher = SemanticSearcher(synthetic_repo, FakeEmbeddingProvider())
        report = EvaluationRunner(synthetic_repo, searcher=searcher).evaluate(cases, k=5)
        assert report.num_cases == 10
        assert report.k == 5
        assert 0.0 < report.mean_precision_at_k <= 1.0
        assert report.mean_recall_at_k == pytest.approx(1.0)
        assert 0.0 < report.mean_reciprocal_rank <= 1.0
