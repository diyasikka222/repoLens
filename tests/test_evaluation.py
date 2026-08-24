"""Tests for the deterministic retrieval evaluation framework."""

import json
import shutil
from pathlib import Path

import pytest

from repolens.evaluation import (
    DEFAULT_K,
    CaseEvaluation,
    EvaluationCase,
    EvaluationReport,
    EvaluationRunner,
    first_relevant_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from repolens.scanner import RepositoryScanner

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC_REPO_DIR = FIXTURES_DIR / "synthetic_repository"
CASES_JSON_PATH = FIXTURES_DIR / "evaluation_cases.json"


def write_file(tmp_path: Path, relative: str, source: str) -> None:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(source, encoding="utf-8")


def load_fixture_cases() -> list[EvaluationCase]:
    payload = json.loads(CASES_JSON_PATH.read_text(encoding="utf-8"))
    return [
        EvaluationCase(query=item["query"], relevant_files=item["relevant_files"])
        for item in payload["cases"]
    ]


# --- Metric helper functions --------------------------------------------------


def test_metric_helpers_on_plain_sequences() -> None:
    retrieved = [Path("a.py"), Path("b.py"), Path("c.py")]

    assert precision_at_k(retrieved, ["b.py"]) == pytest.approx(1 / 3)
    assert precision_at_k([], ["a.py"]) == 0.0
    assert recall_at_k(retrieved, ["b.py", "missing.py"]) == pytest.approx(0.5)
    assert recall_at_k(retrieved, []) == 0.0
    assert first_relevant_rank(retrieved, ["c.py"]) == 3
    assert first_relevant_rank(retrieved, ["absent.py"]) is None
    assert reciprocal_rank(retrieved, ["c.py"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, ["absent.py"]) == 0.0


# --- 1. All retrieved files are relevant ----------------------------------------


def test_all_retrieved_files_are_relevant(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")

    evaluation = EvaluationRunner(tmp_path).evaluate_case(
        EvaluationCase("create_invoice", ["billing/invoice.py"])
    )

    assert evaluation.retrieved_files == (Path("billing/invoice.py"),)
    assert evaluation.precision_at_k == 1.0
    assert evaluation.recall_at_k == 1.0
    assert evaluation.reciprocal_rank == 1.0
    assert evaluation.first_relevant_rank == 1


# --- 2. Some retrieved files are irrelevant --------------------------------------


def test_irrelevant_retrievals_lower_precision(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
    write_file(tmp_path, "notes.py", "# invoice thoughts\nvalue = 1\n")

    evaluation = EvaluationRunner(tmp_path).evaluate_case(
        EvaluationCase("invoice", ["billing/invoice.py"])
    )

    assert evaluation.retrieved_files[0] == Path("billing/invoice.py")
    assert Path("notes.py") in evaluation.retrieved_files
    assert evaluation.precision_at_k == 0.5
    assert evaluation.recall_at_k == 1.0
    assert evaluation.reciprocal_rank == 1.0


# --- 3. No relevant files are retrieved --------------------------------------------


def test_no_relevant_file_retrieved_scores_zero(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
    write_file(tmp_path, "notes.py", "# invoice thoughts\nvalue = 1\n")

    evaluation = EvaluationRunner(tmp_path).evaluate_case(
        EvaluationCase("invoice", ["reports/quarterly.py"])
    )

    assert evaluation.retrieved_files != ()
    assert evaluation.precision_at_k == 0.0
    assert evaluation.recall_at_k == 0.0
    assert evaluation.first_relevant_rank is None


# --- 4. Precision@K -------------------------------------------------------------------


def test_precision_at_k_only_considers_top_k(tmp_path: Path) -> None:
    write_file(tmp_path, "payments/core.py", "def payment_gateway():\n    pass\n")
    write_file(
        tmp_path, "gateway/routes.py", "def route():\n    return 'payment flow'\n"
    )
    write_file(tmp_path, "misc.py", "# payment gateway notes\nx = 1\n")
    write_file(tmp_path, "extra.py", "# payment\ny = 2\n")
    runner = EvaluationRunner(tmp_path)
    case = EvaluationCase("payment gateway", ["payments/core.py"])

    at_two = runner.evaluate_case(case, k=2)
    at_four = runner.evaluate_case(case, k=4)

    assert at_two.retrieved_files[0] == Path("payments/core.py")
    assert at_two.precision_at_k == 0.5
    assert at_four.precision_at_k == 0.25


# --- 5. Recall@K -------------------------------------------------------------------------


def test_recall_at_k_ignores_files_beyond_cutoff(tmp_path: Path) -> None:
    write_file(tmp_path, "payments/core.py", "def payment_gateway():\n    pass\n")
    write_file(
        tmp_path, "gateway/routes.py", "def route():\n    return 'payment flow'\n"
    )
    write_file(tmp_path, "misc.py", "# payment gateway notes\nx = 1\n")
    write_file(tmp_path, "extra.py", "# payment\ny = 2\n")
    runner = EvaluationRunner(tmp_path)
    case = EvaluationCase(
        "payment gateway", ["payments/core.py", "misc.py"]
    )

    at_two = runner.evaluate_case(case, k=2)
    at_four = runner.evaluate_case(case, k=4)

    assert at_two.recall_at_k == 0.5
    assert at_four.recall_at_k == 1.0


# --- 6. MRR when the first result is relevant ------------------------------------------------


def test_reciprocal_rank_when_first_result_is_relevant(tmp_path: Path) -> None:
    write_file(tmp_path, "auth/login.py", "def authenticate():\n    pass\n")

    evaluation = EvaluationRunner(tmp_path).evaluate_case(
        EvaluationCase("authenticate", ["auth/login.py"])
    )

    assert evaluation.first_relevant_rank == 1
    assert evaluation.reciprocal_rank == 1.0


# --- 7. MRR when the first relevant result is rank 2 -------------------------------------------


def test_reciprocal_rank_when_relevant_result_is_second(tmp_path: Path) -> None:
    write_file(tmp_path, "aaa.py", "value = 1  # needle\n")
    write_file(tmp_path, "zzz.py", "value = 1  # needle\n")

    evaluation = EvaluationRunner(tmp_path).evaluate_case(
        EvaluationCase("needle", ["zzz.py"])
    )

    assert evaluation.retrieved_files == (
        Path("aaa.py"),
        Path("zzz.py"),
    )
    assert evaluation.first_relevant_rank == 2
    assert evaluation.reciprocal_rank == 0.5
    assert evaluation.precision_at_k == 0.5
    assert evaluation.recall_at_k == 1.0


# --- 8. MRR when no relevant result exists ---------------------------------------------------------


def test_reciprocal_rank_without_relevant_result_is_zero(tmp_path: Path) -> None:
    write_file(tmp_path, "aaa.py", "value = 1  # needle\n")

    evaluation = EvaluationRunner(tmp_path).evaluate_case(
        EvaluationCase("needle", ["unrelated/elsewhere.py"])
    )

    assert evaluation.first_relevant_rank is None
    assert evaluation.reciprocal_rank == 0.0


# --- 9. Multiple evaluation cases --------------------------------------------------------------------


def test_multiple_cases_aggregate_into_report(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
    write_file(
        tmp_path, "billing/receipts.py", "def generate_receipt():\n    pass\n"
    )
    write_file(tmp_path, "notes.py", "# invoice thoughts\nvalue = 1\n")
    write_file(tmp_path, "aaa.py", "value = 1  # needle\n")
    write_file(tmp_path, "zzz.py", "value = 1  # needle\n")

    cases = [
        EvaluationCase("generate receipt", ["billing/receipts.py"]),
        EvaluationCase("invoice", ["billing/invoice.py"]),
        EvaluationCase("needle", ["zzz.py"]),
    ]

    report = EvaluationRunner(tmp_path).evaluate(cases)

    assert isinstance(report, EvaluationReport)
    assert report.k == DEFAULT_K
    assert report.num_cases == 3
    assert [item.query for item in report.case_evaluations] == [
        "generate receipt",
        "invoice",
        "needle",
    ]
    precisions = [item.precision_at_k for item in report.case_evaluations]
    assert precisions == [1.0, 0.5, 0.5]
    assert report.mean_precision_at_k == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert report.mean_recall_at_k == pytest.approx(1.0)
    ranks = [item.reciprocal_rank for item in report.case_evaluations]
    assert ranks == [1.0, 1.0, 0.5]
    assert report.mean_reciprocal_rank == pytest.approx((1.0 + 1.0 + 0.5) / 3)


# --- 10. Empty relevant-file set -----------------------------------------------------------------------


def test_empty_relevant_file_set_is_handled_gracefully(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")

    evaluation = EvaluationRunner(tmp_path).evaluate_case(
        EvaluationCase("invoice", [])
    )

    assert evaluation.relevant_files == frozenset()
    assert evaluation.retrieved_files != ()
    assert evaluation.precision_at_k == 0.0
    assert evaluation.recall_at_k == 0.0
    assert evaluation.reciprocal_rank == 0.0
    assert evaluation.first_relevant_rank is None

    report = EvaluationRunner(tmp_path).evaluate([EvaluationCase("invoice", [])])
    assert report.num_cases == 1
    assert report.mean_recall_at_k == 0.0
    assert report.mean_reciprocal_rank == 0.0


# --- 11. K larger than the number of results --------------------------------------------------------------


def test_k_larger_than_number_of_results_is_safe(tmp_path: Path) -> None:
    write_file(tmp_path, "widget/models.py", "class Widget:\n    pass\n")
    write_file(tmp_path, "widget/util.py", "# widget helpers\nvalue = 1\n")
    write_file(tmp_path, "widget/views.py", "# widget rendering\nother = 2\n")

    report = EvaluationRunner(tmp_path).evaluate(
        [
            EvaluationCase(
                "widget",
                ["widget/models.py", "widget/views.py"],
            )
        ],
        k=25,
    )

    assert report.k == 25
    evaluation = report.case_evaluations[0]
    assert len(evaluation.retrieved_files) == 3
    assert evaluation.precision_at_k == pytest.approx(2 / 3)
    assert evaluation.recall_at_k == 1.0


# --- 12. Deterministic repeated evaluation ------------------------------------------------------------------


def test_repeated_evaluation_produces_identical_reports(tmp_path: Path) -> None:
    write_file(tmp_path, "billing/invoice.py", "def create_invoice():\n    pass\n")
    write_file(tmp_path, "notes.py", "# invoice thoughts\nvalue = 1\n")
    write_file(tmp_path, "aaa.py", "value = 1  # needle\n")
    write_file(tmp_path, "zzz.py", "value = 1  # needle\n")
    cases = [
        EvaluationCase("create_invoice", ["billing/invoice.py"]),
        EvaluationCase("invoice", ["billing/invoice.py"]),
        EvaluationCase("needle", ["zzz.py"]),
    ]
    runner = EvaluationRunner(tmp_path)

    first = runner.evaluate(cases, k=3)
    second = runner.evaluate(cases, k=3)

    assert first == second


def test_non_positive_k_raises_value_error(tmp_path: Path) -> None:
    write_file(tmp_path, "invoice.py", "def create_invoice():\n    pass\n")
    runner = EvaluationRunner(tmp_path)
    case = EvaluationCase("invoice", [])

    with pytest.raises(ValueError):
        runner.evaluate([case], k=0)
    with pytest.raises(ValueError):
        runner.evaluate_case(case, k=-1)


# --- Fixture dataset integration ------------------------------------------------------------------------------


@pytest.fixture()
def synthetic_repo(tmp_path: Path) -> Path:
    target = tmp_path / "synthetic-repo"
    shutil.copytree(SYNTHETIC_REPO_DIR, target)
    return target


def test_fixture_cases_are_valid_data() -> None:
    payload = json.loads(CASES_JSON_PATH.read_text(encoding="utf-8"))

    assert 8 <= len(payload["cases"]) <= 12
    for item in payload["cases"]:
        assert item["query"].strip()
        assert item["relevant_files"]


def test_fixture_ground_truth_files_exist_in_synthetic_repo(
    synthetic_repo: Path,
) -> None:
    discovered = set(RepositoryScanner(synthetic_repo).discover_python_files())

    for case in load_fixture_cases():
        assert case.relevant_files
        assert case.relevant_files <= discovered


def test_synthetic_evaluation_is_deterministic_with_expected_baseline(
    synthetic_repo: Path,
) -> None:
    cases = load_fixture_cases()
    runner = EvaluationRunner(synthetic_repo)

    first = runner.evaluate(cases, k=5)
    second = runner.evaluate(cases, k=5)

    assert first == second

    by_query = {item.query: item for item in first.case_evaluations}

    invoice = by_query["invoice calculation"]
    assert invoice.retrieved_files[0] == Path("billing/invoice.py")
    assert invoice.precision_at_k == 1.0
    assert invoice.recall_at_k == 1.0

    logging_eval = by_query["structured logging"]
    assert logging_eval.retrieved_files == (Path("logging/logger.py"),)

    passwords = by_query["hash password"]
    assert passwords.reciprocal_rank == 1.0
    assert passwords.recall_at_k == 1.0

    assert by_query["user authentication"].recall_at_k == 1.0
    assert by_query["refund card payment"].recall_at_k == 1.0

    assert first.mean_recall_at_k == pytest.approx(1.0)
    assert first.mean_reciprocal_rank == pytest.approx(0.875)
    assert 0.0 < first.mean_precision_at_k < 1.0

    for evaluation in first.case_evaluations:
        assert isinstance(evaluation, CaseEvaluation)
        assert 0.0 <= evaluation.precision_at_k <= 1.0
        assert 0.0 <= evaluation.recall_at_k <= 1.0
        assert 0.0 <= evaluation.reciprocal_rank <= 1.0
        assert len(evaluation.retrieved_files) <= 5


def test_synthetic_evaluation_with_k_of_one(synthetic_repo: Path) -> None:
    runner = EvaluationRunner(synthetic_repo)

    report = runner.evaluate(load_fixture_cases(), k=1)

    assert report.k == 1
    invoice = next(
        item
        for item in report.case_evaluations
        if item.query == "invoice calculation"
    )
    assert invoice.retrieved_files == (Path("billing/invoice.py"),)
    assert all(len(item.retrieved_files) <= 1 for item in report.case_evaluations)
