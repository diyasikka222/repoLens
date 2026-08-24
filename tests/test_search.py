"""Tests for deterministic lexical code search."""

from pathlib import Path

from repolens.index import Symbol, SymbolKind
from repolens.search import (
    WEIGHT_PATH_TOKEN,
    WEIGHT_SOURCE_TOKEN,
    CodeSearcher,
)


def write_file(tmp_path: Path, relative: str, source: str) -> None:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(source, encoding="utf-8")


def paths_of(results):
    return [result.file_path for result in results]


# --- 1. Exact function-name query -----------------------------------------


def test_exact_function_name_query_ranks_correct_file_first(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "billing/invoice.py",
        "def create_invoice():\n    pass\n",
    )
    write_file(
        tmp_path,
        "reports/summary.py",
        "# create_invoice is mentioned in a comment only\nvalue = 1\n",
    )
    write_file(tmp_path, "unrelated.py", "def delete_cache():\n    pass\n")

    results = CodeSearcher(tmp_path).search("create_invoice")

    assert paths_of(results)[0] == Path("billing") / "invoice.py"
    assert Path("unrelated.py") not in paths_of(results)
    assert Path("reports") / "summary.py" in paths_of(results)
    assert results[0].score > results[-1].score


# --- 2. Exact class-name query --------------------------------------------


def test_exact_class_name_query_ranks_correct_file_first(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "billing/invoice.py",
        "class InvoiceService:\n    def calculate_total(self):\n        pass\n",
    )
    write_file(tmp_path, "helpers/report.py", "class ReportHelper:\n    pass\n")

    results = CodeSearcher(tmp_path).search("InvoiceService")

    assert paths_of(results)[0] == Path("billing") / "invoice.py"


# --- 3. Case-insensitivity -------------------------------------------------


def test_query_matching_is_case_insensitive(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "billing/invoice.py",
        "class InvoiceService:\n    pass\n",
    )

    searcher = CodeSearcher(tmp_path)

    assert paths_of(searcher.search("invoiceservice")) == [
        Path("billing") / "invoice.py"
    ]
    assert paths_of(searcher.search("INVOICE_SERVICE")) == [
        Path("billing") / "invoice.py"
    ]


# --- 4. Separator / camelCase normalization --------------------------------


def test_separator_normalization_matches_tokens(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "billing/calculator.py",
        "def calculate_invoice_total():\n    pass\n",
    )
    write_file(tmp_path, "shipping/labels.py", "def print_label():\n    pass\n")

    searcher = CodeSearcher(tmp_path)
    spaced = searcher.search("invoice calculation")
    underscored = searcher.search("invoice_calculation")
    camel = searcher.search("InvoiceCalculation")

    assert paths_of(spaced) == [Path("billing") / "calculator.py"]
    assert [(r.file_path, r.score) for r in spaced] == [
        (r.file_path, r.score) for r in underscored
    ]
    assert [(r.file_path, r.score) for r in spaced] == [
        (r.file_path, r.score) for r in camel
    ]
    assert Path("shipping") / "labels.py" not in paths_of(spaced)


# --- 5. File-path matching ---------------------------------------------------


def test_file_path_token_matches_are_found(tmp_path: Path) -> None:
    write_file(tmp_path, "auth/session.py", "timeout_seconds = 30\n")
    write_file(tmp_path, "auth/login.py", "def login():\n    pass\n")

    results = CodeSearcher(tmp_path).search("session")

    assert paths_of(results) == [Path("auth") / "session.py"]
    assert results[0].score == WEIGHT_PATH_TOKEN
    assert results[0].matched_terms == ("session",)
    assert results[0].symbols == ()


# --- 6. Source-code text matching -------------------------------------------


def test_source_code_text_matches_are_found(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "utils/text.py",
        "# supports zanzibar encoding\ndef noop():\n    pass\n",
    )

    results = CodeSearcher(tmp_path).search("zanzibar")

    assert paths_of(results) == [Path("utils") / "text.py"]
    assert results[0].score >= WEIGHT_SOURCE_TOKEN


# --- 7. Multiple matching files ----------------------------------------------


def test_multiple_matching_files_are_all_returned(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "refunds/service.py",
        "def process_refund():\n    pass\n",
    )
    write_file(tmp_path, "api.py", "# exposes refund endpoints\nx = 1\n")
    write_file(tmp_path, "ledger.py", "# refund audit log\ny = 2\n")
    write_file(tmp_path, "other.py", "def unrelated():\n    pass\n")

    results = CodeSearcher(tmp_path).search("refund")

    assert len(results) == 3
    assert all(result.score > 0 for result in results)
    assert Path("other.py") not in paths_of(results)


# --- 8. Sorting by relevance --------------------------------------------------


def test_results_are_sorted_by_relevance(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "payments/core.py",
        "def payment_gateway():\n    pass\n",
    )
    write_file(
        tmp_path,
        "gateway/routes.py",
        "def route():\n    return 'payment flow'\n",
    )
    write_file(tmp_path, "misc.py", "# payment gateway notes\nx = 1\n")

    results = CodeSearcher(tmp_path).search("payment gateway")

    assert paths_of(results) == [
        Path("payments") / "core.py",
        Path("gateway") / "routes.py",
        Path("misc.py"),
    ]
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


# --- 9. Limit ------------------------------------------------------------------


def test_limit_is_respected(tmp_path: Path) -> None:
    for number in range(15):
        write_file(tmp_path, f"items/item_{number:02d}.py", f"def item_{number}():\n    pass\n")

    searcher = CodeSearcher(tmp_path)

    assert len(searcher.search("item", limit=3)) == 3
    assert len(searcher.search("item")) == 10
    assert len(searcher.search("item", limit=100)) == 15


def test_non_positive_limit_returns_no_results(tmp_path: Path) -> None:
    write_file(tmp_path, "invoice.py", "def create_invoice():\n    pass\n")
    searcher = CodeSearcher(tmp_path)

    assert searcher.search("invoice", limit=0) == []
    assert searcher.search("invoice", limit=-1) == []


# --- 10. Deterministic tie-breaking ---------------------------------------------


def test_tied_scores_have_deterministic_order(tmp_path: Path) -> None:
    write_file(tmp_path, "bb/tool.py", "value = 1\n")
    write_file(tmp_path, "aa/tool.py", "value = 1\n")

    searcher = CodeSearcher(tmp_path)
    first = searcher.search("tool")
    second = searcher.search("tool")

    assert first[0].score == second[0].score == first[1].score == second[1].score
    assert paths_of(first) == [Path("aa") / "tool.py", Path("bb") / "tool.py"]
    assert paths_of(first) == paths_of(second)


# --- 11. Empty query --------------------------------------------------------------


def test_empty_query_returns_no_results(tmp_path: Path) -> None:
    write_file(tmp_path, "invoice.py", "def create_invoice():\n    pass\n")
    searcher = CodeSearcher(tmp_path)

    assert searcher.search("") == []
    assert searcher.search("   ") == []
    assert searcher.search("___") == []


# --- 12. Empty repository ------------------------------------------------------------


def test_repository_without_python_files_returns_no_results(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("docs only\n", encoding="utf-8")

    searcher = CodeSearcher(tmp_path)

    assert searcher.search("anything") == []
    assert searcher.search("README") == []


# --- 13. Repository-relative paths ------------------------------------------------------


def test_paths_are_repository_relative(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "billing/invoice.py",
        "class InvoiceService:\n    pass\n",
    )

    results = CodeSearcher(tmp_path).search("invoice")

    assert len(results) == 1
    assert results[0].file_path == Path("billing") / "invoice.py"
    assert not results[0].file_path.is_absolute()
    assert results[0].file_path.as_posix() == "billing/invoice.py"


# --- 14. Symbol metadata ------------------------------------------------------------------


def test_symbol_metadata_is_returned(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "billing/invoice.py",
        "class InvoiceService:\n"
        "    def calculate_total(self):\n"
        "        pass\n",
    )

    results = CodeSearcher(tmp_path).search("calculate_total")

    assert len(results) == 1
    result = results[0]
    assert result.symbols == (
        Symbol(
            name="calculate_total",
            kind=SymbolKind.METHOD,
            file_path=Path("billing") / "invoice.py",
            line=2,
            parent_class="InvoiceService",
        ),
    )
    assert result.matched_terms == ("calculate", "total")


def test_exact_match_reports_all_symbols_and_terms(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "billing/invoice.py",
        "def create_invoice():\n    pass\n",
    )

    results = CodeSearcher(tmp_path).search("create_invoice")

    symbol = results[0].symbols[0]
    assert symbol.name == "create_invoice"
    assert symbol.kind is SymbolKind.FUNCTION
    assert symbol.parent_class is None
    assert symbol.line == 1


# --- Import/module-name layer ---------------------------------------------------------------


def test_import_module_names_are_searchable(tmp_path: Path) -> None:
    write_file(tmp_path, "app/main.py", "import billing.invoice\nx = 1\n")
    write_file(tmp_path, "app/other.py", "import os\ny = 2\n")

    results = CodeSearcher(tmp_path).search("invoice")

    assert paths_of(results) == [Path("app") / "main.py"]
