"""Tests for the repository-wide symbol index."""

from pathlib import Path

import pytest

from repolens.index import Symbol, SymbolIndexBuilder, SymbolKind


def write_file(tmp_path: Path, relative: str, source: str) -> None:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(source, encoding="utf-8")


def build_index(tmp_path: Path):
    return SymbolIndexBuilder(tmp_path).build()


def test_indexes_top_level_function(tmp_path: Path) -> None:
    write_file(tmp_path, "billing.py", "def create_invoice():\n    pass\n")

    index = build_index(tmp_path)

    assert index.find("create_invoice") == [
        Symbol(
            name="create_invoice",
            kind=SymbolKind.FUNCTION,
            file_path=Path("billing.py"),
            line=1,
        )
    ]


def test_indexes_class(tmp_path: Path) -> None:
    write_file(tmp_path, "invoice.py", "class InvoiceService:\n    pass\n")

    index = build_index(tmp_path)

    matches = index.find("InvoiceService")
    assert len(matches) == 1
    assert matches[0].kind is SymbolKind.CLASS
    assert matches[0].parent_class is None


def test_indexes_class_methods(tmp_path: Path) -> None:
    source = (
        "class InvoiceService:\n"
        "    def calculate_total(self):\n"
        "        pass\n"
        "\n"
        "    async def refresh(self):\n"
        "        pass\n"
    )
    write_file(tmp_path, "billing/invoice.py", source)

    index = build_index(tmp_path)

    method_names = {
        symbol.name for symbol in index.find("calculate_total")
    }
    assert method_names == {"calculate_total"}
    assert {symbol.name for symbol in index.find("refresh")} == {"refresh"}


def test_stores_correct_line_numbers(tmp_path: Path) -> None:
    source = (
        "\n"
        "\n"
        "class InvoiceService:\n"          # line 3
        "    def calculate_total(self):\n"  # line 4
        "        pass\n"
        "\n"
        "\n"
        "def create_invoice():\n"           # line 8
        "    pass\n"
    )
    write_file(tmp_path, "invoice.py", source)

    index = build_index(tmp_path)

    lines = {symbol.name: symbol.line for symbol in index.get_all_symbols()}
    assert lines == {
        "InvoiceService": 3,
        "calculate_total": 4,
        "create_invoice": 8,
    }


def test_stores_repository_relative_paths(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "billing/invoice.py",
        "def create_invoice():\n    pass\n",
    )

    index = build_index(tmp_path)

    symbol = index.get_all_symbols()[0]
    assert symbol.file_path == Path("billing") / "invoice.py"
    assert not symbol.file_path.is_absolute()


def test_stores_parent_class_for_methods(tmp_path: Path) -> None:
    source = (
        "class InvoiceService:\n"
        "    def calculate_total(self):\n"
        "        pass\n"
    )
    write_file(tmp_path, "invoice.py", source)

    index = build_index(tmp_path)

    method = index.find_by_kind(
        "calculate_total", SymbolKind.METHOD
    )[0]
    assert method.parent_class == "InvoiceService"


def test_finds_symbol_by_name_across_kinds(tmp_path: Path) -> None:
    source = (
        "def helper():\n"
        "    pass\n"
        "\n"
        "\n"
        "class Helper:\n"
        "    pass\n"
    )
    write_file(tmp_path, "mixed.py", source)

    index = build_index(tmp_path)

    kinds = {symbol.kind for symbol in index.find("Helper")}
    assert kinds == {SymbolKind.CLASS}
    helper = index.find("helper")
    assert [symbol.kind for symbol in helper] == [SymbolKind.FUNCTION]
    assert index.find("does_not_exist") == []


def test_duplicate_names_return_multiple_results(tmp_path: Path) -> None:
    write_file(tmp_path, "app/users.py", "class User:\n    pass\n")
    write_file(tmp_path, "app/admin.py", "class User:\n    pass\n")

    index = build_index(tmp_path)

    matches = index.find("User")
    files = {symbol.file_path for symbol in matches}
    assert files == {
        Path("app") / "users.py",
        Path("app") / "admin.py",
    }
    assert len(matches) == 2


def test_find_by_kind_distinguishes_symbols(tmp_path: Path) -> None:
    source = (
        "class User:\n"
        "    def save(self):\n"
        "        pass\n"
        "\n"
        "\n"
        "def save():\n"
        "    pass\n"
    )
    write_file(tmp_path, "users.py", source)

    index = build_index(tmp_path)

    methods = index.find_by_kind("save", SymbolKind.METHOD)
    functions = index.find_by_kind("save", SymbolKind.FUNCTION)
    classes = index.find_by_kind("save", SymbolKind.CLASS)
    assert [symbol.parent_class for symbol in methods] == ["User"]
    assert len(functions) == 1
    assert functions[0].parent_class is None
    assert classes == []
    assert index.find_by_kind("save", "method") == methods


def test_handles_repository_with_multiple_files(tmp_path: Path) -> None:
    write_file(tmp_path, "main.py", "from billing.invoice import create_invoice\n")
    write_file(
        tmp_path,
        "billing/__init__.py",
        "",
    )
    write_file(
        tmp_path,
        "billing/invoice.py",
        "class InvoiceService:\n"
        "    def calculate_total(self):\n"
        "        pass\n"
        "\n"
        "\n"
        "def create_invoice():\n"
        "    pass\n",
    )
    write_file(tmp_path, "utils/format.py", "def fmt(value):\n    pass\n")

    index = build_index(tmp_path)

    names = sorted(symbol.name for symbol in index.get_all_symbols())
    assert names == [
        "InvoiceService",
        "calculate_total",
        "create_invoice",
        "fmt",
    ]


def test_get_all_symbols_is_sorted_deterministically(tmp_path: Path) -> None:
    write_file(tmp_path, "zeta.py", "def beta():\n    pass\n")
    write_file(tmp_path, "alpha.py", "def gamma():\n    pass\n")

    index = build_index(tmp_path)
    again = build_index(tmp_path)

    order = [
        (s.file_path.as_posix(), s.line, s.name)
        for s in index.get_all_symbols()
    ]
    assert order == [("alpha.py", 1, "gamma"), ("zeta.py", 1, "beta")]
    assert order == [
        (s.file_path.as_posix(), s.line, s.name)
        for s in again.get_all_symbols()
    ]


def test_empty_repository_produces_empty_index(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("docs only\n")

    index = build_index(tmp_path)

    assert index.get_all_symbols() == []
    assert index.find("anything") == []


def test_builder_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SymbolIndexBuilder(tmp_path / "does-not-exist").build()
