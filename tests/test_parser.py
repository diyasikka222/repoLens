"""Tests for the AST-based Python parser."""

from pathlib import Path

import pytest

from repolens.parser import PythonParser

SIMPLE_FUNCTION = """\
def greet():
    return "hello"
"""

FUNCTION_WITH_ARGUMENTS = '''\
def add(first: int, second: int = 1, *rest, **options):
    return first + second
'''

SIMPLE_CLASS = """\
class Widget:
    pass
"""

CLASS_WITH_METHODS_AND_BASES = """\
class Base:
    def start(self):
        pass


class Widget(Base, mixins.Mixin):
    size = 10

    def __init__(self, name, size=10):
        self.name = name

    async def refresh(self):
        pass
"""

IMPORT_STATEMENTS = """\
import os
import sys as system
import collections.abc


def use_them():
    import json
"""

FROM_IMPORT_STATEMENTS = """\
from pathlib import Path
from collections.abc import Iterable as Iter
from os.path import join


def use_them():
    from typing import Any
"""


def write_source(tmp_path: Path, filename: str, source: str) -> Path:
    file_path = tmp_path / filename
    file_path.write_text(source, encoding="utf-8")
    return file_path


def test_extracts_top_level_function_from_file(tmp_path: Path) -> None:
    file_path = write_source(tmp_path, "greet.py", SIMPLE_FUNCTION)

    analysis = PythonParser().parse_file(file_path)

    assert len(analysis.functions) == 1
    assert analysis.functions[0].name == "greet"
    assert analysis.file_path == file_path


def test_parses_source_code_without_a_file() -> None:
    analysis = PythonParser().parse_source(SIMPLE_FUNCTION)

    assert [function.name for function in analysis.functions] == ["greet"]
    assert analysis.file_path is None


def test_extracts_function_arguments_in_order(tmp_path: Path) -> None:
    file_path = write_source(tmp_path, "add.py", FUNCTION_WITH_ARGUMENTS)

    analysis = PythonParser().parse_file(file_path)

    arguments = analysis.functions[0].arguments
    assert [argument.name for argument in arguments] == [
        "first",
        "second",
        "rest",
        "options",
    ]
    assert arguments[0].annotation == "int"
    assert arguments[1].annotation == "int"


def test_extracts_class(tmp_path: Path) -> None:
    file_path = write_source(tmp_path, "widget.py", SIMPLE_CLASS)

    analysis = PythonParser().parse_file(file_path)

    assert len(analysis.classes) == 1
    assert analysis.classes[0].name == "Widget"
    assert analysis.classes[0].methods == []


def test_extracts_class_methods_with_parent_class(tmp_path: Path) -> None:
    file_path = write_source(
        tmp_path, "widget.py", CLASS_WITH_METHODS_AND_BASES
    )

    analysis = PythonParser().parse_file(file_path)

    widget = next(cls for cls in analysis.classes if cls.name == "Widget")
    methods = {method.name: method for method in widget.methods}
    assert set(methods) == {"__init__", "refresh"}
    assert all(
        method.parent_class == "Widget" for method in methods.values()
    )
    assert [argument.name for argument in methods["__init__"].arguments] == [
        "self",
        "name",
        "size",
    ]


def test_extracts_base_classes(tmp_path: Path) -> None:
    file_path = write_source(
        tmp_path, "widget.py", CLASS_WITH_METHODS_AND_BASES
    )

    analysis = PythonParser().parse_file(file_path)

    widget = next(
        cls for cls in analysis.classes if cls.name == "Widget"
    )
    assert widget.base_classes == ["Base", "mixins.Mixin"]


def test_extracts_import_statements(tmp_path: Path) -> None:
    file_path = write_source(tmp_path, "imports.py", IMPORT_STATEMENTS)

    analysis = PythonParser().parse_file(file_path)

    imports = {(item.module, item.alias) for item in analysis.imports}
    assert imports == {
        ("os", None),
        ("sys", "system"),
        ("collections.abc", None),
        ("json", None),
    }


def test_extracts_from_import_statements(tmp_path: Path) -> None:
    file_path = write_source(tmp_path, "imports.py", FROM_IMPORT_STATEMENTS)

    analysis = PythonParser().parse_file(file_path)

    from_imports = {
        (item.module, item.name, item.alias)
        for item in analysis.from_imports
    }
    assert from_imports == {
        ("pathlib", "Path", None),
        ("collections.abc", "Iterable", "Iter"),
        ("os.path", "join", None),
        ("typing", "Any", None),
    }


def test_handles_multiple_classes_and_functions(tmp_path: Path) -> None:
    source = "\n".join([CLASS_WITH_METHODS_AND_BASES, FUNCTION_WITH_ARGUMENTS])
    file_path = write_source(tmp_path, "mixed.py", source)

    analysis = PythonParser().parse_file(file_path)

    assert [cls.name for cls in analysis.classes] == ["Base", "Widget"]
    assert [function.name for function in analysis.functions] == ["add"]


def test_preserves_line_numbers(tmp_path: Path) -> None:
    source = (
        "import os\n"
        "\n"
        "\n"
        "def late_function():\n"
        "    pass\n"
        "\n"
        "\n"
        "class LateClass:\n"
        "    def method(self):\n"
        "        pass\n"
    )
    file_path = write_source(tmp_path, "lines.py", source)

    analysis = PythonParser().parse_file(file_path)

    assert analysis.imports[0].line == 1
    assert analysis.functions[0].line == 4
    assert analysis.classes[0].line == 8
    assert analysis.classes[0].methods[0].line == 9


def test_invalid_syntax_raises_syntax_error(tmp_path: Path) -> None:
    file_path = write_source(tmp_path, "broken.py", "def broken(:\n")

    with pytest.raises(SyntaxError):
        PythonParser().parse_file(file_path)


def test_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PythonParser().parse_file(tmp_path / "does-not-exist.py")
