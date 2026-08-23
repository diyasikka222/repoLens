"""Tests for the repository import dependency graph."""

from pathlib import Path

import pytest

from repolens.graph import DependencyEdge, DependencyGraphBuilder


def write_file(tmp_path: Path, relative: str, source: str) -> None:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(source, encoding="utf-8")


def build_graph(tmp_path: Path):
    return DependencyGraphBuilder(tmp_path).build()


def test_local_import_creates_edge(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "main.py",
        "from auth import login\n",
    )
    write_file(tmp_path, "auth.py", "def login():\n    pass\n")

    graph = build_graph(tmp_path)

    assert graph.get_all_edges() == [
        DependencyEdge(source=Path("main.py"), target=Path("auth.py"))
    ]


def test_multiple_local_imports_create_multiple_edges(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "main.py",
        "from auth import login\nfrom database import connect\n",
    )
    write_file(tmp_path, "auth.py", "x = 1\n")
    write_file(tmp_path, "database.py", "y = 2\n")

    graph = build_graph(tmp_path)

    assert set(graph.get_all_edges()) == {
        DependencyEdge(source=Path("main.py"), target=Path("auth.py")),
        DependencyEdge(source=Path("main.py"), target=Path("database.py")),
    }


def test_dependency_chain_is_represented(tmp_path: Path) -> None:
    write_file(tmp_path, "a.py", "from b import thing_b\n")
    write_file(tmp_path, "b.py", "from c import thing_c\n")
    write_file(tmp_path, "c.py", "thing_c = 1\n")

    graph = build_graph(tmp_path)

    assert graph.get_dependencies(Path("a.py")) == [Path("b.py")]
    assert graph.get_dependencies(Path("b.py")) == [Path("c.py")]
    assert graph.get_dependents(Path("c.py")) == [Path("b.py")]


def test_standard_library_imports_are_ignored(tmp_path: Path) -> None:
    write_file(tmp_path, "app.py", "import os\nimport sys\nimport json\n")

    graph = build_graph(tmp_path)

    assert graph.get_all_edges() == []
    assert graph.get_all_nodes() == [Path("app.py")]


def test_unresolvable_third_party_imports_are_ignored(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "app.py",
        "import requests\nfrom pytest import fixture\n",
    )
    write_file(tmp_path, "other.py", "z = 3\n")

    graph = build_graph(tmp_path)

    assert graph.get_all_edges() == []


def test_duplicate_imports_create_single_edge(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        "main.py",
        "import auth\nimport auth as authentication\n"
        "from auth import login\nfrom auth import logout\n",
    )
    write_file(tmp_path, "auth.py", "login = logout = 1\n")

    graph = build_graph(tmp_path)

    assert graph.get_all_edges() == [
        DependencyEdge(source=Path("main.py"), target=Path("auth.py"))
    ]


def test_package_and_nested_module_imports_are_resolved(tmp_path: Path) -> None:
    write_file(tmp_path, "utils/__init__.py", "setting = 1\n")
    write_file(tmp_path, "utils/helpers.py", "def format_name():\n    pass\n")
    write_file(
        tmp_path,
        "main.py",
        "import utils\n"
        "import utils.helpers\n"
        "from utils import helpers\n"
        "from utils.helpers import format_name\n"
        "from utils import setting\n",
    )

    graph = build_graph(tmp_path)

    assert set(graph.get_dependencies("main.py")) == {
        Path("utils") / "__init__.py",
        Path("utils") / "helpers.py",
    }
    assert Path("utils") / "__init__.py" in graph.get_all_nodes()


def test_from_import_symbol_resolves_to_owning_module(tmp_path: Path) -> None:
    write_file(tmp_path, "auth.py", "def login():\n    pass\n")
    write_file(tmp_path, "main.py", "from auth import login\n")

    graph = build_graph(tmp_path)

    assert graph.get_dependencies("main.py") == [Path("auth.py")]


def test_get_dependencies_returns_sorted_unique_targets(tmp_path: Path) -> None:
    write_file(tmp_path, "main.py", "from zeta import one\nfrom alpha import two\n")
    write_file(tmp_path, "alpha.py", "two = 1\n")
    write_file(tmp_path, "zeta.py", "one = 2\n")

    graph = build_graph(tmp_path)

    assert graph.get_dependencies("main.py") == [Path("alpha.py"), Path("zeta.py")]


def test_get_dependents_returns_sorted_sources(tmp_path: Path) -> None:
    write_file(tmp_path, "shared.py", "value = 1\n")
    write_file(tmp_path, "zconsumer.py", "from shared import value\n")
    write_file(tmp_path, "aconsumer.py", "from shared import value\n")

    graph = build_graph(tmp_path)

    assert graph.get_dependents("shared.py") == [
        Path("aconsumer.py"),
        Path("zconsumer.py"),
    ]


def test_nodes_use_repository_relative_paths(tmp_path: Path) -> None:
    write_file(tmp_path, "pkg/nested/mod.py", "value = 1\n")

    graph = build_graph(tmp_path)

    nodes = graph.get_all_nodes()
    assert nodes == [Path("pkg") / "nested" / "mod.py"]
    assert not any(node.is_absolute() for node in nodes)


def test_relative_imports_inside_packages_are_resolved(tmp_path: Path) -> None:
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/service.py", "from .storage import save\n")
    write_file(tmp_path, "pkg/storage.py", "def save():\n    pass\n")
    write_file(tmp_path, "pkg/entry.py", "from ..outside import gone\n")

    graph = build_graph(tmp_path)

    assert graph.get_dependencies("pkg/service.py") == [
        Path("pkg") / "storage.py"
    ]
    assert graph.get_dependencies("pkg/entry.py") == []


def test_self_referencing_import_does_not_create_edge(tmp_path: Path) -> None:
    write_file(tmp_path, "pkg/__init__.py", "from pkg import missing_symbol\n")
    write_file(tmp_path, "pkg/mod.py", "value = 1\n")

    graph = build_graph(tmp_path)

    assert graph.get_dependencies("pkg/__init__.py") == []


def test_module_importing_its_own_package_targets_the_init_file(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, "pkg/__init__.py", "value = 1\n")
    write_file(tmp_path, "pkg/mod.py", "from pkg import value\n")

    graph = build_graph(tmp_path)

    assert graph.get_dependencies("pkg/mod.py") == [Path("pkg") / "__init__.py"]


def test_unknown_node_raises_value_error(tmp_path: Path) -> None:
    write_file(tmp_path, "main.py", "x = 1\n")

    graph = build_graph(tmp_path)

    with pytest.raises(ValueError):
        graph.get_dependencies("missing.py")


def test_empty_repository_produces_empty_graph(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not python\n")

    graph = build_graph(tmp_path)

    assert graph.get_all_nodes() == []
    assert graph.get_all_edges() == []


def test_build_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DependencyGraphBuilder(tmp_path / "does-not-exist").build()
