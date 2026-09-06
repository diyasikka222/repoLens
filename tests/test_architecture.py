"""Tests for the repository-level architecture graph (Milestone 23).

:class:`repolens.architecture.ArchitectureGraph` is a deterministic projection
of existing RepoLens data (incremental index + dependency graph) into three
structural levels — files, Python modules, and packages — with bounded,
deterministic dependency queries. No file is re-parsed by the architecture
builder itself.

The fixture repository under ``tests/fixtures/architecture_repository`` is the
proprietary ``app`` tree used by most tests; incremental behavior is exercised
on throwaway copies of it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from repolens.architecture import (
    ArchitectureGraphBuilder,
    ArchitectureNodeKind,
    ArchitectureRelationship,
    REPO_ROOT_PACKAGE,
)
from repolens.incremental_index import IncrementalIndexBuilder

ROOT = Path(__file__).parent / "fixtures" / "architecture_repository"


@pytest.fixture(scope="module")
def graph():
    return ArchitectureGraphBuilder(ROOT).build()


@pytest.fixture()
def repo(tmp_path: Path):
    target = tmp_path / "repo"
    shutil.copytree(ROOT, target)
    return target


def _rebuild(repo: Path):
    return ArchitectureGraphBuilder(repo).build()


def _ids(nodes) -> set[str]:
    return {node.id for node in nodes}


def _write(repo: Path, rel: str, content: str) -> Path:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Node creation + identity
# ---------------------------------------------------------------------------


def test_node_kinds_and_counts(graph) -> None:
    stats = graph.stats()
    assert stats.files == 10
    assert stats.nodes == stats.files + stats.modules + stats.packages
    kinds = {node.kind for node in graph.get_nodes()}
    assert kinds == {
        ArchitectureNodeKind.FILE,
        ArchitectureNodeKind.MODULE,
        ArchitectureNodeKind.PACKAGE,
    }


def test_file_module_package_nodes(graph) -> None:
    file_node = graph.get_file_node("app/api/routes.py")
    assert file_node is not None
    assert file_node.kind is ArchitectureNodeKind.FILE

    module_node = graph.get_module_node("app.api.routes")
    assert module_node is not None
    assert module_node.kind is ArchitectureNodeKind.MODULE

    package_node = graph.get_package_node("app/api")
    assert package_node is not None
    assert package_node.kind is ArchitectureNodeKind.PACKAGE
    assert graph.get_package_node(".").id == REPO_ROOT_PACKAGE


def test_string_resolution_preference(graph) -> None:
    assert graph.get_file_node("app/api/routes.py") is not None
    assert _ids(graph.dependencies_of("app")) == _ids(graph.package_dependencies("app"))
    assert _ids(graph.dependencies_of("app")) == {"app/models"}
    ambiguity = graph.get_package_node("app")
    assert ambiguity is not None
    assert graph.get_module_node("app.api").id == "app.api"
    assert _ids(graph.dependencies_of("app/models/user.py")) == _ids(
        graph.dependencies_of("app.models.user")
    )


def test_missing_nodes_return_none(graph) -> None:
    assert graph.get_file_node("nope.py") is None
    assert graph.get_module_node("nope") is None
    assert graph.get_package_node("nope") is None
    with pytest.raises(KeyError):
        graph.dependencies_of("does.not.exist")


# ---------------------------------------------------------------------------
# Containment: packages, files, modules
# ---------------------------------------------------------------------------


def test_package_hierarchy(graph) -> None:
    contained = {node.id: node.kind for node in graph.contains("app")}
    assert contained == {
        "app/__init__.py": ArchitectureNodeKind.FILE,
        "app/api": ArchitectureNodeKind.PACKAGE,
        "app/models": ArchitectureNodeKind.PACKAGE,
        "app/repositories": ArchitectureNodeKind.PACKAGE,
        "app/services": ArchitectureNodeKind.PACKAGE,
    }
    assert {node.id for node in graph.contained_by("app")} == {REPO_ROOT_PACKAGE}
    assert {node.id for node in graph.contains(REPO_ROOT_PACKAGE)} == {"app"}


def test_files_in_package(graph) -> None:
    assert _ids(graph.files_in_package("app")) == {"app/__init__.py"}
    assert _ids(graph.files_in_package("app/services")) == {
        "app/services/__init__.py",
        "app/services/_helpers.py",
        "app/services/users.py",
    }
    with pytest.raises(TypeError):
        graph.files_in_package("app/services/users.py")


def test_modules_in_package(graph) -> None:
    assert _ids(graph.modules_in_package("app/services")) == {
        "app.services",
        "app.services._helpers",
        "app.services.users",
    }
    with pytest.raises(TypeError):
        graph.modules_in_package("app/services/users.py")


def test_file_contains_module(graph) -> None:
    file_node = graph.get_file_node("app/models/user.py")
    assert _ids(graph.contains(file_node)) == {"app.models.user"}
    assert _ids(graph.contained_by(graph.get_module_node("app.models.user"))) == {
        "app/models/user.py"
    }


# ---------------------------------------------------------------------------
# Dependency edges
# ---------------------------------------------------------------------------


def test_module_dependencies(graph) -> None:
    assert _ids(graph.module_dependencies("app.api.routes")) == {
        "app.services",
        "app.services.users",
        "app.repositories.users",
        "app.models.user",
    }


def test_file_node_delegates_to_its_module(graph) -> None:
    assert _ids(graph.dependencies_of("app/api/routes.py")) == _ids(
        graph.dependencies_of("app.api.routes")
    )
    assert _ids(graph.dependents_of("app/models/user.py")) == _ids(
        graph.dependents_of("app.models.user")
    )


def test_dependents_of_module(graph) -> None:
    assert _ids(graph.dependents_of("app.models.user")) == {
        "app",
        "app.api.routes",
        "app.models",
        "app.repositories.users",
        "app.services.users",
    }


def test_package_dependencies(graph) -> None:
    assert _ids(graph.package_dependencies("app/api")) == {
        "app/models",
        "app/repositories",
        "app/services",
    }
    assert _ids(graph.package_dependencies("app/services")) == {
        "app/models",
        "app/repositories",
    }
    assert _ids(graph.package_dependencies("app/models")) == set()
    assert _ids(graph.package_dependencies(REPO_ROOT_PACKAGE)) == set()


def test_relative_and_same_package_imports(graph) -> None:
    assert _ids(graph.module_dependencies("app.services.users")) == {
        "app.services._helpers",
        "app.repositories.users",
        "app.models.user",
    }
    assert graph.get_module_node("app.repositories.users") is not None
    assert "app.repositories.users" in _ids(graph.dependents_of("app.models.user"))


def test_no_duplicate_edges_or_nodes(graph) -> None:
    edge_keys = {
        (e.relationship.value, e.source.id, e.target.id) for e in graph.get_edges()
    }
    assert len(edge_keys) == len(graph.get_edges())
    node_keys = {(n.kind.value, n.id) for n in graph.get_nodes()}
    assert len(node_keys) == len(graph.get_nodes())
    assert graph.stats().edges == graph.stats().contains_edges + graph.stats().depends_on_edges


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_ordering_and_serialization(graph) -> None:
    assert graph.serialize() == graph.serialize()
    assert [n.id for n in graph.get_nodes()] == [
        n.id for n in sorted(graph.get_nodes(), key=lambda n: (n.kind.value, n.id))
    ]
    second = ArchitectureGraphBuilder(ROOT).build()
    assert second.serialize() == graph.serialize()


def test_get_edges_by_relationship_all_four(graph) -> None:
    contains = graph.get_edges_by_relationship(ArchitectureRelationship.CONTAINS)
    contained_by = graph.get_edges_by_relationship(ArchitectureRelationship.CONTAINED_BY)
    depends = graph.get_edges_by_relationship(ArchitectureRelationship.DEPENDS_ON)
    depended = graph.get_edges_by_relationship(ArchitectureRelationship.DEPENDED_ON_BY)
    assert len(contains) == graph.stats().contains_edges
    assert len(contained_by) == len(contains)
    assert len(depends) == graph.stats().depends_on_edges
    assert len(depended) == len(depends)
    seen: dict[str, list[str]] = {}
    for edge in depends:
        seen.setdefault((edge.source.id, edge.target.id), []).append("fwd")
    for edge in depended:
        seen.setdefault((edge.target.id, edge.source.id), []).append("rev")
    assert all("fwd" in value and "rev" in value for value in seen.values())


# ---------------------------------------------------------------------------
# Bounded transitive traversal
# ---------------------------------------------------------------------------


def test_transitive_dependencies_bounded(graph) -> None:
    one = _ids(graph.transitive_dependencies("app.api.routes", max_depth=1))
    two = _ids(graph.transitive_dependencies("app.api.routes", max_depth=2))
    assert one == {
        "app.services",
        "app.services.users",
        "app.repositories.users",
        "app.models.user",
    }
    assert two == one | {"app.services._helpers"}
    assert len(two) < graph.stats().modules


def test_transitive_dependents_bounded(graph) -> None:
    names = _ids(graph.transitive_dependents("app.models.user", max_depth=1))
    assert names == {
        "app",
        "app.api.routes",
        "app.models",
        "app.repositories.users",
        "app.services.users",
    }
    assert "app.models.user" not in names


def test_transitive_depth_zero_and_negative(graph) -> None:
    assert graph.transitive_dependencies("app.api.routes", max_depth=0) == []
    assert graph.transitive_dependents("app.models.user", max_depth=0) == []
    with pytest.raises(ValueError):
        graph.transitive_dependencies("app.api.routes", max_depth=-1)
    with pytest.raises(ValueError):
        graph.transitive_dependents("app.models.user", max_depth=-1)


# ---------------------------------------------------------------------------
# Incremental behavior
# ---------------------------------------------------------------------------


def test_builder_reuses_index_without_rescan(graph, repo: Path) -> None:
    index = IncrementalIndexBuilder(repo, persist=False).build()
    from_index = ArchitectureGraphBuilder(repo, index=index).build()
    cold = _rebuild(repo)
    assert from_index.serialize() == cold.serialize()


def test_modified_file_updates_dependencies(repo: Path) -> None:
    _write(repo, "app/config.py", "SERVICE_NAME = 'accounts'\n")
    _write(repo, "app/services/_helpers.py", "from app import config\n\ndef identity(value):\n    return value\n")
    graph = _rebuild(repo)
    assert _ids(graph.module_dependencies("app.services._helpers")) == {"app.config"}
    assert "app" in _ids(graph.package_dependencies("app/services"))


def test_deleted_file_removes_node_and_edges(repo: Path) -> None:
    (repo / "app" / "models" / "user.py").unlink()
    graph = _rebuild(repo)
    assert graph.get_file_node("app/models/user.py") is None
    assert graph.get_module_node("app.models.user") is None
    for edge in graph.get_edges():
        assert edge.source.id != "app.models.user"
        assert edge.target.id != "app.models.user"
    assert "app/models" in _ids(graph.packages())
    assert _ids(graph.module_dependencies("app.repositories.users")) == {"app.models"}
    assert _ids(graph.package_dependencies("app/repositories")) == {"app/models"}


def test_deleted_directory_removes_package(repo: Path) -> None:
    shutil.rmtree(repo / "app" / "services")
    graph = _rebuild(repo)
    assert graph.get_package_node("app/services") is None
    assert "app/services" not in {node.id for node in graph.get_nodes()}
    assert _ids(graph.package_dependencies("app/api")) == {
        "app",
        "app/models",
        "app/repositories",
    }


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_repository(tmp_path: Path) -> None:
    graph = _rebuild(tmp_path)
    first = graph.stats()
    assert first.nodes == 0
    assert first.files == 0
    assert first.modules == 0
    assert first.packages == 0
    assert graph.get_nodes() == []
    assert graph.get_edges() == []
    assert graph.packages() == []


def test_single_file_repository(tmp_path: Path) -> None:
    _write(tmp_path, "main.py", "import json\nprint('hello')\n")
    graph = _rebuild(tmp_path)
    stats = graph.stats()
    assert stats.files == 1 and stats.modules == 1 and stats.packages == 1
    assert graph.get_file_node("main.py") is not None
    assert graph.get_module_node("main") is not None
    assert graph.get_package_node(".") is not None
    assert _ids(graph.files_in_package(".")) == {"main.py"}
    assert graph.dependencies_of("main.py") == []


def test_external_imports_never_crash_and_never_appear(graph) -> None:
    assert graph.get_module_node("json") is None
    assert graph.get_file_node("json.py") is None
    assert "json" not in _ids(graph.dependencies_of("app/api/routes.py"))


def test_unresolvable_relative_import_never_crashes(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "from .nope import thing\n")
    _write(tmp_path, "pkg/a.py", "from ..missing import gone\n")
    graph = _rebuild(tmp_path)
    assert graph.get_file_node("pkg/__init__.py") is not None
    assert graph.get_file_node("pkg/a.py") is not None
    assert graph.stats().depends_on_edges == 0