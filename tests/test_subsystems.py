"""Tests for deterministic subsystem discovery (Milestone 23.2).

:class:`repolens.subsystems.discover_subsystems` is a bounded, deterministic
projection of the architecture graph into top-level architectural areas
(subsystems). These tests pin the exact structure of the mult-subsystem
``architecture_retrieval_repository`` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.architecture import ArchitectureGraphBuilder
from repolens.subsystems import discover_subsystems, subsystem_of

RETRIEVAL = Path(__file__).parent / "fixtures" / "architecture_retrieval_repository"
APP_ONLY = Path(__file__).parent / "fixtures" / "architecture_repository"


@pytest.fixture(scope="module")
def graph():
    return ArchitectureGraphBuilder(RETRIEVAL).build()


@pytest.fixture(scope="module")
def subsystems(graph):
    return discover_subsystems(graph)


def _by_id(subsystems):
    return {subsystem.id: subsystem for subsystem in subsystems}


# ---------------------------------------------------------------------------
# Discovery + determinism
# ---------------------------------------------------------------------------


def test_discovery_is_deterministic(graph) -> None:
    first = discover_subsystems(graph)
    second = ArchitectureGraphBuilder(RETRIEVAL).build()
    assert discover_subsystems(second) == first
    assert [s.id for s in first] == sorted(s.id for s in first)


def test_discovery_inventories_subsystems(subsystems) -> None:
    assert [s.id for s in subsystems] == ["admin", "app", "billing", "store", "tests", "web"]
    by_id = _by_id(subsystems)
    assert set(by_id) == {
        "admin", "app", "billing", "store", "tests", "web",
    }


def test_stats_are_deterministic(subsystems) -> None:
    store = _by_id(subsystems)["store"]
    assert store.stats() == {
        "packages": 5,
        "modules": 12,
        "files": 12,
        "entry_modules": 2,
        "dependencies": 0,
        "dependents": 3,
        "layers": 4,
    }


def test_max_subsystems_bounds_by_id_order(graph) -> None:
    assert [s.id for s in discover_subsystems(graph, max_subsystems=2)] == [
        "admin",
        "app",
    ]
    assert [s.id for s in discover_subsystems(graph, max_subsystems=0)] == []


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


def test_membership_packages_modules_files(subsystems) -> None:
    store = _by_id(subsystems)["store"]
    assert store.packages == (
        "store",
        "store/api",
        "store/models",
        "store/repositories",
        "store/services",
    )
    assert "store.services.checkout" in store.modules
    assert "store/services/checkout.py" in store.files
    assert store.contains_package("store/models")
    assert store.contains_module("store.models.order")
    assert store.contains_file("store/services/checkout.py")
    assert not store.contains_module("billing.invoices")


def test_root_subsystem_holds_only_top_level_files(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "main.py").write_text("import json\n", encoding="utf-8")
    by_id = _by_id(discover_subsystems(ArchitectureGraphBuilder(tmp_path).build()))
    assert by_id["root"].files == ("main.py",)
    assert by_id["root"].modules == ("main",)
    assert by_id["root"].packages == (".",)


# ---------------------------------------------------------------------------
# Dependency projection between subsystems
# ---------------------------------------------------------------------------


def test_subsystem_dependencies(subsystems) -> None:
    by_id = _by_id(subsystems)
    assert by_id["billing"].dependencies == ("store",)
    assert by_id["tests"].dependencies == ("billing", "store")
    assert by_id["admin"].dependencies == ("store",)
    assert by_id["web"].dependencies == ("billing",)
    assert by_id["store"].dependencies == ()


def test_subsystem_dependents(subsystems) -> None:
    by_id = _by_id(subsystems)
    assert by_id["store"].dependents == ("admin", "billing", "tests")
    assert by_id["billing"].dependents == ("tests", "web")
    assert by_id["admin"].dependents == ()
    assert by_id["web"].dependents == ()


def test_entry_modules_are_imported_from_other_subsystems(subsystems) -> None:
    by_id = _by_id(subsystems)
    assert by_id["store"].entry_modules == (
        "store.models.order",
        "store.services.checkout",
    )
    assert by_id["billing"].entry_modules == ("billing", "billing.invoices")
    assert by_id["admin"].entry_modules == ()
    assert by_id["app"].entry_modules == ()


# ---------------------------------------------------------------------------
# Intra-subsystem layers
# ---------------------------------------------------------------------------


def test_layers_stratify_package_dependency_depth(subsystems) -> None:
    store = _by_id(subsystems)["store"]
    assert store.layers == (
        ("store/models",),
        ("store", "store/repositories"),
        ("store/services",),
        ("store/api",),
    )


def test_single_layer_subsystems(subsystems) -> None:
    by_id = _by_id(subsystems)
    assert by_id["app"].layers == (("app",),)
    assert by_id["billing"].layers == (("billing",),)
    assert by_id["tests"].layers == (("tests",),)


# ---------------------------------------------------------------------------
# subsystem_of
# ---------------------------------------------------------------------------


def test_subsystem_of_module_package_and_file(subsystems, graph) -> None:
    assert subsystem_of(graph, "store.services.checkout", subsystems).id == "store"
    assert subsystem_of(graph, "store/api", subsystems).id == "store"
    assert subsystem_of(graph, "store/services/checkout.py", subsystems).id == "store"
    assert subsystem_of(graph, "billing.invoices", subsystems).id == "billing"


def test_subsystem_of_unknown_returns_none(graph) -> None:
    assert subsystem_of(graph, "does.not.exist") is None


def test_subsystem_of_without_precomputed_subsystems(graph) -> None:
    assert subsystem_of(graph, "store.services.checkout").id == "store"
    assert subsystem_of(graph, "web.graph").id == "web"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_subsystem_repository() -> None:
    graph = ArchitectureGraphBuilder(APP_ONLY).build()
    discovered = discover_subsystems(graph)
    assert [s.id for s in discovered] == ["app"]
    (app,) = discovered
    assert app.stats() == {
        "packages": 5,
        "modules": 10,
        "files": 10,
        "entry_modules": 0,
        "dependencies": 0,
        "dependents": 0,
        "layers": 4,
    }


def test_empty_repository_has_no_subsystems(tmp_path: Path) -> None:
    assert discover_subsystems(ArchitectureGraphBuilder(tmp_path).build()) == []


def test_single_file_repository_roots(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print(1)\n", encoding="utf-8")
    discovered = discover_subsystems(ArchitectureGraphBuilder(tmp_path).build())
    assert [s.id for s in discovered] == ["root"]
    assert discovered[0].packages == (".",)
    assert discovered[0].modules == ("main",)