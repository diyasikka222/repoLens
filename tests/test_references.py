"""Tests for cross-file reference extraction and its incremental cache (M22).

All tests are fully offline and deterministic. Extraction must be stable and
deduplicated, imports stay in :class:`ModuleAnalysis` (never re-extracted),
and the persistent reference cache follows the same content-hash discipline as
the repository index: cold build parses N files, a warm build parses 0, and a
one-file change parses exactly 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repolens.incremental_index import IncrementalIndexBuilder
from repolens.references import (
    REFERENCE_CACHE_SCHEMA_VERSION,
    ReferenceExtractor,
    ReferenceIndexBuilder,
    ReferenceKind,
)

ROOT = Path(__file__).parent / "fixtures" / "callgraph_repository"

SRC = (
    "from .models import Order\n"
    "\n"
    "def process_payment(order: Order, card):\n"
    "    total = tax(order.total())\n"
    "    result = charge_card(card)\n"
    "    return result\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "a.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "b.py").touch()
    (root / "a.py").write_text(
        "import os\n\ndef alpha(x: int) -> int:\n    return x + 1\n"
        "\nclass Beta:\n    def meth(self):\n        pass\n",
        encoding="utf-8",
    )
    (root / "b.py").write_text(
        "from .a import alpha, Beta\n\ndef gamma():\n    return alpha(2)\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "ref_cache"


def _builder(repo: Path, cache: Path) -> ReferenceIndexBuilder:
    return ReferenceIndexBuilder(repo, cache_dir=cache)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extraction_records_calls_names_and_attributes() -> None:
    refs = ReferenceExtractor().extract(SRC)
    calls = {r.name: r.kind for r in refs.references if r.kind is ReferenceKind.CALL}
    assert calls == {
        "tax": ReferenceKind.CALL,
        "order.total": ReferenceKind.CALL,
        "charge_card": ReferenceKind.CALL,
    }


def test_extraction_records_parameter_usage_names() -> None:
    refs = ReferenceExtractor().extract(SRC)
    names = {r.name for r in refs.references if r.kind is ReferenceKind.NAME}
    assert "card" in names
    assert "result" in names


def test_extraction_scopes_to_callable_and_class() -> None:
    refs = ReferenceExtractor().extract(
        "class A:\n    def run(self):\n        go()\n\ndef top():\n    go()\n"
    )
    sources = {r.source_symbol for r in refs.references}
    assert sources == {"A.run", "top"}


def test_extraction_parameter_type_hints() -> None:
    refs = ReferenceExtractor().extract(SRC)
    params = {(p.scope, p.name, p.annotation) for p in refs.parameter_types}
    assert ("process_payment", "order", "Order") in params


def test_extraction_self_attribute_assignment_hint() -> None:
    refs = ReferenceExtractor().extract(
        "class CartController:\n"
        "    def __init__(self):\n"
        "        self.cart = models.Cart()\n"
    )
    hints = {(a.scope, a.name, a.target) for a in refs.assignments}
    assert ("CartController.__init__", "self.cart", "models.Cart") in hints


def test_extraction_call_assignment_hint() -> None:
    refs = ReferenceExtractor().extract("def go():\n    c = CartController()\n")
    hints = {(a.scope, a.name, a.target) for a in refs.assignments}
    assert ("go", "c", "CartController") in hints


def test_extraction_dotted_value_assignment_hint() -> None:
    refs = ReferenceExtractor().extract("def go():\n    c = models.Cart\n")
    hints = {(a.scope, a.name, a.target) for a in refs.assignments}
    assert ("go", "c", "models.Cart") in hints


def test_extraction_plain_value_assignments_ignored() -> None:
    refs = ReferenceExtractor().extract(
        "def go():\n    x = 5\n    y = 's'\n    z = [1, 2]\n"
    )
    assert refs.assignments == ()


def test_extraction_deterministic_and_deduplicated() -> None:
    ex = ReferenceExtractor()
    first = ex.extract(ROOT.joinpath("app/checkout.py").read_text(encoding="utf-8"))
    second = ex.extract(ROOT.joinpath("app/checkout.py").read_text(encoding="utf-8"))
    assert first.references == second.references
    assert first.assignments == second.assignments
    assert first.parameter_types == second.parameter_types
    assert len(first.references) == len(set(first.references))


def test_extraction_syntax_error_yields_empty() -> None:
    refs = ReferenceExtractor().extract("def broken(:\n")
    assert refs.references == ()
    assert refs.assignments == ()


def test_extraction_calls_do_not_double_record() -> None:
    refs = ReferenceExtractor().extract("def go():\n    f(g(1))\n")
    calls = sorted(r.name for r in refs.references if r.kind is ReferenceKind.CALL)
    assert calls == ["f", "g"]


def test_extraction_base_class_reference() -> None:
    refs = ReferenceExtractor().extract("from .base import Base\n\nclass X(Base):\n    pass\n")
    bases = [r for r in refs.references if "base_class" in r.evidence]
    assert [b.name for b in bases] == ["Base"]


# ---------------------------------------------------------------------------
# Incremental reference cache
# ---------------------------------------------------------------------------


def test_cold_build_scans_every_file(repo: Path, cache: Path) -> None:
    index = _builder(repo, cache).build()
    stats = index.stats.as_dict()
    assert stats["files_discovered"] == 2
    assert stats["files_scanned"] == 2
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 2


def test_warm_rebuild_scans_nothing(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    index = _builder(repo, cache).build()
    stats = index.stats.as_dict()
    assert stats["files_scanned"] == 0
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 0


def test_single_file_change_reparses_one_file(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    (repo / "a.py").write_text(
        (repo / "a.py").read_text(encoding="utf-8").replace("x + 1", "x + 2"),
        encoding="utf-8",
    )
    index = _builder(repo, cache).build()
    assert index.stats.files_scanned == 1
    assert index.stats.cache_hits == 1


def test_new_file_is_scanned(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    (repo / "c.py").write_text("def new_func():\n    return 1\n", encoding="utf-8")
    index = _builder(repo, cache).build()
    assert index.stats.files_discovered == 3
    assert index.stats.files_scanned == 1
    assert index.stats.cache_hits == 2


def test_deleted_file_prunes_cache(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    (repo / "b.py").unlink()
    index = _builder(repo, cache).build()
    assert index.stats.files_discovered == 1
    assert index.stats.files_scanned == 0
    assert index.stats.files_removed == 1
    assert index.files == (Path("a.py"),)


def test_renamed_file_reparses_and_prunes(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    (repo / "b.py").rename(repo / "c.py")
    index = _builder(repo, cache).build()
    assert index.stats.files_scanned == 1
    assert index.stats.files_removed == 1
    assert Path("c.py") in index.files


def test_corrupt_cache_entry_is_a_miss(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    for entry in cache.glob("*.json"):
        entry.write_text("{not json", encoding="utf-8")
    index = _builder(repo, cache).build()
    assert index.stats.cache_hits == 0
    assert index.stats.cache_misses == 2


def test_schema_bump_invalidates_cache(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    for entry in cache.glob("*.json"):
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload["schema"] = REFERENCE_CACHE_SCHEMA_VERSION + 1
        entry.write_text(json.dumps(payload), encoding="utf-8")
    index = _builder(repo, cache).build()
    assert index.stats.cache_hits == 0
    assert index.stats.files_scanned == 2


def test_wrong_content_hash_is_a_miss(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    for entry in cache.glob("*.json"):
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload["content_hash"] = "different"
        entry.write_text(json.dumps(payload), encoding="utf-8")
    index = _builder(repo, cache).build()
    assert index.stats.cache_hits == 0


def test_clear_restores_clean_build(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    for entry in cache.glob("*.json"):
        entry.unlink()
    index = _builder(repo, cache).build()
    assert index.stats.files_scanned == 2


def test_multiple_repos_are_isolated(tmp_path: Path, cache: Path) -> None:
    a, b = tmp_path / "a_repo", tmp_path / "b_repo"
    a.mkdir()
    b.mkdir()
    (a / "m.py").write_text("def only_a():\n    pass\n", encoding="utf-8")
    (b / "m.py").write_text("def only_b():\n    pass\n", encoding="utf-8")
    r1 = _builder(a, cache).build()
    r2 = _builder(b, cache).build()
    assert r1.references_for(Path("m.py")) != r2.references_for(Path("m.py"))
    assert r1.stats.cache_hits == 0
    assert r2.stats.cache_hits == 0


def test_persist_false_stays_in_memory(repo: Path, cache: Path) -> None:
    builder = ReferenceIndexBuilder(repo, cache_dir=cache, persist=False)
    first = builder.build()
    assert first.stats.files_scanned == 2
    second = builder.build()
    assert second.stats.files_scanned == 0
    assert not cache.exists()


def test_index_snapshot_backed_build_matches_standalone(repo: Path) -> None:
    standalone = ReferenceIndexBuilder(repo, persist=False).build()
    index = IncrementalIndexBuilder(repo, persist=False).build()
    backed = ReferenceIndexBuilder(repo, index=index, persist=False).build()
    for path in index.files:
        assert backed.references_for(path) == standalone.references_for(path)


def test_fixture_extracts_readable_references() -> None:
    index = IncrementalIndexBuilder(ROOT, persist=False).build()
    refs = ReferenceIndexBuilder(ROOT, index=index, persist=False).build()
    checkout = refs.references_for(Path("app/checkout.py"))
    names = {r.name for r in checkout.references}
    assert "models.Cart" in names
    assert "payments.charge_card" in names
    assert "bookings.make" in names
    assert any(a.name == "self.cart" for a in checkout.assignments)
    assert any(
        p.scope == "process_payment" and p.name == "order"
        for p in refs.references_for(Path("app/payments.py")).parameter_types
    )