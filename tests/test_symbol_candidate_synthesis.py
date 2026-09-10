"""P26.1 Step 2: symbol-discovered files become explicit candidates.

Regression tests for the discovery-to-candidate gap: a file that defines a
symbol referenced by the query was discovered (``DISCOVERED_NOT_RANKED``) but
never materialized as a :class:`~repolens.context.ContextCandidate` unless
retrieval or dependency expansion also surfaced it. These tests prove the file
now flows through the normal candidate pipeline (ranking, budget, firewall) and
that retrieval/change-plan behaviour is preserved.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from repolens.context import (
    INCLUSION_SYMBOL_MATCH,
    CandidateRole,
    ContextBudget,
    ContextEngine,
    ContextFirewall,
    DependencyExpansionConfig,
)
from repolens.context_trace import TraceCollector
from repolens.required_file_diagnostics import RequiredFileOutcome, classify_required_files

CHANGE_REPO = (
    Path(__file__).resolve().parent / "fixtures" / "change_plan_repository"
)


@dataclass(frozen=True)
class _Result:
    """Minimal searcher result exposing the attributes the engine reads."""

    file_path: Path
    score: float = 0.5


class _FixedSearcher:
    """Deterministic searcher returning a fixed ranked result list."""

    def __init__(self, results: list[Path]) -> None:
        self._results = results

    def search(self, query: str, limit: int):
        return [_Result(path, score=0.5) for path in self._results[:limit]]


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _build_engine(
    root: Path,
    searcher,
    *,
    budget_tokens: int | None = 10**9,
    depth: int = 0,
    tracer=None,
) -> ContextEngine:
    return ContextEngine(
        root,
        searcher=searcher,
        dependency=DependencyExpansionConfig(depth=depth),
        budget=ContextBudget(max_tokens=budget_tokens),
        tracer=tracer,
    )


def _build_symbol_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "app/dashboard.py", "def render_dashboard():\n    return 'hi'\n")
    _write(
        root,
        "app/widget.py",
        "class ZebraWidgetFactory:\n    pass\n",
    )
    return root


def _dashboard_only_searcher():
    """Surface only the retrieved primary — never the symbol-discovered file."""
    return _FixedSearcher([Path("app/dashboard.py")])


class SymbolCandidateSynthesisTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = _build_symbol_repo(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_symbol_discovered_file_becomes_selected_candidate(self) -> None:
        engine = _build_engine(self.root, _dashboard_only_searcher())
        pkg = engine.build_context("how is the zebra_widget_factory structured")
        by_path = {c.path: c for c in pkg.selected_files}
        # The retrieved primary is still selected first.
        self.assertIn(Path("app/dashboard.py"), by_path)
        # The symbol-only file (never retrieved, never expanded) is a candidate
        # and survives selection.
        widget = by_path[Path("app/widget.py")]
        self.assertEqual(widget.inclusion_reason, INCLUSION_SYMBOL_MATCH)
        self.assertIs(widget.role, CandidateRole.PRIMARY)
        self.assertIsNone(widget.retrieval_rank)
        self.assertTrue(widget.estimated_tokens > 0)

    def test_symbol_candidate_reported_among_primary_candidates(self) -> None:
        engine = _build_engine(self.root, _dashboard_only_searcher())
        pkg = engine.build_context("how is the zebra_widget_factory structured")
        primary_paths = {c.path for c in pkg.primary_candidates}
        self.assertIn(Path("app/widget.py"), primary_paths)
        self.assertIn(Path("app/dashboard.py"), primary_paths)

    def test_retrieved_symbol_file_not_duplicated_by_synthesis(self) -> None:
        # When the symbol file is ALSO retrieved, dedupe keeps the retrieved
        # primary identity (retrieval signals) and never emits a duplicate.
        engine = _build_engine(
            self.root,
            _FixedSearcher([Path("app/dashboard.py"), Path("app/widget.py")]),
        )
        pkg = engine.build_context("how is the zebra_widget_factory structured")
        by_path = {c.path: c for c in pkg.selected_files}
        self.assertEqual(len(pkg.selected_files), len(by_path))
        widget = by_path[Path("app/widget.py")]
        self.assertEqual(widget.role, CandidateRole.PRIMARY)
        self.assertEqual(widget.retrieval_rank, 2)
        self.assertEqual(widget.inclusion_reason, INCLUSION_SYMBOL_MATCH)

    def test_ranked_and_selection_order_is_deterministic(self) -> None:
        first = _build_engine(self.root, _dashboard_only_searcher()).build_context(
            "how is the zebra_widget_factory structured"
        )
        again = _build_engine(self.root, _dashboard_only_searcher()).build_context(
            "how is the zebra_widget_factory structured"
        )
        self.assertEqual(first.to_dict(), again.to_dict())
        # Retrieved primary ranks first; symbol-only candidates follow in
        # deterministic path order before any dependency tier.
        roles = [c.role for c in first.selected_files]
        self.assertEqual(roles[0], CandidateRole.PRIMARY)
        self.assertTrue(roles[-1] is CandidateRole.PRIMARY)

    def test_symbol_candidate_goes_through_firewall(self) -> None:
        _write(
            self.root,
            "app/vault.py",
            "class VaultRegistry:\n    api_key = 'sk-live-1234567890abcdef'\n",
        )
        engine = _build_engine(
            self.root,
            _FixedSearcher([Path("app/dashboard.py")]),
        )
        pkg = engine.build_context("vault_registry secrets")
        safe = ContextFirewall().safe_package(
            pkg, ContextFirewall().inspect(pkg)
        )
        self.assertIn("app/vault.py", {c.path for c in safe.safe_files})
        for candidate in safe.safe_files:
            self.assertNotIn("sk-live-1234567890abcdef", candidate.source)


class SymbolCandidateDiagnosticsTests(unittest.TestCase):
    """The previously DISCOVERED_NOT_RANKED file is now ranked (P26.1 classification)."""

    def test_symbol_discovered_file_ranked_but_budget_excluded(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            root = Path(tmp.name)
            _write(root, "main.py", "def entry():\n    pass\n")
            _write(
                root,
                "widget.py",
                "class ZebraWidgetFactory:\n    pass\n# " + "x" * 400,
            )
            collector = TraceCollector()
            engine = ContextEngine(
                root,
                searcher=_FixedSearcher([Path("main.py")]),
                dependency=DependencyExpansionConfig(depth=0),
                budget=ContextBudget(max_tokens=60),
                tracer=collector,
            )
            engine.build_context("how is the zebra_widget_factory structured")
            trace = collector.trace()
            self.assertIn("widget.py", trace.ranked_paths)
            row = classify_required_files(
                "task", "context", ["widget.py"], trace, budget_max_tokens=60
            )[0]
            # Discovery is unchanged (source=symbol) but the outcome moves from
            # DISCOVERED_NOT_RANKED to RANKED_NOT_SELECTED.
            self.assertEqual(
                row.outcome, RequiredFileOutcome.RANKED_NOT_SELECTED
            )
            self.assertEqual(row.discovery_source, "symbol")
            self.assertEqual(row.inclusion_reason, INCLUSION_SYMBOL_MATCH)
            self.assertIsNotNone(row.ranking_position)
            self.assertIn(row.excluded_reason, {"over_budget", "exceeds_total_budget"})
        finally:
            tmp.cleanup()


class ChangePathNotDuplicatedTests(unittest.TestCase):
    def test_change_plan_candidates_are_not_duplicated(self) -> None:
        engine = ContextEngine(
            CHANGE_REPO,
            searcher=_FixedSearcher([Path("app/services/checkout.py")]),
            dependency=DependencyExpansionConfig(depth=1),
            budget=ContextBudget(max_tokens=10**9),
        )
        pkg = engine.build_context(
            "make checkout reject empty carts",
            change_request="make checkout reject empty carts",
            change_target="app/services/checkout.py",
        )
        selected = list(pkg.selected_files)
        paths = [c.path for c in selected]
        self.assertEqual(len(paths), len(set(paths)))
        # Every surviving change-plan candidate is present exactly once.
        for change in pkg.change_candidates:
            self.assertEqual(paths.count(change.path), 1)
            self.assertIn(change.path, paths)

    def test_change_path_is_deterministic_with_symbol_synthesis(self) -> None:
        engine = ContextEngine(
            CHANGE_REPO,
            searcher=_FixedSearcher([Path("app/services/checkout.py")]),
            dependency=DependencyExpansionConfig(depth=1),
            budget=ContextBudget(max_tokens=10**9),
        )
        kwargs = {
            "change_request": "make checkout reject empty carts",
            "change_target": "app/services/checkout.py",
        }
        first = engine.build_context("make checkout reject empty carts", **kwargs)
        again = engine.build_context("make checkout reject empty carts", **kwargs)
        self.assertEqual(first.to_dict(), again.to_dict())


class BaselineStabilityTests(unittest.TestCase):
    def test_no_symbol_matches_leaves_output_unchanged(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            root = Path(tmp.name)
            _write(root, "alpha.py", "def alpha():\n    return 1\n")
            _write(root, "beta.py", "def beta():\n    return 2\n")
            query = "totally unrelated prose with no symbols here"
            collector = TraceCollector()
            engine = ContextEngine(
                root,
                searcher=_FixedSearcher([Path("alpha.py")]),
                dependency=DependencyExpansionConfig(depth=0),
                budget=ContextBudget(max_tokens=10**9),
                tracer=collector,
            )
            pkg = engine.build_context(query)
            trace = collector.trace()
            # No symbol discovery -> no extra candidates synthesized.
            self.assertEqual(trace.symbols, ())
            self.assertEqual(
                [c.path.as_posix() for c in pkg.selected_files], ["alpha.py"]
            )
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()