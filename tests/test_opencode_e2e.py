"""Unit and integration tests for the Phase 25.6 OpenCode end-to-end harness
(benchmarks/opencode_e2e.py).

Unit tests cover corpus integrity, the deterministic routing policy, path /
fingerprint / metric helpers, opencode.json validation and debris-safe
materialisation.  Integration tests drive the real MCP tool functions through
the harness against the small offline fixture repository
(``tests/fixtures/change_plan_repository``) and the real repository corpus,
asserting the honest pass/fail contract (required recall == 1.0 is required,
never assumed).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import opencode_e2e as e2e  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "change_plan_repository"


def _task(**overrides) -> e2e.OpenCodeTask:
    kwargs = {
        "task_id": "t1",
        "title": "Example task",
        "request": "change retrieval behavior",
        "workflow_type": "bug_fix",
    }
    kwargs.update(overrides)
    return e2e.OpenCodeTask(**kwargs)


def _op(name: str, output: dict | None, error: str | None = None) -> e2e.OpResult:
    return e2e.OpResult(name=name, output=output, elapsed_ms=1.0, error=error)


class CorpusTests(unittest.TestCase):
    def test_covers_every_workflow_type(self) -> None:
        tasks = e2e.build_corpus(REPO_ROOT)
        self.assertEqual(
            {t.workflow_type for t in tasks},
            set(e2e.ASSISTED_TOOLS_BY_TYPE),
        )

    def test_unique_task_ids(self) -> None:
        tasks = e2e.build_corpus(REPO_ROOT)
        ids = [t.task_id for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 8)

    def test_surfaces_are_disjoint(self) -> None:
        for task in e2e.build_corpus(REPO_ROOT):
            required = set(task.required_files)
            self.assertFalse(
                required & set(task.supporting_files),
                f"{task.task_id}: required overlaps supporting",
            )

    def test_surface_files_exist_in_repository(self) -> None:
        e2e.validate_corpus(REPO_ROOT, e2e.build_corpus(REPO_ROOT))

    def test_routing_policy_is_closed(self) -> None:
        for tools in e2e.ASSISTED_TOOLS_BY_TYPE.values():
            self.assertTrue(tools)
            self.assertTrue(set(tools) <= e2e.TOOLS)


class TaskValidationTests(unittest.TestCase):
    def test_rejects_empty_fields(self) -> None:
        for kwargs in (
            {"task_id": " "},
            {"title": ""},
            {"request": None},
            {"workflow_type": "unknown"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    _task(**kwargs)

    def test_rejects_duplicate_surface_entries(self) -> None:
        with self.assertRaises(ValueError):
            _task(required_files=("a.py", "a.py"))

    def test_rejects_overlapping_required_and_supporting(self) -> None:
        with self.assertRaises(ValueError):
            _task(
                required_files=("a.py",),
                supporting_files=("a.py",),
            )

    def test_allows_target_overlapping_required(self) -> None:
        task = _task(
            required_files=("a.py",),
            target_files=("a.py",),
        )
        self.assertIn("a.py", task.target_files)

    def test_to_dict_roundtrip(self) -> None:
        task = _task(
            required_files=("repolens/search.py",),
            expected_symbols=("CodeSearcher",),
        )
        data = task.to_dict()
        self.assertEqual(data["task_id"], "t1")
        self.assertEqual(data["required_files"], ["repolens/search.py"])


class HelperTests(unittest.TestCase):
    def test_posix(self) -> None:
        self.assertEqual(e2e._posix(Path("a/b.py")), "a/b.py")
        self.assertEqual(e2e._posix("a\\b.py"), "a/b.py")
        self.assertEqual(e2e._posix(None), "")

    def test_module_of(self) -> None:
        self.assertEqual(
            e2e._module_of("repolens/context/engine.py"),
            "repolens.context.engine",
        )
        self.assertEqual(e2e._module_of("repolens.search"), "repolens.search")

    def test_enum_value(self) -> None:
        from enum import Enum

        class Risk(Enum):
            HIGH = "high"

        self.assertEqual(e2e._enum_value(Risk.HIGH), "high")
        self.assertEqual(e2e._enum_value("low"), "low")

    def test_strip_root_recursive(self) -> None:
        root = Path("/tmp/repo")
        payload = {
            "path": "/tmp/repo/repolens/search.py",
            "nested": ["/tmp/repo/a.py", "plain", 1],
            "other": "/elsewhere/b.py",
        }
        cleaned = e2e._strip_root(payload, root)
        self.assertEqual(cleaned["path"], "<root>/repolens/search.py")
        self.assertEqual(cleaned["nested"], ["<root>/a.py", "plain", 1])
        self.assertEqual(cleaned["other"], "/elsewhere/b.py")

    def test_fingerprint_is_deterministic_and_root_stable(self) -> None:
        payload = {"tool": [{"path": "repolens/search.py"}], "n": 3}
        a = e2e.canonical_fingerprint(payload, Path("/tmp/repoA"))
        b = e2e.canonical_fingerprint(payload, Path("/tmp/repoB"))
        self.assertEqual(a, b)
        self.assertEqual(a, e2e.canonical_fingerprint(payload, Path("/tmp/repoA")))

    def test_fingerprint_is_input_sensitive(self) -> None:
        a = e2e.canonical_fingerprint({"tool": ["a"]}, Path("/tmp/r"))
        b = e2e.canonical_fingerprint({"tool": ["b"]}, Path("/tmp/r"))
        self.assertNotEqual(a, b)

    def test_fingerprint_strips_wall_clock_latency(self) -> None:
        payload = {
            "statistics": {"build_time": 0.7065, "count": 40},
            "path": "repolens/context/config.py",
        }
        a = e2e.canonical_fingerprint(payload, Path("/tmp/r"))
        shifted = {
            "statistics": {"build_time": 0.7441, "count": 40},
            "path": "repolens/context/config.py",
        }
        self.assertEqual(a, e2e.canonical_fingerprint(shifted, Path("/tmp/r")))
        self.assertIn("build_time", e2e._LATENCY_KEYS)

    def test_target_matches_module_and_file(self) -> None:
        task = _task(target_files=("repolens/context/engine.py",))
        self.assertTrue(
            e2e._target_matches(task, {"target": "repolens.context.engine"})
        )
        self.assertTrue(
            e2e._target_matches(task, {"target": "repolens/context/engine.py"})
        )
        self.assertFalse(
            e2e._target_matches(task, {"target": "something.else"})
        )

    def test_plan_surfaces_target(self) -> None:
        task = _task(target_files=("repolens/context/engine.py",))
        plan = {
            "primary_target": {"target": "not.it"},
            "affected_files": [
                {"path": Path("repolens/context/engine.py")},
            ],
            "tests": [],
        }
        self.assertTrue(e2e._plan_surfaces_target(task, plan, set()))
        task_empty = _task(target_files=())
        self.assertTrue(e2e._plan_surfaces_target(task_empty, None, set()))

    def test_surfaced_files_unions_all_tools(self) -> None:
        ops = [
            _op(
                "change_context",
                {
                    "selected_files": [
                        {"path": Path("repolens/search.py")},
                        {"path": "tests/test_search.py"},
                    ]
                },
            ),
            _op(
                "change_plan",
                {"affected_files": [{"path": "repolens/parser.py"}], "tests": []},
            ),
            _op(
                "analyze_impact",
                {"items": [{"path": "repolens/graph.py"}]},
            ),
            _op(
                "inspect_symbol",
                {
                    "nodes": [{"file": "repolens/context/expansion.py"}],
                    "callers": [{"file": "repolens/context/engine.py"}],
                    "callees": [],
                },
            ),
            _op(
                "architecture_candidates",
                {"candidates": [{"file": "repolens/architecture.py"}]},
            ),
            _op("architecture_candidates_echo", {"candidates": []}),
            _op("get_context", {"selected_files": [{"path": "repolens/retrieval.py"}]}),
            _op("broken", None, error="boom"),
        ]
        self.assertEqual(
            e2e._surfaced_files(ops),
            {
                "repolens/search.py",
                "tests/test_search.py",
                "repolens/parser.py",
                "repolens/graph.py",
                "repolens/context/expansion.py",
                "repolens/context/engine.py",
                "repolens/architecture.py",
                "repolens/retrieval.py",
            },
        )

    def test_materialise_repo_skips_debris(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            (src / "repolens").mkdir(parents=True)
            (src / ".benchmark_data" / "repo").mkdir(parents=True)
            (src / ".venv").mkdir()
            (src / "node_modules").mkdir()
            (src / ".git").mkdir()
            (src / "__pycache__").mkdir()
            (src / "models").mkdir()
            (src / "repolens" / "search.py").write_text("")
            (src / ".benchmark_data" / "repo" / "console.py").write_text("")
            (src / "keep.txt").write_text("keep")
            e2e.materialise_repo(src, dst)
            self.assertTrue((dst / "repolens" / "search.py").is_file())
            self.assertTrue((dst / "keep.txt").is_file())
            self.assertFalse((dst / ".benchmark_data").exists())
            self.assertFalse((dst / ".venv").exists())
            self.assertFalse((dst / "node_modules").exists())
            self.assertFalse((dst / ".git").exists())
            self.assertFalse((dst / "__pycache__").exists())
            self.assertFalse((dst / "models").exists())


def _valid_config(root: Path, enabled: bool = True, repo: str | None = None) -> dict:
    return {
        "mcp": {
            "repolens": {
                "type": "local",
                "command": [
                    f"{root / '.venv' / 'bin' / 'python'}",
                    "-m",
                    "repolens.mcp",
                    "--repo",
                    repo or str(root),
                ],
                "cwd": str(root),
                "enabled": enabled,
            }
        }
    }


class ConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _write(self, payload: str) -> Path:
        path = self.root / "opencode.json"
        path.write_text(payload, encoding="utf-8")
        return path

    def test_valid_config(self) -> None:
        self._write(json.dumps(_valid_config(self.root)))
        result = e2e.validate_opencode_config(self.root)
        self.assertTrue(result["present"])
        self.assertTrue(result["valid"])

    def test_missing_config_is_invalid(self) -> None:
        result = e2e.validate_opencode_config(self.root)
        self.assertFalse(result["present"])
        self.assertFalse(result["valid"])

    def test_invalid_json(self) -> None:
        self._write("{not json")
        self.assertFalse(e2e.validate_opencode_config(self.root)["valid"])

    def test_missing_repolens_entry(self) -> None:
        self._write(json.dumps({"mcp": {"other": {"enabled": True}}}))
        self.assertFalse(e2e.validate_opencode_config(self.root)["valid"])

    def test_disabled_server(self) -> None:
        self._write(json.dumps(_valid_config(self.root, enabled=False)))
        self.assertFalse(e2e.validate_opencode_config(self.root)["valid"])

    def test_wrong_repo_path(self) -> None:
        self._write(
            json.dumps(_valid_config(self.root, repo="/somewhere/else"))
        )
        self.assertFalse(e2e.validate_opencode_config(self.root)["valid"])

    def test_real_opencode_json_validates(self) -> None:
        result = e2e.validate_opencode_config(REPO_ROOT)
        self.assertTrue(result["present"])
        self.assertTrue(result["valid"])


class FixtureWorkflowTests(unittest.TestCase):
    """End-to-end wiring against the small offline fixture repository."""

    def _fixture_task(self) -> e2e.OpenCodeTask:
        return e2e.OpenCodeTask(
            task_id="fixture-checkout",
            title="Checkout rejects empty carts",
            request="make checkout reject empty carts",
            workflow_type="bug_fix",
            required_files=("app/services/checkout.py",),
            supporting_files=("app/validators/checkout.py", "app/api/checkout.py"),
            expected_tests=("tests/test_checkout.py",),
            expected_symbols=("checkout",),
            target_files=("app/services/checkout.py",),
        )

    def test_assisted_workflow_passes_on_fixture(self) -> None:
        env = e2e.WorkflowEnvironment(FIXTURE)
        task = self._fixture_task()
        result = e2e.evaluate_task(task, env, mode="assisted")
        self.assertTrue(result["workflow_ok"])
        self.assertEqual(result["required_recall"], 1.0)
        self.assertEqual(result["test_recall"], 1.0)
        self.assertTrue(result["budget"]["compliant"])
        self.assertEqual(result["invalid_files"], [])
        self.assertEqual(len(result["errors"]), 0)

    def test_assisted_tool_selection_is_expectation_driven(self) -> None:
        env = e2e.WorkflowEnvironment(FIXTURE)
        base = self._fixture_task()
        ops = e2e.run_workflow(base, env, mode="assisted")
        self.assertEqual({op.name for op in ops}, {"change_plan", "change_context"})

        investigate = e2e.OpenCodeTask(
            task_id="fixture-impact",
            title="Impact",
            request="what is affected if app/services/checkout.py changes",
            workflow_type="impact_investigation",
            required_files=("app/services/checkout.py",),
            expected_symbols=("checkout",),
            target_files=("app/services/checkout.py",),
        )
        ops = e2e.run_workflow(investigate, env, mode="assisted")
        self.assertIn("analyze_impact", {op.name for op in ops})
        self.assertNotIn("change_context", {op.name for op in ops})

    def test_baseline_only_runs_get_context(self) -> None:
        env = e2e.WorkflowEnvironment(FIXTURE)
        ops = e2e.run_workflow(self._fixture_task(), env, mode="baseline")
        self.assertEqual({op.name for op in ops}, {"get_context"})


if __name__ == "__main__":
    unittest.main()