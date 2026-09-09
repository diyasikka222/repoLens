"""Phase 25.6 — OpenCode end-to-end workflow evaluation.

This benchmark validates RepoLens through realistic OpenCode-style agent
workflows by driving the *actual* MCP tool implementations through the same
factories and functions the opencode.json server config starts (see
:mod:`repolens.mcp.launcher`).  No agentic loop is simulated, no LLM is
consulted and nothing in the production package is modified: the harness only
executes public ``run_*`` MCP functions and reasons about their structured
responses.

Two workflow modes are compared per task:

- ``baseline`` — the core ``get_context`` tool only (the pre-M24 surface).
- ``assisted``  — the additive change-plan / change-context tools plus, for
  the investigation and architecture tasks, the additive impact / inspect /
  architecture tools.  This mirrors the tool set registered through
  ``build_mcp_server`` today.

Only the tools each task's workflow actually needs are invoked (a fixed,
documented routing policy below); tools that were unused are neither measured
nor penalised.

Task ground truth (``required_files``, ``supporting_files``,
``expected_tests``, ``expected_symbols``) is an explicit, pre-declared surface
per task — the same surface discipline used and validated by the P25.2/P25.3
evaluation framework (:mod:`repolens.agent_evaluation`).  Success is honest:
an assisted/baseline run passes only when the surfaced tool output actually
satisfies that surface (required-file recall == 1.0, budget compliance, no
invalid files, correct plan/risk/architecture/impact signals).

Determinism
-----------
The repository is materialised once into a pristine temp copy that mirrors
``repolens.production_benchmark._SKIP_NAMES`` (virtualenvs, caches, quickfix
dirs and the ``.benchmark_data`` external-benchmark download are skipped) so
measurements reflect the tracked project tree on a fresh checkout rather than
local benchmark debris.  Every tool output, metric and fingerprint excludes
wall-clock latency and is computed from the deterministic, JSON-serializable
tool responses (repo-relative paths), so two consecutive runs of this script
against the same tree produce byte-identical fingerprints.

Usage
-----
    python benchmarks/opencode_e2e.py [root] [--mode assisted|baseline|both]
        [--diagnostics] [--task ID ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repolens.mcp.deps import build_firewall
from repolens.mcp.launcher import (
    make_architecture_factory,
    make_change_plan_factory,
    make_engine_factory,
    make_impact_analyzer_factory,
    make_inspect_factory,
)
from repolens.mcp.tool import run_get_context
from repolens.mcp.impact_tool import run_analyze_impact
from repolens.mcp.inspect_tool import run_inspect_symbol
from repolens.mcp.architecture_tool import run_architecture_candidates
from repolens.mcp.change_plan_tool import (
    run_change_context,
    run_change_plan,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Matches the opencode.json server start (launcher default-max-tokens).
DEFAULT_MAX_TOKENS = 8000

#: Architecture-candidate double-call used to measure ordering determinism.
ARCH_CANDIDATE_LIMIT = 20
ARCH_CANDIDATE_DEPTH = 2

#: Top-level names skipped when materialising the pristine working copy.
#: Mirrors ``repolens.production_benchmark._SKIP_NAMES`` so the evaluated tree
#: is the tracked project on a fresh checkout, not local benchmark debris.
SKIP_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".benchmark_data",
    ".pytest_cache",
    ".repolens-cache",
    ".repolens-index",
    ".repolens_embeddings",
    "models",
    ".cache",
    ".eggs",
    "*.egg-info",
}

#: MCP tool names the harness may invoke.
TOOLS = frozenset(
    {
        "change_plan",
        "change_context",
        "architecture_candidates",
        "analyze_impact",
        "inspect_symbol",
        "get_context",
    }
)

#: Fixed, deterministic tool-selection policy per workflow type (assisted mode).
#: This encodes *which* additive tools suit each task; tools not listed for a
#: task are never invoked, so neither measured nor penalised.
ASSISTED_TOOLS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "bug_fix": ("change_plan", "change_context"),
    "feature": ("change_plan", "change_context"),
    "refactor": ("change_plan", "change_context"),
    "api_change": ("change_plan", "change_context"),
    "impact_investigation": ("analyze_impact", "inspect_symbol"),
    "architecture_change": (
        "change_plan",
        "change_context",
        "architecture_candidates",
    ),
    "test_change": ("change_plan", "change_context"),
    # An ambiguous request is met with a graceful, structured plan response
    # rather than a context pull; see ``run_workflow``.
    "ambiguous": ("change_plan",),
}


def _posix(path: Any) -> str:
    """Normalise a path-like value (``Path``, `str`, or ``None``) to POSIX."""
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def _module_of(path: str) -> str:
    """Return the dotted-module form of a repo-relative file path."""
    stripped = path.rstrip("/")
    if stripped.endswith(".py"):
        stripped = stripped[: -len(".py")]
    return stripped.replace("/", ".")


class OpenCodeTask:
    """A single end-to-end OpenCode work request with explicit ground truth.

    ``required_files`` are the files the tool output must surface for the task
    to pass; ``supporting_files`` are useful but optional context;
    ``expected_tests`` are the test files that validate the change;
    ``expected_symbols`` are definitions/invocations the surfaced context must
    mention; ``target_files`` are the primary change targets the plan must
    identify.  ``workflow_type`` selects the assisted tool set above.
    ```

    Surfaces must be disjoint and must reference only real repository paths.
    """

    def __init__(
        self,
        *,
        task_id: str,
        title: str,
        request: str,
        workflow_type: str,
        required_files: tuple[str, ...] = (),
        supporting_files: tuple[str, ...] = (),
        expected_tests: tuple[str, ...] = (),
        expected_symbols: tuple[str, ...] = (),
        target_files: tuple[str, ...] = (),
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task id must be a non-empty string")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("task title must be a non-empty string")
        if not isinstance(request, str) or not request.strip():
            raise ValueError("task request must be a non-empty string")
        if workflow_type not in ASSISTED_TOOLS_BY_TYPE:
            raise ValueError(f"unknown workflow_type: {workflow_type!r}")
        for label in (
            "required_files",
            "supporting_files",
            "expected_tests",
            "expected_symbols",
            "target_files",
        ):
            values = locals()[label]
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must not contain duplicates")
        required = set(required_files)
        if required & set(supporting_files):
            raise ValueError("required_files and supporting_files must be disjoint")
        self.task_id = task_id
        self.title = title
        self.request = request
        self.workflow_type = workflow_type
        self.required_files = tuple(required_files)
        self.supporting_files = tuple(supporting_files)
        self.expected_tests = tuple(expected_tests)
        self.expected_symbols = tuple(expected_symbols)
        self.target_files = tuple(target_files)
        self.max_tokens = max_tokens

    @property
    def surface_files(self) -> tuple[str, ...]:
        return (
            *self.required_files,
            *self.supporting_files,
            *self.expected_tests,
            *self.target_files,
        )

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "request": self.request,
            "workflow_type": self.workflow_type,
            "required_files": list(self.required_files),
            "supporting_files": list(self.supporting_files),
            "expected_tests": list(self.expected_tests),
            "expected_symbols": list(self.expected_symbols),
            "target_files": list(self.target_files),
            "max_tokens": self.max_tokens,
        }


def build_corpus(root: Path) -> tuple[OpenCodeTask, ...]:
    """Return the deterministic OpenCode task corpus.

    Requests reuse the P25.3-validated surface (``repolens/agent_evaluation.py``
    + ``benchmarks/agent_evaluation.py``) where applicable; the two new tasks
    (investigation, ambiguity) extend it.  Ground truth references only real
    repository files; ``validate_corpus`` enforces existence.
    """
    del root
    return (
        OpenCodeTask(
            task_id="search-ranking-bug",
            title="Lexical ranker overweights exact symbol names",
            request=(
                "The CodeSearcher ranking lets an exact symbol-name match "
                "overwhelm every other signal, so broad queries surface false "
                "positives on that symbol. Fix the scoring so import and source "
                "tokens still influence the ranking."
            ),
            workflow_type="bug_fix",
            required_files=("repolens/search.py",),
            supporting_files=("repolens/parser.py",),
            expected_tests=("tests/test_search.py",),
            expected_symbols=(
                "CodeSearcher",
                "_score",
            ),
            target_files=("repolens/search.py",),
        ),
        OpenCodeTask(
            task_id="context-budget-option",
            title="Configurable maximum context budget",
            request=(
                "Thread a configurable maximum token budget through ContextEngine "
                "and ContextBudget: add an option to the context pipeline "
                "configuration that caps the budget in tokens, and make the "
                "context package respect it before any file is selected."
            ),
            workflow_type="feature",
            required_files=(
                "repolens/context/config.py",
                "repolens/context/engine.py",
            ),
            supporting_files=(
                "repolens/context/package.py",
                "repolens/context/budget.py",
            ),
            expected_tests=("tests/test_context_engine.py",),
            expected_symbols=(
                "ContextBudget",
                "ContextEngine",
            ),
            target_files=("repolens/context/config.py",),
        ),
        OpenCodeTask(
            task_id="context-ranking-refactor",
            title="Extract an explicit dependency-expansion ranking stage",
            request=(
                "Extract rank_candidates into an explicit dependency-expansion "
                "ranking stage so ContextEngine ranks dependency-expanded "
                "candidates separately from the primary ranking instead of "
                "mixing them together."
            ),
            workflow_type="refactor",
            required_files=(
                "repolens/context/engine.py",
                "repolens/context/ranking.py",
            ),
            supporting_files=(
                "repolens/context/candidate.py",
                "repolens/context/expansion.py",
            ),
            expected_tests=("tests/test_context_engine.py",),
            expected_symbols=(
                "ContextEngine",
                "rank_candidates",
            ),
            target_files=("repolens/context/engine.py",),
        ),
        OpenCodeTask(
            task_id="change-plan-risk-signal",
            title="Expose a risk indicator on change_plan",
            request=(
                "Modify the MCP change_plan tool so every planned change exposes "
                "an explicit risk indicator, and keep the JSON response "
                "deterministic and bounded."
            ),
            workflow_type="api_change",
            required_files=(
                "repolens/mcp/change_plan_tool.py",
                "repolens/change_plan.py",
            ),
            supporting_files=(
                "repolens/change_context.py",
                "repolens/mcp/server.py",
            ),
            expected_tests=("tests/test_mcp_change_plan.py",),
            expected_symbols=(
                "run_change_plan",
                "ChangePlanEngine",
                "ChangePlanState",
            ),
            target_files=("repolens/mcp/change_plan_tool.py",),
        ),
        OpenCodeTask(
            task_id="dependency-expansion-blast-radius",
            title="Blast radius of dependency expansion",
            request=(
                "Investigate the blast radius of changing dependency expansion "
                "in the context pipeline. Which modules, callers, dependents "
                "and tests rely on repolens/context/expansion.py, in particular "
                "the expand_dependencies function and the ExpandedNode structure?"
            ),
            workflow_type="impact_investigation",
            required_files=("repolens/context/expansion.py",),
            supporting_files=(
                "repolens/context/engine.py",
                "repolens/graph.py",
            ),
            expected_tests=("tests/test_context_engine.py",),
            expected_symbols=(
                "expand_dependencies",
                "ExpandedNode",
            ),
            target_files=("repolens/context/expansion.py",),
        ),
        OpenCodeTask(
            task_id="architecture-retrieval-ordering",
            title="Stable architecture candidate ordering",
            request=(
                "Fix architecture retrieval producing unstable candidate ordering "
                "when architecture signals tie, so repeated queries return the "
                "same ranked surface."
            ),
            workflow_type="architecture_change",
            required_files=("repolens/architecture_retrieval.py",),
            supporting_files=("repolens/architecture.py",),
            expected_tests=("tests/test_architecture_retrieval.py",),
            expected_symbols=(
                "extract_architecture_signals",
                "ArchitectureRetrievalConfig",
                "ArchitectureCandidate",
            ),
            target_files=("repolens/architecture_retrieval.py",),
        ),
        OpenCodeTask(
            task_id="hybrid-rrf-tests",
            title="Unit tests for RRF hybrid fusion",
            request=(
                "Add unit tests for the HybridSearcher RRF fusion path "
                "(_search_rrf) in repolens/retrieval.py, covering the "
                "reciprocal-rank-fusion strategy."
            ),
            workflow_type="test_change",
            required_files=("tests/test_retrieval.py",),
            supporting_files=("repolens/retrieval.py",),
            expected_tests=("tests/test_retrieval.py",),
            expected_symbols=("HybridSearcher", "_search_rrf"),
            target_files=("tests/test_retrieval.py",),
        ),
        OpenCodeTask(
            task_id="ambiguous-request",
            title="Vague dependency-related improvement request",
            request=(
                "Something about dependencies seems off somewhere and I think we "
                "should improve it, can you look into it and figure out what "
                "might need to change? Like, maybe the way things are expanded or "
                "how files depend on each other — I don't know. Anyway, take a "
                "look."
            ),
            workflow_type="ambiguous",
            required_files=(),
            supporting_files=(),
            expected_tests=(),
            expected_symbols=(),
            target_files=(),
        ),
    )


def validate_corpus(root: Path, tasks: tuple[OpenCodeTask, ...]) -> None:
    """Fail loudly when corpus ground truth references missing files."""
    missing: list[str] = []
    for task in tasks:
        for path in task.surface_files:
            if not (root / path).is_file():
                missing.append(f"{task.task_id}: {path}")
    if missing:
        raise ValueError(
            "corpus references missing repository files:\n- " + "\n- ".join(missing)
        )


# ---------------------------------------------------------------------------
# Pristine working copy
# ---------------------------------------------------------------------------


def materialise_repo(root: Path, dest: Path) -> None:
    """Copy ``root`` into ``dest`` skipping local-only debris (offline)."""
    shutil.copytree(root, dest, ignore=shutil.ignore_patterns(*SKIP_NAMES, "*.pyc"))


def _strip_root(value: Any, root: Path) -> Any:
    """Recursively normalise the working-root prefix out of a JSON value.

    Tool outputs are repo-relative, but this guards fingerprints against the
    (unstable per-run) temp copy path regardless.
    """
    root_prefix = str(root)
    if isinstance(value, dict):
        return {k: _strip_root(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_root(v, root) for v in value]
    if isinstance(value, str):
        if value.startswith(root_prefix):
            return "<root>" + value[len(root_prefix) :]
        return value
    return value


#: Wall-clock fields embedded in production tool payloads (e.g.
#: ``change_plan.statistics.build_time``).  The fingerprint contract excludes
#: latency by definition, so these are stripped before hashing; everything else
#: in the payload is byte-identical across runs and hash seeds.
_LATENCY_KEYS = frozenset({"build_time", "elapsed_ms", "latency_ms", "duration_s", "elapsed_seconds"})


def _strip_latency(value: Any) -> Any:
    """Recursively drop wall-clock measurement keys from a JSON value."""
    if isinstance(value, dict):
        return {
            k: _strip_latency(v)
            for k, v in value.items()
            if k not in _LATENCY_KEYS
        }
    if isinstance(value, list):
        return [_strip_latency(v) for v in value]
    return value


def canonical_fingerprint(payload: dict, root: Path) -> str:
    """Stable sha256 over the deterministic portion of a payload."""
    clean = _strip_root(payload, root)
    clean = _strip_latency(clean)
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_metrics(metrics: dict, root: Path) -> str:
    """Fingerprint a task's metrics plus its tool outputs (no latency)."""
    return canonical_fingerprint(metrics, root)


# ---------------------------------------------------------------------------
# Runtime environment (one per mode = one MCP-server-like set of factories)
# ---------------------------------------------------------------------------


class WorkflowEnvironment:
    """Lazily built, shared MCP factories over one repo (like one server)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.firewall = build_firewall()
        self._engine = None
        self._change_plan = None
        self._impact = None
        self._inspect = None
        self._architecture = None

    def engine_factory(self, *, max_tokens=None, dependency_depth=None):
        if self._engine is None:
            self._engine = make_engine_factory(
                self.root, default_max_tokens=DEFAULT_MAX_TOKENS,
                default_dependency_depth=1,
            )
        return self._engine(
            max_tokens=max_tokens, dependency_depth=dependency_depth
        )

    def change_plan_factory(self):
        if self._change_plan is None:
            self._change_plan = make_change_plan_factory(self.root)
        return self._change_plan

    def impact_factory(self):
        if self._impact is None:
            self._impact = make_impact_analyzer_factory(self.root)
        return self._impact

    def inspect_factory(self):
        if self._inspect is None:
            self._inspect = make_inspect_factory(self.root)
        return self._inspect

    def architecture_factory(self):
        if self._architecture is None:
            self._architecture = make_architecture_factory(self.root)
        return self._architecture


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


@dataclass
class OpResult:
    """Result of one tool invocation."""

    name: str
    output: dict | None
    elapsed_ms: float
    error: str | None = None

    def ok(self) -> bool:
        return self.error is None


def _call(name: str, fn) -> OpResult:
    start = time.perf_counter()
    try:
        output = fn()
        elapsed = (time.perf_counter() - start) * 1000.0
        return OpResult(name=name, output=output, elapsed_ms=round(elapsed, 2))
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - start) * 1000.0
        return OpResult(
            name=name,
            output=None,
            elapsed_ms=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}",
        )


def run_workflow(
    task: OpenCodeTask,
    env: WorkflowEnvironment,
    *,
    mode: str,
) -> list[OpResult]:
    """Execute the deterministic tool workflow for ``task`` in ``mode``."""
    tools = ("get_context",) if mode == "baseline" else ASSISTED_TOOLS_BY_TYPE[task.workflow_type]
    ops: list[OpResult] = []

    if "get_context" in tools:
        ops.append(
            _call(
                "get_context",
                lambda: run_get_context(
                    env.engine_factory,
                    env.firewall,
                    task.request,
                    max_tokens=task.max_tokens,
                    dependency_depth=1,
                ),
            )
        )
    if "change_plan" in tools:
        ops.append(
            _call(
                "change_plan",
                lambda: run_change_plan(
                    env.change_plan_factory(),
                    task.request,
                ),
            )
        )
    if "change_context" in tools:
        ops.append(
            _call(
                "change_context",
                lambda: run_change_context(
                    env.engine_factory,
                    env.firewall,
                    env.change_plan_factory(),
                    task.request,
                    max_tokens=task.max_tokens,
                ),
            )
        )
    if "analyze_impact" in tools and task.target_files:
        target = task.target_files[0]
        ops.append(
            _call(
                "analyze_impact",
                lambda: run_analyze_impact(
                    env.impact_factory(), target
                ),
            )
        )
    if "inspect_symbol" in tools and task.expected_symbols:
        symbol = task.expected_symbols[0]
        ops.append(
            _call(
                "inspect_symbol",
                lambda: run_inspect_symbol(
                    env.inspect_factory(), symbol
                ),
            )
        )
    if "architecture_candidates" in tools:
        first = _call(
            "architecture_candidates",
            lambda: run_architecture_candidates(
                env.architecture_factory(),
                task.request,
                limit=ARCH_CANDIDATE_LIMIT,
                max_depth=ARCH_CANDIDATE_DEPTH,
            ),
        )
        ops.append(first)
        if first.ok():
            second = _call(
                "architecture_candidates_echo",
                lambda: run_architecture_candidates(
                    env.architecture_factory(),
                    task.request,
                    limit=ARCH_CANDIDATE_LIMIT,
                    max_depth=ARCH_CANDIDATE_DEPTH,
                ),
            )
            ops.append(second)
    return ops


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _surfaced_files(ops: list[OpResult]) -> set[str]:
    """All repo-relative paths surfaced by the executed tools (what an agent
    would actually see)."""
    surfaced: set[str] = set()
    for op in ops:
        if not op.ok() or not op.output:
            continue
        out = op.output
        if op.name == "get_context" or op.name == "change_context":
            for item in out.get("selected_files", []):
                surfaced.add(_posix(item.get("path")))
        elif op.name == "change_plan":
            for key in ("affected_files", "tests"):
                for item in out.get(key, []):
                    surfaced.add(_posix(item.get("path")))
        elif op.name == "analyze_impact":
            for item in out.get("items", []):
                surfaced.add(_posix(item.get("path")))
        elif op.name.startswith("inspect_symbol"):
            for key in ("nodes", "callers", "callees"):
                for item in out.get(key, []):
                    surfaced.add(_posix(item.get("file")))
        elif op.name.startswith("architecture_candidates"):
            for item in out.get("candidates", []):
                surfaced.add(_posix(item.get("file")))
    return {p for p in surfaced if p}


def _symbol_anchor_re(symbol: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(symbol) + r"\b")


def _matched_symbols(
    task: OpenCodeTask,
    surfaced: set[str],
    root: Path,
    ops: list[OpResult],
) -> tuple[str, ...]:
    """Symbols that surface: identifiers in surfaced files or tool metadata."""
    matched: list[str] = []
    for symbol in task.expected_symbols:
        ok = False
        for path in surfaced:
            file_path = root / path
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _symbol_anchor_re(symbol).search(text):
                ok = True
                break
        if not ok:
            for op in ops:
                if not op.ok() or not op.output:
                    continue
                metadata_symbols = set(op.output.get("matched_symbols", ())) or set()
                if symbol in metadata_symbols:
                    ok = True
                    break
        if ok:
            matched.append(symbol)
    return tuple(matched)


def _enum_value(value):
    """Unwrap an enum to its scalar value (JSON-safe)."""
    return getattr(value, "value", value)


def _primary_target(ops: list[OpResult]) -> dict | None:
    for op in ops:
        if op.name == "change_plan" and op.ok() and op.output:
            return op.output.get("primary_target")
    return None


def _plan_output(ops: list[OpResult]) -> dict | None:
    for op in ops:
        if op.name == "change_plan" and op.ok() and op.output:
            return op.output
    return None


def _target_matches(task: OpenCodeTask, plan_primary: dict | None) -> bool:
    """Whether the plan's primary target corresponds to a task target file."""
    if not task.target_files:
        return True
    target_file = task.target_files[0]
    target_module = _module_of(target_file)
    if plan_primary:
        primary = _posix(plan_primary.get("target") or "")
        if primary in (target_file, target_module):
            return True
        if target_module and (
            primary.endswith("." + target_module)
            or target_module.endswith("." + primary)
        ):
            return True
        if _posix(primary) == _module_of(target_file):
            return True
    return False


def _plan_surfaces_target(
    task: OpenCodeTask, plan_output: dict | None, surfaced: set[str]
) -> bool:
    """The plan identifies a task target directly (primary, affected, or a
    test) or the target surfaced in context."""
    if not task.target_files:
        return True
    target = task.target_files[0]
    target_module = _module_of(target)
    if plan_output is not None:
        if _target_matches(task, plan_output.get("primary_target")):
            return True
        plan_paths: set[str] = set()
        for key in ("affected_files", "tests"):
            for item in plan_output.get(key, []):
                plan_paths.add(_posix(item.get("path")))
        if target in plan_paths:
            return True
        if any(_module_of(path) == target_module for path in plan_paths):
            return True
    return target in surfaced


def evaluate_task(
    task: OpenCodeTask,
    env: WorkflowEnvironment,
    *,
    mode: str,
) -> dict:
    """Run ``task`` in ``mode`` and compute the full metric record."""
    ops = run_workflow(task, env, mode=mode)
    surfaced = _surfaced_files(ops)
    errors = [op.error for op in ops if not op.ok()]
    plan_primary = _primary_target(ops)
    tool_outputs = {op.name: op.output for op in ops if op.ok() and op.output is not None}
    matched_symbols = _matched_symbols(task, surfaced, env.root, ops)

    required = set(task.required_files)
    supporting = set(task.supporting_files)
    tests = set(task.expected_tests)

    def recall(wanted: set[str]) -> float:
        return len(wanted & surfaced) / len(wanted) if wanted else 1.0

    required_files_missed = tuple(sorted(required - surfaced))
    supporting_files_missed = tuple(sorted(supporting - surfaced))
    tests_missed = tuple(sorted(tests - surfaced))
    symbols_missed = tuple(
        sorted(set(task.expected_symbols) - set(matched_symbols))
    )

    budget = {"compliant": True, "used": None, "max": task.max_tokens}
    context_ops = [op for op in ops if op.name in ("get_context", "change_context")]
    for op in context_ops:
        if op.ok() and op.output:
            bucket = op.output.get("budget") or {}
            max_tokens = bucket.get("max_tokens") or task.max_tokens
            used = op.output.get("total_estimated_tokens") or 0
            budget = {
                "compliant": used <= max_tokens,
                "used": used,
                "max": max_tokens,
            }

    invalid_files = sorted(
        path for path in surfaced if not (env.root / path).is_file()
    )

    plan = None
    plan_op = next((op for op in ops if op.name == "change_plan"), None)
    if plan_op is not None and plan_op.ok() and plan_op.output:
        plan = {
            "status": plan_op.output.get("status"),
            "primary_target": plan_op.output.get("primary_target"),
            "risk": _enum_value(plan_op.output.get("risk")),
            "confidence": plan_op.output.get("confidence"),
            "affected_file_count": len(plan_op.output.get("affected_files", [])),
            "test_count": len(plan_op.output.get("tests", [])),
            "deterministic": plan_op.output.get("deterministic"),
        }

    plan_risk = _enum_value(plan.get("risk")) if plan else None
    plan_signal = (
        plan is not None
        and plan["status"] == "ok"
        and plan_risk in ("low", "medium", "high")
    )
    target_valid = _plan_surfaces_target(task, _plan_output(ops), surfaced)

    impact_signal = None
    impact_op = next((op for op in ops if op.name == "analyze_impact"), None)
    if impact_op is not None and impact_op.ok() and impact_op.output:
        impact_signal = {
            "risk": _enum_value(impact_op.output.get("risk")),
            "item_count": impact_op.output.get("item_count"),
        }

    inspect_signal = None
    inspect_op = next((op for op in ops if op.name == "inspect_symbol"), None)
    if inspect_op is not None and inspect_op.ok() and inspect_op.output:
        inspect_signal = {
            "node_count": inspect_op.output.get("node_count"),
            "caller_count": inspect_op.output.get("caller_count"),
            "callee_count": inspect_op.output.get("callee_count"),
        }

    arch_signal = None
    arch_ops = [op for op in ops if op.name == "architecture_candidates"]
    if arch_ops:
        deterministic = False
        if arch_ops[0].ok() and arch_ops[0].output:
            first = arch_ops[0].output.get("candidates", [])
            if len(arch_ops) == 2 and arch_ops[1].ok() and arch_ops[1].output:
                deterministic = arch_ops[1].output.get("candidates", []) == first
            elif len(arch_ops) == 1:
                deterministic = True
            arch_signal = {
                "candidate_count": len(first),
                "deterministic_repeat": deterministic,
            }

    metrics = {
        "task_id": task.task_id,
        "mode": mode,
        "request": task.request,
        "tool_outputs": tool_outputs,
        "surfaced_files": sorted(surfaced),
        "required_recall": recall(required),
        "supporting_recall": recall(supporting),
        "test_recall": recall(tests),
        "symbol_recall": (len(matched_symbols) / len(task.expected_symbols))
        if task.expected_symbols
        else 1.0,
        "required_files_missed": required_files_missed,
        "supporting_files_missed": supporting_files_missed,
        "tests_missed": tests_missed,
        "symbols_missed": symbols_missed,
        "matched_symbols": matched_symbols,
        "plan_signal": plan_signal,
        "target_valid": target_valid,
        "plan": plan,
        "impact_signal": impact_signal,
        "inspect_signal": inspect_signal,
        "arch_signal": arch_signal,
        "budget": budget,
        "invalid_files": invalid_files,
        "errors": errors,
    }

    ambiguous = task.workflow_type == "ambiguous"
    surfaced_ok = len(errors) == 0 and budget["compliant"] and not invalid_files
    covered = required <= surfaced and len(symbols_missed) == 0
    if mode == "baseline":
        # The core surface only: get_context must surface the required files
        # within budget; plan/architecture/impact signals do not apply.
        workflow_ok = surfaced_ok and covered
    elif ambiguous:
        # Graceful, structured response to a vague request; no targets
        # expected, and the plan must not fabricate a confident target.
        plan_unresolved = (
            plan is not None
            and (plan.get("primary_target") or {}).get("target")
            and (plan.get("confidence") or "") == "unresolved"
        )
        workflow_ok = surfaced_ok and plan_signal and not plan_unresolved
    elif task.workflow_type == "impact_investigation":
        # Investigation is validated by the impact + call-graph signals the
        # request asked for, not by a change plan.
        workflow_ok = (
            surfaced_ok
            and covered
            and impact_signal is not None
            and (impact_signal["item_count"] or 0) > 0
            and inspect_signal is not None
            and (inspect_signal["node_count"] or 0) > 0
            and target_valid
        )
    elif task.workflow_type == "architecture_change":
        workflow_ok = (
            surfaced_ok
            and covered
            and plan_signal
            and target_valid
            and arch_signal is not None
            and (arch_signal["candidate_count"] or 0) > 0
            and arch_signal["deterministic_repeat"]
        )
    else:
        workflow_ok = (
            surfaced_ok
            and covered
            and plan_signal
            and target_valid
        )
    metrics["workflow_ok"] = workflow_ok
    metrics["fingerprint"] = fingerprint_metrics(metrics, env.root)
    return metrics


# ---------------------------------------------------------------------------
# opencode.json validation (read-only)
# ---------------------------------------------------------------------------


def validate_opencode_config(root: Path) -> dict:
    """Validate the opencode.json RepoLens server entry (never modifies it)."""
    config_path = root / "opencode.json"
    checks: dict[str, bool] = {}
    details: list[str] = []
    if not config_path.is_file():
        return {
            "config_path": str(config_path),
            "present": False,
            "valid": False,
            "checks": {"opencode.json exists": False},
            "details": ["opencode.json is missing (config not validated)."],
        }
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "config_path": str(config_path),
            "present": True,
            "valid": False,
            "checks": {"opencode.json is valid JSON": False},
            "details": [f"JSON parse error: {exc}"],
        }
    if not isinstance(data, dict) or not isinstance(data.get("mcp"), dict):
        return {
            "config_path": str(config_path),
            "present": True,
            "valid": False,
            "checks": {"mcp section is a mapping": False},
            "details": ["expected mcp (dictionary) section in opencode.json."],
        }
    entry = data["mcp"].get("repolens")
    checks["repolens server configured"] = isinstance(entry, dict)
    details.append("mcp.repolens: %s" % ("found" if isinstance(entry, dict) else "missing"))

    expected_repo = str(root.resolve())
    command_ok = False
    repo_arg_ok = False
    enabled_ok = False
    cwd_ok = False
    bootstrap_ok = False
    if isinstance(entry, dict):
        command = entry.get("command")
        command_ok = isinstance(command, list) and bool(command) and all(
            isinstance(part, str) for part in command
        )
        details.append(
            "command: %s" % ("list" if command_ok else "invalid/missing")
        )
        if command_ok:
            parts = [str(p) for p in command]
            if parts[0].endswith("/.venv/bin/python"):
                bootstrap_ok = True
            if "-m" in parts and "repolens.mcp" in parts:
                if "--repo" in parts:
                    idx = parts.index("--repo")
                    if idx + 1 < len(parts):
                        repo_arg_ok = Path(parts[idx + 1]).resolve() == Path(expected_repo)
                    details.append(
                        "--repo points at the evaluated root: %s"
                        % ("yes" if repo_arg_ok else "no")
                    )
        enabled = entry.get("enabled")
        enabled_ok = enabled is True or enabled in {"true", "True"}
        details.append(
            "enabled: %s" % ("true" if enabled_ok else str(enabled))
        )
        cwd = entry.get("cwd")
        cwd_ok = cwd in (None, expected_repo, str(root))
        if cwd:
            details.append("cwd: %s" % str(cwd))
    checks["bootstrap uses .venv python"] = bootstrap_ok
    checks["command runs repolens.mcp --repo <root>"] = command_ok and repo_arg_ok
    checks["server enabled"] = enabled_ok
    checks["cwd is the repository root"] = cwd_ok
    valid = all(checks.values())
    return {
        "config_path": str(config_path),
        "present": True,
        "valid": valid,
        "checks": checks,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------


def _metrics_text(row: dict) -> str:
    budget = row.get("budget") or {}
    used = budget.get("used")
    used = f"{used}/{budget.get('max')}" if used is not None else "n/a"
    signals = []
    plan = row.get("plan_signal")
    signals.append("ok" if plan else "noadj")
    if row.get("impact_signal"):
        signals.append("impact")
    if row.get("inspect_signal"):
        signals.append("inspect")
    if row.get("arch_signal"):
        arch = row["arch_signal"]
        signals.append(
            "arch%d%s" % (arch.get("candidate_count") or 0,
                          "" if arch.get("deterministic_repeat") else "!")
        )
    return (
        f"req {row['required_recall']:.2f} sup {row['supporting_recall']:.2f} "
        f"tst {row['test_recall']:.2f} sym {row['symbol_recall']:.2f} "
        f"tok {used} plan {'+'.join(signals)} "
        f"files {len(row.get('surfaced_files', []))} "
        f"invalid {len(row.get('invalid_files', []))} "
        f"errs {len(row.get('errors', []))}"
    )


def _print_diagnostics(task: OpenCodeTask, row: dict) -> None:
    print(f"\n  -- diagnostics: {task.task_id} ({row['mode']})")
    if row.get("errors"):
        print(f"    tool errors: {row['errors']}")
    for label, missed in (
        ("required files missed", row["required_files_missed"]),
        ("supporting files missed", row["supporting_files_missed"]),
        ("tests missed", row["tests_missed"]),
        ("symbols missed", row["symbols_missed"]),
    ):
        if missed:
            print(f"    {label}: {', '.join(missed)}")
    if row.get("invalid_files"):
        print(f"    invalid surfaced files: {row['invalid_files']}")
    if not row.get("budget", {}).get("compliant", True):
        budget = row["budget"]
        print(f"    budget exceeded: used {budget['used']} > max {budget['max']}")
    if row.get("arch_signal") and not row["arch_signal"].get("deterministic_repeat"):
        print("    architecture candidate ordering NOT deterministic across repeat")
    if row.get("plan") and row.get("plan_signal"):
        plan = row["plan"]
        primary = (plan.get("primary_target") or {}).get("target")
        print(
            f"    plan: primary={primary!r} risk={plan.get('risk')} "
            f"affected={plan.get('affected_file_count')} tests={plan.get('test_count')}"
        )


def build_report(
    root: Path,
    tasks: tuple[OpenCodeTask, ...],
    *,
    modes: tuple[str, ...],
    selected_tasks: tuple[str, ...] | None = None,
) -> dict:
    """Run every task in each requested mode and return the full report."""
    tasks = tuple(t for t in tasks if selected_tasks is None or t.task_id in selected_tasks)
    runs: dict[str, list[dict]] = {}
    for mode in modes:
        env = WorkflowEnvironment(root)
        runs[mode] = [evaluate_task(task, env, mode=mode) for task in tasks]
    return {"root": str(root), "modes": list(modes), "tasks": runs}


def _fmt_report(report: dict) -> str:
    lines: list[str] = []
    for mode in report["modes"]:
        lines.append(f"mode: {mode}")
        lines.append(
            f"  {'task':<34} {'req':>12} {'sup':>12} {'tst':>12} {'sym':>12}  ok"
        )
        for row in report["tasks"][mode]:
            lines.append(
                f"  {row['task_id']:<34} "
                f"{row['required_recall']:>12.3f} {row['supporting_recall']:>12.3f} "
                f"{row['test_recall']:>12.3f} {row['symbol_recall']:>12.3f}  "
                f"{'PASS' if row['workflow_ok'] else 'FAIL'}"
            )
            lines.append("    " + _metrics_text(row))
        passed = sum(1 for row in report["tasks"][mode] if row["workflow_ok"])
        lines.append(f"  passed {passed}/{len(report['tasks'][mode])}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="opencode-e2e",
        description=(
            "Drive the real RepoLens MCP tools through OpenCode-style workflows "
            "and report honest end-to-end metrics."
        ),
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--mode",
        choices=("assisted", "baseline", "both"),
        default="both",
        help="Workflow mode to evaluate (default: both).",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        metavar="ID",
        help="Restrict evaluation to the given task id (repeatable).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print per-task missed files/symbols, invalid files and signals.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"error: repository root is not a directory: {root}")
        return 2
    root = root.resolve()

    tasks = build_corpus(root)
    try:
        validate_corpus(root, tasks)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    selected = tuple(args.task) if args.task else None
    modes = ("assisted", "baseline") if args.mode == "both" else (args.mode,)

    config = validate_opencode_config(root)
    print(f"opencode.json: {'valid' if config['valid'] else 'INVALID'}")
    for detail in config["details"]:
        print(f"  - {detail}")
    if not config["valid"]:
        return 1

    staging = Path(tempfile.mkdtemp(prefix="opencode_e2e_"))
    work_root = staging / "repo"
    print(f"materialising pristine working copy under {staging}")
    materialise_repo(root, work_root)
    print(f"evaluating tasks against {work_root}")

    report = build_report(work_root, tasks, modes=modes, selected_tasks=selected)
    print(_fmt_report(report))

    if args.diagnostics:
        for mode in modes:
            for task in tasks:
                if selected and task.task_id not in selected:
                    continue
                row = next(r for r in report["tasks"][mode] if r["task_id"] == task.task_id)
                _print_diagnostics(task, row)

    overall = all(
        row["workflow_ok"]
        for mode in modes
        for row in report["tasks"][mode]
    )

    per_mode_rows = {mode: report["tasks"][mode] for mode in modes}
    aggregate = {
        "passed_tasks": {
            mode: sum(1 for row in rows if row["workflow_ok"])
            for mode, rows in per_mode_rows.items()
        },
        "total_tasks": {
            mode: len(rows) for mode, rows in per_mode_rows.items()
        },
        "assisted_required_recall": (
            sum(r["required_recall"] for r in per_mode_rows.get("assisted", []))
            / len(per_mode_rows.get("assisted", []))
            if per_mode_rows.get("assisted")
            else None
        ),
        "baseline_required_recall": (
            sum(r["required_recall"] for r in per_mode_rows.get("baseline", []))
            / len(per_mode_rows.get("baseline", []))
            if per_mode_rows.get("baseline")
            else None
        ),
    }
    print("\naggregate:")
    for mode, passed in aggregate["passed_tasks"].items():
        total = aggregate["total_tasks"].get(mode, 0)
        print(f"  {mode}: {passed}/{total} tasks passed")
    if aggregate["assisted_required_recall"] is not None:
        print(
            f"  assisted mean required-recall: {aggregate['assisted_required_recall']:.3f}"
        )
    if aggregate["baseline_required_recall"] is not None:
        print(
            f"  baseline mean required-recall: {aggregate['baseline_required_recall']:.3f}"
        )
    assisted_fp = canonical_fingerprint(
        {
            "mode": "assisted",
            "tasks": [
                row["fingerprint"] for row in report["tasks"].get("assisted", [])
            ],
        },
        work_root,
    )
    baseline_fp = canonical_fingerprint(
        {
            "mode": "baseline",
            "tasks": [
                row["fingerprint"] for row in report["tasks"].get("baseline", [])
            ],
        },
        work_root,
    )
    print("\nfingerprints:")
    print(f"  assisted: {assisted_fp}")
    print(f"  baseline: {baseline_fp}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())