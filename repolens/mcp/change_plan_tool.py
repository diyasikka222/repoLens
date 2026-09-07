"""Change-plan MCP tools (Milestone 24.2).

A thin, additive adapter over :class:`~repolens.change_plan.ChangePlanEngine`
and the change-aware :meth:`repolens.context.ContextEngine.build_context`:

- ``change_plan`` — produce a deterministic change plan for a natural-language
  change request (target discovery, affected files/symbols, callers, callees,
  dependencies, architecture, tests, inspection order, risk, and statistics).
- ``change_context`` — build a change-aware context package by folding the
  plan's inspection candidates into the existing context pipeline (ranking and
  budget), then returning only firewall-cleared, safe context together with a
  deterministic per-file explanation of why each file was included.

Following the established MCP tool pattern each tool:

1. validates its arguments strictly;
2. builds the shared change-plan state via an injected factory (lazy, once);
3. calls only existing public RepoLens APIs;
4. returns only structured, safe, JSON-serializable output.

Shared state (:class:`ChangePlanState`) reuses a single incremental index and
a single default :class:`ChangePlanEngine` for every call; calls that raise
the plan bounds build a *per-call* engine that reuses the shared engine's
already-built dependency graph, symbol index, call graph, architecture graph,
subsystems, impact analyzer, and searcher — so no repository re-parse and no
component rebuild happens per call.

No planning, retrieval, ranking, budgeting, or security logic is reimplemented
here, and no internal Python object is ever returned to the caller.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from repolens.change_plan import ChangePlan, ChangePlanConfig, ChangePlanEngine
from repolens.change_context import (
    ChangeContextOptions,
    explain_change_context,
    plan_response_payload,
)
from repolens.context import ContextEngine, ContextFirewall, ContextPackage
from repolens.context.firewall import FirewallResult, SafeContextPackage
from repolens.mcp.architecture_tool import _target_is_safe
from repolens.mcp.errors import (
    ChangeContextError,
    ChangePlanError,
    InternalError,
    InvalidArgumentsError,
)
from repolens.mcp.tool import validate_max_tokens, validate_query

#: Deterministic caps the MCP layer imposes on top of the plan limits.
MAX_TARGETS = 50
MAX_FILES = 500
MAX_TESTS = 100
MAX_MAX_DEPTH = 6

#: Valid ``target_kind`` values (echoed in the response; the engine resolves
#: the actual target kind deterministically).
TARGET_KINDS = ("file", "module", "package", "symbol")


def validate_change_target(value) -> str | None:
    """Validate the optional ``target`` argument (repo-relative, safe)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidArgumentsError(
            "The 'target' argument must be a string.",
            diagnostic=f"target had type {type(value).__name__}",
        )
    stripped = value.strip()
    if not stripped:
        raise InvalidArgumentsError("The 'target' argument must not be empty.")
    if not _target_is_safe(stripped):
        raise InvalidArgumentsError(
            "The 'target' argument must be a repository-relative file path, "
            "module, package, or symbol name.",
            diagnostic=f"unsafe change target: {stripped!r}",
        )
    return stripped


def validate_target_kind(value) -> str | None:
    """Validate the optional ``target_kind`` argument."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidArgumentsError(
            "The 'target_kind' argument must be a string.",
            diagnostic=f"target_kind had type {type(value).__name__}",
        )
    stripped = value.strip().lower()
    if stripped not in TARGET_KINDS:
        choices = ", ".join(TARGET_KINDS)
        raise InvalidArgumentsError(
            f"The 'target_kind' argument must be one of: {choices}."
        )
    return stripped


def validate_limit_int(value, *, name: str, maximum: int) -> int | None:
    """Validate an optional positive, capped integer limit."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentsError(
            f"The '{name}' argument must be a positive integer."
        )
    if value <= 0:
        raise InvalidArgumentsError(f"The '{name}' argument must be positive.")
    if value > maximum:
        raise InvalidArgumentsError(
            f"The '{name}' argument must not exceed {maximum}."
        )
    return value


def validate_bool_flag(value, *, name: str) -> bool:
    """Validate an optional boolean flag argument."""
    if value is None:
        return True
    if not isinstance(value, bool):
        raise InvalidArgumentsError(
            f"The '{name}' argument must be true or false."
        )
    return value


def _plan_limits(max_targets: int | None, max_files: int | None,
                 max_tests: int | None, max_depth: int | None) -> dict:
    """Build the change-plan config override dict from optional limits."""
    overrides: dict = {}
    if max_targets is not None:
        overrides["max_target_candidates"] = max_targets
        overrides["max_primary_targets"] = max(max_targets // 2, 1)
    if max_files is not None:
        overrides["max_affected_files"] = max_files
    if max_tests is not None:
        overrides["max_tests"] = max_tests
    if max_depth is not None:
        overrides["impact_max_depth"] = max_depth
    return overrides


def _build_plan_config(overrides: dict) -> ChangePlanConfig:
    base = ChangePlanConfig()
    if not overrides:
        return base
    return ChangePlanConfig(**{**base.__dict__, **overrides})


def parse_change_plan_arguments(arguments) -> dict:
    """Coerce raw ``change_plan`` arguments into validated options."""
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to change_plan.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "change_plan arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )
    if "request" not in arguments:
        raise InvalidArgumentsError("The 'request' argument is required.")

    allowed = {
        "request",
        "target",
        "target_kind",
        "max_targets",
        "max_files",
        "max_tests",
        "max_depth",
    }
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(f"Unsupported argument(s): {names}.")

    return {
        "request": validate_query(arguments.get("request")),
        "target": validate_change_target(arguments.get("target")),
        "target_kind": validate_target_kind(arguments.get("target_kind")),
        "max_targets": validate_limit_int(
            arguments.get("max_targets"), name="max_targets", maximum=MAX_TARGETS
        ),
        "max_files": validate_limit_int(
            arguments.get("max_files"), name="max_files", maximum=MAX_FILES
        ),
        "max_tests": validate_limit_int(
            arguments.get("max_tests"), name="max_tests", maximum=MAX_TESTS
        ),
        "max_depth": validate_limit_int(
            arguments.get("max_depth"), name="max_depth", maximum=MAX_MAX_DEPTH
        ),
    }


def parse_change_context_arguments(arguments) -> dict:
    """Coerce raw ``change_context`` arguments into validated options."""
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to change_context.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "change_context arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )
    if "request" not in arguments:
        raise InvalidArgumentsError("The 'request' argument is required.")

    allowed = {
        "request",
        "target",
        "target_kind",
        "max_tokens",
        "max_targets",
        "max_files",
        "max_tests",
        "max_depth",
        "include_tests",
        "include_dependencies",
        "include_callers",
        "include_callees",
        "include_architecture",
    }
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(f"Unsupported argument(s): {names}.")

    return {
        "request": validate_query(arguments.get("request")),
        "target": validate_change_target(arguments.get("target")),
        "target_kind": validate_target_kind(arguments.get("target_kind")),
        "max_tokens": validate_max_tokens(arguments.get("max_tokens")),
        "max_targets": validate_limit_int(
            arguments.get("max_targets"), name="max_targets", maximum=MAX_TARGETS
        ),
        "max_files": validate_limit_int(
            arguments.get("max_files"), name="max_files", maximum=MAX_FILES
        ),
        "max_tests": validate_limit_int(
            arguments.get("max_tests"), name="max_tests", maximum=MAX_TESTS
        ),
        "max_depth": validate_limit_int(
            arguments.get("max_depth"), name="max_depth", maximum=MAX_MAX_DEPTH
        ),
        "include_tests": validate_bool_flag(
            arguments.get("include_tests"), name="include_tests"
        ),
        "include_dependencies": validate_bool_flag(
            arguments.get("include_dependencies"), name="include_dependencies"
        ),
        "include_callers": validate_bool_flag(
            arguments.get("include_callers"), name="include_callers"
        ),
        "include_callees": validate_bool_flag(
            arguments.get("include_callees"), name="include_callees"
        ),
        "include_architecture": validate_bool_flag(
            arguments.get("include_architecture"), name="include_architecture"
        ),
    }


# ---------------------------------------------------------------------------
# Shared lazy change-plan state
# ---------------------------------------------------------------------------

#: A factory that builds (once) the shared change-plan state for the repo.
#: Injected so tests can provide fakes; the launcher provides the real one.
ChangePlanFactory = Callable[..., "ChangePlanState"]


class ChangePlanState:
    """Lazily-built, shared change-plan state for the MCP tools.

    The incremental index and the default :class:`ChangePlanEngine` (with its
    dependency graph, symbol index, call graph, architecture graph, subsystems,
    impact analyzer, and searcher) are all computed **once** on first use and
    then reused for every subsequent call. Calls that request non-default plan
    bounds derive a per-call :class:`ChangePlanEngine` that reuses the shared
    components, so no re-parse and no component rebuild happens per call.
    """

    def __init__(self, root: str | Path) -> None:
        from repolens.mcp.deps import _build_index

        self._root = root
        self._index = None
        self._parsed = None
        self._default_engine = None
        self._parallel_engines: dict[ChangePlanConfig, ChangePlanEngine] = {}
        self._lock = threading.Lock()
        self._build_index = _build_index

    @property
    def root(self) -> Path:
        return Path(self._root)

    @property
    def index(self):
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = self._build_index(self.root)
                    self._parsed = self._index.stats.files_parsed
        return self._index

    @property
    def parsed_file_count(self) -> int | None:
        """Files parsed by the incremental index (``None`` until first build)."""
        return self._parsed

    @property
    def default_engine(self) -> ChangePlanEngine:
        """The shared default :class:`ChangePlanEngine` (built once)."""
        if self._default_engine is None:
            index = self.index  # outside the lock: index has its own guard
            with self._lock:
                if self._default_engine is None:
                    self._default_engine = ChangePlanEngine(
                        self.root, index=index
                    )
        return self._default_engine

    def engine(self, config: ChangePlanConfig | None = None) -> ChangePlanEngine:
        """Return the shared default engine, or a per-call engine for the
        given non-default bounds (reusing every shared component)."""
        if config is None:
            return self.default_engine
        base = self.default_engine
        if config == base._config:
            return base
        if config in self._parallel_engines:
            return self._parallel_engines[config]
        with self._lock:
            cached = self._parallel_engines.get(config)
            if cached is not None:
                return cached
            per_call = ChangePlanEngine(
                self.root,
                index=self._index,
                dependency_graph=base._dep_graph,
                symbol_index=base._symbol_index,
                call_graph=base._call_graph,
                arch_graph=base._arch_graph,
                subsystems=base._subsystems,
                impact_analyzer=base._impact_analyzer,
                searcher=base._searcher,
                config=config,
            )
            self._parallel_engines[config] = per_call
            return per_call


def _require_state(factory: ChangePlanFactory) -> ChangePlanState:
    if not callable(factory):
        raise InternalError(
            "The change-plan service is unavailable.",
            diagnostic="change-plan factory is not callable",
        )
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001
        raise ChangePlanError(
            "The change-plan engine could not be initialized.",
            diagnostic=f"change-plan factory failed: {type(exc).__name__}",
        ) from exc


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def run_change_plan(
    change_plan_factory: ChangePlanFactory,
    request: str,
    *,
    target: str | None = None,
    target_kind: str | None = None,
    max_targets: int | None = None,
    max_files: int | None = None,
    max_tests: int | None = None,
    max_depth: int | None = None,
) -> dict:
    """Execute ``change_plan`` and return a structured, safe response."""
    state = _require_state(change_plan_factory)
    config = _build_plan_config(
        _plan_limits(max_targets, max_files, max_tests, max_depth)
    )
    try:
        engine = state.engine(config)
        plan: ChangePlan = engine.plan(request, target=target)
    except InvalidArgumentsError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ChangePlanError(
            "The change plan could not be produced.",
            diagnostic=f"change_plan failed at runtime: {type(exc).__name__}",
        ) from exc

    payload = plan_response_payload(plan, request)
    payload["status"] = "ok"
    payload["target_kind"] = target_kind
    return payload


def run_change_context(
    engine_factory,
    firewall: ContextFirewall,
    change_plan_factory: ChangePlanFactory,
    request: str,
    *,
    target: str | None = None,
    target_kind: str | None = None,
    max_tokens: int | None = None,
    max_targets: int | None = None,
    max_files: int | None = None,
    max_tests: int | None = None,
    max_depth: int | None = None,
    include_tests: bool = True,
    include_dependencies: bool = True,
    include_callers: bool = True,
    include_callees: bool = True,
    include_architecture: bool = True,
) -> dict:
    """Execute ``change_context`` and return a safe, structured response.

    The change plan is generated once by the shared change-plan state, then
    passed verbatim into :meth:`ContextEngine.build_context` so no plan is
    ever generated twice for the same request. Only the firewall-cleared
    safe package reaches the caller, along with a deterministic per-file
    explanation of its change-plan provenance.
    """
    if not callable(engine_factory):
        raise InternalError(
            "The context service is unavailable.",
            diagnostic="engine factory is not callable",
        )
    if not isinstance(firewall, ContextFirewall):
        raise InternalError(
            "The context security service is unavailable.",
            diagnostic="injected firewall is not a ContextFirewall",
        )
    state = _require_state(change_plan_factory)
    config = _build_plan_config(
        _plan_limits(max_targets, max_files, max_tests, max_depth)
    )
    options = ChangeContextOptions(
        include_tests=include_tests,
        include_dependencies=include_dependencies,
        include_callers=include_callers,
        include_callees=include_callees,
        include_architecture=include_architecture,
        max_files=max_files,
    )

    try:
        engine = state.engine(config)
        plan: ChangePlan = engine.plan(request, target=target)
    except InvalidArgumentsError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ChangePlanError(
            "The change plan could not be produced.",
            diagnostic=f"change_context planning failed: {type(exc).__name__}",
        ) from exc

    # 1. Build a change-aware ContextEngine honoring ``max_tokens``.
    try:
        context_engine: ContextEngine = engine_factory(max_tokens=max_tokens)
    except InvalidArgumentsError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ChangeContextError(
            "The context engine could not be configured for this request.",
            diagnostic=f"change_context engine factory failed: {type(exc).__name__}",
        ) from exc

    # 2. Build the change-aware package (plan folded into the shared pipeline).
    try:
        package: ContextPackage = context_engine.build_context(
            request,
            change_request=request,
            change_target=target,
            change_plan=plan,
            change_options=options,
        )
    except Exception as exc:  # noqa: BLE001
        raise ChangeContextError(
            "The change-aware context could not be built.",
            diagnostic=f"build_context(change) failed: {type(exc).__name__}",
        ) from exc

    # 3. Inspect with the firewall; only the safe result is returned.
    try:
        result: FirewallResult = firewall.inspect(package)
        safe: SafeContextPackage = firewall.safe_package(package, result)
    except Exception as exc:  # noqa: BLE001
        raise ChangeContextError(
            "The context firewall could not inspect the change-aware result.",
            diagnostic=f"change_context firewall failed: {type(exc).__name__}",
        ) from exc

    return _build_response(plan, package, safe, result, request, target, target_kind)


def _build_response(
    plan: ChangePlan,
    package: ContextPackage,
    safe: SafeContextPackage,
    result: FirewallResult,
    request: str,
    target: str | None,
    target_kind: str | None,
) -> dict:
    """Assemble the structured, safe ``change_context`` response."""
    from repolens.context.firewall.render import render_safe_context

    selected_explanations: list[dict] = []
    for c in safe.safe_files:
        selected_explanations.append(
            explain_change_context(plan, package, c.path, safe_package=safe)
        )

    payload = plan_response_payload(plan, request)
    change_files = [c.path.as_posix() for c in package.change_candidates]
    return {
        "status": "ok",
        "request": request,
        "target": target,
        "target_kind": target_kind,
        "budget": {"max_tokens": safe.budget.max_tokens},
        "total_estimated_tokens": safe.total_estimated_tokens,
        "intent": safe.intent,
        "matched_symbols": list(safe.matched_symbols),
        "selected_files": [
            {
                "path": c.path,
                "role": c.role,
                "decision": c.decision,
                "estimated_tokens": c.estimated_tokens,
                "selection_reason": c.selection_reason,
                "inclusion_reason": c.inclusion_reason,
                "explanation": explain_change_context(
                    plan, package, c.path, safe_package=safe
                ),
            }
            for c in safe.safe_files
        ],
        "blocked_files": [
            {"path": c.path, "reason": "blocked by firewall"}
            for c in safe.blocked_files
        ],
        "change_files": change_files,
        "change_plan": {
            "primary_target": payload["primary_target"],
            "affected_file_count": len(payload["affected_files"]),
            "affected_symbols": payload["affected_symbols"],
            "test_count": len(payload["tests"]),
            "risk": payload["risk"],
            "risk_factors": payload["risk_factors"],
            "confidence": payload["confidence"],
            "summary": payload["summary"],
            "statistics": payload["statistics"],
        },
        "explanations": selected_explanations,
        "firewall": {
            "enabled": safe.firewall_enabled,
            "policy_version": safe.policy_version,
            "findings": [
                {
                    "path": f.path,
                    "line": f.line,
                    "type": f.type,
                    "severity": f.severity,
                    "decision": f.decision,
                    "reason": f.reason,
                }
                for f in safe.findings
            ],
        },
        "rendered_safe_context": render_safe_context(safe),
    }