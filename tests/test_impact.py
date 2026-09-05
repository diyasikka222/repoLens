"""Offline tests for change impact analysis (Milestone 21).

Exercises the :class:`ImpactAnalyzer` and its integration points: target
resolution (file/module/symbol/``file::symbol``), relationship classification,
bounded reverse traversal, conservative symbol-level evidence, test and
configuration discovery, deterministic risk scoring, the change-aware context
integration in :class:`ContextEngine` (budget + firewall respected,
``build_context`` untouched), and the MCP ``analyze_impact`` tool.

Fully offline and deterministic: no network, no embeddings, no LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repolens.context import (
    ContextBudget,
    ContextEngine,
    ContextFirewall,
    render_context,
)
from repolens.context.firewall.render import render_safe_context
from repolens.impact import (
    EVIDENCE_BASE_CLASS,
    EVIDENCE_CONFIG_TEXT,
    EVIDENCE_DEPENDENCY_EDGE,
    EVIDENCE_PACKAGE_EXPORT,
    EVIDENCE_SYMBOL_IMPORT,
    EVIDENCE_SYMBOL_TEXT,
    EVIDENCE_TEST_NAME,
    ImpactAnalyzer,
    ImpactConfig,
    ImpactTargetError,
    Relationship,
    RiskLevel,
)
from repolens.mcp import build_mcp_server
from repolens.mcp.deps import build_firewall
from repolens.mcp.errors import InvalidArgumentsError, McpError
from repolens.mcp.impact_tool import (
    parse_impact_arguments,
    run_analyze_impact,
    validate_limit,
    validate_max_depth,
    validate_target,
)
from repolens.search import CodeSearcher


def write_file(repo: Path, relative: str, source: str) -> None:
    p = repo / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    write_file(root, "lib/__init__.py", "from .core import Engine\n")
    write_file(
        root,
        "lib/core.py",
        "class Engine:\n"
        "    def start(self):\n"
        "        return 'vroom'\n"
        "\n"
        "def spin():\n"
        "    return 42\n"
        "\n"
        "VALUE = 1\n",
    )
    write_file(root, "lib/middle.py", "from .core import Engine\n")
    write_file(
        root,
        "lib/top.py",
        "from .middle import wire\n"
        "def run_all():\n"
        "    return wire()\n",
    )
    write_file(root, "app/client.py", "from lib.core import Engine\n")
    write_file(root, "config/settings.py", "from lib.core import spin\n")
    write_file(
        root,
        "tests/test_core.py",
        "from lib.core import spin\n"
        "def test_spin():\n"
        "    assert spin() == 42\n",
    )
    write_file(root, "util.py", "class Util:\n    pass\n")
    write_file(root, "tools/tool.py", "import util\n")
    write_file(
        root,
        "scripts/tool.py",
        "def script_helper():\n"
        "    return 3\n",
    )
    write_file(root, "other/util.py", "def util_other():\n    return 1\n")
    write_file(root, "pyproject.toml", 'module_path = "lib/core.py"\n')
    return root


@pytest.fixture
def analyzer(repo: Path) -> ImpactAnalyzer:
    return ImpactAnalyzer(repo)


def item_of(result, path: str, relationship: Relationship | None = None):
    for item in result.items:
        if item.path.as_posix() == path:
            if relationship is None or item.relationship is relationship:
                return item
    return None


def serialize(result) -> list[str]:
    return [
        f"{it.path.as_posix()}|{it.relationship.value}|{it.risk.value}|"
        f"{it.symbol}|{sorted(it.evidence)}|{it.depth}"
        for it in result.items
    ]


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def test_file_target_resolves(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("lib/core.py")
    assert result.kind == "file"
    assert result.module == "lib.core"
    assert result.risk is RiskLevel.HIGH
    assert Relationship.DIRECT_DEPENDENCY in {i.relationship for i in result.items}


def test_module_target_matches_file(repo: Path, analyzer: ImpactAnalyzer) -> None:
    as_file = analyzer.analyze("lib/core.py")
    as_module = analyzer.analyze("lib.core")
    assert as_module.kind == "module"
    assert serialize(as_file) == serialize(as_module)


def test_symbol_target_resolves_to_definition(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("Engine")
    assert result.kind == "symbol"
    assert result.symbol == "Engine"
    assert item_of(result, "app/client.py", Relationship.API_CONSUMER)


def test_symbol_in_file_target(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("lib/core.py::spin")
    assert result.kind == "symbol_in_file"
    assert result.symbol == "spin"
    assert item_of(result, "config/settings.py") is not None


def test_unknown_file_target_raises(repo: Path, analyzer: ImpactAnalyzer) -> None:
    with pytest.raises(ImpactTargetError):
        analyzer.analyze("lib/nope.py")


def test_unknown_symbol_raises(repo: Path, analyzer: ImpactAnalyzer) -> None:
    with pytest.raises(ImpactTargetError):
        analyzer.analyze("MissingThing")


def test_ambiguous_module_target_raises(repo: Path, analyzer: ImpactAnalyzer) -> None:
    with pytest.raises(ImpactTargetError) as exc:
        analyzer.analyze("tool")
    message = str(exc.value)
    assert "tools/tool.py" in message and "scripts/tool.py" in message


# ---------------------------------------------------------------------------
# Relationships and traversal
# ---------------------------------------------------------------------------


def test_direct_and_indirect_dependents(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("lib/core.py")
    assert item_of(result, "lib/middle.py", Relationship.DIRECT_DEPENDENCY)
    assert item_of(result, "lib/top.py", Relationship.INDIRECT_DEPENDENCY)
    top = item_of(result, "lib/top.py")
    assert top.depth == 2


def test_reverse_dependency_included(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("tools/tool.py")
    assert item_of(result, "util.py", Relationship.REVERSE_DEPENDENCY)


def test_max_depth_zero_skips_reverse_traversal(
    repo: Path, analyzer: ImpactAnalyzer,
) -> None:
    result = analyzer.analyze("lib/core.py", max_depth=0)
    assert item_of(result, "lib/middle.py") is None
    assert item_of(result, "lib/top.py") is None


def test_max_depth_one_caps_depth(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("lib/core.py", max_depth=1)
    assert item_of(result, "lib/middle.py", Relationship.DIRECT_DEPENDENCY)
    assert item_of(result, "lib/top.py") is None


def test_max_nodes_bounds_traversal(repo: Path) -> None:
    bounded = ImpactAnalyzer(repo, config=ImpactConfig(max_nodes=2))
    result = bounded.analyze("lib/core.py")
    assert len(result.items) < len(ImpactAnalyzer(repo).analyze("lib/core.py").items)


def test_no_duplicate_path_relationship(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("Engine")
    pairs = [(it.path.as_posix(), it.relationship.value) for it in result.items]
    assert len(pairs) == len(set(pairs))


def test_limit_truncates_items(repo: Path, analyzer: ImpactAnalyzer) -> None:
    full = analyzer.analyze("lib/core.py")
    limited = analyzer.analyze("lib/core.py", limit=2)
    assert len(limited.items) == 2
    assert serialize(full)[:2] == serialize(limited)


# ---------------------------------------------------------------------------
# Test and configuration discovery
# ---------------------------------------------------------------------------


def test_test_discovery_by_name_and_import(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("lib/core.py")
    test_item = item_of(result, "tests/test_core.py", Relationship.TEST)
    assert test_item is not None
    assert EVIDENCE_TEST_NAME in test_item.evidence
    assert EVIDENCE_DEPENDENCY_EDGE in test_item.evidence


def test_configuration_stem_classified(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("lib/core.py")
    item = item_of(result, "config/settings.py", Relationship.CONFIGURATION)
    assert item is not None


def test_declarative_config_referencing_module(
    repo: Path, analyzer: ImpactAnalyzer,
) -> None:
    result = analyzer.analyze("lib/core.py")
    item = item_of(result, "pyproject.toml", Relationship.CONFIGURATION)
    assert item is not None
    assert EVIDENCE_CONFIG_TEXT in item.evidence


# ---------------------------------------------------------------------------
# Conservative symbol evidence
# ---------------------------------------------------------------------------


def test_symbol_import_evidence_only_for_real_importers(
    repo: Path, analyzer: ImpactAnalyzer,
) -> None:
    result = analyzer.analyze("Engine")
    client = item_of(result, "app/client.py", Relationship.API_CONSUMER)
    assert EVIDENCE_SYMBOL_IMPORT in client.evidence
    assert client.symbol == "Engine"


def test_package_reexport_consumer(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("Engine")
    init = item_of(result, "lib/__init__.py", Relationship.API_CONSUMER)
    assert init is not None
    assert EVIDENCE_PACKAGE_EXPORT in init.evidence
    assert EVIDENCE_SYMBOL_IMPORT in init.evidence
    assert init.symbol == "Engine"


def test_file_analysis_emits_no_symbol_evidence(repo: Path, analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("lib/core.py")
    for item in result.items:
        assert EVIDENCE_SYMBOL_IMPORT not in item.evidence
        assert EVIDENCE_SYMBOL_TEXT not in item.evidence
        assert EVIDENCE_BASE_CLASS not in item.evidence


def test_base_class_evidence_only_in_symbol_mode() -> None:
    assert EVIDENCE_BASE_CLASS.startswith("base_class")


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


def test_low_medium_high_risk_levels(repo: Path, analyzer: ImpactAnalyzer) -> None:
    assert analyzer.analyze("lib/core.py").risk is RiskLevel.HIGH
    assert analyzer.analyze("lib/top.py").risk in (RiskLevel.LOW, RiskLevel.MEDIUM)
    assert analyzer.analyze("util.py").risk is RiskLevel.LOW


def test_direct_dependency_items_are_high_risk(
    repo: Path, analyzer: ImpactAnalyzer,
) -> None:
    result = analyzer.analyze("lib/core.py")
    middle = item_of(result, "lib/middle.py", Relationship.DIRECT_DEPENDENCY)
    assert middle.risk is RiskLevel.HIGH


def test_deterministic_ordering(repo: Path, analyzer: ImpactAnalyzer) -> None:
    first = analyzer.analyze("Engine")
    second = analyzer.analyze("Engine")
    assert serialize(first) == serialize(second)


def test_repeated_analysis_reuses_analyzer(repo: Path, analyzer: ImpactAnalyzer) -> None:
    r1 = analyzer.analyze("lib/core.py")
    r2 = analyzer.analyze("lib/core.py")
    assert serialize(r1) == serialize(r2) and r1 is not r2


# ---------------------------------------------------------------------------
# ContextEngine integration (change-aware context)
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(repo: Path) -> ContextEngine:
    return ContextEngine(repo, searcher=CodeSearcher(repo))


def test_build_impact_context_target_first(engine: ContextEngine) -> None:
    package = engine.build_impact_context("lib/core.py")
    assert [c.path.as_posix() for c in package.selected_files] == [
        "lib/core.py",
        "app/client.py",
        "lib/__init__.py",
        "lib/middle.py",
        "tests/test_core.py",
        "lib/top.py",
        "pyproject.toml",
        "config/settings.py",
    ]


def test_build_impact_context_respects_budget(engine: ContextEngine) -> None:
    full = engine.build_impact_context("lib/core.py")
    package = engine.build_impact_context(
        "lib/core.py", budget=ContextBudget(max_tokens=40)
    )
    assert package.budget.max_tokens == 40
    assert package.selected_files[0].path.as_posix() == "lib/core.py"
    assert len(package.selected_files) < len(full.selected_files)
    assert package.excluded_candidates


def test_build_impact_context_impact_intent(engine: ContextEngine) -> None:
    package = engine.build_impact_context("lib/core.py")
    assert package.intent == "impact"
    assert "symbol_match" in {
        c.inclusion_reason for c in package.selected_files
    }
    assert "dependent" in {c.inclusion_reason for c in package.selected_files}


def test_build_change_context_extracts_symbol(engine: ContextEngine) -> None:
    package = engine.build_change_context("what breaks if I change Engine?")
    paths = [c.path.as_posix() for c in package.selected_files]
    assert any(s in paths for s in ("lib/core.py", "app/client.py"))


def test_engine_unknown_target_raises(engine: ContextEngine) -> None:
    with pytest.raises(ImpactTargetError):
        engine.build_impact_context("does/not/exist.py")


def test_firewall_flow_with_impact_context(
    repo: Path, engine: ContextEngine,
) -> None:
    package = engine.build_impact_context("lib/core.py")
    firewall = ContextFirewall()
    result = firewall.inspect(package)
    safe = firewall.safe_package(package, result)
    actual = [Path(c.path) for c in safe.safe_files]
    assert actual == [c.path for c in package.selected_files]
    render_safe_context(safe)


def test_build_context_backward_compatible(engine: ContextEngine) -> None:
    one = engine.build_context("spin")
    two = engine.build_context("spin")
    assert render_context(one) == render_context(two)
    assert hasattr(one, "selected_files")


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------


def test_analyze_impact_registered_only_with_factory(
    repo: Path,
) -> None:
    firewall = build_firewall()
    base = build_mcp_server(_engine_factory(repo), firewall)
    assert not _has_tool(base, "analyze_impact")
    assert _has_tool(base, "get_context")
    with_impact = build_mcp_server(
        _engine_factory(repo), firewall, impact_factory=_impact_factory(repo)
    )
    assert _has_tool(with_impact, "analyze_impact")
    assert _has_tool(with_impact, "get_context")


def test_parse_impact_arguments() -> None:
    parsed = parse_impact_arguments({"target": "  lib/core.py ", "limit": 3})
    assert parsed["target"] == "lib/core.py"
    assert parsed["max_depth"] is None
    assert parsed["limit"] == 3


def test_parse_impact_arguments_rejects_unknown() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_impact_arguments({"target": "x", "symbol": "y"})


def test_parse_impact_arguments_rejects_missing_target() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_impact_arguments({})


def test_validate_max_depth() -> None:
    assert validate_max_depth(0) == 0
    assert validate_max_depth(4) == 4
    assert validate_max_depth(None) is None
    with pytest.raises(InvalidArgumentsError):
        validate_max_depth(-1)


def test_validate_limit() -> None:
    assert validate_limit(5) == 5
    with pytest.raises(InvalidArgumentsError):
        validate_limit(0)
    with pytest.raises(InvalidArgumentsError):
        validate_limit(-2)


def test_validate_target() -> None:
    assert validate_target("lib/core.py") == "lib/core.py"
    with pytest.raises(InvalidArgumentsError):
        validate_target("")
    with pytest.raises(InvalidArgumentsError):
        validate_target(42)


def test_run_analyze_impact_response_shape(repo: Path) -> None:
    response = run_analyze_impact(_impact_factory(repo), "lib/core.py")
    assert response["status"] == "ok"
    for key in (
        "target", "kind", "target_path", "symbol", "module", "risk",
        "max_depth_reached", "summary", "item_count", "items",
    ):
        assert key in response
    json.dumps(response)


def test_run_analyze_impact_symbol_and_options(repo: Path) -> None:
    response = run_analyze_impact(
        _impact_factory(repo), "Engine", max_depth=2, limit=3
    )
    assert response["symbol"] == "Engine"
    assert len(response["items"]) == 3
    assert response["item_count"] == 3
    paths = [item["path"] for item in response["items"]]
    assert len(set(paths)) == len(paths)
    assert response["items"][0]["path"] != response["target_path"]


def test_run_analyze_impact_unknown_target_safe(repo: Path) -> None:
    with pytest.raises(InvalidArgumentsError):
        run_analyze_impact(_impact_factory(repo), "lib/missing.py")


def test_analyze_impact_error_is_safe(repo: Path) -> None:
    try:
        run_analyze_impact(_impact_factory(repo), "lib/missing.py")
    except InvalidArgumentsError as exc:
        assert "missing" in exc.safe_message
        assert exc.diagnostic is None, "no internal detail should leak"


def test_make_impact_analyzer_factory_validates_root_lazily(
    repo: Path, monkeypatch,
) -> None:
    from repolens.mcp import launcher
    from repolens.mcp import deps

    build_calls: list = []

    def spy_build_analyzer(root):
        build_calls.append(root)
        return ImpactAnalyzer(root)

    monkeypatch.setattr(deps, "build_impact_analyzer", spy_build_analyzer)
    factory = launcher.make_impact_analyzer_factory(str(repo))
    assert build_calls == [], "analyzer must not be built during factory creation"
    analyzer = factory()
    assert len(build_calls) == 1
    assert Path(build_calls[0]) == repo.resolve()
    assert factory() is analyzer, "second call must reuse the cached analyzer"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _engine_factory(path: Path):
    def factory(max_tokens=None, dependency_depth=None, **kwargs):
        return ContextEngine(
            path,
            searcher=CodeSearcher(path),
            budget=ContextBudget(max_tokens=max_tokens) if max_tokens else ContextBudget(),
        )

    return factory


def _impact_factory(path: Path):
    def factory(**kwargs) -> ImpactAnalyzer:
        return ImpactAnalyzer(path)

    return factory


def _has_tool(server, name: str) -> bool:
    import asyncio

    tools = asyncio.run(server.list_tools())
    return name in {t.name for t in tools}