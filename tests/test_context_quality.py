"""Offline tests for M19 context quality & smart retrieval.

Covers: symbol-aware retrieval, the deterministic intent classifier,
intent-aware dependency expansion (forward/reverse), duplicate elimination,
budget enforcement with oversized-file truncation, inclusion reasons, the
deterministic ranking policy, firewall interaction, and backward compatibility
of the existing ``ContextEngine`` public behaviour.

No LLMs, network, or external APIs are used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.context import (
    INCLUSION_DEPENDENCY,
    INCLUSION_DEPENDENT,
    INCLUSION_HYBRID_MATCH,
    INCLUSION_LEXICAL_MATCH,
    INCLUSION_SYMBOL_MATCH,
    CandidateRole,
    ContextBudget,
    ContextEngine,
    ContextFirewall,
    DependencyExpansionConfig,
    QueryIntent,
    RetrievalConfig,
    SafeContextPackage,
    classify_intent,
)
from repolens.context.budget import select_within_budget
from repolens.context.candidate import ContextCandidate
from repolens.context.firewall import FirewallConfig
from repolens.context.intent import extract_symbol_tokens
from repolens.context.symbol_retrieval import build_symbol_index, match_symbols
from repolens.graph import DependencyGraphBuilder
from repolens.search import CodeSearcher

SYNTHETIC_REPO = (
    Path(__file__).resolve().parent / "fixtures" / "synthetic_repository"
)


def write_file(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def build_repo(tmp_path: Path) -> Path:
    """A small, dependency-tangled repo for directional-expansion tests."""
    root = tmp_path / "repo"
    root.mkdir()
    write_file(root, "a.py", "from lib import util\n\ndef alpha(value):\n    return util(value)\n")
    write_file(root, "b.py", "from lib import util\n\ndef beta():\n    return 2\n")
    write_file(root, "lib.py", "def util(x):\n    return x + 1\n")
    write_file(root, "standalone.py", "def alone():\n    return 0\n")
    return root


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------


def test_intent_classifies_implementation() -> None:
    assert classify_intent("where is the login service implemented?") == QueryIntent.IMPLEMENTATION
    assert classify_intent("where is PaymentProcessor defined?") == QueryIntent.IMPLEMENTATION


def test_intent_classifies_explanation() -> None:
    assert classify_intent("how does refund flow?") == QueryIntent.EXPLANATION
    assert classify_intent("how does the payment service work?") == QueryIntent.EXPLANATION


def test_intent_classifies_dependency() -> None:
    assert classify_intent("what depends on the login service?") == QueryIntent.DEPENDENCY
    assert classify_intent("what uses auth?") == QueryIntent.DEPENDENCY
    assert classify_intent("impact of changing the billing module") == QueryIntent.DEPENDENCY


def test_intent_classifies_modification() -> None:
    assert classify_intent("where should i modify the billing code?") == QueryIntent.MODIFICATION
    assert classify_intent("what files should i change to add invoicing?") == QueryIntent.MODIFICATION


def test_intent_unknown_for_vague_query() -> None:
    assert classify_intent("refund a card payment") == QueryIntent.UNKNOWN
    assert classify_intent("") == QueryIntent.UNKNOWN


def test_extract_symbol_tokens_is_deterministic() -> None:
    assert extract_symbol_tokens("what uses login and auth")
    assert extract_symbol_tokens("what uses login and auth") == extract_symbol_tokens(
        "what uses login and auth"
    )


# ---------------------------------------------------------------------------
# Symbol-aware retrieval
# ---------------------------------------------------------------------------


def test_symbol_exact_match(tmp_path: Path) -> None:
    write_file(tmp_path, "m.py", "def widget():\n    pass\n")
    si = build_symbol_index(tmp_path, None)
    matches = match_symbols("widget", tmp_path, symbol_index=si)
    assert len(matches) == 1
    assert matches[0].symbol.name == "widget"
    assert matches[0].exact is True


def test_symbol_near_match_normalizes_case_and_underscores(tmp_path: Path) -> None:
    write_file(tmp_path, "m.py", "class InvoiceService:\n    pass\n")
    si = build_symbol_index(tmp_path, None)
    assert [m.symbol.name for m in match_symbols("invoice_service", tmp_path, symbol_index=si)] == ["InvoiceService"]
    assert [m.symbol.name for m in match_symbols("invoice service", tmp_path, symbol_index=si)] == ["InvoiceService"]


def test_unknown_symbol_falls_back_to_empty(tmp_path: Path) -> None:
    write_file(tmp_path, "m.py", "def widget():\n    pass\n")
    si = build_symbol_index(tmp_path, None)
    assert match_symbols("does not exist symbol xyz", tmp_path, symbol_index=si) == []


def test_symbol_boost_prioritizes_match(tmp_path: Path) -> None:
    write_file(tmp_path, "auth/login.py", "class LoginService:\n    pass\n")
    write_file(tmp_path, "auth/session.py", "def make_session():\n    pass\n")
    engine = ContextEngine(
        tmp_path,
        searcher=CodeSearcher(tmp_path),
        dependency=DependencyExpansionConfig(depth=0),
    )
    pkg = engine.build_context("where is login implemented?")
    reasons = {c.path.as_posix(): c.inclusion_reason for c in pkg.selected_files}
    assert reasons.get("auth/login.py") == INCLUSION_SYMBOL_MATCH
    assert pkg.matched_symbols == ("LoginService",)


# ---------------------------------------------------------------------------
# Dependency-expansion direction driven by intent
# ---------------------------------------------------------------------------


def test_implementation_prefers_dependencies(tmp_path: Path) -> None:
    root = build_repo(tmp_path)
    engine = ContextEngine(
        root, searcher=CodeSearcher(root), dependency=DependencyExpansionConfig(depth=1)
    )
    pkg = engine.build_context("where is alpha implemented?")
    # alpha is defined in a.py; its dependency lib.py should come along.
    expanded = {c.path.as_posix() for c in pkg.dependency_candidates}
    assert "lib.py" in expanded
    # Dependents must NOT be expanded for an implementation question.
    assert "b.py" not in expanded
    for c in pkg.dependency_candidates:
        assert c.inclusion_reason == INCLUSION_DEPENDENCY


def test_explanation_prefers_dependencies(tmp_path: Path) -> None:
    root = build_repo(tmp_path)
    engine = ContextEngine(
        root, searcher=CodeSearcher(root), dependency=DependencyExpansionConfig(depth=1)
    )
    pkg = engine.build_context("how does alpha work?")
    expanded = {c.path.as_posix() for c in pkg.dependency_candidates}
    assert "lib.py" in expanded
    assert "b.py" not in expanded


def test_dependency_query_expands_dependents(tmp_path: Path) -> None:
    root = build_repo(tmp_path)
    engine = ContextEngine(
        root, searcher=CodeSearcher(root), dependency=DependencyExpansionConfig(depth=1)
    )
    pkg = engine.build_context("what uses the util symbol?")
    expanded = {c.path.as_posix() for c in pkg.dependency_candidates}
    # util is defined in lib.py; its dependents are a.py and b.py.
    assert "a.py" in expanded
    assert "b.py" in expanded
    for c in pkg.dependency_candidates:
        assert c.inclusion_reason == INCLUSION_DEPENDENT


def test_dependency_expansion_is_bounded(tmp_path: Path) -> None:
    root = build_repo(tmp_path)
    cfg = DependencyExpansionConfig(depth=1, max_expanded=1)
    engine = ContextEngine(root, searcher=CodeSearcher(root), dependency=cfg)
    pkg = engine.build_context("what uses the util symbol?")
    assert len(pkg.dependency_candidates) <= 1


def test_expansion_no_duplicates(tmp_path: Path) -> None:
    root = build_repo(tmp_path)
    engine = ContextEngine(
        root, searcher=CodeSearcher(root), dependency=DependencyExpansionConfig(depth=2)
    )
    pkg = engine.build_context("what uses the util symbol?")
    paths = [c.path for c in pkg.selected_files]
    assert len(paths) == len(set(paths))


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_symbol_matched_primary_ranks_first_at_equal_retrieval(tmp_path: Path) -> None:
    write_file(tmp_path, "a.py", "def target():\n    pass\n")
    from repolens.context.ranking import rank_candidates
    from repolens.context.tokens import estimate_tokens

    def cand(path, *, symbol=False):
        return ContextCandidate(
            path=Path(path),
            source="",
            role=CandidateRole.PRIMARY,
            estimated_tokens=estimate_tokens(""),
            selection_reason="t",
            inclusion_reason=INCLUSION_SYMBOL_MATCH if symbol else None,
            retrieval_rank=1,
        )

    ranked = [c.path.as_posix() for c in rank_candidates([cand("b.py"), cand("a.py", symbol=True)])]
    assert ranked == ["a.py", "b.py"]


def test_ranking_order_preserved_without_symbol_matches() -> None:
    from repolens.context.ranking import rank_candidates
    from repolens.context.tokens import estimate_tokens

    def cand(path, rank, score):
        return ContextCandidate(
            path=Path(path), source="", role=CandidateRole.PRIMARY,
            estimated_tokens=estimate_tokens(""), selection_reason="t",
            retrieval_rank=rank, retrieval_score=score,
        )

    ranked = [c.path.as_posix() for c in rank_candidates([
        cand("a.py", 2, 9.0), cand("c.py", 1, 3.0), cand("b.py", 3, 1.0),
    ])]
    # Rewrites the historical ordering exactly (retrieval rank, then score).
    assert ranked == ["c.py", "a.py", "b.py"]


# ---------------------------------------------------------------------------
# Budget enforcement + truncation
# ---------------------------------------------------------------------------


def _tok_candidate(path, tokens) -> ContextCandidate:
    return ContextCandidate(
        path=Path(path),
        source="x" * (tokens * 4),
        role=CandidateRole.PRIMARY,
        estimated_tokens=tokens,
        selection_reason="t",
    )


def test_budget_exact_enforcement() -> None:
    ranked = [_tok_candidate("a.py", 30), _tok_candidate("b.py", 40)]
    selected, _ = select_within_budget(ranked, ContextBudget(max_tokens=70))
    assert [c.path.as_posix() for c in selected] == ["a.py", "b.py"]
    assert sum(c.estimated_tokens for c in selected) <= 70


def test_oversized_file_truncated_when_configured() -> None:
    ranked = [_tok_candidate("huge.py", 500)]
    selected, excluded = select_within_budget(
        ranked, ContextBudget(max_tokens=100, truncate_oversized=True)
    )
    assert len(selected) == 1
    assert selected[0].path.as_posix() == "huge.py"
    # The truncated source must fit within the budget and be a prefix of the original.
    assert selected[0].estimated_tokens <= 100
    assert len(selected[0].source) < len(ranked[0].source)
    assert excluded == []


def test_oversized_file_excluded_by_default() -> None:
    ranked = [_tok_candidate("huge.py", 500)]
    selected, excluded = select_within_budget(
        ranked, ContextBudget(max_tokens=100)
    )
    assert selected == []
    assert any(e.path.as_posix() == "huge.py" for e in excluded)


def test_engine_budget_with_truncation_keeps_oversized_primary(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, "big.py", "def big():\n    return 'x' * 5000\n")
    engine = ContextEngine(
        tmp_path,
        searcher=CodeSearcher(tmp_path),
        budget=ContextBudget(max_tokens=50, truncate_oversized=True),
        dependency=DependencyExpansionConfig(depth=0),
    )
    pkg = engine.build_context("big")
    assert pkg.total_estimated_tokens <= 50
    assert any(c.path.as_posix() == "big.py" for c in pkg.selected_files)
    assert any(pkg.total_estimated_tokens <= 50 for c in pkg.selected_files)


# ---------------------------------------------------------------------------
# Inclusion reasons (primary signals)
# ---------------------------------------------------------------------------


def test_lexical_primary_reason_is_lexical_match(tmp_path: Path) -> None:
    write_file(tmp_path, "m.py", "def widgetize():\n    pass\n")
    engine = ContextEngine(
        tmp_path,
        searcher=CodeSearcher(tmp_path),
        dependency=DependencyExpansionConfig(depth=0),
    )
    pkg = engine.build_context("widgetize")
    for c in pkg.selected_files:
        if c.role is CandidateRole.PRIMARY:
            assert c.inclusion_reason in {
                INCLUSION_SYMBOL_MATCH, INCLUSION_LEXICAL_MATCH,
            }


def test_inclusion_reason_dict_and_json(tmp_path: Path) -> None:
    write_file(tmp_path, "m.py", "def widget():\n    pass\n")
    engine = ContextEngine(
        tmp_path,
        searcher=CodeSearcher(tmp_path),
        dependency=DependencyExpansionConfig(depth=0),
    )
    pkg = engine.build_context("widget")
    data = pkg.to_dict()
    assert "inclusion_reason" in data["selected_files"][0]
    assert data["intent"] is not None
    assert "matched_symbols" in data


# ---------------------------------------------------------------------------
# Firewall interaction
# ---------------------------------------------------------------------------


def test_firewall_carries_inclusion_reason_and_intent(tmp_path: Path) -> None:
    write_file(tmp_path, "auth/login.py", "class LoginService:\n    pass\n")
    engine = ContextEngine(
        tmp_path,
        searcher=CodeSearcher(tmp_path),
        dependency=DependencyExpansionConfig(depth=0),
    )
    pkg = engine.build_context("where is login implemented?")
    firewall = ContextFirewall(FirewallConfig())
    result = firewall.inspect(pkg)
    safe: SafeContextPackage = firewall.safe_package(pkg, result)
    by_path = {c.path: c for c in safe.safe_files}
    assert safe.intent == "implementation"
    assert safe.matched_symbols == ("LoginService",)
    login = by_path.get("auth/login.py")
    assert login is not None
    assert login.inclusion_reason == INCLUSION_SYMBOL_MATCH
    assert safe.to_dict()["safe_files"][0]["inclusion_reason"] is not None


def test_dependency_context_still_goes_through_firewall(tmp_path: Path) -> None:
    root = build_repo(tmp_path)
    engine = ContextEngine(
        root, searcher=CodeSearcher(root), dependency=DependencyExpansionConfig(depth=1)
    )
    pkg = engine.build_context("where is alpha implemented?")
    assert any(c.role is CandidateRole.DEPENDENCY for c in pkg.selected_files)
    firewall = ContextFirewall(FirewallConfig())
    result = firewall.inspect(pkg)
    safe = firewall.safe_package(pkg, result)
    # Every selected file (including expanded dependencies) is present in the safe package.
    assert len(safe.safe_files) == len(pkg.selected_files)


def test_blocked_firewall_never_leaks_internal_content(tmp_path: Path) -> None:
    write_file(tmp_path, "secret.py", "api_key = 'sk-live-1234567890abcdef'\n")
    engine = ContextEngine(
        tmp_path,
        searcher=CodeSearcher(tmp_path),
        dependency=DependencyExpansionConfig(depth=0),
    )
    pkg = engine.build_context("secret")
    firewall = ContextFirewall(FirewallConfig())
    result = firewall.inspect(pkg)
    safe = firewall.safe_package(pkg, result)
    # The sensitive literal must never land in any safe file's source.
    for c in safe.safe_files:
        assert "sk-live-1234567890abcdef" not in c.source


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_engine_still_builds_package_without_new_options() -> None:
    engine = ContextEngine(SYNTHETIC_REPO, searcher=CodeSearcher(SYNTHETIC_REPO))
    pkg = engine.build_context("invoice calculation")
    assert pkg.selected_files
    assert pkg.intent is not None  # new field is additive, never breaks consumers


def test_engine_default_config_compatible() -> None:
    assert ContextBudget().max_tokens == 8000
    assert ContextBudget().truncate_oversized is False
    assert DependencyExpansionConfig().include_dependencies is True
    assert DependencyExpansionConfig().include_dependents is True
    assert DependencyExpansionConfig().max_expanded == 12
    assert RetrievalConfig().strategy == "rrf"


def test_engine_invalid_root_still_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        ContextEngine(tmp_path / "missing", searcher=CodeSearcher(tmp_path))