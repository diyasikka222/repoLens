"""Focused context representation tests (P26.2 feasibility).

Proves the additive, internal focused-source model works without changing any
default pipeline behaviour: symbol span extraction, deterministic focused
ordering, token accounting, multiple focused symbols from one file, firewall
compatibility, and serialization.  None of these tests invoke the default
budget-selection fix (which is intentionally not implemented).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repolens.context import (
    INCLUSION_SYMBOL_MATCH,
    CandidateRole,
    ContextBudget,
    ContextCandidate,
    ContextFirewall,
    ContextPackage,
    SourceSpan,
    SymbolSpan,
    estimate_tokens,
    extract_symbol_spans,
    focus_candidate,
    focus_package,
    focused_items_for_file,
    render_context,
)
from repolens.context.budget import select_within_budget
from repolens.context_trace import snapshot_candidate
from repolens.index import SymbolKind

_SPAN_SRC = (
    "@decorator\n"  # 1
    "def routed():\n"  # 2
    "    return 1\n"  # 3
    "\n"  # 4
    "class Service:\n"  # 5
    '    """doc."""\n'  # 6
    "    def run(self):\n"  # 7
    "        return 2\n"  # 8
    "\n"  # 9
    "def plan():\n"  # 10
    "    def nested():\n"  # 11 -> not a top-level symbol (parser scope)
    "        return 3\n"  # 12
    "    return nested()\n"  # 13
)

_MULTI_SRC = (
    "def alpha():\n"  # 1
    "    return 1\n"  # 2
    "\n"  # 3
    "def beta():\n"  # 4
    "    return 2\n"  # 5
)


def _candidate(
    path: str = "pkg/a.py",
    source: str = _MULTI_SRC,
    *,
    role: CandidateRole = CandidateRole.PRIMARY,
) -> ContextCandidate:
    return ContextCandidate(
        path=Path(path),
        source=source,
        role=role,
        estimated_tokens=estimate_tokens(source),
        selection_reason="test selection",
        inclusion_reason=INCLUSION_SYMBOL_MATCH,
        retrieval_rank=1,
        retrieval_score=0.5,
    )


def _package(*candidates: ContextCandidate, budget: int = 10_000) -> ContextPackage:
    return ContextPackage(
        query="focus test",
        budget=ContextBudget(max_tokens=budget),
        selected_files=tuple(candidates),
    )


# ---------------------------------------------------------------------------
# Symbol span extraction
# ---------------------------------------------------------------------------


def test_extract_symbol_spans_mirrors_parser_scope() -> None:
    spans = extract_symbol_spans(_SPAN_SRC, Path("pkg/a.py"))
    assert [(s.name, s.kind, s.start_line, s.end_line) for s in spans] == [
        ("routed", SymbolKind.FUNCTION, 1, 3),
        ("Service", SymbolKind.CLASS, 5, 8),
        ("run", SymbolKind.METHOD, 7, 8),
        ("plan", SymbolKind.FUNCTION, 10, 13),
    ]
    # The nested helper is not a top-level symbol: parser scope, unchanged.
    assert not any(s.name == "nested" for s in spans)


def test_extract_symbol_spans_records_method_parent_class() -> None:
    spans = extract_symbol_spans(_SPAN_SRC, Path("pkg/a.py"))
    run = next(s for s in spans if s.name == "run")
    assert run.parent_class == "Service"
    service = next(s for s in spans if s.name == "Service")
    assert service.parent_class is None


def test_extract_symbol_spans_includes_decorator_lines() -> None:
    spans = extract_symbol_spans(_SPAN_SRC, Path("pkg/a.py"))
    routed = next(s for s in spans if s.name == "routed")
    assert routed.start_line == 1
    assert routed.end_line == 3
    assert routed.slice_source(_SPAN_SRC) == "@decorator\ndef routed():\n    return 1\n"


def test_extract_symbol_spans_is_deterministic() -> None:
    first = extract_symbol_spans(_SPAN_SRC, Path("pkg/a.py"))
    again = extract_symbol_spans(_SPAN_SRC, Path("pkg/a.py"))
    assert first == again
    starts = [s.start_line for s in first]
    assert starts == sorted(starts)


def test_extract_symbol_spans_returns_empty_on_parse_error() -> None:
    assert extract_symbol_spans("def broken(:\n", Path("bad.py")) == ()
    assert extract_symbol_spans("", Path("empty.py")) == ()


# ---------------------------------------------------------------------------
# SourceSpan slicing and line mapping
# ---------------------------------------------------------------------------


def test_source_span_slices_exact_lines() -> None:
    text = "a\nbb\nccc\ndddd\n"
    span = SourceSpan(2, 3)
    assert span.slice_source(text) == "bb\nccc\n"
    assert span.line_count == 2


def test_source_span_slice_past_end_returns_existing_lines() -> None:
    span = SourceSpan(2, 5)
    assert span.slice_source("a\nb\n") == "b\n"


def test_source_span_validates_ranges() -> None:
    with pytest.raises(ValueError):
        SourceSpan(0, 3)
    with pytest.raises(ValueError):
        SourceSpan(5, 3)


def test_source_span_original_line_mapping() -> None:
    span = SourceSpan(10, 12)
    assert span.original_line(1) == 10
    assert span.original_line(3) == 12
    assert span.original_line(4) is None
    assert span.original_line(0) is None


# ---------------------------------------------------------------------------
# Deterministic focused representation
# ---------------------------------------------------------------------------


def test_focused_items_for_file_returns_full_then_focused_in_order() -> None:
    candidate = _candidate(source=_MULTI_SRC)
    items = focused_items_for_file(candidate)
    assert len(items) == 3
    # Full-file item is byte-for-byte the original (unfocused) candidate.
    assert items[0].is_focused is False
    assert items[0].source == _MULTI_SRC
    assert items[0].estimated_tokens == estimate_tokens(_MULTI_SRC)
    # Focused items follow in deterministic (start line, name) order.
    assert [i.focus_name for i in items[1:]] == ["alpha", "beta"]
    assert [i.focus_start_line for i in items[1:]] == [1, 4]


def test_focused_items_for_file_is_noop_for_already_focused() -> None:
    focused = focus_candidate(
        _candidate(source=_MULTI_SRC),
        list(extract_symbol_spans(_MULTI_SRC, Path("pkg/a.py")))[0],
    )
    assert focused_items_for_file(focused) == (focused,)


def test_focus_package_is_deterministic_and_only_expands_selected() -> None:
    candidate = _candidate(source=_MULTI_SRC, path="pkg/a.py")
    other = _candidate(source="x = 1\n", path="pkg/b.py")
    package = _package(candidate, other)
    expanded = focus_package(package)
    assert [c.path.as_posix() for c in expanded.selected_files][:2] == [
        "pkg/a.py",
        "pkg/a.py",
    ]
    # Deterministic: expanding twice yields identical packages.
    assert focus_package(package).to_dict() == expanded.to_dict()
    # Restricting paths leaves others as full-file items only.
    restricted = focus_package(package, paths=[Path("pkg/a.py")])
    by_path = [c.path.as_posix() for c in restricted.selected_files]
    assert by_path.count("pkg/b.py") == 1


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def test_focused_candidate_recomputes_tokens_from_slice() -> None:
    candidate = _candidate(source=_MULTI_SRC)
    focused = focused_items_for_file(candidate)[1]
    assert focused.estimated_tokens == estimate_tokens(focused.source)
    assert focused.estimated_tokens < candidate.estimated_tokens
    assert focused.source == "def alpha():\n    return 1\n"
    assert focused.focus_start_line == 1
    assert focused.focus_end_line == 2


def test_focused_tokens_account_in_package_total() -> None:
    items = focused_items_for_file(_candidate(source=_MULTI_SRC))
    package = _package(*items)
    assert package.total_estimated_tokens == sum(
        c.estimated_tokens for c in items
    )
    assert package.total_estimated_tokens < estimate_tokens(_MULTI_SRC) * 3


def test_focused_items_can_fit_budget_where_full_file_cannot() -> None:
    # A large file: the whole file exceeds a small budget, but its focused
    # symbols each fit.  (Measurement only — the selection fix is NOT wired in.)
    lines = ["def sym_%d():\n    return %d\n" % (i, i) for i in range(500)]
    big = _candidate(source="".join(lines))
    budget = ContextBudget(max_tokens=60)
    full_selected, full_excluded = select_within_budget([big], budget)
    assert full_selected == []
    assert full_excluded[0].reason == "exceeds_total_budget"

    items = focused_items_for_file(big)
    focused_selected, focused_excluded = select_within_budget(
        list(items[1:]), budget
    )
    assert focused_selected
    assert all(c.estimated_tokens <= 60 for c in focused_selected)
    assert all(c.is_focused for c in focused_selected)


# ---------------------------------------------------------------------------
# Multiple symbols from one file
# ---------------------------------------------------------------------------


def test_multiple_focused_symbols_share_one_path() -> None:
    items = focused_items_for_file(_candidate(source=_MULTI_SRC))
    focused = [i for i in items if i.is_focused]
    assert len(focused) == 2
    assert {i.path for i in focused} == {Path("pkg/a.py")}
    assert {i.focus_name for i in focused} == {"alpha", "beta"}
    assert len({i.focus_start_line for i in focused}) == 2


# ---------------------------------------------------------------------------
# Firewall compatibility
# ---------------------------------------------------------------------------


def test_firewall_allows_focused_items_and_preserves_focus() -> None:
    items = focused_items_for_file(_candidate(source=_MULTI_SRC))
    package = _package(*items)
    firewall = ContextFirewall()
    result = firewall.inspect(package)
    assert result.safe is True
    assert len(result.allowed) == 3
    safe = firewall.safe_package(package, result)
    focused_safe = [c for c in safe.safe_files if c.focus_start_line is not None]
    assert len(focused_safe) == 2
    assert {c.focus_name for c in focused_safe} == {"alpha", "beta"}
    assert [c.focus_name for c in safe.safe_files] == [None, "alpha", "beta"]


def test_firewall_redacts_focused_slice_and_keeps_focus() -> None:
    source = "def exposed():\n    token = 'sk-test123456789012345678901234567'\n"
    candidate = _candidate(source=source)
    focused = focused_items_for_file(candidate)[1]
    package = _package(focused)
    firewall = ContextFirewall()
    result = firewall.inspect(package)
    assert result.has_findings
    safe = firewall.safe_package(package, result)
    safe_focused = [c for c in safe.safe_files if c.focus_start_line is not None]
    assert len(safe_focused) == 1
    assert safe_focused[0].decision == "redact"
    assert safe_focused[0].focus_name == "exposed"
    assert "[REDACTED]" in safe_focused[0].source or "token = " in safe_focused[0].source


def test_firewall_blocked_focused_item_carries_focus() -> None:
    candidate = _candidate(path="cert.pem", source=_MULTI_SRC)
    items = focused_items_for_file(candidate)
    package = _package(*items)
    firewall = ContextFirewall()
    result = firewall.inspect(package)
    assert "cert.pem" in result.blocked
    safe = firewall.safe_package(package, result)
    blocked_files = [c for c in safe.blocked_files if c.focus_start_line is not None]
    assert {c.focus_name for c in blocked_files} == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_unfocused_candidate_serialization_is_unchanged() -> None:
    candidate = _candidate(source=_MULTI_SRC)
    package = _package(candidate)
    item = package.to_dict()["selected_files"][0]
    assert "focus_name" not in item
    assert "focus_start_line" not in item


def test_focused_candidate_serializes_focus_keys() -> None:
    items = focused_items_for_file(_candidate(source=_MULTI_SRC))
    package = _package(*items)
    data = package.to_dict()
    focused = [c for c in data["selected_files"] if "focus_name" in c]
    assert len(focused) == 2
    alpha = focused[0]
    assert alpha["focus_name"] == "alpha"
    assert alpha["focus_kind"] == "function"
    assert alpha["focus_start_line"] == 1
    assert alpha["focus_end_line"] == 2
    assert alpha["inclusion_reason"] == INCLUSION_SYMBOL_MATCH
    assert alpha["retrieval_rank"] == 1
    # JSON round-trip is lossless for focus metadata.
    decoded = json.loads(package.to_json())
    assert {c["focus_name"] for c in decoded["selected_files"] if "focus_name" in c} == {
        "alpha",
        "beta"
    }


def test_focused_serialization_is_deterministic() -> None:
    items = focused_items_for_file(_candidate(source=_MULTI_SRC))
    package = _package(*items)
    assert package.to_dict() == _package(*items).to_dict()


def test_safe_focused_serialization_round_trips() -> None:
    items = focused_items_for_file(_candidate(source=_MULTI_SRC))
    package = _package(*items)
    firewall = ContextFirewall()
    result = firewall.inspect(package)
    safe = firewall.safe_package(package, result)
    data = safe.to_dict()
    safe_focused = [c for c in data["safe_files"] if "focus_name" in c]
    assert {c["focus_name"] for c in safe_focused} == {"alpha", "beta"}
    decoded = json.loads(safe.to_json())
    assert {c["focus_name"] for c in decoded["safe_files"] if "focus_name" in c} == {
        "alpha",
        "beta",
    }


def test_render_marks_focused_blocks() -> None:
    items = focused_items_for_file(_candidate(source=_MULTI_SRC))
    text = render_context(_package(*items))
    assert "Focus: alpha (function) lines 1-2" in text
    assert "Focus: beta (function) lines 4-5" in text


def test_snapshot_candidate_carries_focus_provenance() -> None:
    focused = focused_items_for_file(_candidate(source=_MULTI_SRC))[1]
    snap = snapshot_candidate(focused)
    assert snap["focus_name"] == "alpha"
    assert snap["focus_start_line"] == 1
    assert snap["focus_end_line"] == 2
    assert snap["focus_kind"] == "function"