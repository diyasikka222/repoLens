"""Deterministic text rendering of a context package for an AI agent.

:func:`render_context` turns a :class:`~repolens.context.package.ContextPackage`
into a stable, human/agent-readable text representation. The exact output is
reproducible: the same package always renders the same string.
"""

from __future__ import annotations

from repolens.context.candidate import CandidateRole
from repolens.context.package import ContextPackage


def render_context(package: ContextPackage) -> str:
    """Render ``package`` as deterministic markdown-style text."""
    parts: list[str] = []

    parts.append("# RepoLens Context")
    parts.append("")
    parts.append(f"Query: {package.query!r}")
    parts.append("")
    parts.append(
        f"Budget: {_budget_label(package.budget)} | "
        f"Selected files: {len(package.selected_files)} | "
        f"Total estimated tokens: {package.total_estimated_tokens}"
    )
    parts.append("")

    for section_title, candidates in (
        ("## Primary Context", _selected_by_role(package, CandidateRole.PRIMARY)),
        ("## Dependency Context", _selected_by_role(
            package, CandidateRole.DEPENDENCY, CandidateRole.DEPENDENT)),
    ):
        if not candidates:
            continue
        parts.append(section_title)
        parts.append("")
        for candidate in candidates:
            parts.extend(_render_candidate(candidate))
    parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _selected_by_role(package: ContextPackage, *roles: CandidateRole) -> list:
    wanted = set(roles)
    return [c for c in package.selected_files if c.role in wanted]


def _budget_label(budget) -> str:
    if budget.max_tokens is None:
        return "unlimited"
    return str(budget.max_tokens)


def _render_candidate(candidate) -> list[str]:
    lines: list[str] = []
    lines.append(f"### {candidate.path.as_posix()}")
    lines.append(f"Reason: {candidate.selection_reason}")
    lines.append(f"Estimated tokens: {candidate.estimated_tokens}")
    if candidate.retrieval_rank is not None:
        lines.append(f"Retrieval rank: {candidate.retrieval_rank}")
    if candidate.role is not CandidateRole.PRIMARY:
        lines.append(f"Graph distance: {candidate.graph_distance}")
    lines.append("")
    lines.append("```python")
    lines.append(candidate.source.rstrip())
    lines.append("```")
    lines.append("")
    return lines
