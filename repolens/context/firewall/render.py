"""Deterministic text rendering of a safe context package (Milestone 13).

:func:`render_safe_context` turns a
:class:`~repolens.context.firewall.safe_package.SafeContextPackage` into a
stable, human/agent-readable text representation.

- Allowed files render normally.
- Redacted files are marked as redacted.
- Blocked files never appear in the output.

The exact output is reproducible: the same package always renders the
same string.
"""

from __future__ import annotations

from repolens.context.firewall.safe_package import SafeContextCandidate, SafeContextPackage


def render_safe_context(package: SafeContextPackage) -> str:
    """Render ``package`` as deterministic markdown-style text."""
    parts: list[str] = []

    parts.append("# RepoLens Safe Context")
    parts.append("")
    parts.append(f"Query: {package.query!r}")
    parts.append("")
    parts.append(
        f"Budget: {_budget_label(package.budget)} | "
        f"Safe files: {len(package.safe_files)} | "
        f"Blocked files: {len(package.blocked_files)} | "
        f"Total estimated tokens: {package.total_estimated_tokens}"
    )
    parts.append("")

    if package.findings:
        parts.append(f"Findings: {len(package.findings)}")
        parts.append("")

    for candidate in package.safe_files:
        parts.extend(_render_safe_candidate(candidate))

    # List blocked files as a summary (content never shown).
    if package.blocked_files:
        parts.append("## Blocked Files")
        parts.append("")
        for candidate in package.blocked_files:
            parts.append(f"- {candidate.path}")
        parts.append("")

    parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _budget_label(budget) -> str:
    if budget.max_tokens is None:
        return "unlimited"
    return str(budget.max_tokens)


def _render_safe_candidate(candidate: SafeContextCandidate) -> list[str]:
    lines: list[str] = []
    lines.append(f"### {candidate.path}")
    lines.append(f"Decision: {candidate.decision.upper()}")
    lines.append(f"Reason: {candidate.selection_reason}")
    lines.append(f"Estimated tokens: {candidate.estimated_tokens}")
    if candidate.retrieval_rank is not None:
        lines.append(f"Retrieval rank: {candidate.retrieval_rank}")
    if candidate.graph_distance is not None:
        lines.append(f"Graph distance: {candidate.graph_distance}")
    lines.append("")
    lines.append("```python")
    lines.append(candidate.source.rstrip())
    lines.append("```")
    lines.append("")
    return lines
