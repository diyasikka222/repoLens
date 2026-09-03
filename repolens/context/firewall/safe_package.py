"""Safe context package model for the context firewall (Milestone 13).

A :class:`SafeContextPackage` is the output of :meth:`ContextFirewall.inspect`.
It preserves useful metadata from the original :class:`~repolens.context.package.ContextPackage`
but contains only content that the firewall has deemed safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from repolens.context.candidate import ContextCandidate
from repolens.context.config import ContextBudget
from repolens.context.firewall.finding import Finding


@dataclass(frozen=True)
class SafeContextCandidate:
    """A context candidate whose content has been inspected by the firewall.

    Blocked candidates are replaced with a sentinel message.  Redacted
    candidates have their source text sanitized.
    """

    path: str
    source: str
    role: str
    estimated_tokens: int
    selection_reason: str
    decision: str
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    graph_distance: int | None = None
    inclusion_reason: str | None = None


@dataclass(frozen=True)
class SafeContextPackage:
    """A firewall-inspected context package safe for agent consumption.

    Attributes:
        query: The original developer query.
        budget: The token budget applied.
        safe_files: Candidates that passed inspection (allowed or redacted).
        blocked_files: Candidates that were removed by the firewall.
        findings: All findings from the inspection.
        firewall_enabled: Whether the firewall was active.
        policy_version: The policy version applied.
    """

    query: str
    budget: ContextBudget
    safe_files: tuple[SafeContextCandidate, ...] = ()
    blocked_files: tuple[SafeContextCandidate, ...] = ()
    findings: tuple[Finding, ...] = ()
    firewall_enabled: bool = True
    policy_version: str = "1.0.0"
    intent: str | None = None
    matched_symbols: tuple[str, ...] = ()

    @property
    def total_estimated_tokens(self) -> int:
        """Sum of estimated tokens across safe files only."""
        return sum(item.estimated_tokens for item in self.safe_files)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "budget": {
                "max_tokens": self.budget.max_tokens,
                "truncate_oversized": self.budget.truncate_oversized,
            },
            "total_estimated_tokens": self.total_estimated_tokens,
            "intent": self.intent,
            "matched_symbols": list(self.matched_symbols),
            "safe_files": [_safe_candidate_dict(c) for c in self.safe_files],
            "blocked_files": [_safe_candidate_dict(c) for c in self.blocked_files],
            "findings": [
                {
                    "path": f.path,
                    "line": f.line,
                    "type": f.type,
                    "severity": f.severity,
                    "decision": f.decision,
                    "reason": f.reason,
                }
                for f in self.findings
            ],
            "firewall_enabled": self.firewall_enabled,
            "policy_version": self.policy_version,
        }

    def to_json(self, **json_kwargs) -> str:
        """Return the package serialized as a JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)


def _safe_candidate_dict(candidate: SafeContextCandidate) -> dict:
    return {
        "path": candidate.path,
        "role": candidate.role,
        "estimated_tokens": candidate.estimated_tokens,
        "selection_reason": candidate.selection_reason,
        "inclusion_reason": candidate.inclusion_reason,
        "decision": candidate.decision,
        "retrieval_rank": candidate.retrieval_rank,
        "retrieval_score": candidate.retrieval_score,
        "lexical_rank": candidate.lexical_rank,
        "semantic_rank": candidate.semantic_rank,
        "graph_distance": candidate.graph_distance,
    }
