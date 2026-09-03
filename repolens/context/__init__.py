"""Dependency-aware context engine with firewall (Milestones 12–13).

RepoLens goes beyond retrieving relevant files: :class:`ContextEngine` turns a
developer query into the smallest useful package of repository context an AI
coding agent needs to understand a task.  The :class:`ContextFirewall` then
inspects that package for potentially sensitive information before exposure.

Pipeline::

    query → retrieval → candidates → dependency expansion → ranking →
    budget → ContextPackage → ContextFirewall → SafeContextPackage

Public surface (kept small and composable):

- :class:`ContextEngine` — the main entry point (``build_context(query)``).
- :class:`~repolens.context.config.RetrievalConfig`,
  :class:`~repolens.context.config.DependencyExpansionConfig`,
  :class:`~repolens.context.config.ContextBudget` — configuration.
- :class:`~repolens.context.candidate.ContextCandidate`,
  :class:`~repolens.context.candidate.ExcludedCandidate`,
  :class:`~repolens.context.candidate.CandidateRole` — candidate model.
- :class:`~repolens.context.package.ContextPackage` — the serializable result.
- :func:`~repolens.context.tokens.estimate_tokens` — deterministic token estimate.
- :func:`~repolens.context.render.render_context` — deterministic text rendering.
- :class:`~repolens.context.firewall.ContextFirewall` — context firewall.
- :class:`~repolens.context.firewall.FirewallConfig` — firewall policy.
- :class:`~repolens.context.firewall.FirewallResult` — inspection result.
- :class:`~repolens.context.firewall.SafeContextPackage` — safe context.
- :class:`~repolens.context.firewall.FirewallDecision` — ALLOW/REDACT/BLOCK.

No agents, MCP, CLI, or server are part of this milestone.
"""

from repolens.context.candidate import (
    INCLUSION_DEPENDENCY,
    INCLUSION_DEPENDENT,
    INCLUSION_HYBRID_MATCH,
    INCLUSION_LEXICAL_MATCH,
    INCLUSION_SEMANTIC_MATCH,
    INCLUSION_SYMBOL_MATCH,
    CandidateRole,
    ContextCandidate,
    ExcludedCandidate,
)
from repolens.context.config import (
    ContextBudget,
    DependencyExpansionConfig,
    RetrievalConfig,
)
from repolens.context.engine import ContextEngine
from repolens.context.firewall import (
    ContextFirewall,
    Finding,
    FirewallConfig,
    FirewallDecision,
    FirewallResult,
    SafeContextCandidate,
    SafeContextPackage,
    Severity,
)
from repolens.context.intent import QueryIntent, classify_intent, extract_symbol_tokens
from repolens.context.package import ContextPackage
from repolens.context.render import render_context
from repolens.context.tokens import estimate_tokens

__all__ = [
    "INCLUSION_DEPENDENCY",
    "INCLUSION_DEPENDENT",
    "INCLUSION_HYBRID_MATCH",
    "INCLUSION_LEXICAL_MATCH",
    "INCLUSION_SEMANTIC_MATCH",
    "INCLUSION_SYMBOL_MATCH",
    "CandidateRole",
    "ContextBudget",
    "ContextCandidate",
    "ContextEngine",
    "ContextFirewall",
    "ContextPackage",
    "DependencyExpansionConfig",
    "ExcludedCandidate",
    "Finding",
    "FirewallConfig",
    "FirewallDecision",
    "FirewallResult",
    "QueryIntent",
    "RetrievalConfig",
    "SafeContextCandidate",
    "SafeContextPackage",
    "Severity",
    "classify_intent",
    "estimate_tokens",
    "extract_symbol_tokens",
    "render_context",
]
