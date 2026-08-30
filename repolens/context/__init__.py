"""Dependency-aware context engine (Milestone 12).

RepoLens goes beyond retrieving relevant files: :class:`ContextEngine` turns a
developer query into the smallest useful package of repository context an AI
coding agent needs to understand a task.

Pipeline::

    query → retrieval → candidates → dependency expansion → ranking →
    budget → final context package

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

No agents, MCP, CLI, or server are part of this milestone.
"""

from repolens.context.candidate import CandidateRole, ContextCandidate, ExcludedCandidate
from repolens.context.config import (
    ContextBudget,
    DependencyExpansionConfig,
    RetrievalConfig,
)
from repolens.context.engine import ContextEngine
from repolens.context.package import ContextPackage
from repolens.context.render import render_context
from repolens.context.tokens import estimate_tokens

__all__ = [
    "CandidateRole",
    "ContextBudget",
    "ContextCandidate",
    "ContextEngine",
    "ContextPackage",
    "DependencyExpansionConfig",
    "ExcludedCandidate",
    "RetrievalConfig",
    "estimate_tokens",
    "render_context",
]
