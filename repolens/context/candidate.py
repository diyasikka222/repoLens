"""Structured context-candidate model for the dependency-aware context engine.

A :class:`ContextCandidate` is one repository file that was selected as part
of a context package, together with the metadata that explains *why* it was
selected: how it entered the pipeline (retrieved directly, or reached via a
dependency relationship), the retrieval signals it carries, its distance in
the dependency graph, and its estimated token cost.

The model deliberately exposes a small, stable surface and hides retrieval /
graph implementation internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CandidateRole(str, Enum):
    """Why a file entered the context candidate set."""

    #: Retrieved directly by the retrieval layer (a primary candidate).
    PRIMARY = "primary"
    #: Reached because a primary (or another candidate) imports it.
    DEPENDENCY = "dependency"
    #: Reached because it imports a primary (a reverse dependency / dependent).
    DEPENDENT = "dependent"


@dataclass(frozen=True)
class ContextCandidate:
    """One repository file proposed for the final context package."""

    path: Path
    source: str
    role: CandidateRole
    estimated_tokens: int
    selection_reason: str

    # Retrieval signals (present when the file also appeared in retrieval).
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    lexical_rank: int | None = None
    semantic_rank: int | None = None

    # Graph information (None for primary candidates).
    graph_distance: int | None = None


@dataclass(frozen=True)
class ExcludedCandidate:
    """A candidate considered but not included in the final package."""

    path: Path
    estimated_tokens: int
    reason: str
