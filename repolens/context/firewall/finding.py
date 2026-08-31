"""Structured finding model for the context firewall (Milestone 13).

Each :class:`Finding` is safe to send to an agent — it never contains the
actual secret value, only metadata about what was detected and why.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    """One detected issue in a context candidate.

    Findings are always safe to serialize: no matched secret value is
    ever included.
    """

    path: str
    line: int | None
    type: str
    severity: str
    decision: str
    reason: str
