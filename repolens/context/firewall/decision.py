"""Decision model for the context firewall (Milestone 13).

Three explicit decisions for how each candidate is handled:

- :attr:`FirewallDecision.ALLOW` — expose as-is.
- :attr:`FirewallDecision.REDACT` — expose with sensitive portions replaced.
- :attr:`FirewallDecision.BLOCK` — do not expose at all.

Severity levels indicate the confidence and importance of a finding.
"""

from __future__ import annotations

from enum import Enum


class FirewallDecision(str, Enum):
    """What the firewall decided to do with a candidate."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class Severity(str, Enum):
    """How serious a finding is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
