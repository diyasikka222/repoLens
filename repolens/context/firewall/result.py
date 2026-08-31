"""Structured result returned by :class:`ContextFirewall.inspect` (Milestone 13).

A :class:`FirewallResult` contains the full outcome of inspection: the overall
safe/unsafe status, categorised candidates, findings, and the policy that was
applied.
"""

from __future__ import annotations

from dataclasses import dataclass

from repolens.context.firewall.decision import FirewallDecision
from repolens.context.firewall.finding import Finding


@dataclass(frozen=True)
class FirewallResult:
    """The structured output of a firewall inspection.

    Attributes:
        safe: Whether the entire context package is safe without changes.
        allowed: Candidates that pass inspection unchanged.
        redacted: Candidates that pass inspection with content redacted.
        blocked: Candidates that were completely removed.
        findings: All individual findings across every inspected file.
        firewall_enabled: Whether the firewall was enabled for this run.
        policy_version: Version identifier for the applied policy.
    """

    safe: bool
    allowed: tuple[str, ...]
    redacted: tuple[str, ...]
    blocked: tuple[str, ...]
    findings: tuple[Finding, ...]
    firewall_enabled: bool
    policy_version: str

    @property
    def has_findings(self) -> bool:
        """Return ``True`` if any findings were produced."""
        return len(self.findings) > 0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "safe": self.safe,
            "allowed": list(self.allowed),
            "redacted": list(self.redacted),
            "blocked": list(self.blocked),
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
