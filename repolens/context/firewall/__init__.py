"""Context firewall for agent-bound context (Milestone 13).

The firewall inspects a :class:`~repolens.context.package.ContextPackage`
and produces a :class:`~repolens.context.firewall.safe_package.SafeContextPackage`
that is safe for exposure to an AI coding agent.

Pipeline::

    ContextPackage
        → ContextFirewall
        → SafeContextPackage

Three decisions per candidate:

- **ALLOW** — expose as-is.
- **REDACT** — expose with sensitive portions replaced.
- **BLOCK** — do not expose at all.

Security is ON by default.  The firewall is deterministic, explainable,
and independent of any LLM.
"""

from repolens.context.firewall.config import FirewallConfig
from repolens.context.firewall.decision import FirewallDecision, Severity
from repolens.context.firewall.finding import Finding
from repolens.context.firewall.firewall import ContextFirewall
from repolens.context.firewall.result import FirewallResult
from repolens.context.firewall.safe_package import (
    SafeContextCandidate,
    SafeContextPackage,
)

__all__ = [
    "ContextFirewall",
    "Finding",
    "FirewallConfig",
    "FirewallDecision",
    "FirewallResult",
    "SafeContextCandidate",
    "SafeContextPackage",
    "Severity",
]
