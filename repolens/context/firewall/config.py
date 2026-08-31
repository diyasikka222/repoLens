"""Configurable policy for the context firewall (Milestone 13).

A :class:`FirewallConfig` controls which detectors are active, which path
patterns trigger a BLOCK, and the placeholder used for redacted content.

Sensible defaults are provided: security is ON by default, all detectors
are enabled, and common secret-file patterns are blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet

FIREWALL_POLICY_VERSION = "1.0.0"

DEFAULT_REDACTION_PLACEHOLDER = "[REDACTED]"

#: Filenames that always trigger a BLOCK (lowercase, matched against ``name``).
DEFAULT_BLOCKED_FILENAMES: FrozenSet[str] = frozenset({
    ".env",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
})

#: File extensions that always trigger a BLOCK (lowercase, with leading dot).
DEFAULT_BLOCKED_EXTENSIONS: FrozenSet[str] = frozenset({
    ".pem",
    ".key",
    ".p12",
    ".pfx",
})

#: Content-detector names that are enabled by default.
ALL_CONTENT_DETECTORS: FrozenSet[str] = frozenset({
    "openai_api_key",
    "aws_access_key",
    "github_token",
    "github_app_token",
    "private_key_block",
    "generic_secret_assignment",
    "database_url_with_credentials",
    "bearer_token",
})


@dataclass(frozen=True)
class FirewallConfig:
    """Policy configuration for the :class:`ContextFirewall`.

    Attributes:
        enabled: Master switch.  When ``False`` the firewall allows everything.
        blocked_filenames: Exact filenames (lowercase) that trigger a BLOCK.
        blocked_extensions: File extensions (lowercase) that trigger a BLOCK.
        content_detectors: Names of enabled content detectors.  Pass an
            empty set to disable all content scanning.
        redaction_placeholder: The string that replaces detected secrets.
        default_action: The fallback decision when no rule matches.
        policy_version: A version identifier embedded in results.
    """

    enabled: bool = True
    blocked_filenames: FrozenSet[str] = DEFAULT_BLOCKED_FILENAMES
    blocked_extensions: FrozenSet[str] = DEFAULT_BLOCKED_EXTENSIONS
    content_detectors: FrozenSet[str] = ALL_CONTENT_DETECTORS
    redaction_placeholder: str = DEFAULT_REDACTION_PLACEHOLDER
    default_action: str = "allow"
    policy_version: str = FIREWALL_POLICY_VERSION
