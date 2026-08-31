"""Content-based secret detectors for the context firewall (Milestone 13).

Each detector is a named function that scans a file's source text for a
specific pattern of sensitive material.  Detectors are designed for **high
precision**: false positives are more harmful than false negatives because
legitimate source code could be unnecessarily removed from agent context.

The actual matched secret value is *never* included in any finding.
"""

from __future__ import annotations

import re

from repolens.context.firewall.config import FirewallConfig
from repolens.context.firewall.finding import Finding

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for performance — each file is scanned once)
# ---------------------------------------------------------------------------

_OPENAI_API_KEY = re.compile(r"sk-[A-Za-z0-9]{20,}")
_AWS_ACCESS_KEY = re.compile(r"AKIA[0-9A-Z]{16}")
_GITHUB_TOKEN = re.compile(r"ghp_[A-Za-z0-9]{36}")
_GITHUB_APP_TOKEN = re.compile(r"gho_[A-Za-z0-9]{36}|ghu_[A-Za-z0-9]{36}")
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN\s+(RSA|DSA|EC|OPENSSH)?\s*PRIVATE\s+KEY-----"
)
_BEARER_TOKEN = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
_DATABASE_URL = re.compile(
    r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s\"']+:[^\s\"']+@[^\s\"']+",
    re.IGNORECASE,
)
_GENERIC_SECRET = re.compile(
    r"""(?:secret|api_?key|access_?key|private_?key|auth_?token|client_?secret)\s*=\s*['\"][A-Za-z0-9\-._]{20,}['"]""",
    re.IGNORECASE,
)
_SLACK_TOKEN = re.compile(r"xox[bpsa]-[0-9]{10,}-[A-Za-z0-9\-]+")

# All detectors in evaluation order.  Each entry is
# ``(name, pattern, type_label, severity, reason)``.
_DETECTORS: list[tuple[str, re.Pattern[str], str, str, str]] = [
    (
        "private_key_block",
        _PRIVATE_KEY_BLOCK,
        "private_key",
        "high",
        "Potential private key block detected",
    ),
    (
        "openai_api_key",
        _OPENAI_API_KEY,
        "api_key",
        "high",
        "Potential API credential detected",
    ),
    (
        "aws_access_key",
        _AWS_ACCESS_KEY,
        "api_key",
        "high",
        "Potential AWS access key ID detected",
    ),
    (
        "github_token",
        _GITHUB_TOKEN,
        "api_token",
        "high",
        "Potential GitHub personal access token detected",
    ),
    (
        "github_app_token",
        _GITHUB_APP_TOKEN,
        "api_token",
        "high",
        "Potential GitHub app/OAuth token detected",
    ),
    (
        "database_url_with_credentials",
        _DATABASE_URL,
        "database_url",
        "high",
        "Potential database URL with embedded credentials detected",
    ),
    (
        "bearer_token",
        _BEARER_TOKEN,
        "bearer_token",
        "medium",
        "Potential bearer token detected",
    ),
    (
        "slack_token",
        _SLACK_TOKEN,
        "api_token",
        "high",
        "Potential Slack API token detected",
    ),
    (
        "generic_secret_assignment",
        _GENERIC_SECRET,
        "secret_assignment",
        "medium",
        "Potential secret value assignment detected",
    ),
]


def check_content(
    source: str,
    relative_path: str,
    config: FirewallConfig,
) -> list[Finding]:
    """Scan ``source`` for sensitive content patterns.

    Returns a list of :class:`Finding` objects.  Each finding includes the
    line number, detector type, severity, and a safe explanation — but
    *never* the matched secret value.
    """
    findings: list[Finding] = []
    enabled = config.content_detectors
    lines = source.split("\n")

    for detector_name, pattern, type_label, severity, reason in _DETECTORS:
        if detector_name not in enabled:
            continue

        for line_idx, line in enumerate(lines, start=1):
            for match in pattern.finditer(line):
                findings.append(
                    Finding(
                        path=relative_path,
                        line=line_idx,
                        type=type_label,
                        severity=severity,
                        decision="redact",
                        reason=reason,
                    )
                )

    return findings


def redact_source(
    source: str,
    findings: list[Finding],
    placeholder: str,
) -> str:
    """Return ``source`` with all detected secrets replaced by ``placeholder``.

    Operates line-by-line to preserve structure.  Each finding is applied
    to its reported line only.
    """
    lines = source.split("\n")
    # Build a map: line_number -> list of (pattern, reason) for this file.
    line_findings: dict[int, list[tuple[re.Pattern[str], str]]] = {}
    for finding in findings:
        if finding.line is not None:
            line_findings.setdefault(finding.line, [])

    # Re-run the relevant patterns on the lines that have findings.
    # We rebuild to avoid storing matched values.
    enabled = _enabled_detectors(findings)
    for detector_name, pattern, type_label, severity, reason in _DETECTORS:
        if detector_name not in enabled:
            continue
        for finding in findings:
            if finding.line is not None and finding.type == type_label:
                line_findings.setdefault(finding.line, []).append(
                    (pattern, reason)
                )

    for line_num in sorted(line_findings):
        idx = line_num - 1
        if idx < len(lines):
            line = lines[idx]
            for pattern, _reason in line_findings[line_num]:
                line = pattern.sub(placeholder, line)
            lines[idx] = line

    return "\n".join(lines)


def _enabled_detectors(findings: list[Finding]) -> set[str]:
    """Infer which detector names produced the given findings."""
    type_to_names: dict[str, list[str]] = {}
    for name, _pattern, type_label, _severity, _reason in _DETECTORS:
        type_to_names.setdefault(type_label, []).append(name)

    enabled: set[str] = set()
    for finding in findings:
        for name in type_to_names.get(finding.type, []):
            enabled.add(name)
    return enabled
