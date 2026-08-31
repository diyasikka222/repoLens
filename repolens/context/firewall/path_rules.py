"""Path-based detection rules for the context firewall (Milestone 13).

Rules operate on the candidate's repository-relative path.  A path rule
produces a BLOCK decision for files that are almost always secret material
regardless of content (private keys, certificates, environment files).

Ordinary source files are *not* blocked merely because their name contains
words such as ``token``, ``password``, or ``secret``.  Only well-known
secret-file names and extensions are matched.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from repolens.context.firewall.config import FirewallConfig
from repolens.context.firewall.decision import FirewallDecision
from repolens.context.firewall.finding import Finding


def check_path_rules(
    relative_path: str,
    config: FirewallConfig,
) -> tuple[FirewallDecision | None, list[Finding]]:
    """Evaluate path-based rules against ``relative_path``.

    Returns ``(decision, findings)`` where ``decision`` is ``BLOCK`` if a
    rule matched, or ``None`` if no path rule applies.
    """
    posix = PurePosixPath(relative_path)
    name = posix.name.lower()

    # --- exact filename match ---
    if name in config.blocked_filenames:
        return (
            FirewallDecision.BLOCK,
            [
                Finding(
                    path=relative_path,
                    line=None,
                    type="path_rule",
                    severity="high",
                    decision="block",
                    reason=f"Sensitive file detected: {name}",
                )
            ],
        )

    # --- extension match ---
    suffix = posix.suffix.lower()
    if suffix in config.blocked_extensions:
        return (
            FirewallDecision.BLOCK,
            [
                Finding(
                    path=relative_path,
                    line=None,
                    type="path_rule",
                    severity="high",
                    decision="block",
                    reason=f"Sensitive file extension detected: {suffix}",
                )
            ],
        )

    # --- filename contains "secret" only for non-source config/data files ---
    # Deliberately *not* applied to ``.py`` (or other source) modules, because
    # the Python ``secrets`` module and helpers named ``secrets.py`` are
    # ordinary, legitimate source files.  This keeps high precision.
    stem = posix.stem.lower()
    secret_config_suffixes = {".json", ".yaml", ".yml", ".env", ".ini",
                              ".toml", ".txt", ".cfg", ".pem", ".key", ".p12"}
    if (
        (stem.startswith("secret") or stem.startswith("secrets"))
        and posix.suffix.lower() in secret_config_suffixes
    ):
        return (
            FirewallDecision.BLOCK,
            [
                Finding(
                    path=relative_path,
                    line=None,
                    type="path_rule",
                    severity="medium",
                    decision="block",
                    reason=f"Secret config/data file detected: {name}",
                )
            ],
        )

    return None, []
