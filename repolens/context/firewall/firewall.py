"""Context firewall for agent-bound context (Milestone 13).

:class:`ContextFirewall` inspects a :class:`~repolens.context.package.ContextPackage`
and produces a :class:`~repolens.context.firewall.safe_package.SafeContextPackage`
that is safe for exposure to an AI agent.

Pipeline::

    ContextPackage
        → path-based rules
        → content-based scanning
        → per-candidate decision (ALLOW / REDACT / BLOCK)
        → SafeContextPackage

The firewall is deterministic, explainable, and independent of any LLM.
"""

from __future__ import annotations

from repolens.context.candidate import CandidateRole
from repolens.context.config import ContextBudget
from repolens.context.firewall.config import (
    FIREWALL_POLICY_VERSION,
    FirewallConfig,
)
from repolens.context.firewall.content_detectors import (
    check_content,
    redact_source,
)
from repolens.context.firewall.decision import FirewallDecision
from repolens.context.firewall.finding import Finding
from repolens.context.firewall.path_rules import check_path_rules
from repolens.context.firewall.result import FirewallResult
from repolens.context.firewall.safe_package import (
    SafeContextCandidate,
    SafeContextPackage,
)
from repolens.context.package import ContextPackage


class ContextFirewall:
    """Inspect a context package and produce a safe version for agent exposure.

    Example::

        firewall = ContextFirewall()
        result = firewall.inspect(package)
        safe = firewall.safe_package(package, result)

    The firewall is instantiated once with a configuration and may inspect
    many packages.  When ``config.enabled`` is ``False`` everything is
    allowed through unmodified.
    """

    def __init__(self, config: FirewallConfig | None = None) -> None:
        self._config = config if config is not None else FirewallConfig()

    @property
    def config(self) -> FirewallConfig:
        """Return the active configuration (read-only)."""
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self, package: ContextPackage) -> FirewallResult:
        """Inspect every selected file in ``package`` and return a result.

        Each selected file is inspected at most once.  No files outside the
        package's ``selected_files`` are read from disk.
        """
        if not self._config.enabled:
            return self._allow_all(package)

        allowed_paths: list[str] = []
        redacted_paths: list[str] = []
        blocked_paths: list[str] = []
        all_findings: list[Finding] = []

        for candidate in package.selected_files:
            relative = candidate.path.as_posix()
            findings: list[Finding] = []

            # --- path rules ---
            path_decision, path_findings = check_path_rules(
                relative, self._config,
            )
            findings.extend(path_findings)

            # --- content rules (skip if already BLOCKed by path) ---
            if path_decision is not FirewallDecision.BLOCK:
                try:
                    content_findings = check_content(
                        candidate.source, relative, self._config,
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    # Fail-closed: if we cannot inspect a file we must not
                    # expose it.  The diagnostic is safe (type name only,
                    # never the content or exception message with secrets).
                    content_findings = [
                        Finding(
                            path=relative,
                            line=None,
                            type="scan_error",
                            severity="high",
                            decision="block",
                            reason=(
                                "File could not be inspected safely; "
                                f"scan failed ({type(exc).__name__})"
                            ),
                        )
                    ]
                findings.extend(content_findings)

            # --- aggregate decision ---
            decision = self._aggregate_decision(path_decision, findings)
            all_findings.extend(findings)

            if decision is FirewallDecision.BLOCK:
                blocked_paths.append(relative)
            elif decision is FirewallDecision.REDACT:
                redacted_paths.append(relative)
            else:
                allowed_paths.append(relative)

        return FirewallResult(
            safe=len(all_findings) == 0,
            allowed=tuple(allowed_paths),
            redacted=tuple(redacted_paths),
            blocked=tuple(blocked_paths),
            findings=tuple(all_findings),
            firewall_enabled=True,
            policy_version=self._config.policy_version,
        )

    def safe_package(
        self,
        package: ContextPackage,
        result: FirewallResult,
    ) -> SafeContextPackage:
        """Produce a :class:`SafeContextPackage` from ``package`` and ``result``.

        Redacted files have their content sanitized.  Blocked files are
        replaced with a sentinel message.  Allowed files pass through
        unchanged.
        """
        if not self._config.enabled:
            return self._passthrough_package(package)

        safe_files: list[SafeContextCandidate] = []
        blocked_files: list[SafeContextCandidate] = []

        # Build lookup: path -> decision
        decision_map: dict[str, str] = {}
        for p in result.allowed:
            decision_map[p] = "allow"
        for p in result.redacted:
            decision_map[p] = "redact"
        for p in result.blocked:
            decision_map[p] = "block"

        for candidate in package.selected_files:
            relative = candidate.path.as_posix()
            decision = decision_map.get(relative, "allow")

            if decision == "block":
                blocked_files.append(
                    SafeContextCandidate(
                        path=relative,
                        source="[BLOCKED by firewall]",
                        role=candidate.role.value,
                        estimated_tokens=1,
                        selection_reason=candidate.selection_reason,
                        decision="block",
                        retrieval_rank=candidate.retrieval_rank,
                        retrieval_score=candidate.retrieval_score,
                        lexical_rank=candidate.lexical_rank,
                        semantic_rank=candidate.semantic_rank,
                        graph_distance=candidate.graph_distance,
                        inclusion_reason=candidate.inclusion_reason,
                    )
                )
            elif decision == "redact":
                # Find findings for this path to drive redaction.
                path_findings = [
                    f for f in result.findings if f.path == relative
                ]
                redacted_source = redact_source(
                    candidate.source,
                    list(path_findings),
                    self._config.redaction_placeholder,
                )
                from repolens.context.tokens import estimate_tokens

                safe_files.append(
                    SafeContextCandidate(
                        path=relative,
                        source=redacted_source,
                        role=candidate.role.value,
                        estimated_tokens=estimate_tokens(redacted_source),
                        selection_reason=candidate.selection_reason,
                        decision="redact",
                        retrieval_rank=candidate.retrieval_rank,
                        retrieval_score=candidate.retrieval_score,
                        lexical_rank=candidate.lexical_rank,
                        semantic_rank=candidate.semantic_rank,
                        graph_distance=candidate.graph_distance,
                        inclusion_reason=candidate.inclusion_reason,
                    )
                )
            else:
                safe_files.append(
                    SafeContextCandidate(
                        path=relative,
                        source=candidate.source,
                        role=candidate.role.value,
                        estimated_tokens=candidate.estimated_tokens,
                        selection_reason=candidate.selection_reason,
                        decision="allow",
                        retrieval_rank=candidate.retrieval_rank,
                        retrieval_score=candidate.retrieval_score,
                        lexical_rank=candidate.lexical_rank,
                        semantic_rank=candidate.semantic_rank,
                        graph_distance=candidate.graph_distance,
                        inclusion_reason=candidate.inclusion_reason,
                    )
                )

        return SafeContextPackage(
            query=package.query,
            budget=package.budget,
            safe_files=tuple(safe_files),
            blocked_files=tuple(blocked_files),
            findings=result.findings,
            firewall_enabled=True,
            policy_version=result.policy_version,
            intent=package.intent,
            matched_symbols=package.matched_symbols,
        )

    def is_safe(self, package: ContextPackage) -> bool:
        """Return ``True`` if the package has no findings."""
        return not self.inspect(package).has_findings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_decision(
        self,
        path_decision: FirewallDecision | None,
        findings: list[Finding],
    ) -> FirewallDecision:
        """Determine the final decision for a candidate.

        Priority: path BLOCK > scan_error BLOCK > content HIGH > content
        MEDIUM > path decision > default.
        """
        if path_decision is FirewallDecision.BLOCK:
            return FirewallDecision.BLOCK

        # Fail-closed: an unreadable/unscannable file is always BLOCKed.
        if any(f.type == "scan_error" for f in findings):
            return FirewallDecision.BLOCK

        has_high = any(f.severity == "high" for f in findings)
        has_medium = any(f.severity == "medium" for f in findings)

        if has_high:
            return FirewallDecision.REDACT
        if has_medium:
            return FirewallDecision.REDACT

        if path_decision is not None:
            return path_decision

        return FirewallDecision.ALLOW

    def _allow_all(self, package: ContextPackage) -> FirewallResult:
        """Return a result that allows everything (firewall disabled)."""
        paths = tuple(c.path.as_posix() for c in package.selected_files)
        return FirewallResult(
            safe=True,
            allowed=paths,
            redacted=(),
            blocked=(),
            findings=(),
            firewall_enabled=False,
            policy_version=self._config.policy_version,
        )

    def _passthrough_package(
        self, package: ContextPackage
    ) -> SafeContextPackage:
        """Produce a safe package that mirrors the original (firewall disabled)."""
        safe_files = tuple(
            SafeContextCandidate(
                path=c.path.as_posix(),
                source=c.source,
                role=c.role.value,
                estimated_tokens=c.estimated_tokens,
                selection_reason=c.selection_reason,
                decision="allow",
                retrieval_rank=c.retrieval_rank,
                retrieval_score=c.retrieval_score,
                lexical_rank=c.lexical_rank,
                semantic_rank=c.semantic_rank,
                graph_distance=c.graph_distance,
                inclusion_reason=c.inclusion_reason,
            )
            for c in package.selected_files
        )
        return SafeContextPackage(
            query=package.query,
            budget=package.budget,
            safe_files=safe_files,
            blocked_files=(),
            findings=(),
            firewall_enabled=False,
            policy_version=self._config.policy_version,
            intent=package.intent,
            matched_symbols=package.matched_symbols,
        )
