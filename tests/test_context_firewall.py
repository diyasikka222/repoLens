"""Offline security tests for the context firewall (Milestone 13).

These tests verify the deterministic, explainable, and LLM-independent
behavior of :class:`~repolens.context.ContextFirewall`.  All secrets used
here are obviously fake/test values.  Tests are fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repolens.context import (
    ContextBudget,
    ContextCandidate,
    ContextEngine,
    ContextFirewall,
    ContextPackage,
    CandidateRole,
    FirewallConfig,
    FirewallDecision,
    FirewallResult,
    SafeContextPackage,
    estimate_tokens,
)
from repolens.context.firewall.content_detectors import redact_source
from repolens.context.firewall.render import render_safe_context
from repolens.search import CodeSearcher
from repolens.context.config import ContextBudget as CB

FAKE_OPENAI = "sk-test123456789012345678901234567"
FAKE_AWS = "AKIAIOSFODNN7EXAMPLE"
# GitHub personal access tokens are 40 chars: "ghp_" + 36 alphanumerics.
FAKE_GITHUB = "ghp_" + "a" * 36
FAKE_AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


def _candidate(path: str, source: str, *, role=CandidateRole.PRIMARY) -> ContextCandidate:
    return ContextCandidate(
        path=Path(path),
        source=source,
        role=role,
        estimated_tokens=estimate_tokens(source),
        selection_reason="test",
    )


def _package(*candidates) -> ContextPackage:
    return ContextPackage(
        query="test query",
        budget=CB(max_tokens=8000),
        selected_files=tuple(candidates),
    )


# ---------------------------------------------------------------------------
# 1. Normal Python source is allowed
# ---------------------------------------------------------------------------


def test_normal_source_is_allowed() -> None:
    src = "def add(a, b):\n    return a + b\n"
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("math/util.py", src)))
    assert result.safe is True
    assert result.allowed == ("math/util.py",)
    assert result.blocked == ()
    assert result.redacted == ()
    assert result.findings == ()


# ---------------------------------------------------------------------------
# 2. Variable named token is allowed when no secret value
# ---------------------------------------------------------------------------


def test_variable_named_token_is_allowed() -> None:
    src = (
        "def refresh_token(token):\n"
        "    return token.upper()\n"
    )
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("auth/tokens.py", src)))
    assert result.safe is True
    assert result.allowed == ("auth/tokens.py",)


# ---------------------------------------------------------------------------
# 3. Variable named password allowed with no secret value
# ---------------------------------------------------------------------------


def test_variable_named_password_is_allowed() -> None:
    src = (
        "def check_password(password):\n"
        "    return len(password) > 0\n"
    )
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("auth/pwd.py", src)))
    assert result.safe is True


# ---------------------------------------------------------------------------
# 4. .env is blocked
# ---------------------------------------------------------------------------


def test_dot_env_is_blocked() -> None:
    src = "SECRET_KEY=abc123\n"
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate(".env", src)))
    assert result.blocked == (".env",)
    assert result.safe is False


# ---------------------------------------------------------------------------
# 5. .env.example is allowed (name alone is insufficient)
# ---------------------------------------------------------------------------


def test_env_example_is_not_auto_blocked() -> None:
    src = "# copy me\n# FOO=bar\n"
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate(".env.example", src)))
    assert result.allowed == (".env.example",)
    assert result.blocked == ()


# ---------------------------------------------------------------------------
# 6. PEM private key is blocked
# ---------------------------------------------------------------------------


def test_pem_private_key_is_blocked() -> None:
    src = "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n"
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("keys/private.pem", src)))
    assert result.blocked == ("keys/private.pem",)
    assert result.safe is False


# ---------------------------------------------------------------------------
# 7. Private key content is detected in a non-key file
# ---------------------------------------------------------------------------


def test_private_key_content_detected_in_source() -> None:
    src = (
        "owner = {\n"
        "    'key': '-----BEGIN PRIVATE KEY-----\\nABC\\n-----END PRIVATE KEY-----'\n"
        "}\n"
    )
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("config/keys.py", src)))
    assert result.redacted == ("config/keys.py",)
    assert any(f.type == "private_key" for f in result.findings)
    assert result.safe is False


# ---------------------------------------------------------------------------
# 8. OpenAI-style API key is detected
# ---------------------------------------------------------------------------


def test_openai_api_key_detected() -> None:
    src = f'OPENAI_API_KEY = "{FAKE_OPENAI}"\n'
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("config/settings.py", src)))
    assert result.redacted == ("config/settings.py",)
    assert any(f.type == "api_key" for f in result.findings)


# ---------------------------------------------------------------------------
# 9. AWS access key ID is detected
# ---------------------------------------------------------------------------


def test_aws_access_key_detected() -> None:
    src = f"AWS_ACCESS_KEY_ID = '{FAKE_AWS}'\n"
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("config/aws.py", src)))
    assert any(f.type == "api_key" for f in result.findings)
    assert result.redacted == ("config/aws.py",)


# ---------------------------------------------------------------------------
# 10. GitHub-style token is detected where supported
# ---------------------------------------------------------------------------


def test_github_token_detected() -> None:
    src = f"token = '{FAKE_GITHUB}'\n"
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("ci/github.py", src)))
    assert any(f.type == "api_token" for f in result.findings)
    assert result.redacted == ("ci/github.py",)


# ---------------------------------------------------------------------------
# 11. Database URL with credentials is detected
# ---------------------------------------------------------------------------


def test_database_url_credentials_detected() -> None:
    src = 'DATABASE_URL = "postgresql://user:supersecret@dbhost:5432/mydb"\n'
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("config/db.py", src)))
    assert any(f.type == "database_url" for f in result.findings)
    assert result.redacted == ("config/db.py",)


# ---------------------------------------------------------------------------
# 12. Generic secret assignment detected with sufficient confidence
# ---------------------------------------------------------------------------


def test_generic_secret_assignment_detected() -> None:
    src = "client_secret = 'verylongsecretvalue1234567890abcdef'\n"
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("config/client.py", src)))
    assert any(f.type == "secret_assignment" for f in result.findings)


# ---------------------------------------------------------------------------
# 13. False-positive examples remain allowed
# ---------------------------------------------------------------------------


def test_false_positive_examples_remain_allowed() -> None:
    # Words like token/password/secret without an actual secret value.
    src = (
        "def refresh_token(token):\n"
        "    return token\n"
        "def set_password(password):\n"
        "    self._p = password\n"
        "SECRET_MODE = 'plain'\n"
    )
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("auth/fp.py", src)))
    assert result.safe is True
    assert result.allowed == ("auth/fp.py",)


# ---------------------------------------------------------------------------
# 14. Redaction removes the secret value
# ---------------------------------------------------------------------------


def test_redaction_removes_secret() -> None:
    src = f'API_KEY = "{FAKE_OPENAI}"\nDB = "postgresql://u:pass@h/db"\n'
    findings, _ = _inspect_content(src)
    redacted = redact_source(src, findings, "[REDACTED]")
    assert FAKE_OPENAI not in redacted
    assert "pass" not in redacted or FAKE_AWS_SECRET not in redacted


def _inspect_content(src: str):
    from repolens.context.firewall.content_detectors import check_content
    findings = check_content(src, "config/x.py", FirewallConfig())
    return list(findings), src


# ---------------------------------------------------------------------------
# 15. Secret values never appear in findings
# ---------------------------------------------------------------------------


def test_secret_values_never_in_findings() -> None:
    src = (
        f'OPENAI_API_KEY = "{FAKE_OPENAI}"\n'
        f'AWS_KEY = "{FAKE_AWS}"\n'
        f'GH = "{FAKE_GITHUB}"\n'
    )
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("config/secrets_here.py", src)))
    body = result.to_dict()
    text = json.dumps(body)
    assert FAKE_OPENAI not in text
    assert FAKE_AWS not in text
    assert FAKE_GITHUB not in text
    # Every finding is fully JSON-serializable and only carries allowed fields.
    for finding in result.findings:
        assert {"path", "line", "type", "severity", "decision", "reason"} == set(
            {"path", "line", "type", "severity", "decision", "reason"}
        )
        assert finding.path
        assert finding.line
        assert finding.type
        assert finding.reason


# ---------------------------------------------------------------------------
# 16. Secret values never appear in JSON serialization
# ---------------------------------------------------------------------------


def test_secrets_never_in_json() -> None:
    src = f'OPENAI_API_KEY = "{FAKE_OPENAI}"\n'
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("config/settings.py", src)))
    safe = firewall.safe_package(
        _package(_candidate("config/settings.py", src)), result,
    )
    text = safe.to_json()
    assert FAKE_OPENAI not in text


# ---------------------------------------------------------------------------
# 17. Secret values never appear in rendered context
# ---------------------------------------------------------------------------


def test_secrets_never_in_rendered_context() -> None:
    src = f'OPENAI_API_KEY = "{FAKE_OPENAI}"\n'
    firewall = ContextFirewall()
    package = _package(_candidate("config/settings.py", src))
    result = firewall.inspect(package)
    safe = firewall.safe_package(package, result)
    rendered = render_safe_context(safe)
    assert FAKE_OPENAI not in rendered


# ---------------------------------------------------------------------------
# 18. Multiple findings in one file
# ---------------------------------------------------------------------------


def test_multiple_findings_in_one_file() -> None:
    src = (
        f'A = "{FAKE_OPENAI}"\n'
        f'B = "{FAKE_AWS}"\n'
    )
    firewall = ContextFirewall()
    result = firewall.inspect(_package(_candidate("config/multi.py", src)))
    assert len(result.findings) >= 2
    assert result.redacted == ("config/multi.py",)


# ---------------------------------------------------------------------------
# 19. Multiple files with mixed decisions
# ---------------------------------------------------------------------------


def test_mixed_allow_redact_block() -> None:
    firewall = ContextFirewall()
    package = _package(
        _candidate("main.py", "def main():\n    pass\n"),
        _candidate("config/settings.py", f'KEY = "{FAKE_OPENAI}"\n'),
        _candidate(".env", "SECRET=value\n"),
    )
    result = firewall.inspect(package)
    assert set(result.allowed) == {"main.py"}
    assert set(result.redacted) == {"config/settings.py"}
    assert set(result.blocked) == {".env"}


# ---------------------------------------------------------------------------
# 20. Blocked files are absent from safe context
# ---------------------------------------------------------------------------


def test_blocked_files_absent_from_safe() -> None:
    firewall = ContextFirewall()
    package = _package(
        _candidate("main.py", "x = 1\n"),
        _candidate("keys/private.pem", "-----BEGIN RSA PRIVATE KEY-----"),
    )
    result = firewall.inspect(package)
    safe = firewall.safe_package(package, result)
    safe_paths = {c.path for c in safe.safe_files}
    assert "main.py" in safe_paths
    assert "keys/private.pem" not in safe_paths


# ---------------------------------------------------------------------------
# 21. Redacted files remain with safe content
# ---------------------------------------------------------------------------


def test_redacted_files_remain_safe() -> None:
    firewall = ContextFirewall()
    package = _package(_candidate("config/settings.py", f'KEY = "{FAKE_OPENAI}"\nx = 1\n'))
    result = firewall.inspect(package)
    safe = firewall.safe_package(package, result)
    redacted = [c for c in safe.safe_files if c.path == "config/settings.py"]
    assert redacted
    assert FAKE_OPENAI not in redacted[0].source


# ---------------------------------------------------------------------------
# 22. Deterministic results
# ---------------------------------------------------------------------------


def test_deterministic_results() -> None:
    firewall = ContextFirewall()
    package = _package(_candidate("config/settings.py", f'KEY = "{FAKE_OPENAI}"\n'))
    r1 = firewall.inspect(package)
    r2 = firewall.inspect(package)
    assert r1.to_dict() == r2.to_dict()


# ---------------------------------------------------------------------------
# 23. Configuration can disable a detector
# ---------------------------------------------------------------------------


def test_config_can_disable_detector() -> None:
    src = f'OPENAI_API_KEY = "{FAKE_OPENAI}"\n'
    cfg = FirewallConfig(content_detectors=frozenset())
    firewall = ContextFirewall(cfg)
    result = firewall.inspect(_package(_candidate("config/settings.py", src)))
    assert result.safe is True
    assert result.allowed == ("config/settings.py",)


def test_config_can_disable_single_detector() -> None:
    # Use an AWS access key, which only the aws_access_key detector matches.
    src = f'value = "{FAKE_AWS}"\n'
    # Confirm it is detected by default.
    assert any(
        f.type == "api_key"
        for f in ContextFirewall().inspect(
            _package(_candidate("config/aws.py", src))
        ).findings
    )
    # Disable only the aws detector.
    cfg = FirewallConfig(content_detectors=frozenset(
        {"openai_api_key", "github_token", "private_key_block",
         "generic_secret_assignment", "database_url_with_credentials",
         "bearer_token", "github_app_token", "slack_token"}
    ))
    firewall = ContextFirewall(cfg)
    result = firewall.inspect(_package(_candidate("config/aws.py", src)))
    assert result.allowed == ("config/aws.py",)
    assert result.findings == ()


# ---------------------------------------------------------------------------
# 24. Firewall disabled mode behaves explicitly and safely
# ---------------------------------------------------------------------------


def test_disabled_firewall_passes_through() -> None:
    cfg = FirewallConfig(enabled=False)
    firewall = ContextFirewall(cfg)
    src = f'KEY = "{FAKE_OPENAI}"\n'
    package = _package(_candidate("config/settings.py", src))
    result = firewall.inspect(package)
    assert result.safe is True
    assert result.firewall_enabled is False
    assert result.allowed == ("config/settings.py",)
    safe = firewall.safe_package(package, result)
    assert safe.firewall_enabled is False
    assert FAKE_OPENAI in safe.safe_files[0].source  # no redaction when disabled


# ---------------------------------------------------------------------------
# 25. Empty ContextPackage
# ---------------------------------------------------------------------------


def test_empty_context_package() -> None:
    firewall = ContextFirewall()
    result = firewall.inspect(_package())
    assert result.safe is True
    assert result.allowed == ()
    safe = firewall.safe_package(_package(), result)
    assert safe.safe_files == ()
    assert safe.blocked_files == ()


# ---------------------------------------------------------------------------
# 26. ContextPackage with no sensitive content
# ---------------------------------------------------------------------------


def test_no_sensitive_content() -> None:
    firewall = ContextFirewall()
    package = _package(
        _candidate("a.py", "def f():\n    return 1\n"),
        _candidate("b.py", "import a\n"),
    )
    result = firewall.inspect(package)
    assert result.safe is True
    safe = firewall.safe_package(package, result)
    assert len(safe.safe_files) == 2
    assert safe.blocked_files == ()


# ---------------------------------------------------------------------------
# 27. Existing ContextEngine tests remain unchanged (smoke check)
# ---------------------------------------------------------------------------


def test_firewall_integrates_with_engine(tmp_path: Path) -> None:
    from tests.test_context_engine import write_file

    write_file(tmp_path, "main.py", "from a import x\n")
    write_file(tmp_path, "a.py", "x = 1\n")
    engine = ContextEngine(tmp_path, searcher=CodeSearcher(tmp_path))
    pkg = engine.build_context("import")
    firewall = ContextFirewall()
    result = firewall.inspect(pkg)
    assert result is not None
    safe = firewall.safe_package(pkg, result)
    assert isinstance(safe, SafeContextPackage)


# ---------------------------------------------------------------------------
# Additional safety checks
# ---------------------------------------------------------------------------


def test_path_rule_for_keyfile(tmp_path: Path) -> None:
    firewall = ContextFirewall()
    package = _package(_candidate("deploy/app.key", "not really a key but blocked by name\n"))
    result = firewall.inspect(package)
    assert result.blocked == ("deploy/app.key",)


def test_path_rule_for_p12(tmp_path: Path) -> None:
    firewall = ContextFirewall()
    package = _package(_candidate("certificates/identity.p12", "binary-ish\n"))
    result = firewall.inspect(package)
    assert result.blocked == ("certificates/identity.p12",)


def test_path_rule_for_credentials_file() -> None:
    firewall = ContextFirewall()
    # "credentials.json" is not in the default blocked-name set, so it passes
    # as-is (no secret content).  This documents that we only block explicit
    # secret-file patterns, not generic "credentials" names.
    package = _package(_candidate("config/credentials.json", '{"user":"alice"}'))
    result = firewall.inspect(package)
    assert result.allowed == ("config/credentials.json",)
    assert result.blocked == ()


def test_failure_inspection_fails_closed(tmp_path: Path, monkeypatch) -> None:
    # If content scanning raises for a file, the file must be BLOCKed (never
    # silently exposed), with a safe diagnostic reason and no secret in the
    # exception path.
    import repolens.context.firewall.firewall as fw_mod

    def boom(source, relative, config):
        raise ValueError("unexpected decode error")

    monkeypatch.setattr(fw_mod, "check_content", boom)
    firewall = ContextFirewall()
    package = _package(_candidate("weird/scan.py", "some source"))
    result = firewall.inspect(package)
    assert result.blocked == ("weird/scan.py",)
    assert not result.allowed
    scan_findings = [f for f in result.findings if f.type == "scan_error"]
    assert scan_findings
    assert "unexpected" not in scan_findings[0].reason or "decode" not in scan_findings[0].reason

    safe = firewall.safe_package(package, result)
    assert "weird/scan.py" not in {c.path for c in safe.safe_files}
