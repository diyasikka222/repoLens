"""Documentation and release-hygiene tests (Phase 25.7).

These tests verify the docs/release story is coherent: canonical entry points
exist, the MCP surface documented matches the implemented tools, every
documented command points at a real file, and no user-specific paths or
accidental secrets live in user-facing documentation.

They are fast, offline, and intentionally non-brittle.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"

REQUIRED_DOCS = [
    "README.md",
    "architecture-mcp.md",
    "architecture-retrieval.md",
    "architecture.md",
    "call-graph.md",
    "change-context.md",
    "change-planning.md",
    "impact-analysis.md",
    "mcp-tools.md",
    "opencode-e2e.md",
    "performance.md",
    "production-validation.md",
    "release-checklist.md",
    "reliability.md",
]

MCP_TOOLS = [
    "get_context",
    "analyze_impact",
    "inspect_symbol",
    "inspect_architecture",
    "discover_subsystems",
    "architecture_candidates",
    "explain_architecture_match",
    "change_plan",
    "change_context",
]

# Accidental-secret probes. Env var *names* are fine; we look for secret-shaped
# values only.
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bghp_[0-9A-Za-z]{30,}"),
]

_HOME = str(Path.home())


class DocumentationTests(unittest.TestCase):
    def test_required_docs_exist(self) -> None:
        for name in REQUIRED_DOCS:
            self.assertTrue(
                (DOCS_DIR / name).is_file(),
                f"docs/{name} is missing; docs index links to it",
            )

    def test_readme_exists_with_expected_sections(self) -> None:
        text = README.read_text(encoding="utf-8")
        for section in [
            "# RepoLens",
            "## Installation",
            "## Quick start",
            "## Embedding configuration",
            "## OpenCode integration",
            "## MCP tools",
            "## Running tests",
            "## Running benchmarks",
            "## Important limitations",
            "## Deeper documentation",
        ]:
            self.assertIn(section, text, f"README missing section {section!r}")

    def test_mcp_tools_documented(self) -> None:
        docs_text = (DOCS_DIR / "mcp-tools.md").read_text(encoding="utf-8")
        readme_text = README.read_text(encoding="utf-8")
        for tool in MCP_TOOLS:
            self.assertIn(
                tool, docs_text, f"docs/mcp-tools.md does not document {tool!r}"
            )
            self.assertIn(tool, readme_text, f"README does not mention {tool!r}")

    def test_documented_benchmark_commands_refer_to_existing_files(self) -> None:
        text = README.read_text(encoding="utf-8")
        found = set(re.findall(r"benchmarks/([\w.]+)\.py", text))
        found.discard("real_repo")
        self.assertTrue(found, "README documents no benchmark commands")
        for name in sorted(found):
            self.assertTrue(
                (REPO_ROOT / "benchmarks" / f"{name}.py").is_file(),
                f"README documents benchmarks/{name}.py but it does not exist",
            )
        self.assertTrue(
            (REPO_ROOT / "benchmarks" / "real_repo").is_dir(),
            "README documents the benchmarks/real_repo module but it is missing",
        )

    def test_docs_have_no_user_specific_paths(self) -> None:
        repo_root = str(REPO_ROOT)
        offenders = []
        targets = [README, *DOCS_DIR.glob("*.md")]
        for path in targets:
            text = path.read_text(encoding="utf-8")
            if repo_root in text:
                offenders.append(f"{path.name}: contains the repo root path")
            if _HOME in text:
                offenders.append(f"{path.name}: contains a user home path")
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_docs_have_no_accidental_secrets(self) -> None:
        offenders = []
        targets = [README, *DOCS_DIR.glob("*.md")]
        for path in targets:
            text = path.read_text(encoding="utf-8")
            for pattern in _SECRET_PATTERNS:
                if pattern.search(text):
                    offenders.append(f"{path.name}: matches {pattern.pattern}")
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_pyproject_metadata_points_at_readme_and_urls(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)
        project = pyproject["project"]
        self.assertEqual(project.get("readme"), "README.md")
        self.assertIn("urls", project, "'[project.urls]' missing from pyproject.toml")
        self.assertTrue(
            project["urls"]["Homepage"].startswith("https://"),
            "Homepage URL should be absolute",
        )

    def test_release_checklist_links_verification(self) -> None:
        text = (DOCS_DIR / "release-checklist.md").read_text(encoding="utf-8")
        self.assertIn("python -m pytest -q", text)
        self.assertIn("opencode.json", text, "checklist must cover untracked config")


if __name__ == "__main__":
    unittest.main()