"""Release metadata regression tests (P25.8).

These tests pin the release-hygiene guarantees introduced in P25.8:

- one canonical project version, exposed everywhere through a single source
  (``repolens.__version__``), with the packaging metadata deriving from it;
- generated setuptools metadata (``*.egg-info``) stays out of version control.

They are fast and offline; the git-backed assertions skip gracefully when the
checkout is not a git repository or git is unavailable.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

import repolens
from repolens.mcp import SERVER_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(args: list[str], *, check: bool = False) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if check:
        result.check_returncode()
    return result.returncode, result.stdout


class ReleaseMetadataTests(unittest.TestCase):
    def test_canonical_package_version(self) -> None:
        self.assertEqual(
            repolens.__version__,
            "0.0.1",
            "repolens.__version__ is the single canonical release version",
        )

    def test_mcp_server_reports_package_version(self) -> None:
        self.assertEqual(
            SERVER_VERSION,
            repolens.__version__,
            "MCP server version must derive from repolens.__version__",
        )

    def test_mcp_launcher_version_flag_reports_package_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "repolens.mcp", "--version"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            f"repolens-mcp {repolens.__version__}",
        )

    def test_pyproject_derives_version_from_package(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)
        project = pyproject["project"]
        self.assertIn(
            "version",
            project["dynamic"],
            "pyproject.toml must declare version as dynamic (single source)",
        )
        dynamic = pyproject["tool"]["setuptools"]["dynamic"]
        self.assertEqual(
            dynamic.get("version"),
            {"attr": "repolens.__version__"},
            "pyproject version must derive from repolens.__version__",
        )

    def test_egg_info_not_tracked_and_ignored(self) -> None:
        if subprocess.run(["git", "rev-parse"], capture_output=True).returncode:
            self.skipTest("not a git checkout")
        code, tracked = _git(["ls-files"])
        self.assertEqual(code, 0)
        stale = [line for line in tracked.splitlines() if line.startswith("repolens.egg-info")]
        self.assertEqual(
            stale, [],
            "repolens.egg-info must not be tracked; found: %s" % sorted(stale),
        )
        code, _ = _git(["check-ignore", "repolens.egg-info"])
        self.assertEqual(
            code, 0,
            "repolens.egg-info must be ignored via .gitignore",
        )

    def test_gitignore_covers_setuptools_build_artifacts(self) -> None:
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ["*.egg-info/", "build/", "dist/"]:
            self.assertIn(pattern, text, f".gitignore missing {pattern!r}")


if __name__ == "__main__":
    unittest.main()