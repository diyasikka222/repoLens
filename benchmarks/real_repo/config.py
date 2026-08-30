"""Static configuration for the real-world repository benchmark.

Everything in this module is derived offline from a pinned, publicly known
release. It contains no network access and no credentials.

The external benchmark target is the ``Textualize/rich`` library, pinned to
the ``v14.3.4`` git tag (commit ``ee8378c3bbbd7c75abc2f55c6c19e83b218ae81d``).
It is a mature, well-known Python library with a meaningful multi-module
package structure (console, table, progress, markup, syntax, theme, ...),
classes, functions, imports and dependencies — a realistic production
codebase. See :mod:`benchmarks.real_repo` for how it is used.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# External repository
# ---------------------------------------------------------------------------

REPOSITORY_NAME = "Textualize/rich"
REPOSITORY_URL = "https://github.com/Textualize/rich"
REPOSITORY_REF = "v14.3.4"
REPOSITORY_COMMIT = "ee8378c3bbbd7c75abc2f55c6c19e83b218ae81d"

# GitHub tarball for the pinned ref (deterministic, no git history needed).
TARBALL_URL = (
    f"https://codeload.github.com/Textualize/rich/tar.gz/refs/tags/{REPOSITORY_REF}"
)

# Sanity thresholds used to fail fast if the downloaded archive does not look
# like the expected repository.
MIN_PYTHON_FILES = 100

# ---------------------------------------------------------------------------
# Benchmark data directory (inside the RepoLens repo, gitignored)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR = PROJECT_ROOT / ".benchmark_data"
REPO_DIR = DATA_DIR / "repo"
ARCHIVE_PATH = DATA_DIR / f"{REPOSITORY_NAME.split('/')[1]}-{REPOSITORY_REF}.tar.gz"

QUERIES_PATH = HERE / "queries.json"
