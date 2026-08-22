"""Filesystem discovery of Python source files in a repository."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

IGNORED_DIRECTORIES = frozenset(
    {".git", ".venv", "venv", "__pycache__", "node_modules"}
)


class RepositoryScanner:
    """Recursively discovers Python files under a repository root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"repository root does not exist: {self.root}"
            )
        if not self.root.is_dir():
            raise NotADirectoryError(
                f"repository root is not a directory: {self.root}"
            )

    def discover_python_files(self) -> list[Path]:
        """Return paths to all Python files under the root, relative to it."""
        return sorted(self._walk(self.root))

    def _walk(self, directory: Path) -> Iterator[Path]:
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                if entry.name not in IGNORED_DIRECTORIES:
                    yield from self._walk(entry)
            elif entry.is_file() and entry.suffix == ".py":
                yield entry.relative_to(self.root)
