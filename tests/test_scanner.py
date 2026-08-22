"""Tests for the repository filesystem scanner."""

from pathlib import Path

import pytest

from repolens.scanner import RepositoryScanner


def test_discovers_python_file_at_root(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hello')\n")

    scanner = RepositoryScanner(tmp_path)

    assert scanner.discover_python_files() == [Path("main.py")]


def test_discovers_python_files_in_nested_directories(tmp_path: Path) -> None:
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "top.py").write_text("a = 1\n")
    (tmp_path / "pkg" / "mid.py").write_text("b = 2\n")
    (tmp_path / "pkg" / "sub" / "leaf.py").write_text("c = 3\n")

    scanner = RepositoryScanner(tmp_path)

    assert scanner.discover_python_files() == [
        Path("pkg") / "mid.py",
        Path("pkg") / "sub" / "leaf.py",
        Path("top.py"),
    ]


def test_ignores_non_python_files(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("docs\n")
    (tmp_path / "setup.cfg").write_text("[metadata]\n")
    (tmp_path / "notes.txt").write_text("notes\n")
    (tmp_path / "backup.py.bak").write_text("old\n")

    scanner = RepositoryScanner(tmp_path)

    assert scanner.discover_python_files() == [Path("app.py")]


@pytest.mark.parametrize(
    "ignored", [".git", ".venv", "venv", "__pycache__", "node_modules"]
)
def test_does_not_scan_ignored_directories(tmp_path: Path, ignored: str) -> None:
    (tmp_path / ignored).mkdir()
    (tmp_path / ignored / "hidden.py").write_text("x = 1\n")
    (tmp_path / ignored / "nested").mkdir()
    (tmp_path / ignored / "nested" / "deep.py").write_text("y = 2\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("z = 3\n")

    scanner = RepositoryScanner(tmp_path)

    assert scanner.discover_python_files() == [Path("src") / "real.py"]


def test_missing_root_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RepositoryScanner(tmp_path / "does-not-exist")


def test_file_root_raises_not_a_directory_error(tmp_path: Path) -> None:
    file_path = tmp_path / "plain.txt"
    file_path.write_text("not a directory\n")

    with pytest.raises(NotADirectoryError):
        RepositoryScanner(file_path)
