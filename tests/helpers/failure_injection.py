"""Test-only fault-injection and observability helpers (Phase 25.5).

Everything in this module is deliberately small, offline, and deterministic.
The helpers fall into three families:

- *providers* — wrappers around ``EmbeddingProvider`` implementations that
  inject failures (raise on the N-th call, raise for documents matching a
  predicate, return the wrong number of vectors) or count calls;
- *filesystem faults* — corrupt cache payloads in-place and monkeypatch
  ``repolens.atomic_write.atomic_write_text`` / ``os.fsync`` so write paths
  fail at a chosen point;
- *fingerprints* — lossless-enough, sorted snapshots of repository indexes,
  change plans, and context packages used by the *recovery-equivalence*
  assertions (two fingerprints are equal iff the visible behaviour the tests
  care about is identical, independent of object identity).

None of these helpers are imported by production code; they only exist under
``tests/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from repolens.embeddings import EmbeddingProvider, FakeEmbeddingProvider, Vector


class EmbeddingFailure(RuntimeError):
    """Marker exception raised by injected embedding failures.

    Using a dedicated exception (instead of an arbitrary ``Exception``) makes
    the tests assert exactly that *the injected fault* propagated, rather
    than any unrelated error.
    """


# ---------------------------------------------------------------------------
# Provider fault injection
# ---------------------------------------------------------------------------


class FlakyProvider:
    """Delegating provider that raises :class:`EmbeddingFailure` on demand.

    Failures are declared either by a call budget (``fail_calls`` raises on
    the first ``fail_calls`` provider invocations, then succeeds — a
    *transient* fault) or by a ``predicate`` over the text(s) being embedded
    (a *deterministic per-document* fault). A provider call counts once per
    ``embed_text`` or ``embed_texts`` invocation.
    """

    def __init__(
        self,
        base: EmbeddingProvider | None = None,
        *,
        fail_calls: int = 0,
        predicate: Callable[[str], bool] | None = None,
    ) -> None:
        self._base = base if base is not None else FakeEmbeddingProvider()
        self._remaining = fail_calls
        self._predicate = predicate
        self.call_count = 0

    def embed_text(self, text: str) -> Vector:
        self._invoke_before(len(text))
        return self._base.embed_text(text)

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        self._invoke_before(sum(len(t) for t in texts))
        return self._base.embed_texts(list(texts))

    def _invoke_before(self, _chars: int) -> None:
        self.call_count += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise EmbeddingFailure("injected transient embedding failure")
        if self._predicate is not None:
            raise EmbeddingFailure("injected per-document embedding failure")


class ShortResultProvider:
    """Returns fewer vectors than requested (silent-truncation bug probe).

    ``emit`` controls how many vectors ``embed_texts`` returns regardless of
    how many documents were requested. By default one short call is followed
    by correct behaviour, mirroring a genuinely broken one-off provider.
    """

    def __init__(self, base: EmbeddingProvider | None = None, *, emit: int = 1) -> None:
        self._base = base if base is not None else FakeEmbeddingProvider()
        self._emit = emit

    def embed_text(self, text: str) -> Vector:
        return self._base.embed_text(text)

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        vectors = self._base.embed_texts(list(texts))
        return tuple(vectors[: self._emit])


class WrongDimensionProvider:
    """Returns vectors whose dimension differs from the query vector.

    Used to check that a fundamentally misconfigured provider produces either
    a typed error or a coherent (never-crashing) result — never a cryptic
    index error downstream.
    """

    def __init__(self, base: EmbeddingProvider | None = None, *, extra_dim: int = 1) -> None:
        self._base = base if base is not None else FakeEmbeddingProvider()
        self._extra = extra_dim

    def embed_text(self, text: str) -> Vector:
        return self._base.embed_text(text) + (0.0,) * self._extra

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return tuple(self.embed_text(t) for t in texts)


class CountingProvider:
    """Counts provider invocations while delegating to a base provider."""

    def __init__(self, base: EmbeddingProvider | None = None) -> None:
        self._base = base if base is not None else FakeEmbeddingProvider()
        self.calls: list[tuple[str, int]] = []

    def embed_text(self, text: str) -> Vector:
        self.calls.append(("single", 1))
        return self._base.embed_text(text)

    def embed_texts(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        self.calls.append(("batch", len(list(texts))))
        return self._base.embed_texts(list(texts))


# ---------------------------------------------------------------------------
# Filesystem fault injection
# ---------------------------------------------------------------------------


def write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON payload to ``path`` (like the caches do)."""
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory / f".tmp-{path.name}"
    temp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temp, path)


def corrupt_all_json(directory: Path, replacement: str = "{corrupted") -> int:
    """Overwrite every ``*.json`` under ``directory`` with garbage.

    Returns the number of files corrupted so the test can assert that at
    least one entry was really hit.
    """
    corrupted = 0
    for entry in sorted(directory.glob("*.json")):
        entry.write_text(replacement, encoding="utf-8")
        corrupted += 1
    return corrupted


def mutate_json_entries(
    directory: Path,
    mutator: Callable[[dict[str, Any]], dict[str, Any]],
) -> int:
    """Rewrite each ``*.json`` under ``directory`` after ``mutator(payload)``."""
    mutated = 0
    for entry in sorted(directory.glob("*.json")):
        payload = json.loads(entry.read_text(encoding="utf-8"))
        entry.write_text(json.dumps(mutator(payload)), encoding="utf-8")
        mutated += 1
    return mutated


def drop_stale_partial(directory: Path, filename: str) -> Path:
    """Drop a stale ``*.part-<pid>.tmp`` sibling to simulate an interrupted write."""
    partial = directory / filename
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("partial garbage", encoding="utf-8")
    return partial


def install_atomic_write_failure(monkeypatch, fail_after: int = 0) -> Callable[[], int]:
    """Monkeypatch ``os.replace`` inside ``repolens.atomic_write`` to fail.

    Patching the *replace* step (rather than the whole
    ``atomic_write_text`` function) means the real atomic-writer runs all of
    its temp-file + flush + fsync work and fails at the final commit step, so
    the writer's own cleanup path is what the tests exercise.

    ``fail_after``=0 fails every commit; ``fail_after``=N lets the first N
    replacements succeed and fails the next. Returns a probe reporting the
    number of commit attempts. pytest's ``monkeypatch`` restores the original
    automatically.
    """
    import repolens.atomic_write as atomic_module

    original_replace = atomic_module.os.replace
    attempts = {"count": 0, "invoked": 0, "last_failed": False}

    def failing_replace(src: Any, dst: Any) -> None:
        attempts["invoked"] += 1
        attempts["count"] += 1
        if attempts["count"] > fail_after:
            attempts["last_failed"] = True
            raise OSError("injected replace failure")
        original_replace(src, dst)

    monkeypatch.setattr(atomic_module.os, "replace", failing_replace)

    def probe() -> dict[str, Any]:
        return dict(attempts)

    return probe


def install_fsync_failure(monkeypatch) -> None:
    """Monkeypatch ``os.fsync`` so a write is interrupted mid-commit.

    With the failure active the atomic writer reaches ``fsync`` and then the
    real path must clean up its temporary file without touching the target.
    """
    import repolens.atomic_write as atomic_module

    def failing_fsync(fd: Any) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(atomic_module.os, "fsync", failing_fsync)


# ---------------------------------------------------------------------------
# Fingerprints (recovery-equivalence snapshots)
# ---------------------------------------------------------------------------


def _symbol_snapshot(symbols) -> tuple[tuple[str, str, ...], ...]:
    return tuple(
        sorted(
            (s.name, s.kind.value, s.file_path.as_posix(), getattr(s, "parent_class", None) or "")
            for s in symbols
        )
    )


def index_fingerprint(index) -> tuple[tuple[str, str, tuple[tuple[str, str, ...], ...], ...]]:
    """A sorted, serializable snapshot of an incremental repository index.

    Covers every observable aspect of the index the recovery-equivalence
    tests care about: the file set, each file's content hash, and the parsed
    symbol surface. Two indexes produced from the same sources fingerprint
    identically even when they were built by different builder instances.
    """
    symbols_by_path: dict[str, list] = {}
    for symbol in index.symbols:
        symbols_by_path.setdefault(symbol.file_path.as_posix(), []).append(symbol)
    files = tuple(
        (
            path.as_posix(),
            index.by_path[path].content_hash,
            _symbol_snapshot(symbols_by_path.get(path.as_posix(), ())),
        )
        for path in sorted(index.files)
    )
    return files


def _item_path(item: Any) -> str:
    """Repository-relative posix path of a candidate/impact item, if present."""
    path = getattr(item, "path", None) or getattr(item, "file_path", None)
    if path is None:
        return ""
    return path if isinstance(path, str) else Path(path).as_posix()


def package_fingerprint(package) -> tuple[Any, ...]:
    """A serializable snapshot of a context package's selected-file surface.

    Captures the per-file decision surface the recovery-equivalence tests
    compare: repository-relative path, selection role, and inclusion reason,
    all sorted. That is exactly what a ``ContextEngine`` observer can rely on
    across processes.
    """
    from repolens.context.candidate import CandidateRole

    files = sorted(
        (
            _item_path(item),
            item.role.value if isinstance(getattr(item, "role", None), CandidateRole) else "",
            getattr(item, "inclusion_reason", "") or "",
        )
        for item in package.selected_files
    )
    return tuple(files)


def plan_fingerprint(plan) -> tuple[Any, ...]:
    """A serializable snapshot of a change plan's decision surface.

    Build-time and parse counts are excluded (they are timing/lifecycle
    details); the plan's decisions — primary target, affected files and
    symbols, tests, and risk — are fully captured.
    """
    affected = tuple(
        sorted(
            _item_path(item)
            for item in getattr(plan, "affected_files", ()) or ()
        )
    )
    return (
        getattr(getattr(plan, "primary_target", None), "id", None),
        affected,
        tuple(sorted(getattr(plan, "affected_symbols", ()) or ())),
        tuple(
            sorted(
                _item_path(item)
                for item in getattr(plan, "tests", ()) or ()
            )
        ),
        getattr(plan, "risk", None),
    )


def require_write_access(path: Path) -> bool:
    """True when the process may actually write to ``path`` (non-root)."""
    try:
        return os.access(path, os.W_OK)
    except OSError:
        return False


def make_repo(root: Path, files: dict[str, str]) -> Path:
    """Create a repository tree from ``{relative_path: source}``."""
    root.mkdir(parents=True, exist_ok=True)
    for relative, source in sorted(files.items()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return root