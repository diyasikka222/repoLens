"""Lightweight structured diagnostics for RepoLens operations (Milestone 20).

Diagnostics describe *what* a major operation did — counts, sizes, and
durations — as one parseable JSON line per record. They are opt-in and never
alter the operation's returned results: a disabled diagnostics path is a
single boolean check, and records are emitted at DEBUG level on the dedicated
``repolens.diagnostics`` logger, which has no handler by default.

Never logged: source-code contents, API keys, secrets, or sensitive
repository contents. The documented operations emit paths, count fields, and
durations only.

Enabling
--------
- Set ``REPOLENS_DIAGNOSTICS=1`` in the environment, or
- call :func:`enable` / :func:`disable` at runtime, or
- attach your own handler to ``repolens.diagnostics`` and set its level to
  DEBUG, then call :func:`enable`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("repolens.diagnostics")

_ENV_FLAG = "REPOLENS_DIAGNOSTICS"
_FALSE_VALUES = {"", "0", "false", "False", "no", "No"}

#: Optional runtime override; ``None`` means "consult the environment".
_override: bool | None = None


def enable() -> None:
    """Turn diagnostics on for the current process regardless of the env."""
    global _override
    _override = True


def disable() -> None:
    """Turn diagnostics off for the current process regardless of the env."""
    global _override
    _override = False


def reset() -> None:
    """Forget any runtime override and consult the environment again."""
    global _override
    _override = None


def enabled() -> bool:
    """Whether diagnostics records are currently being emitted."""
    if _override is not None:
        return _override
    return os.environ.get(_ENV_FLAG, "") not in _FALSE_VALUES


def record(operation: str, **fields: Any) -> None:
    """Emit one diagnostics record if enabled.

    ``operation`` names the measured stage (for example ``"index_build"`` or
    ``"context_build"``). Remaining keyword fields are JSON-safe scalars or
    simple structures; values that cannot be JSON-serialized (for example a
    :class:`pathlib.Path`) fall back to ``str``.
    """
    if not enabled():
        return
    payload: dict[str, Any] = {"operation": operation}
    for key, value in fields.items():
        payload[key] = _json_safe(value)
    logger.debug(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")))


def _json_safe(value: Any) -> Any:
    """Return ``value`` unchanged when it is trivially JSON-safe, else its str."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


@contextmanager
def timed(operation: str, **fields: Any):
    """Measure a block and record its wall time as ``elapsed_ms``.

    Fields are recorded on exit along with ``elapsed_ms`` (3 decimal places).
    Records are only emitted when diagnostics are enabled; the timing itself
    is negligible overhead either way.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        record(operation, elapsed_ms=round(elapsed_ms, 3), **fields)