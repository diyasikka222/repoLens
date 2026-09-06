"""Minimal CLI launcher for the RepoLens MCP server (Milestone 14).

This is *not* a general RepoLens CLI.  It exists only so an MCP-compatible
client can specify the repository root and launch the stdio server::

    python -m repolens.mcp --repo /path/to/repository

The server reads stdin / writes stdout for the MCP protocol; all diagnostics
go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from typing import Callable

from repolens.context import ContextEngine
from repolens.mcp.deps import (
    build_engine,
    build_firewall,
    resolve_embedding_provider,
    validate_repository_root,
)
from repolens.mcp.errors import ConfigurationError, McpError
from repolens.mcp.server import build_mcp_server


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="repolens-mcp",
        description="Launch the RepoLens MCP server over stdio for local use.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the repository that RepoLens will index (required).",
    )
    parser.add_argument(
        "--default-max-tokens",
        type=int,
        default=8000,
        help="Default context budget in estimated tokens (default: 8000).",
    )
    parser.add_argument(
        "--default-dependency-depth",
        type=int,
        default=1,
        help="Default dependency graph depth (default: 1).",
    )
    parser.add_argument(
        "--use-local-embeddings",
        action="store_true",
        help="Use the on-device local embedding provider (requires a one-time "
        "model download on first use).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level for diagnostics on stderr (default: WARNING).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="repolens-mcp 0.1.0",
    )
    return parser.parse_args(argv)


def _configure_logging(log_level: str) -> None:
    """Route diagnostics to stderr only.  MCP protocol uses stdout."""
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )


def make_engine_factory(
    root,
    *,
    embedding_provider=None,
    default_max_tokens: int = 8000,
    default_dependency_depth: int = 1,
) -> Callable[..., ContextEngine]:
    """Return a factory that builds a :class:`ContextEngine`.

    The repository root is validated eagerly so that bad paths fail fast at
    server startup.  The default :class:`ContextEngine` is built **lazily** on
    the first invocation of the returned factory, allowing the MCP stdio
    transport to start without waiting for expensive repository indexing.

    Once built, the default engine is cached and reused for subsequent
    requests whose ``max_tokens`` / ``dependency_depth`` match the defaults.
    Requests that override these parameters build a purpose-configured engine
    for that call (existing behaviour preserved).
    """
    root = _resolve_root_for_factory(root, embedding_provider)

    lock = threading.Lock()
    _cached_default_engine: list[ContextEngine | None] = [None]

    def factory(max_tokens=None, dependency_depth=None) -> ContextEngine:
        use_default = (
            max_tokens is None or max_tokens == default_max_tokens
        ) and (
            dependency_depth is None
            or dependency_depth == default_dependency_depth
        )

        if use_default:
            if _cached_default_engine[0] is None:
                with lock:
                    if _cached_default_engine[0] is None:
                        _cached_default_engine[0] = build_engine(
                            root,
                            embedding_provider=embedding_provider,
                            max_tokens=default_max_tokens,
                            dependency_depth=default_dependency_depth,
                        )
            return _cached_default_engine[0]

        return build_engine(
            root,
            embedding_provider=embedding_provider,
            max_tokens=max_tokens,
            dependency_depth=dependency_depth,
        )

    return factory


def _resolve_root_for_factory(root, embedding_provider):
    """Validate the root once; delegate errors to the caller."""
    from repolens.mcp.deps import validate_repository_root

    return validate_repository_root(root)


def make_impact_analyzer_factory(root) -> Callable[..., "ImpactAnalyzer"]:
    """Return a factory that lazily builds and caches an :class:`ImpactAnalyzer`.

    The root is validated eagerly (fail fast at startup); the analyzer — and
    therefore the repository index — is built on the first invocation, keeping
    MCP startup fast and initialization lazy.
    """
    from repolens.impact import ImpactAnalyzer
    from repolens.mcp.deps import build_impact_analyzer

    root = validate_repository_root(root)

    lock = threading.Lock()
    cache: list[ImpactAnalyzer | None] = [None]

    def factory(**kwargs) -> ImpactAnalyzer:
        if cache[0] is None:
            with lock:
                if cache[0] is None:
                    cache[0] = build_impact_analyzer(root)
        return cache[0]

    return factory


def make_inspect_factory(root) -> Callable[..., "CallGraph"]:
    """Return a factory that lazily builds and caches the M22 call graph.

    The root is validated eagerly (fail fast at startup); the graph — and
    therefore the repository index — is built on the first invocation, keeping
    MCP startup fast and initialization lazy. The build shares the persistent
    index with :func:`make_impact_analyzer_factory`, so a graph built after an
    analyzer performs no re-parsing.
    """
    from repolens.call_graph import CallGraph
    from repolens.mcp.deps import build_reference_graph

    root = validate_repository_root(root)

    lock = threading.Lock()
    cache: list[CallGraph | None] = [None]

    def factory(**kwargs) -> CallGraph:
        if cache[0] is None:
            with lock:
                if cache[0] is None:
                    cache[0] = build_reference_graph(root)
        return cache[0]

    return factory


def make_architecture_factory(root) -> Callable[..., "ArchitectureState"]:
    """Return a factory that lazily builds and caches the shared M23.3 state.

    The root is validated eagerly (fail fast at startup); the incremental
    index, the architecture graph (a pure projection of that index), and the
    subsystem discovery result are all built on the first architecture-tool
    call and reused by every subsequent call. Because the index build shares
    the persistent cache with the other factories, no re-parsing occurs when a
    warm index already exists for the repository.
    """
    from repolens.mcp.architecture_tool import ArchitectureState

    root = validate_repository_root(root)

    lock = threading.Lock()
    cache: list[ArchitectureState | None] = [None]

    def factory(**kwargs) -> ArchitectureState:
        if cache[0] is None:
            with lock:
                if cache[0] is None:
                    cache[0] = ArchitectureState(root)
        return cache[0]

    return factory


def run(argv: list[str] | None = None) -> None:
    """Run the MCP server to completion (blocks on the stdio loop)."""
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        embedding_provider = (
            resolve_embedding_provider(args.use_local_embeddings)
        )
    except Exception as exc:  # noqa: BLE001
        _fatal(
            ConfigurationError(
                "Local embedding provider is unavailable.",
                diagnostic=f"resolve_embedding_provider: {type(exc).__name__}",
            )
        )
        return

    try:
        engine_factory = make_engine_factory(
            args.repo,
            embedding_provider=embedding_provider,
            default_max_tokens=args.default_max_tokens,
            default_dependency_depth=args.default_dependency_depth,
        )
        impact_factory = make_impact_analyzer_factory(args.repo)
        inspect_factory = make_inspect_factory(args.repo)
        architecture_factory = make_architecture_factory(args.repo)
    except McpError as exc:
        _fatal(exc)
        return

    firewall = build_firewall()
    server = build_mcp_server(
        engine_factory,
        firewall,
        impact_factory=impact_factory,
        inspect_factory=inspect_factory,
        architecture_factory=architecture_factory,
    )

    try:
        asyncio.run(server.run_stdio_async())
    except KeyboardInterrupt:
        pass


def _fatal(exc: McpError) -> None:
    """Print a safe fatal error to stderr and report the safe message."""
    logging.getLogger("repolens.mcp.launcher").error(
        "%s", exc.safe_message
    )
    if exc.diagnostic:
        logging.getLogger("repolens.mcp.launcher").error("%s", exc.diagnostic)
    sys.exit(2)


def main(argv: list[str] | None = None) -> None:
    """Console entry point."""
    run(argv)


if __name__ == "__main__":
    main()
