"""Dependency wiring for the RepoLens MCP server (Milestone 14).

The MCP layer is a *thin adapter*: it constructs the existing public RepoLens
objects (:class:`~repolens.context.ContextEngine` and
:class:`~repolens.context.ContextFirewall`) and calls their public methods.  No
retrieval, ranking, budgeting, or security logic lives here.

Providers
---------
By default the server uses a deterministic, fully-offline fake embedding
provider so it works without any API key or model download.  An explicit
:class:`~repolens.embeddings.EmbeddingProvider` (for example
:class:`~repolens.local_embeddings.LocalEmbeddingProvider`) may be supplied to
enable local, on-device semantic retrieval without OpenAI.
"""

from __future__ import annotations

import os
from pathlib import Path

from repolens.context import (
    ContextBudget,
    ContextEngine,
    ContextFirewall,
    DependencyExpansionConfig,
    RetrievalConfig,
)
from repolens.context.firewall import FirewallConfig
from repolens.mcp.errors import ConfigurationError, RepositoryError

#: Paths that are never accepted as a repository root, regardless of config.
_DISALLOWED_ROOT_NAMES = {".", "..", "/", "~"}


def validate_repository_root(root: str | Path) -> Path:
    """Validate and resolve an explicitly configured repository root.

    The repository root is configured at server launch time and is *not*
    supplied by tool callers.  This validates that it exists, is a directory,
    and is readable.  Rejects attempts to pass trivial escape values.
    """
    if not root:
        raise RepositoryError(
            "No repository configured.",
            diagnostic="MCP server requires an explicit --repo path.",
        )
    raw = str(root).strip()
    if raw in _DISALLOWED_ROOT_NAMES:
        raise RepositoryError(
            "The configured repository root is not a valid path.",
            diagnostic=f"Refusing ambiguous root path: {raw!r}",
        )
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise RepositoryError(
            "The configured repository does not exist.",
            diagnostic=f"Repository root not found: {path}",
        )
    if not path.is_dir():
        raise RepositoryError(
            "The configured repository path is not a directory.",
            diagnostic=f"Repository root is not a directory: {path}",
        )
    if not os.access(path, os.R_OK):
        raise RepositoryError(
            "The configured repository is not readable.",
            diagnostic=f"Repository root not readable: {path}",
        )
    return path


def build_engine(
    root: str | Path,
    *,
    embedding_provider=None,
    max_tokens: int | None = None,
    dependency_depth: int | None = None,
) -> ContextEngine:
    """Build a :class:`ContextEngine` for ``root`` using existing configuration.

    ``embedding_provider`` overrides the provider used by retrieval; when
    ``None``, a deterministic offline fake provider is used so the server
    never requires an API key or model download.
    """
    root_path = validate_repository_root(root)

    budget = ContextBudget(max_tokens=max_tokens) if max_tokens is not None else None
    dep_cfg = (
        DependencyExpansionConfig(depth=dependency_depth)
        if dependency_depth is not None
        else None
    )

    retrieval = RetrievalConfig()
    try:
        searcher = retrieval.build_searcher(
            root_path, embedding_provider=embedding_provider,
        )
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(
            "Could not build retrieval for the configured repository.",
            diagnostic=(
                "build_searcher failed for "
                f"{root_path}: {type(exc).__name__}"
            ),
        ) from exc

    try:
        engine = ContextEngine(
            root_path,
            searcher=searcher,
            budget=budget,
            dependency=dep_cfg,
        )
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(
            "Could not initialize the context engine.",
            diagnostic=(
                f"ContextEngine init failed for {root_path}: "
                f"{type(exc).__name__}"
            ),
        ) from exc

    return engine


def build_firewall(firewall_config: FirewallConfig | None = None) -> ContextFirewall:
    """Build a :class:`ContextFirewall` with the given (or default) policy."""
    return ContextFirewall(firewall_config)


def resolve_embedding_provider(prefer_local: bool):
    """Return an embedding provider, or ``None`` for the default fake.

    When ``prefer_local`` is ``False`` (the default) we return ``None`` so the
    engine uses its deterministic offline fake provider.  When ``True`` we
    construct the local on-device provider; if unavailable we delegate the
    error to the engine rather than masking it.
    """
    if not prefer_local:
        return None
    from repolens.local_embeddings import LocalEmbeddingProvider

    _model = os.environ.get("REPOLENS_LOCAL_EMBEDDING_MODEL")
    if _model:
        return LocalEmbeddingProvider(model=_model)
    return LocalEmbeddingProvider()
