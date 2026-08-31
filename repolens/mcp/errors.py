"""Safe error types for the RepoLens MCP layer (Milestone 14).

MCP is an untrusted caller boundary.  Errors returned to the agent must be
concise and safe — never exposing stack traces, environment variables, API
keys, secret values, or arbitrary filesystem contents.

Each error type carries a user-facing (safe) message and an optional private
diagnostic that is only sent to local logs (stderr), never to the agent.
"""

from __future__ import annotations


class McpError(Exception):
    """Base class for RepoLens MCP errors.

    Attributes:
        safe_message: A concise, safe message returned to the agent.
        diagnostic: An optional private detail for local logs (never sent
            to the agent).
    """

    def __init__(self, safe_message: str, diagnostic: str | None = None) -> None:
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.diagnostic = diagnostic


class InvalidArgumentsError(McpError):
    """Input validation failed (bad query, bad max_tokens, etc.)."""


class ConfigurationError(McpError):
    """The MCP server could not be configured (bad repo root, bad deps)."""


class RepositoryError(McpError):
    """The configured repository is missing, unreadable, or unusable."""


class ContextEngineError(McpError):
    """Building a context package failed."""


class FirewallError(McpError):
    """The context firewall failed during inspection."""


class InternalError(McpError):
    """An unexpected internal failure (generic safe message)."""
