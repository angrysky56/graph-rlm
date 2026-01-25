"""
Runtime module - provides global call_mcp_tool function for generated wrappers.

This module maintains a context-isolated CoordinatorClient instance that supports
concurrent async sessions without race conditions.
"""

import contextvars
from pathlib import Path
from typing import Any

from .client import CoordinatorClient

# Context-local client - isolated per async task/session
# This replaces the old global singleton pattern for concurrency safety
_client_var: contextvars.ContextVar[CoordinatorClient | None] = contextvars.ContextVar(
    "mcp_client", default=None
)
_config_path_var: contextvars.ContextVar[str | Path | None] = contextvars.ContextVar(
    "mcp_config_path", default=None
)

# Backwards compatibility: keep module-level default config
_default_config_path: str | Path | None = None


def initialize_runtime(config_path: str | Path | None = None) -> None:
    """
    Initialize the MCP runtime client for the current context.

    This should be called once at startup, typically by Coordinator.generate_tools()
    or manually if using the generated tools directly.

    Args:
        config_path: Path to MCP server configuration
    """
    global _default_config_path
    _default_config_path = config_path
    _config_path_var.set(config_path)

    # Create client for current context
    client = CoordinatorClient(config_path)
    _client_var.set(client)


def get_client() -> CoordinatorClient:
    """
    Get or create the client instance for the current async context.

    This is concurrency-safe: each async task gets its own client instance
    via contextvars, preventing race conditions between concurrent sessions.

    Returns:
        CoordinatorClient instance for current context

    Note:
        Prefer this over get_global_client() for new code.
    """
    client = _client_var.get()

    if client is None:
        # Auto-initialize with config from context or default
        config_path = _config_path_var.get() or _default_config_path
        client = CoordinatorClient(config_path)
        _client_var.set(client)

    return client


def get_global_client() -> CoordinatorClient:
    """
    Get the client instance (backwards compatible alias).

    Deprecated: Use get_client() for new code.

    Returns:
        CoordinatorClient instance for current context
    """
    return get_client()


async def call_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """
    Call an MCP tool through the context-local client.

    This function is imported by all generated tool wrappers and provides
    the actual MCP communication layer.

    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to call
        arguments: Tool arguments dictionary

    Returns:
        Tool execution result

    Example:
        >>> # Generated tools call this internally:
        >>> result = await call_mcp_tool(
        ...     server_name="chroma",
        ...     tool_name="query",
        ...     arguments={"collection": "papers", "query_text": "transformers"}
        ... )
    """
    client = get_client()
    return await client.call_tool(server_name, tool_name, arguments)


async def close_runtime() -> None:
    """
    Close the runtime client for the current context and clean up resources.

    Call this when shutting down your application or ending a session.
    """
    client = _client_var.get()

    if client is not None:
        await client.close()
        _client_var.set(None)
