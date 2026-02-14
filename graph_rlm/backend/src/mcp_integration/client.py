"""Coordinator client for calling actual MCP servers.

This module provides the runtime communication layer between
generated tool wrappers and real MCP servers, using the robust
McpClientManager for connection management.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from .core import McpClientManager

logger = logging.getLogger("graph_rlm.mcp_integration.client")


class CoordinatorClient:
    """
    Client for communicating with MCP servers at runtime.

    Wraps McpClientManager to provide a simple interface for tool execution.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """
        Initialize coordinator client.

        Args:
            config_path: Path to MCP server configuration
        """
        self.manager = McpClientManager()
        self.manager.initialize(Path(config_path) if config_path else None)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        from .runtime import get_stop_event

        stop_event = get_stop_event()
        return await self.manager.call_tool(
            server_name, tool_name, arguments, stop_event=stop_event
        )

    async def close(self) -> None:
        """Close all connections."""
        await self.manager.cleanup()

    async def __aenter__(self) -> "CoordinatorClient":
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        await self.close()


# Global client instance for simple usage
_global_client: CoordinatorClient | None = None


async def call_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> Any:
    """
    High-level function to call an MCP tool.

    This is the function that generated tool wrappers call.

    Args:
        server_name: Name of the server
        tool_name: Name of the tool
        arguments: Tool arguments
        config_path: Optional config path (uses global client if None)

    Returns:
        Tool execution result
    """
    global _global_client

    # Simple loop integrity check for the global instance
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    # Use global client for efficiency
    if _global_client is None or (
        current_loop
        and getattr(_global_client.manager, "_loop", None) is not current_loop
    ):
        if _global_client is not None:
            logger.warning("Global MCP client loop mismatch. Re-initializing.")
            # We don't await cleanup here as the old loop might be closed/stale
            # The new manager in the new client will start fresh.
        _global_client = CoordinatorClient(config_path)

    return await _global_client.call_tool(server_name, tool_name, arguments)


async def cleanup_global_client_async() -> None:
    """Clean up global client asynchronously."""
    global _global_client

    if _global_client is not None:
        await _global_client.close()
        _global_client = None
