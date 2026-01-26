"""MCP Client Manager with state machine architecture for lazy loading and connection.

This module provides the core runtime client manager that connects to MCP servers
on-demand, caches tools, and manages the lifecycle of server connections using
an explicit state machine.
"""

import asyncio
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import Tool

from ..config import ConfigManager
from .config import McpConfig, ServerConfig

logger = logging.getLogger("graph_rlm.mcp_integration.core.client")


class PermissiveClientSession(ClientSession):
    """ClientSession that skips strict output validation but logs warnings."""

    async def _validate_tool_result(self, name: str, result: Any) -> None:
        try:
            await super()._validate_tool_result(name, result)
        except Exception as e:
            logger.warning(
                f"Validation failed for tool '{name}', suppressing error: {e}"
            )


class ConnectionState(str, Enum):
    """Explicit states for the MCP Client Manager lifecycle.

    States:
        UNINITIALIZED: Manager created but not initialized
        INITIALIZED: Configuration loaded, no server connections
        CONNECTED: At least one server connection established
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    CONNECTED = "connected"


class McpClientManager:
    """Lazy-loading MCP client manager with explicit state machine.

    This manager implements a state machine pattern for managing MCP server
    connections with the following characteristics:

    1. Lazy Loading: Servers are only connected when their tools are first requested
    - Lazy initialization: Config loaded on initialize(), servers NOT connected
    - Lazy connection: Servers connect on first call_tool() call
    - Tool caching: Cache tools per server to avoid repeated list_tools calls
    - Defensive unwrapping: Handle response.value and fallback patterns
    - Explicit state tracking: Clear state transitions with validation

    State Transitions:
        UNINITIALIZED -> INITIALIZED (via initialize())
        INITIALIZED -> CONNECTED (via _connect_to_server())
        any state -> UNINITIALIZED (via cleanup())
    """

    def __init__(self) -> None:
        """Initialize an uninitialized MCP Client Manager."""
        self.state = ConnectionState.UNINITIALIZED
        self.config: McpConfig | None = None
        self.sessions: dict[str, ClientSession] = {}
        self.tools_cache: dict[str, list[Tool]] = {}

        # Connection supervision
        self._connection_tasks: dict[str, asyncio.Task] = {}
        self._stop_events: dict[str, asyncio.Event] = {}

        # Load timeouts
        config_manager = ConfigManager()
        timeouts = config_manager.get_timeouts()
        self.connect_timeout = timeouts["connect"]
        self.read_timeout = timeouts["read"]

    def _validate_state(self, required_state: ConnectionState, operation: str) -> None:
        """Validate that the manager is in the required state."""
        if self.state != required_state:
            raise RuntimeError(
                f"Cannot {operation}: Manager is in state '{self.state.value}', but requires state '{required_state.value}'"
            )

    def _validate_state_at_least(
        self, minimum_state: ConnectionState, operation: str
    ) -> None:
        """Validate that the manager has at least reached the minimum state."""
        state_order = [
            ConnectionState.UNINITIALIZED,
            ConnectionState.INITIALIZED,
            ConnectionState.CONNECTED,
        ]
        current_idx = state_order.index(self.state)
        required_idx = state_order.index(minimum_state)

        if current_idx < required_idx:
            raise RuntimeError(
                f"Cannot {operation}: Manager is in state '{self.state.value}', but requires at least state '{minimum_state.value}'"
            )

    def _mark_initialized(self) -> None:
        """Transition to INITIALIZED state."""
        self.state = ConnectionState.INITIALIZED
        logger.debug("State transition: UNINITIALIZED -> INITIALIZED")

    def _mark_connected(self) -> None:
        """Transition to CONNECTED state."""
        if self.state == ConnectionState.INITIALIZED:
            self.state = ConnectionState.CONNECTED
            logger.debug("State transition: INITIALIZED -> CONNECTED")

    def _mark_uninitialized(self) -> None:
        """Transition back to UNINITIALIZED state."""
        self.state = ConnectionState.UNINITIALIZED
        logger.debug("State transition: -> UNINITIALIZED")

    def initialize(self, config_path: Path | None = None) -> None:
        """Initialize the manager by loading configuration.

        Args:
            config_path: Path to mcp_config.json. If None, looks in default locations.
        """
        if self.state != ConnectionState.UNINITIALIZED:
            logger.warning("Manager already initialized, reloading config")

        # Load configuration
        if config_path:
            with open(config_path) as f:
                self.config = McpConfig.from_json(f.read())
        else:
            # Try default locations
            possible_paths = [
                Path("mcp_servers.json"),
                Path(os.getenv("MCP_JSON", "")),
            ]
            for path in possible_paths:
                if path and path.exists():
                    with open(path) as f:
                        self.config = McpConfig.from_json(f.read())
                    break

            if not self.config:
                raise FileNotFoundError("Could not find MCP configuration file")

        self._mark_initialized()

    async def _maintain_connection(
        self,
        server_name: str,
        server_config: ServerConfig,
        ready_event: asyncio.Event,
        stop_event: asyncio.Event,
    ) -> None:
        """
        Persistent task that manages the lifecycle of a connection within a single task scope.
        This ensures 'anyio' contexts are entered and exited in the same task.
        """
        try:
            # Choose transport
            if server_config.type == "stdio":
                if not server_config.command:
                    raise ValueError(f"Server {server_name} missing command")

                env = dict(os.environ)
                if server_config.env:
                    env.update(server_config.env)

                command = self._resolve_command(server_config.command, env)
                server_params = StdioServerParameters(
                    command=command, args=server_config.args or [], env=env
                )

                # Context 1: Stdio
                async with stdio_client(server_params) as (read_stream, write_stream):
                    # Context 2: Session
                    async with PermissiveClientSession(
                        read_stream, write_stream
                    ) as session:
                        await session.initialize()

                        # Register session and signal readiness
                        self.sessions[server_name] = session
                        ready_event.set()

                        # Wait until stop signal
                        await stop_event.wait()

                        # Give server time to settle/drain before closing streams
                        await asyncio.sleep(0.5)

            elif server_config.type == "sse":
                if not server_config.url:
                    raise ValueError(f"Server {server_name} missing url")

                async with sse_client(url=server_config.url) as (
                    read_stream,
                    write_stream,
                ):
                    async with PermissiveClientSession(
                        read_stream, write_stream
                    ) as session:
                        await session.initialize()
                        self.sessions[server_name] = session
                        ready_event.set()
                        await stop_event.wait()
            else:
                raise ValueError(f"Unsupported transport: {server_config.type}")

        except asyncio.CancelledError:
            # Expected on shutdown if we cancel tasks directly
            pass
        except Exception as e:
            logger.error(f"Connection task failure for {server_name}: {e}")
            # If we failed before ready, we should set it so the waiter doesn't hang
            if not ready_event.is_set():
                # We can't really signal error via event, but we can prevent hang.
                # The waiter will check self.sessions and fail.
                ready_event.set()
        finally:
            # Cleanup registry
            if server_name in self.sessions:
                del self.sessions[server_name]
            logger.debug(f"Connection task for {server_name} exited")

    def _resolve_command(self, command: str, env: dict[str, str]) -> str:
        """
        Resolve command path for subprocess execution.

        Searches common user paths (nvm, pyenv, cargo, etc.) if the command
        is not found in the provided PATH environment variable.
        """
        import shutil
        import sys

        # 1. Try with provided PATH
        path_env = env.get("PATH", os.environ.get("PATH", ""))
        resolved = shutil.which(command, path=path_env)
        if resolved:
            return resolved

        # 2. Try with common user paths based on platform
        home = Path.home()
        common_paths = []

        if sys.platform == "win32":
            appdata = os.environ.get("APPDATA")
            localappdata = os.environ.get("LOCALAPPDATA")
            if appdata:
                common_paths.append(Path(appdata) / "npm")
            if localappdata:
                common_paths.append(
                    Path(localappdata) / "Programs" / "Python" / "Scripts"
                )
                common_paths.append(Path(localappdata) / "uv")
            common_paths.append(home / ".cargo" / "bin")
        else:
            common_paths.extend(
                [
                    home / ".pyenv" / "shims",
                    home / ".cargo" / "bin",
                    home / ".local" / "bin",
                    Path("/usr/local/bin"),
                    Path("/usr/bin"),
                    Path("/bin"),
                    Path("/opt/homebrew/bin"),
                ]
            )
            nvm_versions = home / ".nvm" / "versions" / "node"
            if nvm_versions.exists():
                for version_dir in nvm_versions.iterdir():
                    if version_dir.is_dir():
                        common_paths.append(version_dir / "bin")

        search_path = os.pathsep.join(str(p) for p in common_paths)
        if env.get("PATH"):
            env["PATH"] = f"{search_path}{os.pathsep}{env['PATH']}"
        else:
            env["PATH"] = search_path

        resolved = shutil.which(command, path=search_path)
        if resolved:
            return resolved
        return command

    async def _connect_to_server(
        self, server_name: str, server_config: ServerConfig
    ) -> ClientSession:
        """Connect to a specific server via supervisor task."""
        if server_name in self.sessions:
            return self.sessions[server_name]

        logger.info(f"Connecting to server: {server_name} ({server_config.type})")

        ready_event = asyncio.Event()
        stop_event = asyncio.Event()

        task = asyncio.create_task(
            self._maintain_connection(
                server_name, server_config, ready_event, stop_event
            )
        )
        self._connection_tasks[server_name] = task
        self._stop_events[server_name] = stop_event

        try:
            # Wait for connection ready
            await asyncio.wait_for(ready_event.wait(), timeout=self.connect_timeout)
        except Exception as e:
            # Connection failed or timed out
            logger.error(f"Failed to connect to {server_name}: {e}")
            stop_event.set()  # Signal task to exit
            # Clean up
            if server_name in self._connection_tasks:
                del self._connection_tasks[server_name]
            if server_name in self._stop_events:
                del self._stop_events[server_name]
            raise RuntimeError(f"Failed to connect to {server_name}: {e}") from e

        if server_name not in self.sessions:
            raise RuntimeError(
                f"Connection task for {server_name} finished without session"
            )

        self._mark_connected()
        return self.sessions[server_name]

    async def _cleanup_server(self, server_name: str) -> None:
        """Clean up resources for a specific server."""
        # Signal stop
        if server_name in self._stop_events:
            self._stop_events[server_name].set()

        # Wait for task
        if server_name in self._connection_tasks:
            try:
                # Wait for the task to finish (it should exit context on stop_event)
                await asyncio.wait_for(self._connection_tasks[server_name], timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                logger.warning(f"Error awaiting connection task for {server_name}: {e}")
            del self._connection_tasks[server_name]

        if server_name in self._stop_events:
            del self._stop_events[server_name]

        if server_name in self.sessions:
            del self.sessions[server_name]

    async def _ensure_connection(
        self, server_name: str, server_config: ServerConfig, max_retries: int = 2
    ) -> ClientSession:
        """
        Ensure a healthy connection to a server, with auto-reconnect on failure.

        If the server connection is stale or broken, cleans up and retries.
        """
        last_error = None

        for attempt in range(max_retries + 1):
            # Check if we have an existing session
            if server_name in self.sessions:
                # Verify the connection is still alive by checking the task
                task = self._connection_tasks.get(server_name)
                if task and not task.done():
                    # Connection task is still running, session should be valid
                    return self.sessions[server_name]
                else:
                    # Task died - clean up stale session
                    logger.warning(
                        f"Server {server_name} connection task died, cleaning up for reconnect"
                    )
                    await self._cleanup_server(server_name)

            # Try to connect
            try:
                session = await self._connect_to_server(server_name, server_config)
                return session
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Connection attempt {attempt + 1}/{max_retries + 1} to {server_name} failed: {e}. Retrying..."
                    )
                    # Clean up before retry
                    await self._cleanup_server(server_name)
                    # Brief delay before retry
                    await asyncio.sleep(0.5)

        raise RuntimeError(
            f"Failed to connect to {server_name} after {max_retries + 1} attempts: {last_error}"
        )

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Call an MCP tool with lazy server connection and auto-reconnect."""
        # Custom dispatch for internal 'skills' server
        if server_name == "skills":
            from graph_rlm.backend.mcp_tools import skills as skills_tool

            func = getattr(skills_tool, tool_name, None)
            if not func:
                raise ValueError(f"Tool {tool_name} not found in skills")
            return await func(**(arguments or {}))

        self._validate_state_at_least(ConnectionState.INITIALIZED, "call_tool")

        if not self.config:
            raise RuntimeError("Configuration not loaded")

        server_config = self.config.get_server(server_name)
        if not server_config:
            raise ValueError(f"Server {server_name} not found in configuration")

        if server_config.disabled:
            raise ValueError(f"Server {server_name} is disabled")

        # Connect with auto-reconnect
        session = await self._ensure_connection(server_name, server_config)

        # Call tool with timeout and reconnect on connection errors
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments or {}), timeout=self.read_timeout
            )
        except TimeoutError:
            raise TimeoutError(
                f"Tool call {tool_name} on {server_name} timed out after {self.read_timeout}s"
            ) from None
        except (ConnectionError, BrokenPipeError, EOFError) as e:
            # Connection died during call - try to reconnect once
            logger.warning(
                f"Connection error during tool call, attempting reconnect: {e}"
            )
            await self._cleanup_server(server_name)
            session = await self._ensure_connection(server_name, server_config)
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments or {}), timeout=self.read_timeout
            )

        # Unwrap result similar to reference implementation
        if hasattr(result, "content"):
            return "\n".join([c.text for c in result.content if c.type == "text"])
        return result

    async def list_tools(self, server_name: str) -> list[Tool]:
        """List tools for a server, using cache if available."""
        # Custom dispatch for internal 'skills' server
        if server_name == "skills":
            from graph_rlm.backend.mcp_tools import skills as skills_tool

            # Synthesize Tool objects
            tools = []
            for tool_def in skills_tool.TOOLS:
                name = str(tool_def["name"])
                desc = str(tool_def.get("description", ""))
                schema_data = tool_def.get("input_schema", {})
                schema: dict[str, Any] = (
                    dict(schema_data) if isinstance(schema_data, dict) else {}
                )

                tools.append(Tool(name=name, description=desc, inputSchema=schema))
            return tools

        self._validate_state_at_least(ConnectionState.INITIALIZED, "list_tools")

        if server_name in self.tools_cache:
            return self.tools_cache[server_name]

        if not self.config:
            raise RuntimeError("Configuration not loaded")

        server_config = self.config.get_server(server_name)
        if not server_config:
            raise ValueError(f"Server {server_name} not found")

        # Connect with auto-reconnect
        session = await self._ensure_connection(server_name, server_config)

        try:
            result = await session.list_tools()
            tools = result.tools
        except (ConnectionError, BrokenPipeError, EOFError) as e:
            # Connection died - try to reconnect once
            logger.warning(
                f"Connection error during list_tools, attempting reconnect: {e}"
            )
            await self._cleanup_server(server_name)
            session = await self._ensure_connection(server_name, server_config)
            result = await session.list_tools()
            tools = result.tools

        self.tools_cache[server_name] = list(tools)
        return self.tools_cache[server_name]

    async def cleanup(self) -> None:
        """Close all connections and reset manager."""
        logger.info("Cleaning up MCP Client Manager")

        # Stop all connection tasks
        shutdown_tasks = []
        for server_name in list(self._connection_tasks.keys()):
            # Signal stop
            if server_name in self._stop_events:
                self._stop_events[server_name].set()
            shutdown_tasks.append(self._connection_tasks[server_name])

        if shutdown_tasks:
            # Wait for all to exit contexts cleanly
            try:
                # Type ignore: gather returns Future[list], wait_for expects Awaitable
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),  # type: ignore
                    timeout=2.0,
                )
            except Exception as e:
                logger.warning(f"Error during cleanup wait: {e}")

        self.sessions.clear()
        self.tools_cache.clear()
        self._connection_tasks.clear()
        self._stop_events.clear()

        self._mark_uninitialized()
