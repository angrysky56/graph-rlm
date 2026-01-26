"""
MCP server discovery and introspection.

This module connects to MCP servers and discovers their capabilities:
- Available tools
- Tool schemas (parameters, descriptions)
- Resources
- Prompts

Used by the generator to create Python wrapper libraries.
"""

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
except ImportError:
    raise ImportError("MCP SDK is required. Install with: uv pip install mcp") from None

try:
    from anyio import BrokenResourceError
except ImportError:
    # anyio should be installed with mcp, but provide fallback
    BrokenResourceError = Exception


class MCPServerConfig:
    """Configuration for a single MCP server."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        transport_type: str = "stdio",
        tags: list[str] | None = None,
    ) -> None:
        """
        Initialize server configuration.

        Args:
            name: Server identifier
            command: Command to launch server (e.g., 'uvx', 'python') or URL for SSE
            args: Command arguments
            env: Environment variables
            transport_type: Transport type ("stdio" or "sse")
            tags: Optional tags for categorization (e.g., "data", "ai")
        """
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.transport_type = transport_type
        self.tags = tags or []

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> "MCPServerConfig":
        """Create config from dictionary (e.g., from JSON)."""
        return cls(
            name=name,
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env", {}),
            transport_type=config.get("type", "stdio"),
            tags=config.get("tags", []),
        )


class ServerIntrospector:
    """Introspects MCP servers to discover capabilities."""

    def __init__(self, server_config: MCPServerConfig) -> None:
        """
        Initialize introspector for a server.

        Args:
            server_config: Server configuration
        """
        self.config = server_config
        self.tools: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, dict[str, Any]] = {}
        self.prompts: dict[str, dict[str, Any]] = {}

    async def discover_tools(self) -> dict[str, Any]:
        """
        Connect to server and discover all capabilities.

        Returns:
            Dictionary with tools, resources, and prompts
        """
        # Merge with current environment to preserve PATH, etc.
        env = dict(os.environ)
        if self.config.env:
            env.update(self.config.env)

        # Ensure standard paths are in PATH
        current_path = env.get("PATH", "")
        standard_paths = ["/usr/local/bin", "/usr/bin", "/bin"]
        for p in standard_paths:
            if p not in current_path.split(os.pathsep):
                current_path = f"{current_path}{os.pathsep}{p}" if current_path else p

        # Add NVM paths if they exist (fix for npx/node not found)
        nvm_versions = Path.home() / ".nvm" / "versions" / "node"
        if nvm_versions.exists():
            for version_dir in nvm_versions.iterdir():
                if version_dir.is_dir():
                    bin_dir = version_dir / "bin"
                    if str(bin_dir) not in current_path.split(os.pathsep):
                        current_path = f"{bin_dir}{os.pathsep}{current_path}"

        env["PATH"] = current_path

        import tempfile

        # Create a temp file to capture stderr (only used for stdio)
        stderr_file = tempfile.TemporaryFile(mode="w+")

        try:
            if self.config.transport_type == "sse":
                transport_ctx = sse_client(url=self.config.command)
            else:
                # Resolve command path
                command = self._resolve_command(self.config.command, env.get("PATH"))

                # Create server parameters
                server_params = StdioServerParameters(
                    command=command,
                    args=self.config.args,
                    env=env,
                )
                transport_ctx = stdio_client(server_params, errlog=stderr_file)

            async with transport_ctx as (read, write):
                async with ClientSession(read, write) as session:

                    async def _discover() -> None:
                        # Initialize the session
                        await session.initialize()

                        # List available tools
                        tools_result = await session.list_tools()
                        self.tools = {
                            tool.name: {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            }
                            for tool in tools_result.tools
                        }

                        # Try to list resources (not all servers support this)
                        try:
                            resources_result = await session.list_resources()
                            self.resources = {
                                str(res.uri): {
                                    "uri": str(res.uri),
                                    "name": res.name,
                                    "description": res.description,
                                    "mime_type": res.mimeType,
                                }
                                for res in resources_result.resources
                            }
                        except Exception:
                            # Resources not supported
                            self.resources = {}

                        # Try to list prompts
                        try:
                            prompts_result = await session.list_prompts()
                            self.prompts = {
                                prompt.name: {
                                    "name": prompt.name,
                                    "description": prompt.description,
                                    "arguments": prompt.arguments,
                                }
                                for prompt in prompts_result.prompts
                            }
                        except Exception:
                            # Prompts not supported
                            self.prompts = {}

                    # Discovery happens within the context manager
                    # Timeout is handled by the caller (discover_all_servers)
                    await _discover()

                    # Give the server a moment to settle before we close the connection.
                    # Some servers (like desktop-commander) send notifications during
                    # startup that can cause BrokenResourceError if we exit too quickly.
                    await asyncio.sleep(0.5)

            return {
                "name": self.config.name,
                "tools": self.tools,
                "resources": self.resources,
                "prompts": self.prompts,
                "tags": self.config.tags,  # Include tags in result
            }

        except BrokenResourceError:
            # Known race condition in MCP SDK: server sends notifications during
            # context teardown. This is often recoverable on retry.
            return {
                "name": self.config.name,
                "error": "Server connection closed unexpectedly (BrokenResourceError). "
                "This can happen when the server sends notifications during startup. "
                "Consider increasing MCP_DISCOVERY_TIMEOUT or retrying.",
                "tools": {},
                "resources": {},
                "prompts": {},
                "tags": self.config.tags,
            }
        except BaseExceptionGroup as eg:
            # Python 3.11+ wraps async errors in ExceptionGroup
            # Check if it contains a BrokenResourceError
            broken_resource_errors = [
                e for e in eg.exceptions if isinstance(e, BrokenResourceError)
            ]
            if broken_resource_errors:
                return {
                    "name": self.config.name,
                    "error": "Server connection closed unexpectedly (BrokenResourceError in ExceptionGroup). "
                    "This can happen when the server process exits during communication. "
                    "The coordinator will continue with other servers.",
                    "tools": self.tools,  # Return any tools discovered before the error
                    "resources": self.resources,
                    "prompts": self.prompts,
                    "tags": self.config.tags,
                }
            # Re-raise if it doesn't contain BrokenResourceError
            raise
        except BaseException as e:
            import traceback

            tb = traceback.format_exc()

            # Read stderr captured so far if using stdio
            stderr_output = ""
            if self.config.transport_type == "stdio":
                stderr_file.seek(0)
                stderr_output = stderr_file.read()

            return {
                "name": self.config.name,
                "error": f"Failed to connect: {e}\nTraceback:\n{tb}\nStderr:\n{stderr_output}",
                "tools": {},
                "resources": {},
                "prompts": {},
                "tags": self.config.tags,
            }
        finally:
            stderr_file.close()

    def _resolve_command(self, command: str, path_env: str | None) -> str:
        """
        Resolve absolute path for a command, checking common user paths.
        """

        # 1. Try with provided PATH
        resolved = shutil.which(command, path=path_env)
        if resolved:
            return resolved

        # 2. Try with common user paths
        home = Path.home()
        common_paths = [
            home / ".pyenv" / "shims",
            home / ".cargo" / "bin",
            home / ".local" / "bin",
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/bin"),
        ]

        # Add NVM paths if they exist
        nvm_versions = home / ".nvm" / "versions" / "node"
        if nvm_versions.exists():
            for version_dir in nvm_versions.iterdir():
                if version_dir.is_dir():
                    common_paths.append(version_dir / "bin")

        # Construct search path
        search_path = os.pathsep.join(str(p) for p in common_paths)

        resolved = shutil.which(command, path=search_path)
        if resolved:
            return resolved

        # 3. Fallback to original command
        return command


class ConfigLoader:
    """Loads MCP server configurations from various sources."""

    @staticmethod
    def load_from_json(filepath: str | Path) -> dict[str, MCPServerConfig]:
        """
        Load server configs from JSON file.

        Expected format:
        {
          "mcpServers": {
            "server_name": {
              "command": "uvx",
              "args": ["server-package"],
              "env": {"KEY": "value"},
              "tags": ["category1", "category2"]
            }
          }
        }

        Args:
            filepath: Path to JSON configuration file

        Returns:
            Dictionary mapping server names to configurations
        """
        filepath = Path(filepath).expanduser()

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        servers = data.get("mcpServers", {})

        return {
            name: MCPServerConfig.from_dict(name, config)
            for name, config in servers.items()
        }


async def discover_all_servers(
    config_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Discover capabilities of all configured servers in parallel.

    Args:
        config_path: Path to config file (required)

    Returns:
        Dictionary mapping server names to their capabilities
    """
    # Load configuration
    configs = ConfigLoader.load_from_json(config_path)

    # Get timeouts
    from .config import ConfigManager

    config_manager = ConfigManager()
    timeouts = config_manager.get_timeouts()
    discovery_timeout = timeouts["discovery"]

    # Limit concurrency to avoid overwhelming the system
    semaphore = asyncio.Semaphore(10)

    async def _discover_one(
        name: str, config: MCPServerConfig
    ) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                introspector = ServerIntrospector(config)
                # Wrap individual discovery in timeout
                result = await asyncio.wait_for(
                    introspector.discover_tools(), timeout=discovery_timeout
                )
                return name, result
            except TimeoutError:
                return name, {
                    "name": name,
                    "error": f"Discovery timed out after {discovery_timeout}s",
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                    "tags": config.tags,
                }
            except Exception as e:
                return name, {
                    "name": name,
                    "error": f"Discovery failed: {str(e)}",
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                    "tags": config.tags,
                }

    # Launch all discoveries in parallel
    tasks = [_discover_one(name, config) for name, config in configs.items()]

    results_list = await asyncio.gather(*tasks)

    return dict(results_list)


async def discover_single_server(
    config_path: str | Path,
    server_name: str,
) -> dict[str, Any]:
    """
    Discover capabilities of a single server (fast, targeted discovery).

    Args:
        config_path: Path to config file
        server_name: Name of the server to discover

    Returns:
        Server capabilities dictionary
    """
    # Load configuration
    configs = ConfigLoader.load_from_json(config_path)

    if server_name not in configs:
        return {
            "name": server_name,
            "error": f"Server '{server_name}' not found in configuration",
            "tools": {},
            "resources": {},
            "prompts": {},
            "tags": [],
        }

    config = configs[server_name]

    # Get timeouts
    from .config import ConfigManager

    config_manager = ConfigManager()
    timeouts = config_manager.get_timeouts()
    discovery_timeout = timeouts["discovery"]

    try:
        introspector = ServerIntrospector(config)
        result = await asyncio.wait_for(
            introspector.discover_tools(), timeout=discovery_timeout
        )
        return result
    except TimeoutError:
        return {
            "name": server_name,
            "error": f"Discovery timed out after {discovery_timeout}s",
            "tools": {},
            "resources": {},
            "prompts": {},
            "tags": config.tags,
        }
    except Exception as e:
        return {
            "name": server_name,
            "error": f"Discovery failed: {str(e)}",
            "tools": {},
            "resources": {},
            "prompts": {},
            "tags": config.tags,
        }


def discover_tools(config_path: str | Path) -> dict[str, list[str]]:
    """
    Synchronous wrapper to get tool names for all servers.

    Args:
        config_path: Path to config file (required)

    Returns:
        Dictionary mapping server names to tool name lists
    """
    results = asyncio.run(discover_all_servers(config_path))

    return {
        name: list(info["tools"].keys())
        for name, info in results.items()
        if "error" not in info
    }


def filter_tools_from_results(
    results: dict[str, dict[str, Any]],
    query: str,
    max_results: int = 50,
    min_score: int = 2,
) -> list[dict[str, Any]]:
    """
    Pure synchronous filtering of tool results matching a query.
    Extracts logic from find_tools/asyncio.run for reusability.

    Args:
        results: Server discovery results
        query: Search query string
        max_results: Maximum number of results to return (default 50)
        min_score: Minimum score threshold (default 2)

    Returns:
        List of matching tools, sorted by score descending
    """
    matches = []
    query_parts = set(re.findall(r"\w+", query.lower()))

    for server_name, info in results.items():
        if "error" in info:
            continue

        tags = set(info.get("tags", []))

        for tool_name, tool_def in info["tools"].items():
            score = 0

            # Text to search in
            name_text = tool_name.lower()
            desc_text = (tool_def.get("description") or "").lower()
            server_text = server_name.lower()

            # Keyword matching
            for term in query_parts:
                if term in name_text:
                    score += 5
                if term in server_text:
                    score += 3
                if term in tags:
                    score += 3
                if term in desc_text:
                    score += 1

            if score >= min_score:
                desc = tool_def.get("description") or ""
                if len(desc) > 2000:
                    desc = desc[:1997] + "..."

                matches.append(
                    {
                        "server": server_name,
                        "tool": tool_name,
                        "description": desc,
                        "tags": info.get("tags", []),
                        "score": score,
                    }
                )

    # Sort by score desc and limit results
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:max_results]


def find_tools(config_path: str | Path, query: str) -> list[dict[str, Any]]:
    """
    Find tools matching a semantic query.

    Matches query against tool name, server name, tags, and description.

    Args:
        config_path: Path to config file
        query: Search query

    Returns:
        List of matching tools with metadata and score
    """
    results = asyncio.run(discover_all_servers(config_path))
    return filter_tools_from_results(results, query)


def get_server_details(
    server_name: str,
    config_path: str | Path,
) -> dict[str, Any]:
    """
    Get detailed information about a specific server.

    Args:
        server_name: Name of the server
        config_path: Path to config file (required)

    Returns:
        Server capabilities dictionary
    """
    results = asyncio.run(discover_all_servers(config_path))

    if server_name not in results:
        raise ValueError(f"Server '{server_name}' not found in config")

    return results[server_name]
