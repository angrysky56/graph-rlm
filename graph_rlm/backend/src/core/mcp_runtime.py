"""
MCP Server discovery and proxy logic for the Graph-RLM Agent.
"""

import importlib.util
import inspect
import pkgutil
from importlib import import_module
from typing import TYPE_CHECKING

import graph_rlm.backend.mcp_tools as mcp_tools_pkg

from .logger import get_logger

if TYPE_CHECKING:
    from .rlm_interface import RLMInterface

logger = get_logger("graph_rlm.mcp_runtime")


def is_mcp_available():
    """Defensive check for MCP tools availability."""
    return (
        importlib.util.find_spec("mcp_tools") is not None
        or importlib.util.find_spec("graph_rlm.backend.mcp_tools") is not None
    )


def is_skills_available():
    """Defensive check for skills/manager availability."""
    return (
        importlib.util.find_spec(
            ".skill_storage", package="graph_rlm.backend.src.mcp_integration"
        )
        is not None
    )


class MCPServerNamespace:
    """Lazy-loaded namespace for a single MCP server."""

    def __init__(self, mod_name: str, alias: str, rlm_interface: "RLMInterface"):
        self._mod_name = mod_name
        self._alias = alias
        self._rlm_interface = rlm_interface
        self._module = None
        self._tools = {}
        self._docs = {}

    def set_rlm_interface(self, rlm_interface: "RLMInterface"):
        """Update the RLM interface binding."""
        self._rlm_interface = rlm_interface

    def _ensure_loaded(self):
        if self._module is False:  # Already tried and failed
            return
        if self._module is None:
            try:
                self._module = import_module(
                    f"graph_rlm.backend.mcp_tools.{self._mod_name}"
                )
                for attr in dir(self._module):
                    if not attr.startswith("_"):
                        func = getattr(self._module, attr)
                        if callable(func):
                            # Use actual function name, no aliases
                            def make_wrapper(f, n):
                                async def wrapped(*args, **kwargs):
                                    self._rlm_interface.record_tool_use(n)
                                    res = f(*args, **kwargs)
                                    if inspect.isawaitable(res):
                                        return await res
                                    return res

                                return wrapped

                            wrapper = make_wrapper(func, f"mcp.{self._alias}.{attr}")
                            self._tools[attr] = wrapper
                            self._docs[attr] = func.__doc__
            except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
                logger.warning("Failed to load MCP server %s: %s", self._mod_name, e)
                self._module = False  # Mark as failed

    def __getattr__(self, name):
        self._ensure_loaded()
        if name in self._tools:
            return self._tools[name]

        # Resilience: Handle agent hallucinations for 'analyze'
        if name == "analyze" and "advanced_reasoning" in self._tools:
            return self._tools["advanced_reasoning"]

        raise AttributeError(f"MCP Server '{self._alias}' has no tool '{name}'")

    def __dir__(self):
        self._ensure_loaded()
        return list(self._tools.keys())

    def __repr__(self):
        return f"<MCPServerNamespace '{self._alias}' (from {self._mod_name})>"


class LazyMCPNamespace:
    """Lazy-loaded root namespace for all MCP servers."""

    def __init__(self, rlm_interface: "RLMInterface"):
        self._rlm_interface = rlm_interface
        self._aliases = {}
        self._scan_done = False

    def set_rlm_interface(self, rlm_interface: "RLMInterface"):
        """Update the RLM interface binding and propagate to children."""
        self._rlm_interface = rlm_interface
        for server in self._aliases.values():
            if hasattr(server, "set_rlm_interface"):
                server.set_rlm_interface(rlm_interface)

    def _scan(self):
        if not self._scan_done and is_mcp_available():
            try:

                logger.info("Starting MCP server discovery...")
                for _, mod_name, _ in pkgutil.iter_modules(mcp_tools_pkg.__path__):
                    if mod_name.startswith("_") or mod_name == "skills":
                        logger.debug("Skipping module: %s", mod_name)
                        continue

                    logger.info("Discovered MCP module: %s", mod_name)

                    server = MCPServerNamespace(mod_name, mod_name, self._rlm_interface)
                    self._aliases[mod_name] = server
                    logger.info("Registered MCP server: %s", mod_name)

                self._scan_done = True
                logger.info("MCP server discovery completed.")
            except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
                logger.warning("MCP Scan Error: %s", e)

    def __getattr__(self, name):
        self._scan()
        if name in self._aliases:
            return self._aliases[name]
        raise AttributeError(f"No MCP server found with name or alias '{name}'")

    def list_servers(self):
        """Returns a list of all discovered MCP server names."""
        self._scan()
        return list(self._aliases.keys())

    def __dir__(self):
        return self.list_servers()

    def __repr__(self):
        return f"<LazyMCPNamespace with {len(self._aliases)} server aliases>"


def get_mcp_server_names() -> list[str]:
    """
    Directly scans the mcp_tools directory and returns list of server names.
    Does NOT require a full RLMInterface or LazyMCPNamespace.
    """
    names = []
    if not is_mcp_available():
        return names
    try:
        for _, mod_name, _ in pkgutil.iter_modules(mcp_tools_pkg.__path__):
            if mod_name.startswith("_") or mod_name == "skills":
                continue
            names.append(mod_name)
    except Exception as e:
        logger.warning("Quick MCP name scan failed: %s", e)
    return names
