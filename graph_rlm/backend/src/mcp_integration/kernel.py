"""
Persistent Python kernel for isolated execution.
Handles JSON commands over stdin/stdout and maintains state.
"""

import ast
import asyncio
import builtins
import inspect
import json
import logging
import os
import sys
from typing import Optional

import nest_asyncio

# Allow nested event loops for agent code execution
nest_asyncio.apply()

# Optional scientific computing modules
try:
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    _SCIENTIFIC_MODULES = {
        "np": np,
        "sp": sp,
        "spla": spla,
    }
except ImportError:
    _SCIENTIFIC_MODULES = {}
    # We delay logging this until the logger is configured
# If that directory contains a 'skills.py' (like ours does), it shadows 'skills/' dir.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    # We move it to the end or remove it if we want to be strict.
    # Removing it is safer since we should use absolute imports for the backend.
    sys.path.remove(script_dir)

# Configure logging for the kernel itself (stderr)
logging.basicConfig(
    level=logging.INFO, stream=sys.stderr, format="[KERNEL] %(message)s"
)
logger = logging.getLogger("kernel")

# Global State Container
_MCP_DISCOVERY = {}


class IPCClient:
    """Mock MCP client that proxies tool calls over IPC."""

    def __init__(self):
        """Initialize the IPC client."""

    def __getattr__(self, name):
        if name not in _MCP_DISCOVERY:
            servers = list(_MCP_DISCOVERY.keys())
            raise AttributeError(
                f"MCP Server '{name}' not found. Available servers: {servers}"
            )
        return IPCServerProxy(name)

    def __dir__(self):
        return list(_MCP_DISCOVERY.keys())

    def __repr__(self):
        servers = list(_MCP_DISCOVERY.keys())
        if not servers:
            return "<MCP Client (No servers discovered)>"
        res = ["--- Available MCP Servers ---"]
        for s in servers:
            res.append(f"- mcp.{s}")
        res.append("-----------------------------")
        res.append("Type mcp.<server_name> to see its tools and descriptions.")
        return "\n".join(res)


class IPCServerProxy:
    """Proxy for an MCP server."""

    def __init__(self, server_name: str):
        """Initialize the server proxy."""
        self.server_name = server_name

    def __getattr__(self, name):
        server_data = _MCP_DISCOVERY.get(self.server_name, {})
        if name not in server_data:
            tools = list(server_data.keys())
            raise AttributeError(
                f"Tool '{name}' not found in MCP Server '{self.server_name}'. Available tools: {tools}"
            )
        return IPCToolProxy(self.server_name, name)

    def __dir__(self):
        server_data = _MCP_DISCOVERY.get(self.server_name, {})
        return list(server_data.keys())

    def __repr__(self):
        server_data = _MCP_DISCOVERY.get(self.server_name, {})
        tools = list(server_data.keys())
        if not tools:
            return f"<MCP Server '{self.server_name}' (No tools discovered yet)>"
        res = [f"--- MCP Server '{self.server_name}' Tools ---"]
        for t, td in server_data.items():
            doc = td.get("doc", "No description provided.")
            res.append(f"\n* mcp.{self.server_name}.{t}:")
            for line in doc.strip().split("\n"):
                res.append(f"    {line}")
        res.append("\n------------------------------------------")
        res.append("Execute a tool: mcp.server_name.tool_name(args...)")
        return "\n".join(res)


class IPCToolProxy:
    """Proxy for an MCP tool."""

    def __init__(self, server_name: str, tool_name: str):
        """Initialize the tool proxy and load docstrings from discovery data."""
        self.tool_path = f"mcp.{server_name}.{tool_name}"
        server_data = _MCP_DISCOVERY.get(server_name, {})
        tool_data = server_data.get(tool_name, {})
        self.__doc__ = tool_data.get("doc", "")

    def __repr__(self):
        doc = self.__doc__ or "No description provided."
        return f"<MCP Tool '{self.tool_path}'>\n{doc}"

    async def __call__(self, *args, **kwargs):
        req = {"tool": self.tool_path, "args": args, "kwargs": kwargs}
        print(f"<<IPC_REQUEST>>{json.dumps(req)}", flush=True)

        # Blocking read from stdin for response
        loop = asyncio.get_running_loop()
        response_line = await loop.run_in_executor(None, sys.stdin.readline)

        if not response_line:
            raise EOFError("Host closed connection unexpectedly during IPC.")

        if response_line.startswith("<<IPC_RESPONSE>>"):
            resp_json = response_line.replace("<<IPC_RESPONSE>>", "").strip()
            resp = json.loads(resp_json)
            if resp.get("status") == "success":
                return resp.get("result")

            raise RuntimeError(f"IPC Error: {resp.get('error')}")

        raise RuntimeError(f"Protocol Violation: {response_line[:100]}")


class KBProxy:
    """Proxy for knowledge base paths and directories."""

    _ALIASES: dict = {
        "kb_dir": "root",
        "kb_path": "root",
        "knowledge_base_dir": "root",
        "base_dir": "root",
        "base_path": "root",
    }

    def __init__(self, base_path: Optional[str] = None):
        """Initialize knowledge base proxy with project paths."""

        # Default to project root if no base path provided
        self._base_path = base_path or os.environ.get(
            "PROJECT_ROOT", "/home/ty/Repositories/ai_workspace/graph-rlm"
        )
        self._kb_root = os.path.join(self._base_path, "knowledge_base")

    @property
    def reports_dir(self):
        """Path to reports directory."""

        return os.path.join(self._kb_root, "reports")

    @property
    def plans_dir(self):
        """Path to plans directory."""

        return os.path.join(self._kb_root, "plans")

    @property
    def outputs_dir(self):
        """Path to outputs directory."""

        return os.path.join(self._kb_root, "outputs")

    @property
    def axioms_dir(self):
        """Path to axioms directory."""
        return os.path.join(self._base_path, "graph_rlm", "backend", "axioms_dir")

    @property
    def mcp_tools_dir(self):
        """Path to MCP tools directory."""
        return os.path.join(self._base_path, "graph_rlm", "backend", "mcp_tools")

    @property
    def skills_dir(self):
        """Path to skills directory."""
        return os.path.join(self._base_path, "graph_rlm", "backend", "skills")

    @property
    def workspace_dir(self):
        """Path to workspace directory."""
        return os.path.join(self._kb_root, "workspace")

    @property
    def src_dir(self):
        """Path to source directory."""
        return os.path.join(self._base_path, "graph_rlm", "backend", "src")

    @property
    def root(self):
        """Knowledge base root path."""
        return self._kb_root

    @property
    def root_dir(self):
        """Knowledge base root path alias."""
        return self._kb_root

    def __getitem__(self, key: str) -> str:
        """Allow dictionary-like access to knowledge base paths."""
        if hasattr(self, key):
            val = getattr(self, key)
            if isinstance(val, str):
                return val
        if key == "root" or key == "root_dir":
            return self._kb_root
        raise KeyError(f"KBProxy has no attribute or path: {key}")

    def __dir__(self):
        """Include dynamic keys in directory listing."""
        return list(super().__dir__()) + [
            "reports_dir",
            "plans_dir",
            "outputs_dir",
            "axioms_dir",
            "mcp_tools_dir",
            "skills_dir",
            "src_dir",
            "workspace_dir",
            "root",
            "root_dir",
        ]

    def __getattr__(self, name: str) -> str:
        """Resolve common attribute aliases; raise helpful error otherwise."""
        # Avoid infinite recursion for internal lookups
        if name.startswith("_"):
            raise AttributeError(name)
        alias_target = KBProxy._ALIASES.get(name)
        if alias_target:
            return getattr(self, alias_target)
        valid_attrs = [a for a in dir(self) if not a.startswith("_")]
        raise AttributeError(
            f"'KBProxy' has no attribute '{name}'. Valid attributes: {valid_attrs}"
        )


class RLMClient:
    """Mock RLM client for proxying agent interface calls."""

    def __init__(self):
        """Initialize RLM client with KB proxy."""
        self.kb = KBProxy()

    def __getattr__(self, name: str):
        # Don't proxy 'kb' through IPCRLMProxy - it's handled in __init__
        if name == "kb":
            return self.kb
        return IPCRLMProxy(name)


class IPCRLMProxy:
    """Proxy for an RLM interface method."""

    def __init__(self, method_name: str):
        """Initialize the RLM method proxy."""
        self.tool_path = f"rlm.{method_name}"

    async def __call__(self, *args, **kwargs):
        req = {"tool": self.tool_path, "args": args, "kwargs": kwargs}
        print(f"<<IPC_REQUEST>>{json.dumps(req)}", flush=True)

        loop = asyncio.get_running_loop()
        response_line = await loop.run_in_executor(None, sys.stdin.readline)

        if not response_line:
            raise EOFError("Host closed connection unexpectedly during IPC.")

        if response_line.startswith("<<IPC_RESPONSE>>"):
            resp_json = response_line.replace("<<IPC_RESPONSE>>", "").strip()
            resp = json.loads(resp_json)
            if resp.get("status") == "success":
                return resp.get("result")

            raise RuntimeError(f"IPC RLM Error: {resp.get('error')}")

        raise RuntimeError(f"Protocol Violation: {response_line[:100]}")


# Initialize Global Clients
mcp_client = IPCClient()
rlm = RLMClient()


async def execute_code(code: str, globals_dict: dict):
    """Compiles and executes code allowing top-level await and returning the last expression."""
    result = None
    try:
        # Parse the code to check for a final expression
        flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        tree = ast.parse(code, mode="exec")

        last_node = tree.body[-1] if tree.body else None
        result = None

        # If the last node is an expression, separating it allows us to return its value
        if last_node and isinstance(last_node, ast.Expr):
            # Remove the last expression from the main body
            last_expr = tree.body.pop()

            # Execute the preceding statements (if any)
            if tree.body:
                # We must re-compile the modified tree
                module_obj = compile(tree, "<input>", "exec", flags=flags)
                # Use eval to capture the coroutine if TOP_LEVEL_AWAIT is used
                res = eval(module_obj, globals_dict)  # nosec
                if inspect.iscoroutine(res):
                    await res

            # Compile the last expression as an 'eval' mode object
            # We must wrap the value node in an Expression object for 'eval' mode
            if hasattr(last_expr, "value"):
                expr_val = ast.Expression(last_expr.value)
                # Fix locations for the new node
                ast.fix_missing_locations(expr_val)

                expr_obj = compile(expr_val, "<input>", "eval", flags=flags)
                result = eval(expr_obj, globals_dict)  # nosec
                # Await if it's a coroutine (from top-level await expression)
                if inspect.iscoroutine(result):
                    result = await result
            else:
                result = None

        else:
            # No final expression (e.g., assignment, def, import), just exec whole block
            code_obj = compile(tree, "<input>", "exec", flags=flags)
            # Use eval to capture the coroutine if TOP_LEVEL_AWAIT is used
            res = eval(code_obj, globals_dict)  # nosec
            if inspect.iscoroutine(res):
                await res
            result = None

        # Output the logical result specifically for validation checks
        if result is not None:
            # We use JSON for clean serialization of primitives
            try:
                # We skip complex objects that aren't JSON serializable
                if isinstance(result, (bool, int, float, str, list, dict, type(None))):
                    print(f"<<RESULT>>{json.dumps(result)}", flush=True)
                else:
                    # Fallback to string representation for complex objects
                    print(f"<<RESULT>>{json.dumps(str(result))}", flush=True)
            except (TypeError, ValueError) as e:
                # Specific catch for JSON serialization issues
                logger.warning("Result serialization failed (JSON error): %s", e)
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Fallback catch for other unexpected errors during result processing
                logger.warning(
                    "Result serialization failed (Unexpected): %s", e, exc_info=True
                )

    except (SyntaxError, NameError, TypeError, ValueError) as e:
        logger.error(
            "Execution Error in code block:\n%s\nError: %s", code, str(e), exc_info=True
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "Unexpected Execution Error in code block:\n%s\nError: %s: %s",
            code,
            type(e).__name__,
            str(e),
            exc_info=True,
        )

    return result


async def kernel_loop():
    """Main loop for the persistent kernel."""
    logger.info("Ready. Waiting for commands on stdin...")

    # Log specific warning if scientific modules are missing
    if not _SCIENTIFIC_MODULES:
        logger.warning(
            "Scientific computing modules (numpy, scipy) not available in kernel"
        )

    # Persistent Globals
    # We populate it with the modules we've imported and our helper clients
    user_globals = globals().copy()
    user_globals.update(_SCIENTIFIC_MODULES)

    user_globals.update(
        {
            "mcp_client": mcp_client,
            "mcp": mcp_client,  # Legacy alias - will be shadowed by 'import mcp'
            "rlm": rlm,
            "kb": rlm.kb,
            "print": builtins.print,
            "asyncio": asyncio,
            "json": json,
            "sys": sys,
        }
    )

    loop = asyncio.get_running_loop()

    while True:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break  # EOF

            line = line.strip()
            if not line:
                continue

            # Command Packet: {"command": "EXECUTE", "code": "...", "discovery": {}, "context": {}}
            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                logger.error("Invalid JSON: %s", line[:100])
                continue

            command = packet.get("command")

            if command == "EXIT":
                logger.info("Exiting kernel.")
                break

            if command == "EXECUTE":
                # Update Discovery Data (Skip if empty/cached)
                new_discovery = packet.get("discovery")
                if new_discovery:
                    _MCP_DISCOVERY.update(new_discovery)

                # Update Context (restore variables if needed, though they persist)
                context_update = packet.get("context", {})
                user_globals.update(context_update)

                code = packet.get("code", "")

                # --- EXECUTION BLOCK ---
                # We assume stdout/stderr capture is handled by the parent process reading pipes
                await execute_code(code, user_globals)

                # Signal Completion
                print("<<EXECUTION_COMPLETE>>", flush=True)
                print("<<EXECUTION_COMPLETE>>", file=sys.stderr, flush=True)

        except EOFError:
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Kernel Loop Error: %s: %s", type(e).__name__, str(e), exc_info=True
            )


if __name__ == "__main__":
    try:
        asyncio.run(kernel_loop())
    except KeyboardInterrupt:
        pass
