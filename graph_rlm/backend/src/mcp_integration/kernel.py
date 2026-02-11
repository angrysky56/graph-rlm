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
import traceback

# Prevent the kernel's directory from shadowing the top-level 'skills' package.
# When running a script, Python prepends the script's directory to sys.path.
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


# --- IPC CLIENT (Mock MCP) ---
class IPCClient:
    """Mock MCP client that proxies tool calls over IPC."""

    def __init__(self):
        """Initialize the IPC client."""

    def __getattr__(self, name):
        return IPCServerProxy(name)

    def __dir__(self):
        return list(_MCP_DISCOVERY.keys())


class IPCServerProxy:
    """Proxy for an MCP server."""

    def __init__(self, server_name: str):
        """Initialize the server proxy."""
        self.server_name = server_name

    def __getattr__(self, name):
        return IPCToolProxy(self.server_name, name)

    def __dir__(self):
        server_data = _MCP_DISCOVERY.get(self.server_name, {})
        return list(server_data.keys())


class IPCToolProxy:
    """Proxy for an MCP tool."""

    def __init__(self, server_name: str, tool_name: str):
        """Initialize the tool proxy and load docstrings from discovery data."""
        self.tool_path = f"mcp.{server_name}.{tool_name}"
        server_data = _MCP_DISCOVERY.get(server_name, {})
        tool_data = server_data.get(tool_name, {})
        self.__doc__ = tool_data.get("doc", "")

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


class RLMClient:
    """Mock RLM client for proxying agent interface calls."""

    def __getattr__(self, name: str):
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
mcp = IPCClient()
rlm = RLMClient()


async def execute_code(code: str, globals_dict: dict):
    """Compiles and executes code allowing top-level await."""
    try:
        # Enable top-level await via AST flag
        flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        code_obj = compile(code, "<input>", "exec", flags=flags)

        # Execute in the persistent globals context
        result = eval(code_obj, globals_dict)  # nosec: B307

        # If the code contained top-level await, eval returns a coroutine
        if inspect.iscoroutine(result):
            await result

    except (SyntaxError, NameError, TypeError, ValueError) as e:
        logger.error("Execution Error: %s", str(e))
        traceback.print_exc()
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected Execution Error: %s", str(e))
        # We don't exit here; we just report the error and keep the kernel alive.
        # The parent runtime detects errors by parsing stderr or exit codes,
        # but here we rely on the textual traceback output.


async def kernel_loop():
    """Main loop for the persistent kernel."""
    logger.info("Ready. Waiting for commands on stdin...")

    # Persistent Globals
    # We populate it with the modules we've imported and our helper clients
    user_globals = globals().copy()
    user_globals.update(
        {
            "mcp": mcp,
            "rlm": rlm,
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
                # Update Discovery Data
                _MCP_DISCOVERY.update(packet.get("discovery", {}))

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
        except Exception as e:  # noqa: BLE001
            logger.error("Kernel Loop Error: %s", str(e))
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(kernel_loop())
    except KeyboardInterrupt:
        pass
