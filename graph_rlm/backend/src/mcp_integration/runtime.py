"""
Runtime environment manager for the isolated agent.
Handles virtual environment resolution, subprocess execution, and IPC.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .client import call_mcp_tool

__all__ = ["AgentRuntime", "call_mcp_tool"]

logger = logging.getLogger(__name__)

_THREADING_STOP_EVENT = None


def get_stop_event():
    """Returns the global stop event for the runtime."""
    return _THREADING_STOP_EVENT


def set_stop_event(event):
    """Sets the global stop event (threading.Event) from the Agent."""
    global _THREADING_STOP_EVENT
    _THREADING_STOP_EVENT = event


class AgentRuntime:
    """
    Manages the 'agent_venv' execution context.
    Uses 'uv run' to guarantee isolation from the host process.
    """

    def __init__(self, project_root: Path):
        """
        Initialize the Runtime Manager.

        Args:
            project_root: The root of the repository (e.g. graph-rlm/)
                          Expects 'backend/agent_venv' to exist relative to this.
        """
        self.project_root = project_root

        # Resolve 'backend' logic
        # If project_root is 'graph_rlm', then 'backend' is immediate child?
        # Let's try to find it.
        if (project_root / "backend").exists():
            self.backend_root = project_root / "backend"
        elif (project_root / "graph_rlm" / "backend").exists():
            self.backend_root = project_root / "graph_rlm" / "backend"
        elif (project_root / "src").exists():
            # Fallback if provided root is actually backend
            self.backend_root = project_root
        else:
            # Assume we are in backend?
            self.backend_root = project_root

        self.agent_venv = self.backend_root / "agent_venv"
        self.python_exe = self._get_venv_python()
        # Persistent Session Map: session_id -> Process
        self.sessions: Dict[str, Any] = {}

    def _get_venv_python(self) -> Path:
        """Locates the Python executable within the agent's venv."""
        if sys.platform == "win32":
            exe = self.agent_venv / "Scripts" / "python.exe"
        else:
            exe = self.agent_venv / "bin" / "python"

        if not exe.exists():
            logger.warning(
                "Agent Venv Python not found at %s. Checking alternates...", exe
            )

        return exe

    async def _ensure_session(self, session_id: str) -> Any:
        """
        Ensures a persistent kernel process is running for the session.
        Handles event loop mismatch validation.

        Args:
            session_id: The ID of the session to ensure.

        Returns:
            The running asyncio.subprocess.Process.
        """
        current_loop = asyncio.get_running_loop()

        if session_id in self.sessions:
            # Check for existing session
            session_data = self.sessions[session_id]

            # Handle migration: support both old (Process) and new (dict) formats during update
            # The new format is a dict. If it's not a dict, it must be the old Process object.
            if not isinstance(session_data, dict):
                proc = session_data
                proc_loop = getattr(proc, "_loop", None)
            else:
                proc = session_data["process"]
                proc_loop = session_data["loop"]

            # Validate Loop Integrity
            # StreamReader objects are bound to the loop they were created in.
            # We cannot reuse a process created in a different loop even if we wanted to.
            if proc_loop is not current_loop:
                logger.warning(
                    "Session %s attached to closed/different loop (%s vs %s). Restarting kernel.",
                    session_id,
                    getattr(proc_loop, "_thread_id", "unknown"),
                    getattr(current_loop, "_thread_id", "unknown"),
                )
                # We can't await proc.terminate() if it's on a different loop safely,
                # but we should drop the reference.
                try:
                    proc.kill()
                except Exception:
                    pass
                del self.sessions[session_id]
            elif proc.returncode is not None:
                # Process died, restart
                logger.warning("Session %s died. Restarting kernel.", session_id)
                del self.sessions[session_id]
            else:
                return proc

        # Start new kernel
        kernel_script = self.backend_root / "src" / "mcp_integration" / "kernel.py"
        if not kernel_script.exists():
            # Dev/Test fallback path
            kernel_script = self.backend_root / "mcp_integration" / "kernel.py"

        cmd = [str(self.python_exe), str(kernel_script)]

        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{self.project_root}:{os.environ.get('PYTHONPATH', '')}",
        }

        logger.info("Starting Kernel for Session %s: %s", session_id, cmd)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self.agent_venv),
        )

        # Store process along with the loop it belongs to
        self.sessions[session_id] = {"process": process, "loop": current_loop}
        return process

    async def execute(
        self,
        code: str,
        context: Dict[str, Any],
        mcp_namespace: Optional[Any] = None,
    ) -> Tuple[str, str, int]:
        """
        Runs code in a persistent kernel.
        Args:
            code: Python code
            context: Variables to inject (must contain 'session_id' for persistence)
            mcp_namespace: Tool provider
        """
        session_id = str(context.get("session_id", "default_session"))

        # 1. Get/Start Kernel
        try:
            process = await self._ensure_session(session_id)
        except Exception as e:
            return "", f"Failed to start kernel: {e}", 1

        if not process.stdin or not process.stdout or not process.stderr:
            return "", "System Error: Kernel process streams are missing.", 1

        # 2. Prepare MCP Discovery Data
        discovery_data = {}
        if mcp_namespace:
            try:
                server_names = dir(mcp_namespace)
                for srv_name in server_names:
                    try:
                        srv_obj = getattr(mcp_namespace, srv_name)
                        tools = {}
                        for tool_name in dir(srv_obj):
                            try:
                                tool_obj = getattr(srv_obj, tool_name)
                                doc = getattr(tool_obj, "__doc__", "") or ""
                                tools[tool_name] = {"doc": doc}
                            except AttributeError:
                                # Skip items that don't behave like tools
                                continue
                        discovery_data[srv_name] = tools
                    except AttributeError:
                        # Skip items that don't behave like servers
                        continue
            except Exception as e:
                logger.warning("Failed to extract MCP discovery data: %s", e)

        # 3. Construct Command Packet
        packet = {
            "command": "EXECUTE",
            "code": code,
            "context": {
                k: v
                for k, v in context.items()
                if isinstance(v, (str, int, float, bool, list, dict))
            },
            "discovery": discovery_data,
        }

        # 4. Send to Kernel
        try:
            msg = json.dumps(packet) + "\n"
            process.stdin.write(msg.encode())
            await process.stdin.drain()
        except Exception as e:
            logger.error("Failed to send to kernel: %s", str(e))
            self.sessions.pop(session_id, None)
            return "", f"Kernel Communication Error: {e}", 1

        # 5. Read Output Loop
        async def read_stream(stream, is_stdout=False) -> str:
            acc = []
            while True:
                # Safeguard: Ensure streams are still open
                if not process.stdout or not process.stderr:
                    raise RuntimeError("Process streams are closed unexpectedly.")

                line_bytes = await stream.readline()
                if not line_bytes:
                    # EOF reached, stream closed.
                    break

                line = line_bytes.decode()
                clean_line = line.strip()

                if clean_line == "<<EXECUTION_COMPLETE>>":
                    break

                if is_stdout and line.startswith("<<IPC_REQUEST>>"):
                    try:
                        req_json = line.replace("<<IPC_REQUEST>>", "").strip()
                        req = json.loads(req_json)
                        await self._handle_ipc_request(process, req, mcp_namespace)
                    except Exception as e:
                        logger.error("IPC Error: %s", str(e))
                        err = {"status": "error", "error": str(e)}
                        if process.stdin:
                            try:
                                process.stdin.write(
                                    f"<<IPC_RESPONSE>>{json.dumps(err)}\n".encode()
                                )
                                await process.stdin.drain()
                            except (OSError, BrokenPipeError):
                                # If writing the error back fails, the kernel is likely dead.
                                # We ignore this to avoid crashing the host during cleanup.
                                pass
                else:
                    acc.append(line)
            return "".join(acc)

        try:
            stdout_data, stderr_data = await asyncio.gather(
                read_stream(process.stdout, is_stdout=True),
                read_stream(process.stderr, is_stdout=False),
            )

            if process.returncode is not None:
                return (
                    stdout_data,
                    stderr_data + "\nKernel Process Exited Unexpectedly.",
                    1,
                )

            return stdout_data, stderr_data, 0

        except Exception as e:
            logger.error("Runtime Exception: %s", str(e))
            return "", f"Runtime Error: {e}", 1

    async def _handle_ipc_request(
        self, process, req: Dict, mcp_namespace: Optional[Any] = None
    ):
        """
        Executes the requested tool on the HOST using the provided MCP namespace.
        """
        tool_name = req.get("tool")
        args = req.get("args", [])
        kwargs = req.get("kwargs", {})

        logger.info("IPC Tool Request: %s", tool_name)

        try:
            if not tool_name:
                raise ValueError("Tool name is empty")

            # Validate Format: mcp.server.tool OR rlm.tool
            if tool_name.startswith("rlm."):
                # Handle RLM Direct Calls
                rlm_interface = getattr(mcp_namespace, "_rlm_interface", None)
                if not mcp_namespace or not rlm_interface:
                    raise RuntimeError("RLM Interface not available for IPC.")

                func_name = tool_name.split(".")[1]
                rlm = rlm_interface

                try:
                    func = getattr(rlm, func_name)
                except AttributeError as e:
                    raise ValueError(
                        f"Method '{func_name}' not found on RLM interface."
                    ) from e

                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

            elif tool_name.startswith("mcp."):
                parts = tool_name.split(".")
                if len(parts) < 3:
                    raise ValueError(f"Invalid tool name format: {tool_name}")

                if mcp_namespace:
                    # Navigate the namespace: mcp -> server -> tool
                    # parts[0] is 'mcp'
                    server_name = parts[1]
                    func_name = parts[2]

                    try:
                        server_obj = getattr(mcp_namespace, server_name)
                        func = getattr(server_obj, func_name)
                    except AttributeError as e:
                        raise ValueError(
                            f"Tool not found in MCP namespace: {tool_name}"
                        ) from e

                    # Execute
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                else:
                    raise RuntimeError(
                        "MCP Namespace not provided to Runtime. Cannot resolve tools."
                    )
            else:
                raise ValueError(
                    f"Invalid tool name: {tool_name} (Must start with 'mcp.' or 'rlm.')"
                )

            # Serialize Result
            if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
                result = str(result)

            resp = {"status": "success", "result": result}

        except Exception as e:
            logger.error("IPC Tool Execution Failed: %s", e)
            resp = {"status": "error", "error": str(e)}

        # Send Response
        msg = f"<<IPC_RESPONSE>>{json.dumps(resp)}\n"
        if process and process.stdin:
            try:
                process.stdin.write(msg.encode())
                await process.stdin.drain()
            except BrokenPipeError:
                pass

    def _indent_code(self, code: str) -> str:
        """Indents code for the wrapper."""
        return "\n".join("    " + line for line in code.splitlines())
