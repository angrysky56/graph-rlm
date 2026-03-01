"""
Runtime environment manager for the isolated agent.
Handles virtual environment resolution, subprocess execution, and IPC.
"""

import asyncio
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Dict, Optional, Tuple

from .client import call_mcp_tool

__all__ = ["AgentRuntime", "call_mcp_tool"]

logger = logging.getLogger(__name__)


@dataclass
class _RuntimeState:
    stop_event: Optional[Event] = None


_state = _RuntimeState()


def get_stop_event() -> Optional[Event]:
    """Returns the global stop event for the runtime."""
    return _state.stop_event


def set_stop_event(event: Event):
    """Sets the global stop event (threading.Event) from the Agent."""
    _state.stop_event = event


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
        # Discovery Cache: session_id_discovery -> float (timestamp or True)
        self._discovery_cache: Dict[str, Any] = {}

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
                # We catch typical process-management errors:
                # - ProcessLookupError: Process already gone
                # - OSError: Permission or other system error
                # - AttributeError: If proc is unexpectedly not a Process object
                except (OSError, ProcessLookupError, AttributeError) as e:
                    logger.debug(
                        "Failed to kill stale kernel process %s: %s",
                        getattr(proc, "pid", "unknown"),
                        e,
                    )
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
            "PYTHONPATH": f"{self.project_root}:{self.backend_root / 'skills'}:{os.environ.get('PYTHONPATH', '')}",
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
        timeout: Optional[float] = None,
    ) -> Tuple[str, str, Any, int]:
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
        except (asyncio.TimeoutError, ConnectionError, RuntimeError, OSError) as e:
            logger.error("Failed to start kernel for session %s: %s", session_id, e)
            return "", f"Failed to start kernel: {e}", None, 1
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Fallback for unexpected errors during session creation
            logger.error(
                "Unexpected error starting kernel session %s: %s", session_id, e
            )
            return "", f"System Error (Session Start): {e}", None, 1

        if not process.stdin or not process.stdout or not process.stderr:
            return "", "System Error: Kernel process streams are missing.", None, 1

        # 2. Prepare MCP Discovery Data (With Caching)
        discovery_data = {}
        if mcp_namespace:
            # We cache discovery per-session to avoid giant payloads on every call
            cache_key = f"{session_id}_discovery"
            if cache_key in self._discovery_cache:
                discovery_data = {}  # Signal to kernel: nothing new
            else:
                try:
                    server_names = [
                        n
                        for n in dir(mcp_namespace)
                        if not n.startswith("_")
                        and n
                        not in (
                            "CallToolRequest",
                            "ClientCapabilities",
                            "ClientNotification",
                            "ClientRequest",
                            "ClientResult",
                            "ClientSession",
                            "CompleteRequest",
                            "CreateMessageRequest",
                            "CreateMessageResult",
                            "ErrorData",
                            "GetPromptRequest",
                            "GetPromptResult",
                            "Implementation",
                            "IncludeContext",
                            "InitializeRequest",
                            "InitializeResult",
                            "InitializedNotification",
                            "JSONRPCError",
                            "JSONRPCRequest",
                            "JSONRPCResponse",
                            "ListPromptsRequest",
                            "ListPromptsResult",
                            "ListResourcesRequest",
                            "ListResourcesResult",
                            "ListToolsResult",
                            "LoggingLevel",
                            "LoggingMessageNotification",
                            "McpError",
                            "Notification",
                            "PingRequest",
                            "ProgressNotification",
                            "PromptsCapability",
                            "ReadResourceRequest",
                            "ReadResourceResult",
                            "Resource",
                            "ResourceUpdatedNotification",
                            "ResourcesCapability",
                            "RootsCapability",
                            "SamplingMessage",
                            "SamplingRole",
                            "ServerCapabilities",
                            "ServerNotification",
                            "ServerRequest",
                            "ServerResult",
                            "ServerSession",
                            "SetLevelRequest",
                            "StdioServerParameters",
                            "StopReason",
                            "SubscribeRequest",
                            "Tool",
                            "ToolsCapability",
                            "UnsubscribeRequest",
                            "client",
                            "server",
                            "shared",
                            "stdio_client",
                            "stdio_server",
                            "types",
                            "ClientSessionGroup",
                            "CreateMessageResultWithTools",
                            "SamplingCapability",
                            "SamplingContent",
                            "SamplingContextCapability",
                            "SamplingMessageContentBlock",
                            "SamplingToolsCapability",
                            "ToolChoice",
                            "ToolResultContent",
                            "ToolUseContent",
                            "UrlElicitationRequiredError",
                            "os",
                        )
                    ]
                    for srv_name in server_names:
                        try:
                            # We only care about objects that look like server namespaces or modules
                            srv_obj = getattr(mcp_namespace, srv_name)
                            if not hasattr(srv_obj, "__dir__") and not inspect.ismodule(
                                srv_obj
                            ):
                                continue

                            tools = {}
                            for tool_name in dir(srv_obj):
                                if tool_name.startswith("_"):
                                    continue
                                try:
                                    tool_obj = getattr(srv_obj, tool_name)
                                    # [Optimization] Skip heavy docstrings if not first run
                                    doc = getattr(tool_obj, "__doc__", "") or ""
                                    tools[tool_name] = {"doc": doc}
                                except AttributeError:
                                    continue
                            discovery_data[srv_name] = tools
                        except AttributeError:
                            continue

                    self._discovery_cache[cache_key] = True
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

        # 4. Construct Message
        try:
            msg = json.dumps(packet) + "\n"
        except Exception as e:
            return "", f"Serialization Error: {e}", None, 1

        # 5. Read Output Loop (Closure)
        async def read_stream(stream, is_stdout=False) -> str:
            acc = []
            while True:
                # Safeguard: Ensure streams are still open
                if not process.stdout or not process.stderr:
                    break

                line_bytes = await stream.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode()
                clean_line = line.strip()

                if clean_line == "<<EXECUTION_COMPLETE>>":
                    break

                if is_stdout and line.startswith("<<RESULT>>"):
                    try:
                        res_json = line.replace("<<RESULT>>", "").strip()
                        context["_last_result"] = json.loads(res_json)
                    except Exception as e:
                        logger.warning("Failed to parse result JSON from kernel: %s", e)
                elif is_stdout and line.startswith("<<IPC_REQUEST>>"):
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
                                pass
                else:
                    acc.append(line)
            return "".join(acc)

        # 6. Execute with Deadlock Protection
        # We start reading BEFORE we finish writing/draining to ensure the
        # kernel doesn't block on writing back to us while we are still writing to it.
        try:
            # Create read tasks
            stdout_task = asyncio.create_task(
                read_stream(process.stdout, is_stdout=True)
            )
            stderr_task = asyncio.create_task(
                read_stream(process.stderr, is_stdout=False)
            )

            # Send to Kernel
            process.stdin.write(msg.encode())
            await process.stdin.drain()

            # Wait for output
            if timeout:
                try:
                    stdout_data, stderr_data = await asyncio.wait_for(
                        asyncio.gather(stdout_task, stderr_task), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    stdout_task.cancel()
                    stderr_task.cancel()
                    logger.warning("Kernel execution timeout after %ss", timeout)
                    return (
                        "",
                        f"Execution Timeout Error: Process exceeded {timeout}s",
                        None,
                        124,
                    )
            else:
                stdout_data, stderr_data = await asyncio.gather(
                    stdout_task, stderr_task
                )

            if process.returncode is not None:
                return (
                    stdout_data,
                    stderr_data + "\nKernel Process Exited Unexpectedly.",
                    None,
                    1,
                )

            # Extract result
            exec_result = context.get("_last_result")
            return stdout_data, stderr_data, exec_result, 0

        except Exception as e:
            logger.error("Runtime Exception: %s", str(e))
            return "", f"Runtime Error: {e}", None, 1

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
                except AttributeError:
                    # Fallback: treat as a skill invocation via run_skill
                    logger.info(
                        "IPC: '%s' not a native RLM method, routing to run_skill.",
                        func_name,
                    )
                    func = rlm.run_skill
                    # Rewrite args so run_skill receives (name, args)
                    kwargs = {"name": func_name, "args": dict(kwargs) if kwargs else {}}
                    args = []

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
