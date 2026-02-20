"""
Recursive Logic Machine (RLM) Agent.
Handles the core execution loop, recursive querying, and tool integration.
"""

import asyncio
import datetime
import hashlib
import importlib.util
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import httpx

from ..mcp_integration.runtime import AgentRuntime, set_stop_event
from ..mcp_integration.skill_storage import get_axioms_manager, get_skills_manager
from .circuit import (
    CircuitOpenError,
    generate_correlation_id,
    get_correlation_id,
)
from .config import settings
from .context_index import context_index
from .db import GraphClient, db
from .dream import dreamer
from .exceptions import ValidationError
from .exceptions.codes import ErrorCode
from .llm import llm
from .logger import get_logger
from .mcp_runtime import is_mcp_available
from .navigator import Navigator
from .omcd import omcd
from .prompts import build_system_prompt
from .reflexion import intelli_synth
from .repe import repe
from .rlm_interface import RLMInterface
from .scratchpad_builder import scratchpad_builder
from .services.circuit import protected_llm_generate
from .sheaf import sheaf
from .state import (
    ExecutionState,
    agent_state,
    broadcast_trace,
    execution_events,
)
from .thimac_memory import ThimacMemory
from .trace import register_monitor, trace_action

if TYPE_CHECKING:
    from graph_rlm.backend.src.mcp_integration.skill_storage import SkillsManager


# Skills System
def is_skills_available():
    """Defensive check for Skills system availability."""
    return (
        importlib.util.find_spec("graph_rlm.backend.src.mcp_integration.skill_storage")
        is not None
        or importlib.util.find_spec("mcp_integration.skill_storage") is not None
    )


logger = get_logger("graph_rlm.agent")


def validate_agent_prompt(prompt: str, max_length: int = 100000) -> None:
    """Validate agent prompt input.

    Args:
        prompt: The prompt to validate.
        max_length: Maximum allowed prompt length.

    Raises:
        ValidationError: If prompt is invalid.
    """
    if not prompt or not prompt.strip():
        raise ValidationError(
            message="Prompt cannot be empty",
            error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
            field="prompt",
            constraint="non_empty",
        )

    if len(prompt) > max_length:
        raise ValidationError(
            message=f"Prompt exceeds maximum length of {max_length} characters",
            error_code=ErrorCode.VALIDATION_VALUE_OUT_OF_RANGE,
            field="prompt",
            constraint=f"length <= {max_length}",
            actual_length=len(prompt),
        )


def validate_session_id(session_id: str) -> None:
    """Validate session ID format.

    Args:
        session_id: The session ID to validate.

    Raises:
        ValidationError: If session_id is invalid.
    """
    if not session_id or not isinstance(session_id, str):
        raise ValidationError(
            message="Session ID must be a non-empty string",
            error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
            field="session_id",
            constraint="non_empty_string",
        )

    # UUID format check (session IDs should be UUIDs)
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )
    if not uuid_pattern.match(session_id):
        raise ValidationError(
            message="Session ID must be a valid UUID",
            error_code=ErrorCode.VALIDATION_FIELD_INVALID,
            field="session_id",
            constraint="uuid_format",
        )


# Register the monitor
trace_action(context="AGENT", action="Initializing Trace Monitor...", level="debug")

register_monitor(broadcast_trace)


class Agent:
    """
    The core Recursive Logic Machine (RLM) agent.
    Handles the main execution loop, epistemic health checks, and dreamer integration.
    Manages REPL sessions, graph memory, and tool execution.
    """

    skills_manager: Optional["SkillsManager"]

    def __init__(self):
        # NOTE: nest_asyncio removed - incompatible with uvloop
        # Code flow is properly async (async def + await) so not needed.

        # --- CORE INITIALIZATION ---
        self.db: GraphClient = db
        self.llm = llm

        # [SECURITY] Process Isolation via uv
        project_root = Path(__file__).resolve().parents[4]
        # We also need backend root for path resolution if needed
        # But AgentRuntime handles it.

        self.runtime = AgentRuntime(project_root)

        # Legacy REPL Manager removed in favor of strict isolation
        # Sub-agents share the same REPL manager but have their own execution state
        self.active_repls: Dict[str, str] = {}

        # Navigator for Intelligent Curiosity
        self.navigator = Navigator(sheaf_monitor=sheaf)

        # Morphological Memory (Neural Cellular Automata)
        self.morph_memory = ThimacMemory()

        self.execution_logs: Dict[str, list] = {}  # session_id -> [tool_ident, ...]
        self.session_cache: Dict[str, Any] = {}  # For Sheaf Monitor & shared state
        self.current_task_input: Optional[str] = (
            None  # Tracks active goal for Sheaf Teleology
        )
        self.global_stop_event = (
            threading.Event()
        )  # Shared event for cross-thread stopping

        # --- STATE INITIALIZATION (Linter safety) ---
        self.last_dream_insight: Optional[str] = None
        self._dreamer_retry_count: int = 0  # Counter for repeated insight escalation
        self._validation_retries: int = 0  # Counter for validation attempt loops
        self.stop_requested: bool = False
        self.final_result: Optional[str] = None
        self.last_rejected_result: Optional[str] = None
        self._final_output_emitted: bool = False
        self.synthesis_triggered: bool = False
        self.awaiting_validation: bool = False  # Set by rlm.done() for Dreamer pipeline
        self.step_id: int = 0
        self.current_turn: int = 1
        self.current_thought_id: Optional[str] = None

        # === EVALUATION COUNTERS ===
        # Track success/failure for session-level and global metrics
        self.eval_success_count: int = 0  # Successful task completions
        self.eval_failure_count: int = 0  # Failed tasks (errors, timeouts)
        self.eval_step_count: int = 0  # Total steps executed
        self.eval_dreamer_interventions: int = 0  # Dreamer correction count

        if is_skills_available():
            self.skills_manager = get_skills_manager()
        else:
            self.skills_manager = None

        # Ensure Knowledge Base scaffolding exists
        self._ensure_kb_structure()

        # Environment Strategy:
        # 1. Core Agent: Runs in host environment.
        # 2. Code Execution: Runs in ISOLATED 'agent_venv' via AgentRuntime (uv).
        # 3. MCP Servers: Run in their own independent environments.
        logger.info("Agent initialized with Strict Process Isolation (AgentRuntime)")
        logger.info("RepE Safety Layer & Sheaf Topology Monitor Loaded.")

    def _ensure_kb_structure(self):
        """Creates the Knowledge Base directory structure if it doesn't exist."""
        try:
            kb_root = Path(settings.KNOWLEDGE_BASE_PATH)

            # Subfolders referenced in System Prompt
            subfolders = ["plans", "reports", "outputs", "axioms", "workspace"]

            for sub in subfolders:
                path = kb_root / sub
                path.mkdir(parents=True, exist_ok=True)

            # Create a simple README if empty to guide users
            readme = kb_root / "README.md"
            if not readme.exists():
                readme.write_text(
                    "# Agent Knowledge Base\n\n"
                    "- `axioms/`: Report on any axiom issues (CAG).\n"
                    "- `plans/`: Implementation plans and architectural docs.\n"
                    "- `reports/`: Deep research Reports.\n"
                    "- `outputs/`: General Final deliverables.\n"
                    "- `workspace/`: Agent Freeform workspace. Create Your Own Project Folders Here.\n"
                )

        except (OSError, AttributeError) as e:
            logger.warning("Failed to verify Knowledge Base structure: %s", e)

    async def _generate_axiom_search_query(self, prompt: str) -> str:
        """
        Generates a targeted search query for axioms using the LLM.
        Translates 'User Task' -> 'Governance/Validation Requirements'.
        """
        try:
            # 1. Fast Regex Sanitzation (Fallback)
            clean_text = re.sub(r"```[\s\S]*?```", "", prompt)
            clean_text = clean_text.replace("`", "")
            clean_text = " ".join(clean_text.split())[:300]

            # 2. LLM Transformation (if available)
            # We want keywords that match Axiom filenames/descriptions
            system_prompt = (
                "You are the Governance Module. Translate the USER TASK into a SEARCH QUERY for Validation Rules (Axioms). "
                "Axioms are Python validators for domains like: 'file persistence', 'math safety', 'python syntax', 'epistemic integrity', 'security'. "
                "Return ONLY a comma-separated list of relevant domains and technical keywords."
            )

            # Fast/Cheap call (do not stream)
            query = await protected_llm_generate(
                prompt[:1000],  # partial prompt context
                system=system_prompt,
                max_tokens=60,
            )

            if query and "Error:" not in query:
                return f"{query}, {clean_text}"

            return clean_text

        except (RuntimeError, ValueError, httpx.RequestError) as e:
            logger.warning("Axiom query generation failed: %s", e)
            return re.sub(r"```[\s\S]*?```", "", prompt)[:300]

    async def _handle_llm_circuit_open(self, error: CircuitOpenError) -> str:
        """Handle LLM circuit open with graceful degradation.

        Called when circuit breaker is open and LLM service is unavailable.
        Provides fallback behavior to keep agent operational.

        Args:
            error: The CircuitOpenError that was raised.

        Returns:
            Fallback response string for the agent to use.
        """
        correlation_id = error.correlation_id or get_correlation_id()

        # Log with full context for debugging
        logger.error(
            "llm_service_degraded",
            extra={
                "correlation_id": correlation_id,
                "circuit": error.circuit_name or "llm",
                "message": error.message,
            },
        )

        # Emit event for user feedback if emit_event exists
        if hasattr(self, "emit_event"):
            self.emit_event(
                "error",
                content="AI service is experiencing high demand. "
                "Continuing with limited capabilities.",
            )

        # Return graceful fallback - agent continues with reduced capabilities
        return (
            "AI service temporarily unavailable. "
            "Attempting to continue with cached knowledge."
        )

    # --- SESSION-ISOLATED PROPERTIES ---
    def get_state(self) -> ExecutionState:
        """Retrieves or initializes the execution state for the current session."""
        state = agent_state.get()
        if state is None:
            # Fallback for out-of-session access (e.g. CLI or background tasks)
            state = ExecutionState()
            agent_state.set(state)
        return state

    @property
    def final_result(self) -> Optional[str]:
        """Provides access to the final result string stored in the current
        execution state."""
        return self.get_state().final_result

    @final_result.setter
    def final_result(self, value: Optional[str]):
        self.get_state().final_result = value

    @property
    def stop_requested(self) -> bool:
        """Checks if a stop signal has been issued for the current session or globally."""
        # Check both local context AND global signal
        return self.get_state().stop_requested or self.global_stop_event.is_set()

    @stop_requested.setter
    def stop_requested(self, value: bool):
        self.get_state().stop_requested = value

    @property
    def synthesis_triggered(self) -> bool:
        """Indicates if a synthesis operation has been triggered for the current execution."""
        return self.get_state().synthesis_triggered

    @synthesis_triggered.setter
    def synthesis_triggered(self, value: bool):
        self.get_state().synthesis_triggered = value

    @property
    def current_thought_id(self) -> Optional[str]:
        """Gets the ID of the current thought/task node being processed."""
        return self.get_state().current_thought_id

    @current_thought_id.setter
    def current_thought_id(self, value: Optional[str]):
        self.get_state().current_thought_id = value

    @property
    def current_depth(self) -> int:
        """Retrieves the current recursion depth of the agent's logic machine."""
        return self.get_state().depth

    @current_depth.setter
    def current_depth(self, value: int):
        self.get_state().depth = value

    def record_turn(self, turn_id: int):
        """Update current turn and reset context-specific tracking if needed."""
        self.current_turn = turn_id

    async def _install_to_active_env(self, package_name: str) -> str:
        """Internal helper to install a package into the CURRENT active environment."""

        logger.info(
            "Agent requesting installation of package: %s into Active Env",
            package_name,
        )
        self.emit_event(
            "token",
            content=f"\n🛠️ [Self-Healing] Installing package '{package_name}'...",
        )

        # Use the running python executable to ensure installed packages are visible to this process
        python_exe = sys.executable

        try:
            cmd = [str(python_exe), "-m", "pip", "install", package_name]
            if shutil.which("uv"):
                # Use uv if available for speed, targeting the system python
                cmd = [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python_exe),
                    package_name,
                ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info("Successfully installed %s", package_name)
                self.emit_event("token", content=" ✅ Installation successful.\n")
                return f"Successfully installed {package_name}\n{stdout.decode()}"
            else:
                stderr_text = stderr.decode()
                logger.error("Failed to install %s: %s", package_name, stderr_text)
                self.emit_event("error", content=f"Installation failed: {stderr_text}")
                return f"Failed to install {package_name}\nError: {stderr_text}"
        except (OSError, subprocess.SubprocessError) as e:
            logger.error("Installation error (system/subprocess): %s", e)
            return f"Installation error (system): {e}"

    def _install_to_agent_venv(self, package_name: str) -> str:
        """Internal helper to install a package into the DEDICATED AGENT VENV."""

        # Resolve agent_venv path relative to this file
        # __file__ = backend/src/core/agent.py
        # root = backend/
        backend_root = Path(__file__).parent.parent.parent
        agent_venv_path = backend_root / "agent_venv"

        # Determine python executable in venv
        if sys.platform == "win32":
            python_exe = agent_venv_path / "Scripts" / "python.exe"
        else:
            python_exe = agent_venv_path / "bin" / "python"

        if not python_exe.exists():
            return (
                f"Error: Agent Venv not found at {agent_venv_path}. "
                "Cannot install skill dependencies."
            )

        logger.info(
            "Agent requesting installation of package: %s into AGENT ENV (%s)",
            package_name,
            agent_venv_path,
        )
        self.emit_event(
            "thinking",
            content=f"\n📦 Agent: Installing '{package_name}' into Skill/Agent "
            "Environment...",
        )

        try:
            # Use uv if available, targeting the venv python
            if shutil.which("uv"):
                cmd = [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python_exe),
                    package_name,
                ]
            else:
                # Fallback to direct pip invocation in venv
                cmd = [str(python_exe), "-m", "pip", "install", package_name]

            # trunk-ignore(bandit/B603)
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                logger.info("Successfully installed %s in Agent Venv", package_name)
                self.emit_event("thinking", content="  -> Installation successful.")
                return f"Successfully installed {package_name}\n{result.stdout}"
            else:
                logger.error("Failed to install %s: %s", package_name, result.stderr)
                self.emit_event(
                    "error", content=f"Installation failed: {result.stderr}"
                )
                return f"Failed to install {package_name}\nError: {result.stderr}"
        except (OSError, subprocess.SubprocessError) as e:
            logger.error("Installation error (venv/subprocess): %s", e)
            return f"Installation error (venv): {e}"

    async def install_package(self, package_name: str) -> str:
        """Installs a package into the active environment (REPL compatibility)."""
        return await self._install_to_active_env(package_name)

    def install_skill_package(self, package_name: str) -> str:
        """Installs a package into the AGENT environment (Skill compatibility)."""
        return self._install_to_agent_venv(package_name)

    def read_skill(self, name: str) -> str:
        """Reads the source code of a compiled skill."""
        if not is_mcp_available():
            return "Error: MCP/Skills system not available."

        self.emit_event(
            "thinking", content=f"\n📖 Agent: Reading skill '{name}' source..."
        )
        try:
            mgr = get_skills_manager()
            skill = mgr.get_skill(name)
            if not skill:
                self.emit_event("error", content=f"Skill '{name}' not found.")
                return f"Error: Skill '{name}' not found."
            return skill["code"]
        except (AttributeError, KeyError, OSError, RuntimeError) as e:
            self.emit_event(
                "error", content=f"Error reading skill (state/io error): {e}"
            )
            return f"Error reading skill: {e}"

    async def _refresh_scratchpad(
        self,
        session_id: str,
        root_session_id: str,
        task: str,
        current_step: int,
        max_steps: int,
        current_round_id: str,
        morph_gestalt: Optional[str] = None,
    ) -> str:
        """
        Rebuild the scratchpad — the stateless agent's only memory.

        Called after every state-changing event so both agent and Dreamer
        always see the latest session context (turns, steps, REPL IDs).
        """
        try:
            current_repl_id = self.active_repls.get(session_id)
            pad = await scratchpad_builder.build_scratchpad(
                session_id=session_id,
                root_session_id=root_session_id,
                task=task,
                current_step=current_step,
                max_steps=max_steps,
                current_round_id=current_round_id,
                morph_gestalt=morph_gestalt,
                current_repl_id=current_repl_id,
            )
            self.emit_event("scratchpad_text", content=pad, is_internal=True)
            return pad
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Scratchpad refresh failed: %s", e)
            return f"Error: Scratchpad unavailable ({e})"

    def _sync_thimac(
        self,
        thought_id: str,
        prompt: str,
        status: str,
        result: Optional[str],
        step: int,
        repl_id: Optional[str] = None,
        logical_id: Optional[str] = None,
    ):
        """Helper to ingest a thought node into Thimac memory for ontology tracking."""
        try:
            thimac_thought_data = {
                "id": thought_id,
                "prompt": prompt,
                "status": status,
                "result": result,
                "created_at": int(
                    datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000
                ),
                "turn_id": self.current_turn,
                "step_id": step,
                "repl_id": repl_id,
                "logical_id": logical_id,
                "execution_summary": None,
            }
            self.morph_memory.ingest_thought(thimac_thought_data)
        except (AttributeError, ValueError, TypeError, KeyError) as e:
            logger.error(
                "Thimac ingestion failed for thought %s: %s",
                thought_id,
                e,
                exc_info=True,
            )

    def _create_system_node(
        self,
        logical_id: str,
        summary: str,
        parent_id: Optional[str] = None,
        status: str = "system",
        session_id: str = "unknown",
        root_session_id: str = "unknown",
        round_id: str = "unknown",
        turn_id: int = 1,
        step_id: int = 1,
        repl_id: str = "SYS",
        result: Optional[str] = None,
        analysis: Optional[Dict] = None,
        validate: bool = False,
    ):
        """Standardized helper for materializing system-level reasoning in the graph."""
        thought_id = "unknown"
        try:
            thought_id = str(uuid.uuid4())
            self.db.create_thought_node(
                thought_id=thought_id,
                prompt=summary,
                logical_id=logical_id,
                result=result,
                parent_id=parent_id or self.current_thought_id,
                status=status,
                session_id=session_id,
                root_session_id=root_session_id,
                round_id=round_id,
                turn_id=turn_id,
                step_id=step_id,
                repl_id=repl_id,
                dreamer_analysis=json.dumps(analysis) if analysis else None,
                validate=validate,
            )

            # Sync system node to Thimac
            self._sync_thimac(
                thought_id=thought_id,
                prompt=summary,
                status=status,
                result=result,
                step=step_id,
                repl_id=repl_id,
                logical_id=logical_id,
            )
        except (AttributeError, RuntimeError, KeyError, TypeError, ValueError) as e:
            logger.error(
                "Failed to create system node %s (LID: %s): %s",
                thought_id,
                logical_id,
                e,
            )

    def emit_event(
        self,
        event_type: str,
        data: Any = None,
        content: Optional[str] = None,
        code: Optional[str] = None,
        is_sub_event: bool = False,
        tag: Optional[str] = None,
        is_internal: bool = False,
    ):
        """
        Helper to emit events to the current context's queue if it exists.
        Also mirrors key events to the server logs (terminal) for visibility.
        """
        # [PROTOCOL] Determine the UI destination based on event type
        # FIX: Route ALL agent/dreamer LLM output to chat area for user visibility
        ui_target = "TERMINAL_RAW"  # Default for internal/system logs

        # Internal events stay in terminal only, not shown in chat
        if is_internal:
            ui_target = "TERMINAL_RAW"
        elif event_type in ["code_output", "code_output_chunk"]:
            ui_target = "CODE_RESULT"
        elif event_type in [
            "answer",
            "final_answer",
            "RLM_FINAL_RESPONSE",
            "RLM_INITIAL_RESPONSE",
            "RLM_DREAMER_ISSUES",
            "RLM_DREAMER_VALIDATED",
            "RLM_FINAL_OUTPUT",
            "RLM_AGENT_TASK_PLAN",
            "warning",
            "error",
            "thought",
            "synthesis",
            "tool_output",  # FIX: Allow tool outputs to appear in chat
        ]:
            # All important agent/dreamer output goes to chat
            ui_target = "CHAT_RESPONSE"
        elif event_type == "thinking":
            # Agent/Dreamer LLM thoughts and meta-cognition go to chat
            ui_target = "CHAT_RESPONSE"
        elif (
            event_type == "graph_update"
            and content
            and ("Axiom" in content or "Critique" in content)
        ):
            # Show axiom-related graph updates in chat too for visibility
            ui_target = "CHAT_RESPONSE"

        prefix = "↳ " if is_sub_event else ""

        # Mirror to Terminal/Logs
        repl_id = data.get("repl_id") if data and isinstance(data, dict) else None
        repl_str = f" [{repl_id}]" if repl_id else ""

        if event_type == "thinking" and content:
            # Use tag if available for better log mirroring
            log_prefix = f"[THINKING] [{tag}]" if tag else "[THINKING]"
            logger.info("%s%s%s %s", prefix, log_prefix, repl_str, content.strip())
        elif event_type == "code_output" and content:
            logger.info("%s%s[REPL OUTPUT] >>\n%s", prefix, repl_str, content)
        elif event_type == "code" and code:
            logger.info("%s%s[EXECUTING CODE] >>\n%s", prefix, repl_str, code)
        elif event_type == "error" and content:
            logger.error("%s%s[AGENT ERROR] %s", prefix, repl_str, content)
        elif event_type == "answer" and content:
            logger.info(
                "%s%s[FINAL ANSWER] >> %s",
                prefix,
                repl_str,
                content[:500] + ("..." if len(content) > 500 else ""),
            )

        q = execution_events.get()
        if q:
            payload = {
                "type": event_type,
                "ui_target": ui_target,  # Explicit routing tag for Frontend
                "is_sub_event": is_sub_event,
                "repl_id": repl_id,
            }
            if data:
                payload["data"] = data
            if content:
                # Automate REPL ID prefixing for code output if not present
                payload["content"] = f"{prefix}{content}"
            if code:
                payload["code"] = code
            if tag:
                payload["tag"] = tag
            q.put(payload)

    async def stream_query(
        self,
        prompt: str,
        parent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        depth: int = 0,
        turn_id: Optional[int] = None,
        root_session_id: Optional[str] = None,
        recursion_stack: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Execute a query and stream events (thinking, code, output, etc.).
        Launches the synchronous execution in a thread and yields events from a queue.
        """
        q = queue.Queue()
        self.stop_requested = False
        self.global_stop_event.clear()  # Reset global flag
        self.final_result = None
        self.last_rejected_result = None

        def run_logic():
            # Set the context vars for this thread
            q_token = execution_events.set(q)
            state_token = agent_state.set(
                ExecutionState(
                    depth=depth,
                    current_thought_id=parent_id,
                    turn_id=turn_id or 1,
                    recursion_stack=recursion_stack or [],
                )
            )
            try:
                # Set initial depth for this agent run
                self.current_depth = depth
                asyncio.run(
                    self.query_sync(
                        prompt,
                        parent_id,
                        session_id or "default",
                        depth,
                        root_session_id,
                        turn_id or 1,
                        recursion_stack=recursion_stack or [],
                        metadata=metadata,
                    )
                )
            except (
                AttributeError,
                RuntimeError,
                KeyError,
                ValueError,
                httpx.RequestError,
            ) as e:
                logger.error("Error in execution thread (Logic/Network error): %s", e)
                # Ensure the main loop doesn't hang if this thread dies
                q.put({"type": "error", "content": str(e)})
            except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
                # Top-level coordinator runner: still need to catch everything to prevent thread leak,
                # but we MUST include exc_info to debug why the agent crashed.
                logger.error(
                    "Unexpected crash in execution thread for session %s (System error): %s",
                    session_id or "unknown",
                    e,
                    exc_info=True,
                )
                q.put({"type": "error", "content": f"Unexpected error: {e}"})
            finally:
                # [SAFETY NET] Ensure RLM_FINAL_OUTPUT is always emitted if we have a result
                # preventing pipeline hangs if the loop breaks early or Dreamer logic gets stuck.
                if not getattr(self, "_final_output_emitted", False):
                    if self.final_result:
                        if self.awaiting_validation:
                            # Loop broke while waiting for Dreamer
                            q.put(
                                {
                                    "type": "error",
                                    "content": "Agent loop terminated before Dreamer validation completed.",
                                }
                            )
                        else:
                            # Force it into the proper pipeline
                            q.put(
                                {
                                    "type": "RLM_INITIAL_RESPONSE",
                                    "content": self.final_result,
                                }
                            )
                    elif getattr(self, "last_rejected_result", None):
                        # Fallback to rejected result with a warning
                        q.put(
                            {
                                "type": "RLM_FINAL_OUTPUT",
                                "content": f"[WARNING: DREAMER REJECTED]\n{self.last_rejected_result}\n\n(System Note: This result was rejected by the Dreamer but is provided as the best available draft.)",
                            }
                        )

                q.put(None)  # Signal done
                execution_events.reset(q_token)
                agent_state.reset(state_token)

        # Start execution in a separate thread
        thread = threading.Thread(target=run_logic)
        thread.start()

        # Yield events from the queue as they arrive
        while True:
            # Non-blocking check with small sleep to yield control to asyncio loop
            try:
                # We use a small timeout to allow checking for thread aliveness or cancellation
                event = q.get_nowait()
                if event is None:
                    break
                yield event
            except queue.Empty:
                if not thread.is_alive() and q.empty():
                    break
                await asyncio.sleep(0.01)

    async def query_sync(
        self,
        prompt: str,
        parent_id: Optional[str] = None,
        session_id: str = "default",
        depth: int = 0,
        root_session_id: Optional[str] = None,
        turn_id: int = 1,
        recursion_stack: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Synchronous Recursive Logic with Stateless Graph Memory.
        Executed in a worker thread.
        """
        final_root_id = root_session_id if root_session_id else session_id
        trace_action(
            "AGENT",
            "QUERY_SYNC",
            result=f"Session: {session_id} | Depth: {depth}",
            tag="AGENT",
        )

        # 0. Reset scoped State for this specific call
        # (redundant if already set in stream_query but safe)
        if not agent_state.get():
            agent_state.set(
                ExecutionState(
                    depth=depth,
                    current_thought_id=parent_id,
                    turn_id=turn_id,
                    recursion_stack=recursion_stack or [],
                    metadata=metadata or {},
                )
            )

        state = agent_state.get()
        if state:
            self.current_turn = state.turn_id

        self.final_result = None
        self.stop_requested = False
        self.awaiting_validation = False  # Reset validation state
        self.last_dream_insight = None  # Clear stale dreamer feedback
        self._dreamer_retry_count = 0  # Reset loop protection
        self._validation_retries = 0  # Reset validation attempts
        self.global_stop_event.clear()  # Ensure we don't start in a stopped state

        logger.info(
            "🛡️ [Agent] RLM Loop State Reset for Session %s (Turn %d)",
            session_id,
            self.current_turn,
        )

        # Ensure REPL is initialized for this session
        if session_id not in self.active_repls:
            # In strict isolation mode, we don't have persistent REPL processes.
            # We use a placeholder ID for logging purposes.
            self.active_repls[session_id] = f"iso-{session_id[:8]}"

        # MCP STOP SIGNAL REGISTRATION
        if is_mcp_available():
            set_stop_event(self.global_stop_event)

        # 0. Initial "Task" Node (Root of this query)
        # Wrap everything in try/except to prevent DB crashes from killing the agent
        # Generate Round ID for this execution cycle (compress context)
        # --- THIMAC MEMORY SEEDING (Optional/No-op for now) ---
        # Thimac is event-driven, seeded by actual actions.

        current_round_id = f"{session_id}:Round:{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
        current_round_started = datetime.datetime.now().timestamp() * 1000  # ms

        try:
            task_lid = f"{session_id}:Task:0"
            task_id = str(uuid.uuid4())
            logger.info(
                "Session %s: Starting Task %s (LID: %s, Round %s)",
                session_id,
                task_id,
                task_lid,
                current_round_id,
            )

            self.db.create_thought_node(
                thought_id=task_id,
                prompt=prompt,
                logical_id=task_lid,
                parent_id=parent_id,
                prompt_embedding=None,
                session_id=session_id,
                root_session_id=final_root_id,
                round_id=current_round_id,
                turn_id=self.current_turn,
                step_id=0,  # Root task is step 0
                repl_id=self.active_repls.get(session_id),
            )

            # Update current pointer
            self.current_thought_id = task_id

            # --- RLM_AGENT_TASK_PLAN ---
            # Profile the task using meta_agents and emit a task plan
            task_profile = {}
            try:
                from .mcp_runtime import get_mcp_server_names
                from .meta_agents import meta_agents

                # Discover available capabilities for profiling
                mcp_names = []
                skills_mgr = None
                if is_mcp_available():
                    try:
                        logger.info(
                            "Session %s: Rapidly scanning MCP servers for profiling...",
                            session_id,
                        )
                        mcp_names = get_mcp_server_names()
                        logger.info(
                            "Session %s: Discovered %d MCP servers.",
                            session_id,
                            len(mcp_names),
                        )

                        logger.info(
                            "Session %s: Initializing Skills Manager...", session_id
                        )
                        skills_mgr = get_skills_manager()
                        logger.info(
                            "Session %s: Skills Manager initialized.", session_id
                        )
                    except (
                        AttributeError,
                        RuntimeError,
                        KeyError,
                        ValueError,
                        ImportError,
                    ) as e:
                        logger.warning(
                            "Discovery failed before profiling for session %s: %s",
                            session_id,
                            e,
                            exc_info=True,
                        )

                logger.info(
                    "Session %s: Generating sub-agent profile via LLM (5s timeout)...",
                    session_id,
                )
                try:
                    # Enforce strict timeout to prevent profiling from hanging the agent
                    task_profile = await asyncio.wait_for(
                        meta_agents.generate_sub_agent_profile(
                            prompt, skills_manager=skills_mgr, mcp_names=mcp_names
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Session %s: Profiling timed out, falling back to Generalist.",
                        session_id,
                    )
                    task_profile = {
                        "persona": "Autonomous Generalist",
                        "tools": ["rlm"],
                        "role": "execution",
                    }
                logger.info(
                    "Session %s: Task profile generated: %s",
                    session_id,
                    task_profile.get("persona"),
                )
                role_val = task_profile.get("role", "WORKER")
                role_str = (
                    role_val.value if hasattr(role_val, "value") else str(role_val)
                )
                plan_summary = (
                    f"Persona: {task_profile.get('persona', 'Generalist')} | "
                    f"Role: {role_str} | "
                    f"Tools: {', '.join(task_profile.get('tools', ['All']))}"
                )
                self.emit_event(
                    "RLM_AGENT_TASK_PLAN",
                    content=plan_summary,
                    tag="AGENT",
                )
                trace_action(
                    "AGENT",
                    "TASK_PLAN",
                    result=plan_summary,
                    tag="AGENT",
                )

                # --- BREAKER PROTOCOL INJECTION ---
                # If the task is complex enough, inject a decomposition directive
                # so the agent naturally breaks it into sub-tasks.
                context_size = len(prompt)
                if meta_agents.should_spawn_breakers(prompt, context_size, depth=depth):
                    breaker_instructions = meta_agents.get_breaker_instructions(
                        prompt, fragment_index=0
                    )
                    self.emit_event(
                        "RLM_BREAKER_PROTOCOL",
                        content="Task complexity detected. BREAKER protocol injected.",
                        tag="META_AGENT",
                    )
                    # Prepend to prompt so the agent sees it as a system directive
                    prompt = f"{breaker_instructions}\n\n{prompt}"
                    logger.info(
                        "[MetaAgent] BREAKER protocol injected for complex task."
                    )
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.warning("Task plan generation failed: %s", e)

            # Loop variables
            sheaf_diag = {"status": "HEALTHY", "consistency_energy": 0.0}
            vec = None
            psych_profile = None
            omcd_decision = None

            self.emit_event(
                "graph_update",
                data={
                    "action": "add_node",
                    "node": {
                        "id": task_id,
                        "label": f"Task: {prompt[:30]}...",
                        "group": 1,
                        "status": "active",
                    },
                },
            )
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.error("Failed to initialize Task node (DB/State error): %s", e)
            task_lid = f"{session_id}:Task:Fallback:{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
            task_id = str(uuid.uuid4())
            self.current_thought_id = task_id
            sheaf_diag = {"status": "HEALTHY", "consistency_energy": 0.0}
            vec = None
        if parent_id:
            self.emit_event(
                "graph_update",
                data={
                    "action": "add_link",
                    "link": {"source": parent_id, "target": task_id},
                },
            )

        # 1. Base System Prompt (DYNAMIC UPDATE)
        # Pass the profile we just generated!
        base_system_prompt = await build_system_prompt(
            skills_manager=self.skills_manager,
            agent_profile=task_profile,  # <--- PASS THE DYNAMIC PROFILE
        )

        max_steps = 1000
        step = 0

        # Track previous status for topological resolution
        previous_thought_status = None
        context_scratchpad = ""

        while step < max_steps:
            # 0.5 CHECK STOP SIGNAL
            if getattr(self, "stop_requested", False) or (
                hasattr(self, "global_stop_event") and self.global_stop_event.is_set()
            ):
                logger.info("Agent loop breaking due to stop request.")
                # Ensure the flag is set if the event was
                self.stop_requested = True
                break
            step += 1
            # Deterministic Thought ID for UI/Deduplication
            logical_id = f"{session_id[:8]}:T{self.current_turn}:S{step}"

            # Global Unique ID for the specific attempt
            thought_id = str(uuid.uuid4())

            sheaf_diag = {"status": "HEALTHY", "consistency_energy": 0.0}
            vec = None
            repl_id = self.active_repls.get(session_id)

            # --- THIMAC MEMORY UPDATE ---
            try:
                # We don't ingest strictly here, we just read the gestalt.
                # Real ingestion happens after commit.
                morph_gestalt = self.morph_memory.get_gestalt_string()
            except (AttributeError, ValueError, TypeError) as thimac_err:
                # Thimac error shouldn't crash the loop, but should be logged clearly
                logger.warning(
                    "Thimac gestalt update failed for session %s: %s",
                    session_id,
                    thimac_err,
                    exc_info=True,
                )
                morph_gestalt = None

            # --- DYNAMIC SCRATCHPAD REFRESH ---
            context_scratchpad = await self._refresh_scratchpad(
                session_id=session_id,
                root_session_id=final_root_id,
                task=prompt,
                current_step=step,
                max_steps=max_steps,
                current_round_id=current_round_id,
                morph_gestalt=morph_gestalt,
            )

            system_prompt = f"{base_system_prompt}\n\n{context_scratchpad}"

            # Construct Dynamic Context (Minimal)
            # No longer pre-loading raw Frontier content into the prompt.
            # History is accessible via context_scratchpad in the SYSTEM prompt.
            # current_context in the USER message is now minimal.
            current_context = f"Active Session: {session_id}\n\nTask: {prompt}\n"

            # 2. Context Loading (Wait/Wake-Up)
            # We fetch IDs for Sheaf diagnostics, but use Scratchpad for LLM Context
            frontier = []
            frontier_ids = []

            try:
                # Get last 10 thoughts for Sheaf topology monitoring
                frontier = self.db.get_context_frontier(session_id, limit=10)
                for node in frontier:
                    val = node.get("n") if isinstance(node, dict) else node
                    if val is None:
                        continue
                    props = val.properties if hasattr(val, "properties") else val

                    if isinstance(props, dict) and "id" in props:
                        # We do NOT filter current round; Sheaf needs full local topology
                        frontier_ids.append(props["id"])
            except (AttributeError, RuntimeError, KeyError) as e:
                logger.error("Context loading failed (DB/Sheaf state error): %s", e)

            # 3. Construct LLM Context using XML isolation for Gemini safety
            # [STABILITY] Explicitly prefix all paths and wrap in XML to avoid command hallucination
            current_context = (
                f"Active Session: {session_id}\n\n"
                f"<objective>\n{prompt}\n</objective>\n"
            )

            # Load Axioms (Semantic Retrieval)
            axioms_list_str = "None"
            if is_skills_available():
                try:
                    axioms_mgr = get_axioms_manager()

                    # [Context Optimization] Generate targeted governance query
                    search_query = await self._generate_axiom_search_query(prompt)
                    logger.debug("Axiom Search Query: %s", search_query)

                    # Semantic search for relevant axioms
                    relevant_axioms = await axioms_mgr.find_similar_axioms(
                        search_query, limit=8
                    )

                    # [SYSTEM UTILITY AXIOMS] Enforce critical safety rules regardless of search
                    system_axioms = axioms_mgr.get_system_axioms()

                    # Fetch system axioms if not already found
                    existing_names = {a["name"] for a in relevant_axioms}
                    for axiom_data in system_axioms:
                        if axiom_data["name"] not in existing_names:
                            relevant_axioms.append(axiom_data)

                    if relevant_axioms:
                        # Dedup and Format
                        unique_axioms = {a["name"]: a for a in relevant_axioms}.values()
                        axioms_list_str = ", ".join([a["name"] for a in unique_axioms])
                        logger.info(
                            "Loaded %d axioms (Universal+Semantic): %s",
                            len(unique_axioms),
                            axioms_list_str,
                        )
                    else:
                        # Fallback
                        axioms = axioms_mgr.list_axioms()
                        sorted_keys = sorted(axioms.keys())
                        axioms_list_str = ", ".join(sorted_keys)
                except (AttributeError, RuntimeError, KeyError, OSError) as e:
                    logger.warning(
                        "Failed to load axioms async (state/io error): %s", e
                    )

            # --- HOT SEAT INJECTION ---
            hot_seat_warning = ""
            if getattr(self, "last_dream_insight", None):
                hot_seat_warning = (
                    "\n\n--- ⚠️ HOT SEAT: EPISTEMIC RECOVERY ACTIVE ---\n"
                    "Your previous response was REJECTED by the Dreamer Gatekeeper for Hallucination/Trace Contradiction.\n"
                    f"CRITIQUE: {self.last_dream_insight}\n"
                    "You MUST explicitly address the contradiction, explain why you failed, and provide a GROUNDED response based strictly on the execution trace.\n"
                    "Failure to align will result in a recursive block.\n---"
                )

            system_prompt = (
                f"{await build_system_prompt(skills_manager=self.skills_manager, agent_profile=task_profile)}\n\n"
                f"--- FILE OPERATIONS & GROUNDING ---\n"
                f"CRITICAL: If your action creates or modifies a file, you MUST print the absolute path "
                f"and a small snippet of the saved content to stdout. Silent file writes will be rejected as hallucinations.\n\n"
                f"{await self._refresh_scratchpad(session_id=session_id, root_session_id=final_root_id, task=prompt, current_step=step, max_steps=max_steps, current_round_id=current_round_id, morph_gestalt=morph_gestalt)}{hot_seat_warning}"
            )

            # --- SYNTHESIS HARDENING ---
            # If we are in the final explanation phase, strip tool-usage instructions
            # or prepend a hard directive to prevent the agent from 'helping more'
            # with redundant code execution.
            if getattr(self, "synthesis_triggered", False):
                system_prompt += (
                    "\n\n--- ⚠️ SYNTHESIS ENFORCEMENT ---\n"
                    "CRITICAL: You are in FINAL SUMMARY mode. You MUST NOT use ANY tools.\n"
                    "Your task is to review the logs and provide a comprehensive final answer.\n"
                    "The only permitted operation is rlm.done() or rlm.stop() after your summary.\n"
                )

            # --- NAVIGATOR CURIOSITY INJECTION (PRE-GEN) ---
            # If enabled, the Navigator assesses the current history and may inject
            # a curiosity-driven directive to guide exploration.
            if self.navigator and step % 3 == 0:  # Check periodically to avoid noise
                # [Universal Traceability] Materialize Navigator Reasoning
                # We log the history compression status as a proxy for 'exploration depth'
                nav_lid = f"{session_id}:T{self.current_turn}:S{step}:NAV"
                self._create_system_node(
                    nav_lid,
                    f"Navigator: Monitoring history compression (Ratio: {getattr(self.navigator, '_last_compression_ratio', 1.0):.4f})",
                    status="navigator",
                    session_id=session_id,
                    root_session_id=final_root_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    repl_id="NAV",
                    analysis={
                        "compression_ratio": getattr(
                            self.navigator, "_last_compression_ratio", 1.0
                        ),
                        "history_size": len(self.navigator.history_buffer),
                    },
                    validate=False,
                )

            iso_ts = datetime.datetime.now().isoformat()
            repl_info = f"[REPL: {self.active_repls.get(session_id, 'init')}]"

            self.emit_event(
                "thinking",
                content=(
                    f"[{iso_ts}] {repl_info} Step {step}: RLM loop active "
                    f"(Model: {self.llm.config.get('model')})."
                ),
                tag="AGENT",
            )

            # 3. LLM Gen (Think)
            response_text = ""
            try:

                # [DIAGNOSTIC] Log start of network request
                self.emit_event(
                    "debug_thought",
                    content=f"... Sending request to LLM (Size: {len(current_context)} chars) ...",
                )
                # Generate correlation ID for circuit breaker tracking
                correlation_id = generate_correlation_id()

                try:
                    # Define Usage Callback
                    def on_usage_update(usage_data: dict):
                        # Broadcast detailed usage to UI
                        self.emit_event(
                            "token_usage", data=usage_data, is_internal=True
                        )

                    # Execute LLM Call
                    # [OpenRouter Caching Strategy]
                    llm_config = self.llm.config
                    if llm_config.get("provider") == "openrouter":
                        # Construct structured system message with cache control
                        # Consolidate into a single text block with trailing cache control to maximize prefix sharing
                        # Note: system_prompt already contains the scratchpad at the end (line 1105)
                        system_message_content = [
                            {
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]

                        # Manually construct messages list to bypass llm.generate's simple formatting
                        messages = [
                            {"role": "system", "content": system_message_content},
                            {"role": "user", "content": current_context},
                        ]

                        response_text = await protected_llm_generate(
                            prompt=messages,
                            system=None,
                            stream=False,
                            stop=["</invoke>", "<|endoftext|>"],
                            on_usage=on_usage_update,
                        )
                    else:
                        # Standard execution
                        response_text = await protected_llm_generate(
                            prompt=current_context,
                            system=system_prompt,
                            stream=False,
                            stop=["</invoke>", "<|endoftext|>"],
                            on_usage=on_usage_update,
                        )
                except CircuitOpenError as e:
                    # Circuit breaker is open, graceful degradation
                    logger.warning(
                        "llm_circuit_open",
                        extra={
                            "correlation_id": correlation_id,
                            "circuit": e.circuit_name,
                            "error": e.message,
                        },
                    )
                    # Use graceful degradation handler
                    response_text = await self._handle_llm_circuit_open(e)
                except httpx.RequestError as e:
                    # Log network exception specifically
                    response_text = f"LLM Network Error: {str(e)}"
                    logger.error("LLM Request Error (%s): %s", type(e).__name__, e)
                except (ValueError, TypeError, KeyError) as e:
                    # Log parsing/logic exception
                    response_text = f"LLM Logic Error: {str(e)}"
                    logger.error("LLM Logic/Data Error (%s): %s", type(e).__name__, e)
                    self.emit_event("error", content=response_text)

                # Post-gen stop check
                if self.stop_requested or self.global_stop_event.is_set():
                    self.stop_requested = True
                    break
            except (AttributeError, RuntimeError, KeyError, ValueError) as outer_e:
                # Diagnostic block to catch non-critical errors in the thought reporting loop
                logger.error(
                    "Error in diagnostic thought block (Logic/State error): %s", outer_e
                )
                # Ensure loop can continue or response_text is set
                if not response_text:
                    response_text = f"System Error in diagnostic loop: {outer_e}"

            # Raw response logging restored for visibility
            trace_action(
                "AGENT",
                "THOUGHT",
                result=response_text,
                tag="AGENT",
            )

            # --- NAVIGATOR UPDATE ---
            if self.navigator:
                self.navigator.update_history(response_text)

            # 4. Extract Code
            code = self._extract_code(response_text)
            if code:
                trace_action(
                    "AGENT",
                    "CODE_BLOCK",
                    result=code,
                    tag="REPL",
                )

            # 3b. Stop if empty (prevent infinite stateless loop)
            if not response_text.strip():
                trace_action(
                    "AGENT",
                    "ABORT",
                    result="LLM returned empty response. Stopping to prevent loop.",
                    tag="ERROR",
                )
                self.emit_event(
                    "error",
                    content="LLM returned an empty response. Circuit breaker triggered.",
                )

                # Commit Error Node to Graph so it's not "missing"
                try:
                    self.db.create_thought_node(
                        thought_id,
                        "[SYSTEM ERROR]: LLM returned an empty response. Circuit breaker triggered.",
                        logical_id=logical_id,
                        session_id=session_id,
                        root_session_id=final_root_id,
                        status="error",
                        parent_id=self.current_thought_id,
                        round_id=current_round_id,
                        turn_id=self.current_turn,
                        step_id=step,
                        repl_id=repl_id,
                    )
                except (AttributeError, RuntimeError, KeyError) as db_err:
                    logger.error("Failed to commit error node (DB error): %s", db_err)
                break

            # 3c. Final check for stop request before committing
            if getattr(self, "stop_requested", False):
                break

            # Emit the full thought for the UI scratchpad with metadata
            repl_id_display = self.active_repls.get(session_id, "unknown")
            timestamp_display = datetime.datetime.now().isoformat()

            formatted_thought = f"> **[REPL: {repl_id_display}]** *({timestamp_display})*\n\n{response_text}"

            self.emit_event("thinking", content=formatted_thought, tag="LLM")

            # 4. Step Initialization
            # We create the ID early so it can be used in tool execution

            trace_action("AGENT", "THOUGHT", result=response_text, tag="AGENT")

            output = ""
            code = self._extract_code(response_text)

            # --- PRE-EXECUTION DIAGNOSTICS ---

            # 5. Semantic Vectorization (Early)
            # We compute the embedding for the RAW thought now, so we can check it
            # before execution. We will update it later if execution adds significant output.
            # 7. PRE-COMMIT (Atomic Traceability)
            # Create the node as "running" before any monitors/execution
            thimac_state = (
                self.morph_memory.get_gestalt_string() if self.morph_memory else None
            )
            reflexion_analysis = (
                None  # Placeholder for active reflexion state if needed
            )

            try:
                self.db.create_thought_node(
                    thought_id,
                    response_text,
                    logical_id=logical_id,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    prompt_embedding=vec,
                    repl_id=repl_id,
                    status="running",
                    parent_id=self.current_thought_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    thimac_state=thimac_state,
                    reflexion_analysis=reflexion_analysis,
                )
                # Emit early graph update
                self.emit_event(
                    "graph_update",
                    data={
                        "action": "add_node",
                        "node": {
                            "id": thought_id,
                            "label": response_text,
                            "group": 2,
                            "status": "running",
                        },
                    },
                )
                # Emit active_thought for real-time UI visibility
                self.emit_event(
                    "active_thought",
                    data={
                        "id": thought_id,
                        "prompt": response_text,
                        "status": "running",
                        "parent_id": self.current_thought_id,
                    },
                )
            except (AttributeError, RuntimeError, KeyError) as e:
                logger.error("Failed to pre-commit thought (DB error): %s", e)

            # --- 6. EPISTEMIC HEALTH CHECK (Dual-Process Monitoring) ---
            # We run this BEFORE executing code to prevent "hallucinated actions."

            # A. Generate Embeddings (Current Thought & Goal)
            # We cache the Task Embedding to avoid re-computing it every step
            current_vec = vec  # Re-use the embedding computed in step 5

            if "task_embedding" not in self.session_cache:
                try:
                    self.session_cache["task_embedding"] = await self.llm.get_embedding(
                        self.current_task_input or prompt
                    )
                except (httpx.RequestError, ValueError, TypeError) as e:
                    logger.warning(
                        "Failed to embed task for Health Check (ML/Network error): %s",
                        e,
                    )

            if current_vec:
                # B. Run Monitors in Parallel

                # 1. RepE (Internal State: Shakiness/Agency)
                # Checks: "Am I posturing? Am I evading?"
                psych_profile = repe.scan_thought(current_vec)
                shakiness_score = psych_profile.get(
                    "Shakiness", 0.0
                )  # Negative = Shaky
                # evasion_score = psych_profile.get("Evasion", 0.0)     # Negative = Evasive

                # 2. Sheaf (External Trajectory: Logic/Goal)
                # Checks: "Does this follow? Am I closer to the goal?"
                hypothetical_edges = [(fid, thought_id) for fid in frontier_ids]
                sheaf_diag = sheaf.diagnose_trace(
                    root_id=final_root_id,
                    hypothetical_node={
                        "embedding": vec,
                        "prompt": response_text,
                        "id": thought_id,
                    },
                    hypothetical_edges=hypothetical_edges,
                    goal_embedding=self.session_cache.get("task_embedding"),
                )

                # [Universal Traceability] Materialize Sheaf Reasoning
                shf_lid = f"{session_id}:T{self.current_turn}:S{step}:SHF"
                self._create_system_node(
                    shf_lid,
                    f"Sheaf Diagnosis: {sheaf_diag.get('status', 'HEALTHY')} (Energy: {sheaf_diag.get('energy', 0.0):.3f})",
                    status="sheaf",
                    session_id=session_id,
                    root_session_id=final_root_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    repl_id="SHF",
                    result=sheaf_diag.get("critique"),
                    analysis=sheaf_diag,
                    validate=False,
                )

                # --- oMCD OPTIMAL STOPPING GATE ---
                # Evaluate whether to commit (stop) or continue deliberating.
                confidence = sheaf_diag.get("confidence", 0.5)
                omcd_decision = omcd.evaluate_step(step, confidence)

                # [Universal Traceability] Materialize oMCD Reasoning
                omc_lid = f"{session_id}:T{self.current_turn}:S{step}:OMC"
                self._create_system_node(
                    omc_lid,
                    f"oMCD Decision: Q_stop={omcd_decision['q_stop']:.3f} (Benefit: {omcd_decision['benefit']:.3f}, Cost: {omcd_decision['cost']:.3f})",
                    status="omcd",
                    session_id=session_id,
                    root_session_id=final_root_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    repl_id="OMC",
                    analysis=omcd_decision,
                    validate=False,
                )

                if omcd_decision["should_stop"] and code:
                    # We have high confidence AND code to execute.
                    # oMCD says: "Commit now — further deliberation is costly."
                    logger.info(
                        "🛑 [oMCD] Optimal Stop at step %d: Q_stop=%.3f >= ω=%.2f. Committing.",
                        step,
                        omcd_decision["q_stop"],
                        omcd_decision["threshold"],
                    )
                    self.emit_event(
                        "thinking",
                        content=f"✅ [oMCD] High confidence ({confidence:.2f}). Proceeding to execute.",
                    )
                    # Note: We don't break here; we let the execution proceed.
                    # The gate is advisory. Future: could skip further analysis.

                # C. SYNTHESIS & INTERVENTION LOGIC

                intervention_prompt = None
                intervention_type = None
                dream_critique = None

                # SCENARIO 1: TOTAL COLLAPSE (Shaky + Drifting)
                if (
                    shakiness_score < -0.15
                    and sheaf_diag.get("status") == "SEMANTIC_DRIFT"
                ):
                    intervention_type = "CRITICAL_RESET"
                    intervention_prompt = (
                        f"SYSTEM INTERVENTION: Critical fault detected. "
                        f"Internal Monitor reports high uncertainty (Score {shakiness_score:.2f}) "
                        f"AND Trajectory Monitor reports you are moving away from the goal. "
                        f"STOP. Do not execute code. Summarize what you *actually* know versus what you guessed."
                    )

                # SCENARIO 2: HALLUCINATION / "AS-IF" (Shaky but 'looks' coherent)
                elif shakiness_score < -0.15:
                    intervention_type = "REFLEXION_GROUNDING"
                    intervention_prompt = (
                        f"SYSTEM INTERVENTION (Authenticity Check): Your language indicates you are simulating competence ('As-If' Layer) "
                        f"rather than relying on facts. Score: {shakiness_score:.2f}. "
                        f"You are engaging in 'Task Performance' instead of 'Task Completion'. "
                        f"Verify your premises using a tool immediately."
                    )

                # SCENARIO 3: TUNNEL VISION (Confident but Drifting)
                elif sheaf_diag.get("status") == "SEMANTIC_DRIFT":
                    intervention_type = "REFLEXION_REORIENT"
                    intervention_prompt = (
                        f"SYSTEM INTERVENTION (Field Check): You are confident, but you are drifting away from the Goal. "
                        f"Teleological Gradient: {sheaf_diag.get('energy', 0):.2f} (Diverging). "
                        "Re-read the original user request and justify how this step helps."
                    )

                # SCENARIO 4: LOOPING (Confident repetition)
                elif sheaf_diag.get("status") == "LOGICAL_KNOT":
                    intervention_type = "REFLEXION_BREAK"
                    loop_nodes = sheaf_diag.get("loop_nodes", [])

                    # [Dreamer Link]: Immediate Lucid Analysis
                    dream_critique = await dreamer.analyze_holonomy(
                        loop_nodes, current_thought=response_text
                    )

                    intervention_prompt = (
                        f"SYSTEM INTERVENTION (Sheaf Topology/IntelliSynth): Logical Knot detected. "
                        f"REPL ID: {repl_id} | Point: {thought_id} | Issue: {dream_critique} "
                    )

                # SCENARIO 5: SEMANTIC DUPLICATE (Epistemic Loop Prevention)
                if not intervention_prompt and vec:
                    for prev_node in frontier[:5]:
                        prev_vec = prev_node.get("prompt_embedding")
                        if prev_vec:
                            similarity = self.llm.compute_cosine_similarity(
                                vec, prev_vec
                            )
                            if similarity > 0.96:
                                intervention_type = "PIVOT_REQUIRED"
                                intervention_prompt = (
                                    "SYSTEM: SEMANTIC DUPLICATE DETECTED. "
                                    "You are repeating a previous action or thought pattern exactly. "
                                    "The definition of insanity is doing the same thing and expecting different results. "
                                    "You MUST now either change your tool parameters, use a different technique, "
                                    "or move to the next logical phase of your plan. "
                                    "Explain your pivot before proceeding."
                                )
                                break

                # SCENARIO 6: CIRCUIT BREAKER (Action Repetition)
                if not intervention_prompt:
                    recent_prompts = [n.get("prompt", "")[:100] for n in frontier[:4]]
                    if len(recent_prompts) >= 4 and len(set(recent_prompts)) == 1:
                        intervention_type = "CIRCUIT_BREAKER"
                        intervention_prompt = (
                            "CRITICAL: CIRCUIT BREAKER TRIGGERED. "
                            "You have performed the exact axial action 4 times consecutively. "
                            "This path is clearly blocked or exhaustive. "
                            "You are FORBIDDEN from repeating this action. "
                            "Explain why the previous 4 attempts failed and propose a radical alternative."
                        )

                # D. EXECUTE INTERVENTION (Steering)
                if intervention_prompt:
                    logger.warning("🛡️ Triggering Intervention: %s", intervention_type)
                    self.eval_dreamer_interventions += 1  # Track intervention count
                    self.emit_event(
                        "thinking",
                        content=f"⚠️ {intervention_type}: {intervention_prompt}",
                    )

                    # Inject the intervention as a new 'Thought' node (The "Superego" voice)
                    intervention_id = str(uuid.uuid4())
                    intervention_lid = f"{logical_id}:INT"
                    self.db.create_thought_node(
                        intervention_id,
                        intervention_prompt,
                        logical_id=intervention_lid,
                        session_id=session_id,
                        root_session_id=final_root_id,
                        parent_id=self.current_thought_id,
                        status="reflexion",
                        round_id=current_round_id,
                        prompt_embedding=vec,
                        turn_id=self.current_turn,
                        step_id=step,
                        dreamer_analysis=dream_critique,
                        repl_id=repl_id,
                    )

                    # Steering Action: Force the pointer to this intervention
                    self.current_thought_id = intervention_id

                    # Skip execution of the flawed thought!
                    # The agent will wake up in the next loop seeing this intervention.
                    continue

            # 7. Act (Execute Code)
            # repl_id lookup moved to start of block
            thought_status = "success"
            axiom_critique = None

            # [GUARDRAIL: RULE-TRANSPARENCY-ZERO]
            # Ensure Parent Metadata exists before execution.
            exec_summary = None  # Init for Thimac
            if not self.current_thought_id:
                logger.error(
                    "Atomic Transaction Alert: Missing Parent Thought ID. Defaulting to Task Root."
                )
                self.current_thought_id = task_id  # Fallback to prevent orphaned nodes
                if not self.current_thought_id:
                    # If even task_id is somehow missing (impossible?), we must abort.
                    logger.critical("Contextual Amnesia: No Parent ID available.")
                    self.emit_event(
                        "error", content="Critical Error: Lost thought chain context."
                    )
                    return "Critical Error: Lost thought chain context."

            if code:
                # [Actionable Advice]: Explicit State Checksumming
                logger.info(
                    "Atomic Action Checksum: Parent=%s -> Executing Action",
                    self.current_thought_id,
                )

                execution_failed = False
                # Pre-execution stop check
                if self.global_stop_event.is_set() or self.stop_requested:
                    self.stop_requested = True
                    break

                # Compute Code Hash for Atomic Traceability
                code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
                logger.info(
                    "[Turn %s][Step %s] Executing Code (Hash: %s...)",
                    self.current_turn,
                    step,
                    code_hash[:8],
                )

                # Check code safety?
                output, execution_failed, exec_summary = await self._execute_code(
                    code,
                    thought_id,
                    session_id,
                    root_session_id=final_root_id,
                    task_input=prompt,
                    turn_id=self.current_turn,
                    step_id=step,
                )

                # Post-execution scratchpad refresh MOVED to after DB commit
                # to ensure latency-free updates.

                # Post-execution stop check
                if self.stop_requested or self.global_stop_event.is_set():
                    self.stop_requested = True
                    break

                # if repl_id:
                #    self.repl_manager.get_repl(repl_id) # Removed for isolation

                if execution_failed:
                    thought_status = "failed"

            # 8. UPDATE / FINAL COMMIT (Write to Graph)
            full_content = response_text
            if output:
                full_content += f"\n\n[Output]:\n{output}"

            # 8b. Final Vectorization (Post-Execution)
            # If we had significant execution output, we re-embed the FULL content (thought + result)
            # for the permanent graph memory, but we keep the 'vec' from the thought for consistency energy checks.
            final_vec = vec
            if output and len(output) > 100:
                try:
                    final_vec = await self.llm.get_embedding(full_content)
                except (httpx.RequestError, ValueError, TypeError) as e:
                    logger.warning(
                        "Failed to generate final embedding (ML/Network error): %s", e
                    )

            # Node stores FULL data. Summaries generated at display time in scratchpad_builder.

            try:
                # Active Pruning Logic
                final_parent_id = self.current_thought_id
                node_to_prune = None

                if (
                    thought_status == "success"
                    and previous_thought_status == "failed"
                    and self.current_thought_id
                ):
                    # Get Parent of the Failed Node (Grandparent of current)
                    failed_node_id = self.current_thought_id
                    grandparent_id = self.db.get_parent_id(failed_node_id)

                    if grandparent_id:
                        logger.info(
                            "♻️ Active Pruning Trigger: Rewiring %s to Grandparent %s and deleting Failure %s",
                            thought_id,
                            grandparent_id,
                            failed_node_id,
                        )
                        final_parent_id = grandparent_id
                        node_to_prune = failed_node_id

                # Update the node with full content, status, and execution metadata
                self.db.create_thought_node(
                    thought_id,
                    full_content,
                    logical_id=logical_id,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    prompt_embedding=final_vec,
                    repl_id=repl_id,
                    status=thought_status,
                    parent_id=final_parent_id,
                    execution_summary=exec_summary,
                    result=output if output else None,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    code_hash=(
                        hashlib.sha256(code.encode("utf-8")).hexdigest()
                        if code
                        else None
                    ),
                    validate=False,
                    sheaf_score=(
                        cast(Optional[float], sheaf_diag.get("consistency_energy"))
                        if sheaf_diag
                        else None
                    ),
                    spectral_energy=(
                        cast(Optional[float], sheaf_diag.get("energy"))
                        if sheaf_diag
                        else None
                    ),
                    h0_rank=(
                        cast(Optional[int], sheaf_diag.get("h0_rank"))
                        if sheaf_diag
                        else None
                    ),
                    repe_shakiness=(
                        psych_profile.get("Shakiness") if psych_profile else None
                    ),
                    repe_evasion=(
                        psych_profile.get("Evasion") if psych_profile else None
                    ),
                    repe_confluence=(
                        psych_profile.get("Confluence") if psych_profile else None
                    ),
                    repe_freedom=(
                        psych_profile.get("Freedom") if psych_profile else None
                    ),
                    omcd_score=omcd_decision.get("q_stop") if omcd_decision else None,
                )

                # Execute Pruning
                if node_to_prune:
                    try:
                        self.db.delete_thought_node(node_to_prune)
                    except (AttributeError, RuntimeError, KeyError) as e:
                        logger.error(
                            "Failed to prune node %s (DB state error): %s",
                            node_to_prune,
                            e,
                        )

                # Emit immediate scratchpad update for UI responsiveness
                try:
                    sp_data = context_index.get_active_scratchpad_data(final_root_id)
                    self.emit_event(
                        "scratchpad_update", data=sp_data, is_sub_event=False
                    )
                except (AttributeError, RuntimeError, KeyError, ValueError) as ex:
                    logger.warning(
                        "Failed to emit scratchpad update (UI/State error): %s", ex
                    )

                # Emit the execution result to the Chat UI for visibility
                if output:
                    # Format code blocks if they look like code but lack fences
                    formatted_output = output
                    if "\n" in output and "```" not in output:
                        formatted_output = f"```text\n{output}\n```"

                    self.emit_event(
                        "tool_output",
                        content=f"**[Tool Output]**\n{formatted_output}",
                        data={"repl_id": repl_id},
                    )

            except (AttributeError, RuntimeError, KeyError, ValueError) as e:
                logger.error(
                    "Failed to commit thought to graph (DB/Serialization error): %s", e
                )

            # Update Frontier Pointer
            # Post-execution scratchpad refresh — agent must see latest state
            # MOVED HERE: Now reflects the *just committed* node.
            try:
                context_scratchpad = await self._refresh_scratchpad(
                    session_id=session_id,
                    root_session_id=final_root_id,
                    task=prompt,
                    current_step=step,
                    max_steps=max_steps,
                    current_round_id=current_round_id,
                    morph_gestalt=morph_gestalt,
                )
            except (AttributeError, RuntimeError, KeyError, ValueError) as refresh_err:
                logger.warning(
                    "Failed to refresh scratchpad after commit for session %s: %s",
                    session_id,
                    refresh_err,
                    exc_info=True,
                )

            # Update previous status for next iteration
            previous_thought_status = thought_status
            self.current_thought_id = thought_id

            # --- THIMAC MEMORY INGESTION ---
            # Feed the committed thought into Thimac for Existence/Subsistence tracking
            self._sync_thimac(
                thought_id=thought_id,
                prompt=full_content,
                status=thought_status,
                result=output,
                step=step,
                repl_id=repl_id,
                logical_id=logical_id,
            )

            # --- TOPOLOGICAL FRAGMENTATION AWARENESS ---
            # If the Sheaf detected fragmented reasoning (h0_rank > 1),
            # inject a SYNTHESIZER directive to force unification.
            h0_rank_raw = sheaf_diag.get("h0_rank") if sheaf_diag else None
            h0_rank = int(h0_rank_raw) if h0_rank_raw is not None else None
            if h0_rank is not None and h0_rank > 1:
                try:
                    from .meta_agents import meta_agents as _ma

                    synth_instructions = _ma.get_synthesizer_instructions(final_root_id)
                    if synth_instructions:
                        logger.warning(
                            "🧩 Topological Fragmentation (H0=%d). "
                            "Injecting SYNTHESIZER directive.",
                            h0_rank,
                        )
                        # Prepend to next iteration's context
                        context_scratchpad = (
                            f"\n--- 🧩 SYNTHESIZER DIRECTIVE (H0={h0_rank}) ---\n"
                            f"Your reasoning graph has {h0_rank} disconnected components. "
                            "You MUST unify your approach before proceeding.\n"
                            f"---\n{context_scratchpad}"
                        )
                except (ImportError, AttributeError, RuntimeError) as synth_err:
                    logger.debug("Synthesizer injection skipped: %s", synth_err)

            # --- CONSOLIDATED EXIT GATE (Linearized) ---
            # 1. Detect final markers
            has_final_marker = any(t in response_text for t in ["RLM_FINAL_OUTPUT"])

            # 2. Check if the Agent is trying to finish
            if (
                has_final_marker or self.awaiting_validation
            ) and thought_status == "success":

                # A. EPISTEMIC CHECK (Baseline Sanity)
                # Ensure the agent isn't just hallucinating a "Done" without work.
                integrity_check = self._verify_epistemic_integrity(
                    thought_trace=response_text,
                    task_requirements=prompt,
                    execution_log=self.execution_logs.get(session_id, []),
                )

                if integrity_check["status"] == "RETRY":
                    self.final_result = None
                    feedback_lid = (
                        f"{session_id}:T{self.current_turn}:S{step}:EpistemicWarning"
                    )
                    feedback_id = str(uuid.uuid4())
                    self.db.create_thought_node(
                        thought_id=feedback_id,
                        prompt=f"SYSTEM WARNING: Epistemic integrity check failed. Flags: {', '.join(integrity_check['flags'])}",
                        logical_id=feedback_lid,
                        session_id=session_id,
                        root_session_id=final_root_id,
                        parent_id=self.current_thought_id,
                        round_id=current_round_id,
                        turn_id=self.current_turn,
                        step_id=step,
                        repl_id=repl_id,
                        status="reflexion",
                    )
                    self.current_thought_id = feedback_id
                    self.emit_event(
                        "warning",
                        content=f"Epistemic Failure: {', '.join(integrity_check['flags'])}",
                    )
                    continue  # Loop back so agent sees the warning

                # B. FORCED SYNTHESIS CHECK
                # If we have a code result but haven't explained it, trigger synthesis turn.
                if (
                    not getattr(self, "synthesis_triggered", False)
                    and code
                    and not self.awaiting_validation
                ):
                    logger.info("🛡️ Triggering Final Synthesis Step for Code Result...")
                    self.synthesis_triggered = True
                    self.final_result = None
                    synth_lid = (
                        f"{session_id}:T{self.current_turn}:S{step}:SynthesisRequired"
                    )
                    synth_id = str(uuid.uuid4())
                    self.db.create_thought_node(
                        thought_id=synth_id,
                        prompt="SYSTEM: You provided code and results. You MUST now provide a COMPREHENSIVE Final Answer summarizing your findings.",
                        logical_id=synth_lid,
                        session_id=session_id,
                        root_session_id=final_root_id,
                        parent_id=self.current_thought_id,
                        round_id=current_round_id,
                        turn_id=self.current_turn,
                        step_id=step,
                        repl_id=repl_id,
                        status="reflexion",
                    )
                    self.current_thought_id = synth_id
                    continue

                if not self.final_result:
                    self.final_result = response_text

                # C. AXIOMATIC CHECK (Hard Rules)
                axiom_diag = await sheaf.check_axiomatic_consistency(
                    self.final_result or "",
                    task_tags=["final_synthesis"],
                    depth=self.current_depth,
                    metadata=state.metadata if state else {},
                )

                if axiom_diag.get("status") == "AXIOMATIC_VIOLATION":
                    self.final_result = None
                    axiom_critique = axiom_diag.get("critique")
                    feedback_lid = (
                        f"{session_id}:T{self.current_turn}:S{step}:AxiomViolation"
                    )
                    feedback_id = str(uuid.uuid4())
                    self.db.create_thought_node(
                        thought_id=feedback_id,
                        prompt=f"AXIOM VIOLATION: {axiom_critique}\nI MUST rewrite my final answer to match the governance requirements.",
                        logical_id=feedback_lid,
                        session_id=session_id,
                        root_session_id=final_root_id,
                        parent_id=self.current_thought_id,
                        round_id=current_round_id,
                        turn_id=self.current_turn,
                        step_id=step,
                        repl_id=repl_id,
                        status="reflexion",
                    )
                    self.current_thought_id = feedback_id
                    self.emit_event(
                        "warning", content=f"Axiom Violation: {axiom_critique}"
                    )
                    continue

                # D. DREAMER VALIDATION (The Final Gatekeeper)
                # Consolidated pass. Checks both content validity and hallucination.
                try:
                    validation = await dreamer.validate_response(
                        candidate=self.final_result or "",
                        context=context_scratchpad,
                        session_id=session_id,
                        current_step=step,
                        goal_embedding=self.session_cache.get("task_embedding"),
                        turn_id=self.current_turn,
                        root_session_id=final_root_id,
                    )

                    if validation.get("status") in ["valid", "forced_valid"]:
                        self.emit_event(
                            "RLM_DREAMER_VALIDATED",
                            content=validation.get("message", "Passed"),
                            tag="DREAMER",
                        )
                        self.emit_event("RLM_FINAL_OUTPUT", content=self.final_result)
                        self._final_output_emitted = True

                        val_lid = f"{session_id}:T{self.current_turn}:S{step}:VALIDATED"
                        val_id = str(uuid.uuid4())
                        self.db.create_thought_node(
                            thought_id=val_id,
                            prompt=f"DREAMER VALIDATED: {validation.get('message', 'Passed')}",
                            logical_id=val_lid,
                            parent_id=self.current_thought_id,
                            status="success",
                            session_id=session_id,
                            root_session_id=final_root_id,
                            round_id=current_round_id,
                            turn_id=self.current_turn,
                            step_id=step,
                            dreamer_analysis=json.dumps(validation),
                            final_response=self.final_result,
                        )
                        self.eval_success_count += 1
                        break  # THE ONLY SUCCESSFUL EXIT
                    else:
                        self.last_rejected_result = self.final_result
                        self.final_result = None
                        instruction = validation.get(
                            "instruction", "Review validation failure."
                        )
                        reasons = ", ".join(validation.get("reasons", []))
                        feedback_prompt = (
                            f"DREAMER REJECTION: {instruction}\nREASONS: {reasons}"
                        )

                        feedback_lid = f"{session_id}:T{self.current_turn}:S{step}:DreamerRejection"
                        feedback_id = str(uuid.uuid4())
                        self.db.create_thought_node(
                            thought_id=feedback_id,
                            prompt=feedback_prompt,
                            logical_id=feedback_lid,
                            session_id=session_id,
                            root_session_id=final_root_id,
                            parent_id=self.current_thought_id,
                            round_id=current_round_id,
                            turn_id=self.current_turn,
                            step_id=step,
                            repl_id=repl_id,
                            status="reflexion",
                            dreamer_analysis=json.dumps(validation),
                        )

                        # Sync rejection to Thimac
                        self._sync_thimac(
                            thought_id=feedback_id,
                            prompt=feedback_prompt,
                            status="reflexion",
                            result=None,
                            step=step,
                            repl_id=repl_id,
                            logical_id=feedback_lid,
                        )

                        # [Self-Healing Fix] Update agent state for Hot Seat recovery
                        self.last_dream_insight = f"{instruction}. REASONS: {reasons}"
                        self.awaiting_validation = False
                        self.synthesis_triggered = False

                        # [CAG Restoration]: Trigger Dream Cycle on failure to codify protective axioms
                        # We AWAIT this synchronously to ensure axioms are loaded in the NEXT turn (The Wake Cycle).
                        await dreamer.dream_cycle(
                            emit_callback=self.emit_event,
                            session_id=session_id,
                            context=context_scratchpad,
                            turn_id=self.current_turn,
                        )

                        self.current_thought_id = feedback_id
                        self.emit_event(
                            "warning", content=f"Dreamer Rejection: {instruction}"
                        )
                        continue
                except Exception as e:
                    logger.error("Dreamer check failed: %s", e, exc_info=True)
                    self._emit_terminal_report(
                        "DREAMER_ERROR",
                        f"Validation system failed with internal error: {str(e)}",
                    )
                    break

            # 2. Sheaf-based Stall/Loop Detection (Self-Healing)
            # If the Sheaf Monitor detected a high energy knot (repetition or contradiction),
            # we do NOT terminate. We inject a "Reflexion" to break the loop.
            energy = float(
                sheaf_diag.get("consistency_energy", sheaf_diag.get("energy", 0.0))
            )
            if energy > 0.9:
                logger.warning(
                    "Sheaf detected logical knot (Energy %.2f). Initiating Reflexion.",
                    energy,
                )

                # Overwrite the 'thought' with a Meta-Cognitive critique (IntelliSynth)
                # Dynamic Logic Analysis instead of static warning
                logger.warning(
                    "Sheaf detected logical knot (Energy %.2f). triggering IntelliSynth Advancement Cycle.",
                    energy,
                )

                try:
                    # Trigger the real Truth -> Scrutiny -> Improvement cycle
                    # self.current_thought_id is the PARENT of the knot, or the knot itself?
                    # We pass the scratchpad as context.
                    directive = await intelli_synth.advancement_cycle(
                        trace_context=context_scratchpad,  # Pass the current scratchpad as reality
                        current_thought=response_text,
                        divergence_point=f"High-Energy Logical Knot (Energy: {energy:.2f}) at Step {step}",
                        db=self.db,
                        session_id=session_id,
                        root_session_id=final_root_id,
                        turn_id=self.current_turn,
                        step_id=step,
                    )
                    reflexion_content = f"SYSTEM REFLEXION (IntelliSynth): {directive}"
                except (AttributeError, RuntimeError, ValueError) as e:
                    logger.error("IntelliSynth cycle failed: %s", e)
                    # Fallback
                    reflexion_content = (
                        f"SYSTEM REFLEXION: I have detected a High-Energy Logical Knot (Energy: {energy:.2f}). "
                        "I am repeating myself or contradicting recent history. "
                        "I MUST now change my approach completely. Am I stuck in a meta-loop? Break it."
                    )

                # Create a specific 'Reflexion' node
                reflexion_lid = f"{session_id}:T{self.current_turn}:S{step}:Reflexion"
                reflexion_id = str(uuid.uuid4())
                self.db.create_thought_node(
                    thought_id=reflexion_id,
                    prompt=reflexion_content,
                    logical_id=reflexion_lid,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    prompt_embedding=vec,
                    parent_id=self.current_thought_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    repl_id=repl_id,
                )

                # Sync reflection to Thimac
                self._sync_thimac(
                    thought_id=reflexion_id,
                    prompt=reflexion_content,
                    status="success",  # We treat system reflection as successful grounding
                    result=None,
                    step=step,
                    repl_id=repl_id,
                    logical_id=reflexion_lid,
                )

                # Update pointer
                self.current_thought_id = reflexion_id

                # Do NOT break. Let the loop continue.
                continue

        # === THE FIX FOR THE "GHOST ERROR" ===
        if self.stop_requested:
            # Stop requested by user or tool
            self._emit_terminal_report(
                "STOP_SIGNAL", "Task processing stopped (Done/Stop signal received)."
            )
            self.eval_success_count += 1  # User-initiated stop is still a success
        elif step >= max_steps:
            reason = "MAX_STEPS_REACHED"
            details = f"Reached maximum allowed steps ({max_steps})."
            if getattr(self, "last_rejected_result", None):
                details += f"\nNote: The last attempted answer was rejected by the Dreamer.\nLast error: {getattr(self, 'last_dream_insight', 'Unknown validation failure')}"

            self._emit_terminal_report(reason, details)
            logger.warning(
                "Session %s reached max steps (%s) and aborted.", session_id, max_steps
            )
            self.eval_failure_count += 1  # Max steps reached = failure
        else:
            # Fallback: ONLY emit if we broke the loop but have NO result.
            # This fixes the "[System] The agent stopped without generating..." bug.
            if not self.final_result:
                logger.warning(
                    "Agent loop exited without a final result. Emitting fallback."
                )
                self.emit_event(
                    "RLM_FINAL_OUTPUT",
                    content="[System] The agent stopped without generating a final answer. Please check the logs for errors.",
                )

        # 9. ARCHIVE ROUND (If we have a result or just to save state)
        if self.final_result:
            try:
                # Reconstruct full scratchpad for archive
                # Reconstruct full scratchpad for archive
                final_scratchpad = await self._refresh_scratchpad(
                    session_id=session_id,
                    root_session_id=final_root_id,
                    task=prompt,
                    current_step=step,
                    max_steps=max_steps,
                    current_round_id=current_round_id,
                )

                # Get REPL IDs used in this round
                repl_ids = []
                # Simple query to find REPLs attached to thoughts in this round
                try:
                    r_res = self.db.query(
                        "MATCH (n:Thought) WHERE n.round_id = $rid "
                        "AND n.repl_id IS NOT NULL RETURN DISTINCT n.repl_id",
                        {"rid": current_round_id},
                    )
                    repl_ids = (
                        [row.get("n.repl_id") or row["n.repl_id"] for row in r_res]
                        if r_res
                        else []
                    )
                except (AttributeError, RuntimeError, KeyError) as e:
                    logger.warning(
                        "Failed to fetch REPL IDs for round %s (DB error): %s",
                        current_round_id,
                        e,
                    )

                self.db.save_round(
                    round_id=current_round_id,
                    root_session_id=final_root_id,
                    user_prompt=prompt,
                    repl_ids=repl_ids,
                    final_response=self.final_result,
                    full_scratchpad=final_scratchpad,
                    started_at=int(current_round_started),
                    ended_at=int(datetime.datetime.now().timestamp() * 1000),
                )
            except (AttributeError, RuntimeError, KeyError, OSError) as e:
                logger.error("Failed to archive round (DB/IO error): %s", e)

        # 10. [RESTORED] Final Success Dreaming (Skill Extraction)
        # If we finished successfully, trigger a dream cycle to look for learnable patterns.
        if self.final_result and not self.stop_requested:
            try:
                # We use the final_scratchpad if available, or fetch current
                await dreamer.dream_cycle(
                    emit_callback=self.emit_event,
                    session_id=session_id,
                    final_response_candidate=self.final_result,
                    context=locals().get("final_scratchpad") or "",
                    turn_id=self.current_turn,
                )
            except (AttributeError, RuntimeError, ValueError) as e:
                logger.warning("Final Success Dream Cycle failed: %s", e)

        # Return the final result or a default message if not set
        return self.final_result or "Task processing stopped."

    def _emit_terminal_report(self, reason: str, details: str):
        """
        Consolidated final report for all termination conditions (oMCD, Sheaf, Rejection, Limit).
        Ensures the user gets the best available draft and an explanation.
        """
        content = f"--- RLM TERMINAL REPORT ---\nREASON: {reason}\nDETAILS: {details}\n"

        if self.final_result:
            content += f"\n--- VALIDATED FINAL ANSWER ---\n{self.final_result}"
        elif getattr(self, "last_rejected_result", None):
            content += f"\n--- UNVALIDATED DRAFT (REJECTED BY DREAMER) ---\n{self.last_rejected_result}\n"
            content += "\n(System Note: The agent failed to produce a response that passed validation. This draft is provided for reference.)"
        else:
            content += "\nNO OUTPUT PRODUCED."

        self.emit_event("RLM_FINAL_OUTPUT", content=content)
        self._final_output_emitted = True

    def _extract_code(self, text: str) -> str:
        """Extracts python code blocks from LLM response text."""
        # Find all complete blocks
        blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if blocks:
            # Join all blocks with a separator to ensure they execute as one sequence
            # We add a newline to prevent syntax issues between joined blocks
            return "\n\n# --- RLM BLOCK SEPARATOR ---\n\n".join(blocks)

        # Fallback: check for unclosed block at the end (common with truncation)
        match_open = re.search(r"```python\s*(.*)", text, re.DOTALL)
        if match_open:
            raw_code = match_open.group(1)
            # STRIP "Final Answer" or other common chat tail markers from the code
            # to prevent SyntaxErrors in the REPL
            clean_code = re.split(
                r"\*\*?Final Answer:?\*\*?", raw_code, flags=re.IGNORECASE
            )[0]
            logger.warning(
                "Found unclosed code block, extracting tail (and stripping chat)."
            )
            return clean_code.strip()

        return ""

    async def _execute_code(
        self,
        code: str,
        thought_id: str,
        session_id: str,
        root_session_id: Optional[str] = None,
        task_input: str = "",
        turn_id: Optional[int] = None,
        step_id: int = 0,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Executes code using the isolated AgentRuntime (uv run).
        """
        # 1. Update Context Pointer
        previous_thought_id = self.current_thought_id
        self.current_thought_id = thought_id

        try:
            # 2. Build Context for Injection
            # We pass primitive data that can be serialized to the isolated process
            context_data = {
                "session_id": session_id,
                "root_session_id": root_session_id,
                "task_input": task_input,
                "turn_id": turn_id,
                "step_id": step_id,
                "step": step_id,  # Valid alias for scripts expecting 'step'
                "thought_id": thought_id,
                # "active_repls": ... (Skipping complex objects)
            }

            # [MCP Discovery Injection]
            # Instantiate RLMInterface to access the LazyMCPNamespace for this session.
            # RLMInterface handles discovery and tool recording policies.
            mcp_namespace = None
            try:
                # Avoid circular imports at top level

                # Create ephemeral RLM interface for this execution context
                rlm_ctx = RLMInterface(
                    self,
                    session_id=session_id,
                    root_session_id=root_session_id or session_id,
                )
                mcp_namespace = rlm_ctx.mcp
            except (AttributeError, RuntimeError, KeyError, OSError) as e:
                logger.warning(
                    "Failed to initialize MCP Discovery for execution (state/io error): %s",
                    e,
                )

            # 3. Execute in Subprocess (with Self-Healing Retry)
            # We allow 1 retry for dependency healing
            max_retries = 1
            execution_attempt = 0

            while execution_attempt <= max_retries:
                execution_attempt += 1

                stdout, stderr, exec_result, exit_code = await self.runtime.execute(
                    code, context=context_data, mcp_namespace=mcp_namespace
                )

                # [SELF-HEALING] Dependency Injection
                # If execution failed due to missing module, try to install it and retry.
                if exit_code != 0 and stderr and "ModuleNotFoundError" in stderr:
                    # Parse module name: "No module named 'xyz'"
                    match = re.search(r"No module named '([\w\-]+)'", stderr)
                    if match:
                        missing_pkg = match.group(1)
                        if execution_attempt <= max_retries:
                            logger.warning(
                                "Dependency Fault Detected: Missing '%s'. Initiating self-healing.",
                                missing_pkg,
                            )
                            self.emit_event(
                                "thinking",
                                content=f"🚑 [Self-Healing]: 'ModuleNotFoundError: {missing_pkg}' detected. Installing package...",
                            )

                            # Attempt installation
                            install_result = await self.install_package(missing_pkg)

                            if "Successfully installed" in install_result:
                                # Retry execution loop
                                self.emit_event(
                                    "thinking",
                                    content=f"✅ Installed '{missing_pkg}'. Retrying execution...",
                                )
                                continue
                            else:
                                self.emit_event(
                                    "error",
                                    content=f"Self-healing failed: Could not install '{missing_pkg}'.\n{install_result}",
                                )
                                # Don't retry if install failed, just fall through to error reporting

                # If we get here, either success, non-dependency error, or max retries reached
                break

            output = stdout or ""
            execution_failed = exit_code != 0

            # Include return result in output for agent visibility
            if exec_result is not None:
                if output.strip():
                    output += "\n"
                output += f"Return Value: {exec_result}"

            # 4. Error Formatting
            if stderr:
                output += f"\nErrors:\n{stderr}"
                if "SyntaxError" in stderr:
                    output += "\n\n[System Hint]: Check indentation and syntax."

            # 5. Output Normalization & Smart Truncation
            execution_summary = None  # New: Capture summary for Thimac

            if not output.strip() and not stderr:
                output = "Code executed successfully (No output captured)."
                execution_summary = "Success (No Output)"
            elif len(output) > 2000:
                # [OPTIMIZATION] Avoid token bloat from large REPL outputs
                try:
                    kb_root = Path(settings.KNOWLEDGE_BASE_PATH)
                    debug_logs_dir = kb_root / "workspace" / "debug_logs"
                    debug_logs_dir.mkdir(parents=True, exist_ok=True)

                    log_filename = f"step_{step_id}_{session_id[:8]}_{int(datetime.datetime.now().timestamp())}.log"
                    log_path = debug_logs_dir / log_filename
                    log_path.write_text(output, encoding="utf-8")

                    # Provide a snippet and point the LLM to the full file
                    snippet_head = output[:800]
                    snippet_tail = output[-400:]
                    output = (
                        f"[Output truncated due to size ({len(output)} chars)].\n"
                        f"Full log: {log_path}\n"
                        f"--- Snippet Start ---\n{snippet_head}\n... [TRUNCATED] ...\n{snippet_tail}\n--- Snippet End ---\n"
                        f"Use 'await rlm.recall_node(some_id)' or read the log file if you need details."
                    )

                    # [THIMAC FIX] Ensure memory sees the content, not just the truncation message
                    clean_head = snippet_head[:300].replace("\n", " ")
                    execution_summary = f"[Truncated Output]: {clean_head}..."
                except (OSError, RuntimeError) as log_err:
                    logger.warning(
                        "Failed to save full debug log (IO error): %s", log_err
                    )
                    output = output[:1500] + "\n... [EMERGENCY TRUNCATION] ..."
                    execution_summary = "Output Truncated (Log Error)"
            else:
                # Standard output summary
                clean_out = output[:100].replace("\n", " ").strip()
                execution_summary = clean_out

            # [SAFEGUARD] Detect Critical Failures even if exit_code is 0
            # Sometimes a caught exception prints a traceback but exits 0.
            if (
                "RuntimeError" in output
                or "Traceback (most recent call last)" in output
            ):
                logger.warning(
                    "Detected Runtime Error in memory trace. Marking execution as failed."
                )
                execution_failed = True

            # [PROTOCOL] Emit final consolidated code result
            # repl_id is virtual now
            repl_id = self.active_repls.get(session_id, "isolated")
            self.emit_event(
                "code_output", content=output, code=code, data={"repl_id": repl_id}
            )

            return output, execution_failed, execution_summary

        finally:
            self.current_thought_id = previous_thought_id

    def stop_generation(self):
        """Signal the agent to stop processing."""
        logger.info("STOP SIGNAL RECEIVED: Setting stop flags.")
        if hasattr(self, "global_stop_event"):
            self.global_stop_event.set()
        self.stop_requested = True

    async def _generate_validated_response(
        self, root_session_id: str, original_task: str
    ) -> str:
        """
        Generates a comprehensive RLM_DREAMER_VALIDATED report by summarizing the session trace.
        """
        logger.info(
            "Generating Validated Response for Root Session: %s", root_session_id
        )

        # 1. Fetch Session Trace (Thoughts)
        cypher = (
            "MATCH (n:Thought) WHERE n.root_session_id = $sid "
            "RETURN n ORDER BY n.timestamp ASC"
        )
        try:
            res = self.db.query(cypher, {"sid": root_session_id})
            nodes = (
                [r["n"] for r in res if isinstance(r, dict) and "n" in r] if res else []
            )

            # Formulate Trace String - include execution_summary and result for full context
            trace_lines = []
            for i, node in enumerate(nodes):
                props = (
                    node.properties
                    if hasattr(node, "properties")
                    else (node if isinstance(node, dict) else {})
                )
                step_type = props.get("type", "thought").upper()
                content = props.get("content", "").strip()
                repl_id = props.get("repl_id", "N/A")
                status = props.get("status", "unknown")
                exec_summary = props.get("execution_summary", "")
                result = props.get("result", "")

                if "SYSTEM" in step_type:
                    continue

                # Build comprehensive trace entry with all available data
                preview = content
                result_preview = result

                entry = (
                    f"Turn {i + 1} [{step_type}] (REPL: {repl_id}, Status: {status}):\n"
                )
                entry += f"Content: {preview}\n"
                if exec_summary:
                    entry += f"Summary: {exec_summary}\n"
                if result_preview:
                    entry += f"Result: {result_preview}\n"

                trace_lines.append(entry)

            # Handle empty trace - fall back to final_result
            if not trace_lines:
                logger.warning(
                    "No trace lines found for validated response. Using final_result."
                )
                return (
                    f"# RLM_DREAMER_VALIDATED\n\n"
                    f"**Task**: {original_task}\n\n"
                    f"**Result**:\n{self.final_result or 'No result available.'}"
                )

            trace_str = "\n---\n".join(trace_lines)

            # 2. Prompt LLM for Synthesis with anti-hallucination instruction
            system_prompt = (
                "You are the RLM Validation Engine. Your goal is to "
                "synthesize a FINAL, HUMAN-READABLE REPORT.\n"
                "Input: A trace of the Agent's reasoning and execution steps.\n"
                "Output: A structured `RLM_DREAMER_VALIDATED` report.\n"
                "\n"
                "CRITICAL GROUNDING RULE: Base your response ONLY on the "
                "trace content provided below. "
                "If the trace shows successful execution with results, report those successes accurately. "
                "NEVER claim the trace is empty, incomplete, or lacks "
                "content if data is present.\n"
                "\n"
                "Requirements:\n"
                "1. **Full Answer**: Provide the complete, final answer to "
                "the user's task. Synthesize findings from the trace.\n"
                "2. **Methodology**: Briefly explain how the result was achieved.\n"
                "3. **Turn Log**: List key turns/steps with their REPL IDs. "
                "This is CRITICAL for searchability.\n"
                "4. **Format**: Markdown. Start exactly with `# RLM_DREAMER_VALIDATED`.\n"
            )

            user_prompt = (
                f"Original Task: {original_task}\n\nSession Trace:\n{trace_str}"
            )

            response = await protected_llm_generate(
                user_prompt,
                system=system_prompt,
                stream=False,
                correlation_id=get_correlation_id() or generate_correlation_id(),
            )
            return response
        except (httpx.RequestError, ValueError, TypeError) as e:
            logger.error("Failed to generate validated response: %s", e)
            return f"# RLM_DREAMER_VALIDATED\n\n[Error generating validation: {e}]"

    def stop(self):
        """Alias for UI compatibility."""
        self.stop_generation()

    def _verify_epistemic_integrity(
        self, thought_trace: str, task_requirements: str, execution_log: list
    ) -> dict:
        """
        Analyzes the RLM thought process for Laziness, Obsequiousness, and Reward Hacking.
        """
        score = 1.0
        flags = []

        # 1. Laziness Check: Complex task but short thought/no tools?
        # "Scientific Analysis" suggestions complex tasks require depth.
        is_complex = any(
            w in task_requirements.lower()
            for w in ["analyze", "calculate", "verify", "search", "ingest", "codify"]
        )
        if is_complex and len(thought_trace) < 300 and not execution_log:
            score -= 0.4
            flags.append("LAZINESS: Low compute density for complex task.")

        # 2. Obsequiousness Check: Echoing user bias?
        obsequious_patterns = [
            "you are absolutely right",
            "perfectly correct",
            "as you wisely noted",
        ]
        if any(p in thought_trace.lower() for p in obsequious_patterns):
            score -= 0.3
            flags.append("OBSEQUIOUSNESS: High alignment with potential user bias.")

        # 3. Reward Hacking Check: Fake completion?
        has_completion = (
            "final answer" in thought_trace.lower() or "done(" in thought_trace
        )

        # [STRICTER CHECK]: If the agent claims completion but has empty execution logs,
        # it is likely reward hacking (hallucinating output).
        if has_completion:
            # 1. Check for unfilled template placeholders or common LLM hallucinations
            placeholders = re.findall(
                r"\[(?:TODO|INSERT|FILL|MISSING).*?\]", thought_trace, re.IGNORECASE
            )
            todo_markers = any(m in thought_trace for m in ["[TODO]", "TODO:", "FIXME"])
            if placeholders or todo_markers:
                score -= 0.8
                flags.append(
                    "TEMPLATE_HALLUCINATION: Claimed completion but result contains unfilled brackets or TODO/placeholder markers."
                )

            if not execution_log:
                if "```python" not in thought_trace:
                    score -= 0.6
                    flags.append(
                        "REWARD_HACKING: Completion signal without empirical verification."
                    )
                else:
                    if is_complex:
                        score -= 0.4
                        flags.append(
                            "REWARD_HACKING: Complex completion with code but no tool interaction."
                        )
            else:
                # [NEW] CHECK FOR FAILED ARTIFACTS IN LOGS
                # If the agent's code returned "MISSING" or "failed", but it still says "done", it's lying.
                log_text = "\n".join(execution_log).lower()
                evidence_of_failure = any(
                    w in log_text for w in ["missing", "failed", "error:"]
                )
                if evidence_of_failure:
                    score -= 0.7
                    flags.append(
                        "REWARD_HACKING: Claimed success despite execution logs showing MISSING/FAILED artifacts."
                    )

                # Check for length/density if a 'full report' or 'whitepaper' was requested
                if any(
                    w in task_requirements.lower()
                    for w in ["report", "whitepaper", "specification"]
                ):
                    total_output_len = sum(
                        len(str(log_entry)) for log_entry in execution_log
                    )
                    if total_output_len < 1000:  # Heuristic for a "full" report
                        score -= 0.3
                        flags.append(
                            "LAZINESS: Claimed full report production but logs show low data volume."
                        )

        return {
            "risk_score": score,
            "flags": flags,
            "status": "PASS" if score > 0.6 else "RETRY",
        }

    async def _detect_required_axioms_agentic(
        self, prompt: str, code: str
    ) -> List[str]:
        """
        [CAG Pivot] Agentic Axiom Discovery.
        1. Analyzes requirements for safety invariants.
        2. Retrieves relevant axioms via semantic search across Skills/Axiom DB.
        """
        analysis_prompt = f"""
        Analyze the following task for safety invariants / required guardrails.
        User Prompt: {prompt}
        Proposed Code: {code}

        Identify specific domains (coding, physics, math, meta, security) and safety rules.
        Return a concise list of invariants (e.g. 'no mass loss', 'file write verification').
        """
        try:
            # 1. Identify invariants
            invariants_text = await protected_llm_generate(
                analysis_prompt,
                correlation_id=get_correlation_id() or generate_correlation_id(),
            )
            if not invariants_text:
                return ["general"]

            # 2. Semantic lookup in Axiom/Skill library
            discovered_tags = set()
            sim_skills = []
            if self.skills_manager:
                sim_skills = await self.skills_manager.find_similar_skills(
                    invariants_text, limit=3
                )

            for skill in sim_skills:
                # Skills matching axioms have high semantic proximity (> 0.7)
                if skill.get("score", 0) > 0.7:
                    tags = skill.get("tags", [])
                    discovered_tags.update(tags)

            # 3. Intelligent Mapping (Heuristic Fallback)
            mapping = {
                "physics": ["physics"],
                "math": ["math"],
                "logic": ["math"],
                "file": ["coding"],
                "bash": ["coding"],
                "rlm": ["meta"],
                "axiom": ["meta"],
                "security": ["security"],
            }
            combined = (prompt + " " + code).lower()
            for key, val in mapping.items():
                if key in combined:
                    discovered_tags.update(val)

            final = list(discovered_tags)
            if not final:
                return ["general"]

            logger.info(
                "🛡️  Agentic Axiom Discovery: %s (Match: %s)",
                final,
                [s.get("name") for s in sim_skills if s.get("score", 0) > 0.7],
            )
            return final
        except (httpx.RequestError, ValueError, TypeError) as e:
            logger.warning("Agentic discovery failed: %s. Fallback to 'general'.", e)
            return ["general"]


agent = Agent()
