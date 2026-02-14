"""
Recursive Logic Machine (RLM) Agent.
Handles the core execution loop, recursive querying, and tool integration.
"""

import asyncio
import datetime
import hashlib
import importlib.util
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
from .morphogenesis import MorphologicalMemory
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
        self.morph_memory = MorphologicalMemory()

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
            subfolders = ["plans", "research-reports", "outputs", "axioms", "workspace"]

            for sub in subfolders:
                path = kb_root / sub
                path.mkdir(parents=True, exist_ok=True)

            # Create a simple README if empty to guide users
            readme = kb_root / "README.md"
            if not readme.exists():
                readme.write_text(
                    "# Agent Knowledge Base\n\n"
                    "- `axioms/`: Human-readable rules and validators (CAG).\n"
                    "- `plans/`: Implementation plans and architectural docs.\n"
                    "- `research-reports/`: Deep research findings.\n"
                    "- `outputs/`: Final deliverables.\n"
                    "- `workspace/`: General scratchpad.\n"
                )

            logger.info("Knowledge Base structure verified at: %s", kb_root)
        except (OSError, AttributeError) as e:
            logger.warning("Failed to verify Knowledge Base structure: %s", e)

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

    def _refresh_scratchpad(
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
            pad = scratchpad_builder.build_scratchpad(
                session_id=session_id,
                root_session_id=root_session_id,
                task=task,
                current_step=current_step,
                max_steps=max_steps,
                current_round_id=current_round_id,
                morph_gestalt=morph_gestalt,
            )
            self.emit_event("scratchpad_text", content=pad, is_internal=True)
            return pad
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Scratchpad refresh failed: %s", e)
            return f"Error: Scratchpad unavailable ({e})"

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
                logger.error(
                    "Unexpected crash in execution thread (System error): %s", e
                )
                q.put({"type": "error", "content": f"Unexpected error: {e}"})
            finally:
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
                )
            )

        state = agent_state.get()
        if state:
            self.current_turn = state.turn_id

        self.final_result = None
        self.stop_requested = False
        self.awaiting_validation = False  # Reset validation state
        self.global_stop_event.clear()  # Ensure we don't start in a stopped state

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
        # --- MORPHOLOGICAL MEMORY SEEDING ---
        try:
            # Generate embedding for the core task prompt
            task_embedding = await self.llm.get_embedding(prompt)
            if task_embedding:
                self.morph_memory.seed(task_embedding)
                logger.info("Morphological Memory seeded with task embedding.")
        except (
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            httpx.RequestError,
        ) as e:
            logger.warning(
                "Failed to seed morphological memory (ML/Network error): %s", e
            )

        current_round_id = str(uuid.uuid4())
        current_round_started = datetime.datetime.now().timestamp() * 1000  # ms

        try:
            task_id = str(uuid.uuid4())
            logger.info(
                "Session %s: Starting Task %s (Round %s)",
                session_id,
                task_id,
                current_round_id,
            )

            self.db.create_thought_node(
                task_id,
                prompt,
                parent_id,
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
            try:
                from .meta_agents import meta_agents

                task_profile = meta_agents.generate_sub_agent_profile(prompt)
                plan_summary = (
                    f"Persona: {task_profile.get('persona', 'Generalist')} | "
                    f"Role: {task_profile.get('role', 'WORKER').value if hasattr(task_profile.get('role'), 'value') else task_profile.get('role', 'WORKER')} | "
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

        # 1. Base System Prompt
        base_system_prompt = await build_system_prompt()

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
            thought_id = str(uuid.uuid4())
            sheaf_diag = {"status": "HEALTHY", "consistency_energy": 0.0}
            vec = None
            repl_id = self.active_repls.get(session_id)

            # --- MORPHOLOGICAL MEMORY UPDATE ---
            try:
                # Allow the memory to 'grow' and consolidate
                self.morph_memory.update(steps=5)
                # Read the current gestalt state
                morph_gestalt = self.morph_memory.get_gestalt_string()
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("Morphological update failed (logic error): %s", e)
                morph_gestalt = None

            # --- DYNAMIC SCRATCHPAD REFRESH ---
            context_scratchpad = self._refresh_scratchpad(
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

            # 2b. Language Guard: Check if frontier is primarily non-English
            if frontier:
                # Heuristic...
                pass

            # Load Axioms (Semantic Retrieval)
            axioms_list_str = "None"
            if is_skills_available():
                try:
                    axioms_mgr = get_axioms_manager()

                    # [Context Optimization] Only load relevant axioms to reduce token noise
                    # Use prompt as the semantic key
                    search_query = (
                        getattr(self, "current_task_input", None)
                        or prompt
                        or "general safety"
                    )

                    # Semantic search for top 10 relevant axioms
                    relevant_axioms = await axioms_mgr.find_similar_axioms(
                        search_query, limit=10
                    )

                    if relevant_axioms:
                        axioms_list_str = ", ".join(
                            [a["name"] for a in relevant_axioms]
                        )
                        logger.debug(
                            "Loaded %d semantic axioms: %s",
                            len(relevant_axioms),
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
                f"{await build_system_prompt(skills_manager=self.skills_manager)}\n\n"
                f"{context_scratchpad}{hot_seat_warning}"
            )

            # --- NAVIGATOR CURIOSITY INJECTION (PRE-GEN) ---
            # If enabled, the Navigator assesses the current history and may inject
            # a curiosity-driven directive to guide exploration.
            if self.navigator and step % 3 == 0:  # Check periodically to avoid noise
                # Check for stagnation / low entropy in recent history
                # This is a lightweight check before we generate the next thought
                pass

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
                        system_message_content = [
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "text",
                                "text": "\n\n[CACHE MARKER] System Instructions End.",
                                "cache_control": {"type": "ephemeral"},
                            },
                        ]

                        # Manually construct messages list to bypass llm.generate's simple formatting
                        messages = [
                            {"role": "system", "content": system_message_content},
                            {"role": "user", "content": current_context},
                        ]

                        response_text = await self.llm.generate(
                            prompt=messages,
                            system=None,
                            stream=False,
                            stop=["</invoke>", "<|endoftext|>"],
                            on_usage=on_usage_update,
                        )
                    else:
                        # Standard execution
                        response_text = await self.llm.generate(
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

            formatted_thought = (
                f"[{timestamp_display}] [REPL: {repl_id_display}]\n{response_text}"
            )

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
            try:
                self.db.create_thought_node(
                    thought_id,
                    response_text,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    prompt_embedding=vec,
                    repl_id=repl_id,
                    status="running",
                    parent_id=self.current_thought_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
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
                    root_id=task_id,
                    hypothetical_node={"embedding": current_vec},
                    hypothetical_edges=hypothetical_edges,
                    goal_embedding=self.session_cache.get("task_embedding"),
                )

                # --- oMCD OPTIMAL STOPPING GATE ---
                # Evaluate whether to commit (stop) or continue deliberating.
                confidence = sheaf_diag.get("confidence", 0.5)
                omcd_decision = omcd.evaluate_step(step, confidence)

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
                    self.db.create_thought_node(
                        intervention_id,
                        intervention_prompt,
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
                output, execution_failed = await self._execute_code(
                    code,
                    thought_id,
                    session_id,
                    root_session_id=final_root_id,
                    task_input=prompt,
                    turn_id=self.current_turn,
                    step_id=step,
                )

                # Post-execution scratchpad refresh — agent must see latest state
                context_scratchpad = self._refresh_scratchpad(
                    session_id=session_id,
                    root_session_id=final_root_id,
                    task=prompt,
                    current_step=step,
                    max_steps=max_steps,
                    current_round_id=current_round_id,
                    morph_gestalt=morph_gestalt,
                )

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

            # Generate execution summary for scratchpad display using SUMMARY_MODEL
            # LLM-generated summary provides semantic understanding vs. simple truncation
            exec_summary = None
            if output:
                try:
                    # Use SUMMARY_MODEL if configured, otherwise fallback to main model
                    summary_model = (
                        settings.SUMMARY_MODEL or settings.get_llm_config().get("model")
                    )
                    summary_prompt = f"""Summarize this agent step in ONE sentence (max 100 chars):
ACTION: {code if code else response_text}
RESULT: {output}
STATUS: {thought_status}

Summary (describe WHAT was done and KEY outcome):"""
                    exec_summary = await protected_llm_generate(
                        summary_prompt,
                        model=summary_model,
                        correlation_id=get_correlation_id()
                        or generate_correlation_id(),
                    )
                    exec_summary = exec_summary.strip()[:150]

                    # Mark failure in summary
                    if thought_status == "failed":
                        exec_summary = f"[FAILED] {exec_summary}"
                except (httpx.RequestError, ValueError, TypeError, RuntimeError) as e:
                    logger.warning(
                        "Failed to generate step summary (LLM/Data error): %s", e
                    )
                    # Fallback to first line of output
                    exec_summary = output.strip().split("\n")[0][:100]
                    if thought_status == "failed":
                        exec_summary = f"[FAILED] {exec_summary}"

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
            except (AttributeError, RuntimeError, KeyError, ValueError) as e:
                logger.error(
                    "Failed to commit thought to graph (DB/Serialization error): %s", e
                )

            # Update Frontier Pointer
            # Update previous status for next iteration
            previous_thought_status = thought_status
            self.current_thought_id = thought_id

            # --- CONTINUOUS MORPHOLOGICAL MEMORY ---
            # Feed every committed thought's embedding back into the grid
            # so the gestalt evolves with the agent's actual trajectory.
            if final_vec is not None:
                try:
                    self.morph_memory.seed(final_vec)
                except (ValueError, TypeError, AttributeError) as morph_err:
                    logger.debug("Morph memory continuous seed skipped: %s", morph_err)

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

            # 1. Answer Detection & Terminal Triggers
            code_result = getattr(self, "final_result", None)

            # --- FORCED SYNTHESIS FOR CODE RESULTS ---
            # If we obtained a result via code (rlm.done or heuristic) but haven't explained it yet,
            # we force the agent to perform a "Synthesis Turn" to provide a readable summary.
            if code_result and not getattr(self, "synthesis_triggered", False):
                logger.info("🛡️ Triggering Final Synthesis Step for Code Result...")
                self.synthesis_triggered = True

                # Clear the "hard" result so we don't break the loop yet.
                # We store it in the prompt so the agent sees it.
                self.final_result = None

                synthesis_prompt = (
                    f"SYSTEM: Execution complete. Code returned: '{code_result}'. "
                    "You MUST now review the execution logs above and write a COMPREHENSIVE, "
                    "READABLE Final Answer summarizing your findings. Do not run more code."
                )

                self.db.create_thought_node(
                    str(uuid.uuid4()),
                    synthesis_prompt,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    parent_id=self.current_thought_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    repl_id=repl_id,
                )
                # Skip the rest of the loop to let the Agent generate the synthesis
                continue

            has_final_marker = any(
                t in response_text
                for t in [
                    "RLM_FINAL_OUTPUT",
                ]
            )

            # --- PREVENT PREMATURE VICTORY ---
            # If the agent claims it is "Done" but also provided code in this turn,
            # we MUST force a synthesis turn so it reviews the code's output.
            if (
                has_final_marker
                and code
                and not getattr(self, "synthesis_triggered", False)
            ):
                logger.info(
                    "🛡️ Premature Victory detected (Code + Final Marker). Forcing Synthesis Turn."
                )
                self.synthesis_triggered = True
                synthesis_prompt = (
                    "SYSTEM: You provided code and claimed a final answer in the same turn. "
                    "You MUST now review the execution logs (especially checking if all files exist) "
                    "and provide a final synthesis that accounts for the output below."
                )
                self.db.create_thought_node(
                    str(uuid.uuid4()),
                    synthesis_prompt,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    parent_id=self.current_thought_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    repl_id=repl_id,
                )
                continue

            if (
                has_final_marker or getattr(self, "final_result", None)
            ) and thought_status == "success":
                # --- EPISTEMIC VERIFICATION ---
                # Check for Laziness, Obsequiousness, and Reward Hacking before breaking.
                integrity_check = self._verify_epistemic_integrity(
                    thought_trace=response_text,
                    task_requirements=prompt,
                    execution_log=self.execution_logs.get(session_id, []),
                )

                if integrity_check["status"] == "RETRY" and not getattr(
                    self, "final_result", None
                ):
                    # SAFETY: Don't reset if the user hit STOP
                    if self.global_stop_event.is_set() or self.stop_requested:
                        logger.info(
                            "Integrity failure observed, but stop requested. Breaking."
                        )
                        self.stop_requested = True
                        break

                    logger.warning(
                        "🛡️ Epistemic Failure detected: %s", integrity_check["flags"]
                    )
                    # Reset final answer and inject critique
                    self.final_result = None
                    self.stop_requested = False

                    critique = (
                        "🛡️ SYSTEM INTEGRITY ALERT: Your response was flagged for: "
                        + ", ".join(integrity_check["flags"])
                        + "\nI MUST show my work, avoid being overly obsequious, "
                        + "and verify results with tools."
                    )
                    # EMIT to UI so the user sees this correction
                    self.emit_event("warning", content=critique)
                    self.db.create_thought_node(
                        str(uuid.uuid4()),
                        critique,
                        session_id=session_id,
                        root_session_id=final_root_id,
                        prompt_embedding=vec,
                        round_id=current_round_id,
                        turn_id=self.current_turn,
                        step_id=step,
                        repl_id=repl_id,
                    )
                    continue  # Keep the loop running
                # --- END VERIFICATION ---

                if not self.final_result:
                    self.final_result = response_text

                # 1. Generate Candidate Validated Response (Draft)
                # We do this BEFORE Dreamer so Dreamer can validate whether
                # this response resolves prior failures.
                final_response_candidate = None
                if self.final_result:
                    try:
                        final_response_candidate = (
                            await self._generate_validated_response(
                                final_root_id, prompt
                            )
                        )
                    except (httpx.RequestError, ValueError, TypeError) as e:
                        logger.warning(
                            "Failed to generate candidate response (LLM/Data error): %s",
                            e,
                        )

                # 2. Dreamer Trigger (Auto-Consolidate before exit)
                try:
                    logger.info("💤 Triggering Pre-Exit Dream Cycle (No timeout)...")
                    self.emit_event(
                        "thinking",
                        content="Synthesizing final answer (Lucid Dreaming)...",
                        tag="DREAMER",
                    )
                    try:
                        # Pass emit_event so Dreamer can emit progress to UI
                        def dreamer_emit(event_type, content, is_internal=False):
                            self.emit_event(
                                event_type,
                                content=content,
                                tag="DREAMER",
                                is_internal=is_internal,
                            )

                        # Re-generate scratchpad for Dreamer context if needed
                        # Or use the last known context. The prompt variable might contain it,
                        # but scratchpad_content is cleaner
                        scratchpad_content = self._refresh_scratchpad(
                            session_id=session_id,
                            root_session_id=final_root_id,
                            task=prompt,
                            current_step=step,
                            max_steps=max_steps,
                            current_round_id=current_round_id,
                        )

                        dream_res = await dreamer.dream_cycle(
                            emit_callback=dreamer_emit,
                            session_id=final_root_id,  # Scope to current session only
                            final_response_candidate=final_response_candidate,
                            context=scratchpad_content,  # Pass the full context
                        )
                        logger.info(
                            "💤 Dream Cycle Completed. Status: %s",
                            dream_res.get("status"),
                        )
                    except (
                        httpx.RequestError,
                        ValueError,
                        TypeError,
                        RuntimeError,
                    ) as e:
                        logger.warning(
                            "Dream cycle failed during execution (LLM/State error): %s",
                            e,
                        )
                        dream_res = {}

                    dream_status = dream_res.get("status", "")
                    insight = dream_res.get("insight") or ""

                    # GATEKEEPER LOGIC: Absolute Validation
                    # Validates "Surprise" (Dreamer) against the execution trace.
                    # Validation must be absolute (Pass/Fail based on Axioms), not probabilistic.

                    should_trigger_healing = False

                    # 1. Critical Failures or Insights always trigger healing
                    if dream_status in ("error", "critical"):
                        should_trigger_healing = True
                    elif dream_status == "lucid" and insight:
                        logger.info(
                            "🛡️ Gatekeeper: Systemic Issue/Insight detected. Forcing Self-Healing."
                        )
                        should_trigger_healing = True
                    elif insight:
                        # Any insight at this stage suggests a contradiction or hallucination
                        logger.info(
                            "🛡️ Gatekeeper: Explicit Insight detected. Restricting exit."
                        )
                        should_trigger_healing = True

                    if should_trigger_healing and insight:
                        if insight:
                            dreamer_msg = (
                                f"💤 [Dreamer Gatekeeper]: Systemic Issue Detected. "
                                f"Blocking Exit.\n\n{insight}"
                            )
                            self.emit_event(
                                "thinking",
                                content=dreamer_msg,
                                tag="DREAMER",
                            )
                            # CRITICAL FIX: Persist to DB so UI sees it
                            if self.current_thought_id:
                                try:
                                    # Append to existing summary or content
                                    self.db.query(
                                        "MATCH (t:Thought {id: $tid}) SET t.execution_summary = coalesce(t.execution_summary, '') + $msg RETURN t",
                                        {
                                            "tid": self.current_thought_id,
                                            "msg": f"\n\n---\n{dreamer_msg}",
                                        },
                                    )
                                except (
                                    AttributeError,
                                    RuntimeError,
                                    KeyError,
                                ) as db_err:
                                    logger.error(
                                        "Failed to persist Dreamer msg (DB error): %s",
                                        db_err,
                                    )

                        # 2. Check if we've already tried to fix this exact issue
                        # to prevent infinite loops
                        last_insight = getattr(self, "last_dream_insight", None)
                        if last_insight != insight:
                            # 3. REJECT EXIT. Force Self-Healing.
                            logger.info(
                                "🛡️ Dreamer Gatekeeper REJECTED exit. Injecting insight for self-healing."
                            )
                            self.last_dream_insight = insight
                            self._dreamer_retry_count = 0  # Reset counter for new issue
                            self.final_result = None  # Cancel valid result
                            self.stop_requested = False
                            self.synthesis_triggered = False

                            # Inject the Insight as a High-Priority Thought
                            rejection_msg = (
                                f"DREAMER GATEKEEPER MANDATE: I cannot accept this Result yet.\n"
                                f"I detected a systemic failure pattern (Topic Drift or "
                                f"Meta-Cognitive Loop) in your recent actions:\n{insight}\n\n"
                                "URGENT COMPLIANCE: You must immediately pivot away from meta-diagnosis of yourself. "
                                "Use `rlm.recall(node_id)` as directed above to retrieve the missing research data and fulfill the *original* user task. "
                                "Failure to ground your next thought in technical research data will result in another rejection."
                            )

                            rejection_id = str(uuid.uuid4())
                            self.db.create_thought_node(
                                rejection_id,
                                rejection_msg,
                                session_id=session_id,
                                root_session_id=final_root_id,
                                dreamer_analysis=insight,
                                round_id=current_round_id,
                                status="rejected",
                                turn_id=self.current_turn,
                                step_id=step,
                                repl_id=repl_id,
                            )
                            self.current_thought_id = rejection_id
                            # Set internal state to force recovery behavior
                            self.synthesis_triggered = False
                            continue  # Loop back
                        else:
                            # REMOVED: Force exit bypass that allowed agent to ignore Dreamer
                            # Instead, escalate with different guidance or ask user for help
                            dreamer_retry_count = getattr(
                                self, "_dreamer_retry_count", 0
                            )
                            self._dreamer_retry_count = dreamer_retry_count + 1

                            if dreamer_retry_count >= 3:
                                # Escalate to user - agent is stuck
                                logger.error(
                                    "🚨 Dreamer loop exhausted. Agent cannot self-heal. Escalating to user."
                                )
                                self.emit_event(
                                    "error",
                                    content=(
                                        "🚨 **DREAMER ESCALATION**: The agent has tried 3 times to fix "
                                        f"the following issue but cannot self-heal:\n\n{insight}\n\n"
                                        "**Please provide guidance or adjust the task.**"
                                    ),
                                )
                                # Don't allow final result to be set - wait for user
                                self.final_result = None
                                break  # Exit but without success

                            else:
                                # Give escalating guidance
                                logger.warning(
                                    "Dreamer loop detected (attempt %d/3). Forcing retry with different approach.",
                                    dreamer_retry_count + 1,
                                )
                                escalation_msg = (
                                    f"DREAMER ESCALATION (Attempt {dreamer_retry_count + 1}/3): "
                                    "Your previous approach did not resolve the issue. "
                                    "You MUST try a FUNDAMENTALLY DIFFERENT approach this time.\n\n"
                                    f"PERSISTENT ISSUE:\n{insight}\n\n"
                                    "Do NOT repeat the same actions. Consider:\n"
                                    "1. Using different tools or methods.\n"
                                    "2. Breaking the task into smaller steps.\n"
                                    "3. Asking clarifying questions if requirements are unclear."
                                )
                                escalation_id = str(uuid.uuid4())
                                self.db.create_thought_node(
                                    escalation_id,
                                    escalation_msg,
                                    session_id=session_id,
                                    root_session_id=final_root_id,
                                    dreamer_analysis=insight,
                                    round_id=current_round_id,
                                    status="rejected",
                                    turn_id=self.current_turn,
                                    step_id=step,
                                    repl_id=repl_id,
                                )
                                self.current_thought_id = escalation_id
                                self.final_result = None
                                self.stop_requested = False
                                continue  # Loop back for retry
                except (
                    AttributeError,
                    RuntimeError,
                    KeyError,
                    ValueError,
                    httpx.RequestError,
                ) as e:
                    logger.warning(
                        "Dream cycle failed on exit (state/network error): %s", e
                    )

                # 3. Store the final response (If we passed the Gatekeeper)
                if final_response_candidate:
                    try:
                        # Emit the Agent's own synthesized summary for the UI.
                        # This candidate was generated by the Agent's _generate_validated_response method.
                        # The Dreamer (if run) only inspected it for self-healing; it did not produce it.
                        self.emit_event(
                            "RLM_FINAL_OUTPUT",
                            content=final_response_candidate,
                            tag="AGENT_SUMMARY",
                        )

                        self.db.create_thought_node(
                            str(uuid.uuid4()),
                            final_response_candidate,  # Store the RICH response as content
                            session_id=session_id,
                            root_session_id=final_root_id,
                            status="success",
                            final_response=final_response_candidate,
                            round_id=current_round_id,
                            turn_id=self.current_turn,
                            step_id=step,
                            repl_id=repl_id,
                        )
                    except (AttributeError, RuntimeError, KeyError) as e:
                        logger.warning(
                            "Failed to store final response (DB/State error): %s", e
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
                reflexion_id = str(uuid.uuid4())
                self.db.create_thought_node(
                    reflexion_id,
                    reflexion_content,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    prompt_embedding=vec,
                    parent_id=self.current_thought_id,
                    round_id=current_round_id,
                    turn_id=self.current_turn,
                    step_id=step,
                    repl_id=repl_id,
                )

                # Update pointer
                self.current_thought_id = reflexion_id

                # Do NOT break. Let the loop continue.
                continue

        # 8. Loop Exit: Emit Final Answer if available
        if self.final_result:
            # Phase 2: Axiomatic Validation on FINAL synthesis
            try:
                axiom_diag = await sheaf.check_axiomatic_consistency(
                    self.final_result,
                    task_tags=["final_synthesis"],
                    depth=self.current_depth,
                )
                if axiom_diag.get("status") == "AXIOMATIC_VIOLATION":
                    axiom_critique = axiom_diag.get("critique")
                    logger.warning(
                        "🛡️ Final Synthesis Blocked by Axiom: %s", axiom_critique
                    )

                    # Instead of breaking, we retry one more time with a reflexion
                    reflexion_id = str(uuid.uuid4())
                    self.db.create_thought_node(
                        reflexion_id,
                        f"SYSTEM CRITIQUE: My final response was rejected by axiom validation.\nCritique: {axiom_critique}\nI MUST rewrite the final answer to address this.",
                        session_id=session_id,
                        root_session_id=final_root_id,
                        parent_id=self.current_thought_id,
                        round_id=current_round_id,
                        status="reflexion",
                        step_id=step,
                    )
                    self.current_thought_id = reflexion_id
                    self.final_result = None  # Clear result to continue loop
                    # Return to loop start
                    return await self.query_sync(
                        prompt,
                        parent_id=reflexion_id,
                        session_id=session_id,
                        depth=self.current_depth,
                        root_session_id=final_root_id,
                        turn_id=self.current_turn,
                        recursion_stack=recursion_stack,
                        metadata=metadata,
                    )
            except (
                AttributeError,
                RuntimeError,
                KeyError,
                ValueError,
                httpx.RequestError,
            ) as e:
                self.emit_event(
                    "error",
                    content=f"Axiomatic check on final response failed (state/network error): {e}",
                )

            # --- NEW VALIDATION PROTOCOL (v2) ---
            # 1. Emit Initial Attempt (may already have been emitted by rlm.done())
            if not getattr(self, "awaiting_validation", False):
                self.emit_event("RLM_INITIAL_RESPONSE", content=self.final_result)

            # 2. Fresh scratchpad rebuild for validation — Dreamer needs current state
            context_scratchpad = self._refresh_scratchpad(
                session_id=session_id,
                root_session_id=final_root_id,
                task=prompt,
                current_step=step,
                max_steps=max_steps,
                current_round_id=current_round_id,
            )

            # 3. Dreamer Validation Handshake
            try:
                validation = await dreamer.validate_response(
                    candidate=self.final_result,
                    context=context_scratchpad,
                    session_id=session_id,
                    current_step=step,
                    goal_embedding=self.session_cache.get("task_embedding"),
                    turn_id=self.current_turn,
                )

                status = validation.get("status")

                if status in ["valid", "forced_valid"]:
                    # SUCCESS: Dreamer validated — emit validated signal
                    self.emit_event(
                        "RLM_DREAMER_VALIDATED",
                        content=validation.get("message"),
                        tag="DREAMER",
                    )
                    self.emit_event("RLM_FINAL_OUTPUT", content=self.final_result)
                    self.eval_success_count += 1
                    # Fall through to return (since we are outside the loop)
                elif status == "exhausted":
                    # FAILURE: Budget exhausted — Hard Stop
                    instruction = validation.get("instruction", "Budget exhausted.")
                    self.emit_event(
                        "error",
                        content=f"Terminating due to exhaustion: {instruction}",
                        tag="DREAMER",
                    )
                    self.eval_failure_count += 1
                    return f"Terminated: {instruction}"
                else:
                    # FAILURE: Dreamer found issues — bounded retry (max 2)
                    validation_retries = getattr(self, "_validation_retries", 0)
                    instruction = validation.get(
                        "instruction", "Review validation failure."
                    )

                    if validation_retries >= 2:
                        # Max retries reached — force-emit best result
                        logger.warning(
                            "Validation retry limit reached (%s). Force-emitting best result.",
                            validation_retries,
                        )
                        self.emit_event(
                            "warning",
                            content=f"Validation failed after {validation_retries} retries. Emitting best result.",
                            tag="DREAMER",
                        )
                        self.emit_event("RLM_FINAL_OUTPUT", content=self.final_result)
                        self._validation_retries = 0  # Reset for next turn
                    else:
                        # Retry within this turn (no recursive restart)
                        self._validation_retries = validation_retries + 1
                        self.emit_event(
                            "RLM_DREAMER_ISSUES",
                            content=f"[Retry {self._validation_retries}/2] {instruction}",
                            tag="DREAMER",
                        )

                        # Log Wake Node (scoped to this turn)
                        wake_id = str(uuid.uuid4())
                        self.db.create_thought_node(
                            wake_id,
                            f"SYSTEM WAKE (retry {self._validation_retries}): {instruction}",
                            parent_id=self.current_thought_id,
                            session_id=session_id,
                            root_session_id=final_root_id,
                            round_id=current_round_id,
                            status="wake",
                            step_id=step,
                            turn_id=self.current_turn,
                        )

                        self.current_thought_id = wake_id
                        self.final_result = None  # Clear for re-attempt

                        return await self.query_sync(
                            prompt,
                            parent_id=wake_id,
                            session_id=session_id,
                            depth=self.current_depth,
                            root_session_id=final_root_id,
                            turn_id=self.current_turn,
                            recursion_stack=recursion_stack,
                            metadata=metadata,
                        )

            except (
                AttributeError,
                RuntimeError,
                KeyError,
                ValueError,
                httpx.RequestError,
            ) as e:
                logger.error("Validation Handshake failed (state/network error): %s", e)
                # Fallback to Safe Exit on system error
                self.emit_event("error", content=f"Validation System Error: {e}")
                self.emit_event("RLM_FINAL_OUTPUT", content=self.final_result)
        elif self.stop_requested:
            # Stop requested by user or tool
            self.emit_event(
                "RLM_FINAL_OUTPUT",
                content="Task processing stopped (Done/Stop signal received).",
            )
            self.eval_success_count += 1  # User-initiated stop is still a success
        elif step >= max_steps:
            self.emit_event(
                "error",
                content=f"AGENT LIMIT REACHED: Reached max_steps ({max_steps}). Stopping execution.",
            )
            logger.warning(
                "Session %s reached max steps (%s) and aborted.", session_id, max_steps
            )
            self.eval_failure_count += 1  # Max steps reached = failure
        else:
            # Fallback: Emit a system notice if the loop exits without a result (e.g. error/circuit breaker)
            # This prevents the UI from hanging fastidiously waiting for an event that never comes.
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
                final_scratchpad = self._refresh_scratchpad(
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

        # Return the final result or a default message if not set
        return self.final_result or "Task processing stopped."

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
    ) -> Tuple[str, bool]:
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
            if not output.strip() and not stderr:
                output = "Code executed successfully (No output captured)."
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
                except (OSError, RuntimeError) as log_err:
                    logger.warning(
                        "Failed to save full debug log (IO error): %s", log_err
                    )
                    output = output[:1500] + "\n... [EMERGENCY TRUNCATION] ..."

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

            return output, execution_failed

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
            placeholders = re.findall(r"\{[a-zA-Z0-9_]+\}", thought_trace)
            todo_markers = any(
                m in thought_trace for m in ["[TODO]", "TODO:", "FIXME", "..."]
            )
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
