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
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
import redis

from ..mcp_integration.runtime import AgentRuntime, set_stop_event
from ..mcp_integration.skill_storage import get_skills_manager
from .circuit import (
    CircuitOpenError,
    generate_correlation_id,
    get_correlation_id,
)
from .config import settings
from .db import GraphClient, db
from .dream import dreamer
from .exceptions import ValidationError
from .exceptions.codes import ErrorCode
from .llm import llm
from .logger import get_logger
from .mcp_runtime import get_mcp_server_names, is_mcp_available
from .meta_agents import meta_agents
from .navigator import Navigator

# from .prompts import build_system_prompt
from .reflexion import intelli_synth
from .rlm_interface import RLMInterface
from .scratchpad_builder import scratchpad_builder
from .semantic_summarizer import summarize_event
from .services.circuit import protected_llm_generate
from .sheaf import sheaf
from .state import (
    ExecutionState,
    agent_state,
    broadcast_trace,
    execution_events,
)
from .thimac_memory import ThimacIntention, ThimacMemory
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
        self._last_reflexion_step: int = -10  # Cooldown tracker for Sheaf reflexion

        # === EVALUATION COUNTERS ===
        # Track success/failure for session-level and global metrics
        self.eval_success_count: int = 0  # Successful task completions
        self.eval_failure_count: int = 0  # Failed tasks (errors, timeouts)
        self.eval_step_count: int = 0  # Total steps executed
        self.eval_dreamer_interventions: int = 0  # Dreamer correction count
        self.round_started_at: int = 0

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
                "You are the Governance Module. Translate the USER TASK into a SEARCH QUERY "
                "for Validation Rules (Axioms). Axioms are Python validators for domains like: "
                "'file persistence', 'math safety', 'python syntax', 'epistemic integrity', "
                "'security'. Return ONLY a comma-separated list of relevant domains and "
                "technical keywords."
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
        execution_state: Optional[Any] = None,
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
                execution_state=execution_state,
                memory_trajectory=(
                    self.morph_memory.all_events if self.morph_memory else None
                ),
            )
            self.emit_event("scratchpad_text", content=pad, is_internal=True)
            return pad
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Scratchpad refresh failed: %s", e)
            return f"Error: Scratchpad unavailable ({e})"

    async def _sync_thimac(
        self,
        thought_id: str,
        prompt: str,
        status: str,
        result: Optional[str],
        step: int,
        session_id: str,
        round_id: str,
        turn_id: Optional[int] = None,
        repl_id: Optional[str] = None,
        logical_id: Optional[str] = None,
        tool_calls: Optional[List[str]] = None,
        is_branching: bool = False,
        intent_type: Optional[ThimacIntention] = None,
        embedding: Optional[List[float]] = None,
        parent_id: Optional[str] = None,
        sheaf_score: Optional[float] = None,
        omcd_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Helper to ingest a thought node into Thimac memory and return classification."""
        try:
            # Calculate MDL Compression Gain (Phase 4 Gestalt)
            compression_gain = 0.0
            if getattr(self, "navigator", None):
                content = f"{prompt}\n{result}" if result else prompt
                compression_gain = self.navigator.compute_compression_progress(content)

            thimac_thought_data = {
                "id": thought_id,
                "prompt": prompt,
                "status": status,
                "result": result,
                "created_at": int(
                    datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000
                ),
                "session_id": session_id,
                "root_session_id": self.session_cache.get(
                    "root_session_id", session_id
                ),
                "round_id": round_id or "ROOT",
                "turn_id": turn_id if turn_id is not None else self.current_turn,
                "step_id": step,
                "repl_id": repl_id,
                "logical_id": logical_id,
                "compression_gain": compression_gain,
                "frequency": 1.0,  # Defaults, will be refined in ingest_thought
                "confidence": 0.9,
                "rtm_depth": 0,
                "metadata": metadata or {},
            }

            # Generate Semantic Gist (Phase 4.6)
            semantic_gist = ""
            if result or prompt:
                summary_model = getattr(
                    settings, "SUMMARY_MODEL", "google/gemini-2.0-flash-lite"
                )
                # If it's a failure, we still want a gist explaining what failed
                context_result = result if result else f"[Status: {status}]"
                semantic_gist = await summarize_event(
                    prompt, context_result, model=summary_model
                )

            event = self.morph_memory.ingest_thought(
                thimac_thought_data,
                tool_calls,
                is_branching=is_branching,
                semantic_gist=semantic_gist,
                intent_type=intent_type,
                embedding=embedding,
                parent_id=parent_id,
                sheaf_score=sheaf_score,
                omcd_score=omcd_score,
            )

            # Warm Persistence: Immediate write to DB for UI/refresh consistency
            try:
                self.db.create_thought_node(
                    thought_id=thought_id,
                    prompt=prompt,
                    logical_id=logical_id,
                    parent_id=parent_id,
                    session_id=session_id,
                    root_session_id=thimac_thought_data["root_session_id"],
                    repl_id=repl_id,
                    status=status,
                    result=result,
                    round_id=round_id,
                    turn_id=thimac_thought_data["turn_id"],
                    step_id=step,
                    sheaf_score=sheaf_score,
                    omcd_score=omcd_score,
                    semantic_gist=semantic_gist,
                    inference_pressure=event.inference_pressure if event else None,
                    relational_gravity=event.relational_gravity if event else None,
                    epistemic_eros=event.epistemic_eros if event else None,
                    free_energy=event.free_energy if event else None,
                    metabolic_state=event.metabolic_state if event else None,
                    frequency=event.frequency if event else None,
                    confidence=event.confidence if event else None,
                    rtm_depth=event.rtm_depth if event else None,
                    validate=False,  # Skip redundant guardrails
                )
            except (AttributeError, RuntimeError, ValueError) as db_err:
                logger.warning("Warm persistence failed for %s: %s", thought_id, db_err)

            # Persistent Homology Pipeline
            if compression_gain > 0.1 and status == "success":
                trace_action(
                    "THIMAC",
                    "PERSISTENT_HOMOLOGY",
                    f"Node {thought_id[:8]} compressed state space significantly (+{compression_gain:.2f}). Flagged for survival.",
                )

            res = event.to_dict()
            res["operation_reason"] = event.operation_reason
            return res
        except (AttributeError, ValueError, TypeError, KeyError) as e:
            logger.error(
                "Thimac ingestion failed for thought %s: %s",
                thought_id,
                e,
                exc_info=True,
            )
            return None

    async def _flush_memory_chain(self, final_event_id: Optional[str]):
        """
        Traverses the parent_id chain in RAM and flushes the successful branch
        to the permanent global graph (FalkorDB).
        """
        if not self.morph_memory or not final_event_id:
            return

        # 1. Map all events for quick lookup
        event_map = {e.thought_id: e for e in self.morph_memory.all_events}

        # 2. Reconstruct the chain from final event upward
        chain = []
        curr_id: Optional[str] = final_event_id
        while curr_id and curr_id in event_map:
            ev = event_map[curr_id]
            chain.append(ev)
            curr_id = ev.parent_id
            # Safety break for loops
            if len(chain) > 100:
                break

        # 3. Flush chain in causal order (bottom-up)
        flushed_ids = set()
        for event in reversed(chain):
            if event.thought_id in flushed_ids:
                continue

            try:
                # We reuse create_thought_node which handles the Cypher MERGE/CREATE
                self.db.create_thought_node(
                    thought_id=event.thought_id,
                    prompt=event.full_data,
                    logical_id=event.logical_id,
                    session_id=str(event.session_id),
                    root_session_id=str(event.root_session_id),
                    prompt_embedding=event.embedding,
                    repl_id=event.repl_id,
                    status=event.status,
                    parent_id=event.parent_id,
                    turn_id=event.turn_id,
                    step_id=event.step_id,
                    epistemic_eros=event.epistemic_eros,
                    inference_pressure=event.inference_pressure,
                    relational_gravity=event.relational_gravity,
                    free_energy=event.free_energy,
                    metabolic_state=event.metabolic_state,
                    semantic_gist=event.semantic_gist,
                    step_summary=event.summary,
                    validate=False,  # Skip guardrails during flush
                )
                flushed_ids.add(event.thought_id)
            except (
                redis.exceptions.RedisError,
                redis.exceptions.ResponseError,
                AttributeError,
                ValueError,
                TypeError,
                KeyError,
            ) as e:
                logger.error("Failed to flush event %s to DB: %s", event.thought_id, e)

        logger.info(
            "Memory Flush: Consolidated %d nodes to final graph.", len(flushed_ids)
        )

    async def _create_system_node(
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
        repl_id: Optional[str] = "SYS",
        result: Optional[str] = None,
        thought_id: Optional[str] = None,
        analysis: Optional[Dict] = None,
        validate: bool = False,
        is_branching: bool = False,
        sheaf_score: Optional[float] = None,
        omcd_score: Optional[float] = None,
    ) -> str:
        """Standardized helper for materializing system-level reasoning in the graph."""
        thought_id = thought_id or str(uuid.uuid4())
        try:
            # Sync system node to Thimac first to get metadata
            # NOTE: DB write is deferred to the Batch Flush mechanism.
            await self._sync_thimac(
                thought_id=thought_id,
                prompt=summary,
                status=status,
                result=result,
                step=step_id,
                session_id=session_id,
                round_id=round_id,
                turn_id=turn_id,
                repl_id=repl_id,
                logical_id=logical_id,
                tool_calls=None,
                is_branching=is_branching,
                parent_id=parent_id,
                sheaf_score=sheaf_score,
                omcd_score=omcd_score,
                metadata={
                    "analysis": analysis,
                    "validate": validate,
                    "root_session_id": root_session_id,
                },
            )

            # Note: The system node is now purely in Thimac RAM.
            # It will be flushed to the DB if it is part of a successful branch
            # during the Batch Consolidation phase (RELEASE/ACCEPT).

            # Emit graph_update for UI visibility
            self.emit_event(
                "graph_update",
                data={
                    "action": "add_node",
                    "node": {
                        "id": thought_id,
                        "label": summary[:50],
                        "group": 3 if status == "system" else 2,
                        "status": status,
                    },
                },
            )

            return thought_id

        except (AttributeError, RuntimeError, KeyError, TypeError, ValueError) as e:
            logger.error(
                "Failed to create system node %s (LID: %s): %s",
                thought_id,
                logical_id,
                e,
            )
            return thought_id

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
                        warning_text = (
                            "[WARNING: DREAMER REJECTED]\n"
                            f"{self.last_rejected_result}\n\n"
                            "(System Note: This result was rejected by the Dreamer but "
                            "is provided as the best available draft.)"
                        )
                        q.put({"type": "RLM_FINAL_OUTPUT", "content": warning_text})

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

    async def _initialize_turn(
        self,
        prompt: str,
        parent_id: Optional[str],
        session_id: str,
        depth: int,
        root_session_id: Optional[str],
        turn_id: int,
        recursion_stack: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Initializes the agent state for a new turn (query_sync call).
        Returns a dictionary of context data.
        """
        final_root_id = root_session_id if root_session_id else session_id
        self.session_cache["root_session_id"] = final_root_id
        trace_action(
            "AGENT",
            "QUERY_SYNC",
            result=f"Session: {session_id} | Depth: {depth}",
            tag="AGENT",
        )

        # 0. Reset scoped State for this specific call
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
        self.awaiting_validation = False
        self.last_dream_insight = None
        self._dreamer_retry_count = 0
        self._validation_retries = 0
        self.global_stop_event.clear()
        self.round_started_at = int(time.time())

        logger.info(
            "🛡️ [Agent] RLM Loop State Reset for Session %s (Turn %d)",
            session_id,
            self.current_turn,
        )

        if session_id not in self.active_repls:
            self.active_repls[session_id] = f"so-{session_id[:8]}"

        if is_mcp_available():
            set_stop_event(self.global_stop_event)

        current_round_id = f"{session_id}:Round:{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"

        try:
            task_lid = f"{session_id}:Task:0"
            task_id = str(uuid.uuid4())
            await self._sync_thimac(
                thought_id=task_id,
                prompt=prompt,
                status="task",
                result=None,
                step=0,
                session_id=session_id,
                round_id=current_round_id,
                repl_id=self.active_repls.get(session_id),
                logical_id=task_lid,
            )
            self.current_thought_id = task_id

            # Meta-Agent Profiling (Inlined for now, could be further abstracted)
            task_profile = await self._generate_task_profile(prompt)

            # Breaker Protocol Injection
            if meta_agents.should_spawn_breakers(prompt, len(prompt), depth=depth):
                breaker_instructions = meta_agents.get_breaker_instructions(
                    prompt, fragment_index=0
                )
                self.emit_event(
                    "RLM_BREAKER_PROTOCOL",
                    content="Task complexity detected. BREAKER protocol injected.",
                    tag="META_AGENT",
                )
                prompt = f"{breaker_instructions}\n\n{prompt}"

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
        except (AttributeError, RuntimeError, KeyError, ValueError) as e:
            logger.error("Failed to initialize Task node: %s", e)
            task_id = str(uuid.uuid4())
            self.current_thought_id = task_id
            current_round_id = f"fallback-{int(time.time())}"
            task_profile = {"persona": "Autonomous Generalist", "tools": ["rlm"]}

        if parent_id:
            self.emit_event(
                "graph_update",
                data={
                    "action": "add_link",
                    "link": {"source": parent_id, "target": task_id},
                },
            )

        # Final Context Construction
        max_steps = 1000  # Safety ceiling (increased from 15 per user request)

        # Initial Scratchpad (Step 0)
        pad = await self._refresh_scratchpad(
            session_id=session_id,
            root_session_id=final_root_id,
            task=prompt,
            current_step=0,
            max_steps=max_steps,
            current_round_id=current_round_id,
            morph_gestalt=None,
            execution_state=agent_state.get(),
        )

        # [GROUNDING] Proactively inject task_input into the REPL at Step 0
        try:
            asyncio.create_task(
                self.runtime.execute(
                    f"task_input = {repr(prompt)}",
                    context={"session_id": session_id, "step_id": 0},
                )
            )
        except (AttributeError, RuntimeError) as e:
            logger.warning("Initial REPL grounding failed: %s", e)

        # [PRE-FLIGHT] Axiom Discovery
        # Search for domain-specific validators before the agent even acts.
        relevant_axioms = []
        try:
            from ..mcp_integration.skill_storage import get_axioms_manager

            # 1. Generate Query
            search_query = await self._generate_axiom_search_query(prompt)

            # 2. Semantic Search
            mgr = get_axioms_manager()
            similar = await mgr.find_similar_axioms(search_query, limit=5)

            # 3. Store Metadata (Name + Description)
            for ax in similar:
                if ax.get("score", 0) > 0.4:  # Relevance threshold
                    relevant_axioms.append(
                        {
                            "name": ax.get("name"),
                            "description": ax.get(
                                "description", "No description available."
                            ),
                        }
                    )

            if relevant_axioms:
                logger.info(
                    "🛡️ [Axiom Discovery] Found %d relevant pre-flight axioms for turn.",
                    len(relevant_axioms),
                )
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning("Pre-flight axiom discovery failed: %s", e)

        return {
            "pad": pad or "",
            "root_id": final_root_id,
            "round_id": current_round_id,
            "repl_id": self.active_repls.get(session_id, "isolated"),
            "task_profile": task_profile,
            "prompt": prompt,
            "task_id": task_id,
            "step": 0,
            "max_steps": max_steps,
            "exec_state": agent_state.get() or ExecutionState(),
            "relevant_axioms": relevant_axioms,
        }

    async def _generate_task_profile(self, prompt: str) -> Dict[str, Any]:
        """Helper to generate sub-agent profile with timeout."""
        try:
            mcp_names = get_mcp_server_names() if is_mcp_available() else []
            skills_mgr = get_skills_manager() if is_mcp_available() else None

            task_profile = await asyncio.wait_for(
                meta_agents.generate_sub_agent_profile(
                    prompt, skills_manager=skills_mgr, mcp_names=mcp_names
                ),
                timeout=5.0,
            )

            role_val = task_profile.get("role", "WORKER")
            role_str = role_val.value if hasattr(role_val, "value") else str(role_val)
            persona = task_profile.get("persona", "Generalist")
            tools = ", ".join(task_profile.get("tools", ["All"]))
            plan_summary = f"Persona: {persona} | Role: {role_str} | Tools: {tools}"

            self.emit_event("RLM_AGENT_TASK_PLAN", content=plan_summary, tag="AGENT")
            trace_action("AGENT", "TASK_PLAN", result=plan_summary, tag="AGENT")
            return task_profile
        except (
            AttributeError,
            RuntimeError,
            KeyError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            logger.warning("Task profiling failed: %s", e)
            return {
                "persona": "Autonomous Generalist",
                "tools": ["rlm"],
                "role": "execution",
            }

    async def _initialize_step(
        self, step: int, session_id: str, turn_ctx: Dict[str, Any]
    ) -> None:
        """
        Initializes the context and prompt for a single step.
        """
        task_profile = turn_ctx["task_profile"]
        exec_state = turn_ctx["exec_state"]

        logical_id = f"{session_id[:8]}:T{self.current_turn}:S{step}"
        thought_id = str(uuid.uuid4())

        # Thimac Gestalt
        try:
            morph_gestalt = (
                self.morph_memory.get_gestalt_string() if self.morph_memory else None
            )
        except (AttributeError, RuntimeError) as e:
            logger.warning("Thimac gestalt update failed: %s", e)
            morph_gestalt = None

        # Dashboard Metrics
        dashboard_data = await self._get_dashboard_metrics(exec_state)

        # Build System Prompt - HIGHER STABILITY: State remains in User Message ONLY
        from .prompts import build_system_prompt

        system_prompt = await build_system_prompt(
            skills_manager=self.skills_manager,
            agent_profile=task_profile,
            dashboard_data=dashboard_data,
            relevant_axioms=turn_ctx.get("relevant_axioms", []),
        )
        system_prompt += (
            "\n\n--- FILE OPERATIONS & GROUNDING ---\n"
            "CRITICAL: If your action creates or modifies a file, you MUST print the absolute path "
            "and a small snippet of the saved content to stdout. Silent file writes will be rejected as hallucinations."
        )

        # Hot Seat Injection
        if getattr(self, "last_dream_insight", None):
            system_prompt += (
                "\n\n--- ⚠️ HOT SEAT: EPISTEMIC RECOVERY ACTIVE ---\n"
                f"Your previous response was REJECTED by the Dreamer Gatekeeper.\n"
                f"CRITIQUE: {self.last_dream_insight}\n"
                "Address the contradiction and provide a GROUNDED response.\n---"
            )

        # Synthesis Hardening
        if getattr(self, "synthesis_triggered", False):
            system_prompt += "\n\n--- ⚠️ SYNTHESIS ENFORCEMENT ---\nFINAL SUMMARY mode. NO tools permitted."

        # Navigator/Sheaf/Axioms could be added here similarly or kept in turn_ctx

        self.current_thought_id = thought_id
        turn_ctx.update(
            {
                "logical_id": logical_id,
                "thought_id": thought_id,
                "system_prompt": system_prompt,
                "morph_gestalt": morph_gestalt,
                "dashboard_data": dashboard_data,
            }
        )

    async def _get_dashboard_metrics(self, exec_state: Any) -> Dict[str, Any]:
        """Helper to fetch dashboard metrics."""
        data = {}
        try:
            if self.morph_memory and self.morph_memory.all_events:
                event = self.morph_memory.all_events[-1]
                data = {
                    "sheaf_energy": (
                        f"{event.sheaf_score:.2f}" if event.sheaf_score else "0.00"
                    ),
                    "omcd_score": (
                        f"{event.omcd_score:.2f}" if event.omcd_score else "0.00"
                    ),
                    "thimac_op": (
                        event.operation.value
                        if hasattr(event.operation, "value")
                        else event.operation
                    ),
                    "metabolic_state": event.metabolic_state or "STABLE",
                }
            # Add branching state
            data["branching_state"] = getattr(exec_state, "branching_state", "STABLE")
        except (AttributeError, RuntimeError) as e:
            logger.warning("Failed to fetch metrics: %s", e)
        return data

    async def _generate_thought(
        self, system_prompt: str, current_context: str, session_id: str, step: int
    ) -> str:
        """
        Gathers a thought from the LLM, including introspective healing.
        """
        if (
            self.global_stop_event.is_set()
            or self.stop_requested
            or getattr(self.runtime, "stopping", False)
        ):
            self.stop_requested = True
            return ""

        response_text = ""
        current_healing_attempt = 0
        max_healing_retries = 2
        # [REINFORCEMENT] Explicitly anchor the agent to the mission
        mission_anchor = f"--- 🎯 CURRENT MISSION ---\n{self.current_task_input or 'Complete the user request.'}\n---------------------------\n\n"
        temp_context = mission_anchor + current_context

        while current_healing_attempt <= max_healing_retries:
            try:
                log_label = (
                    "... LLM Generation ..."
                    if current_healing_attempt == 0
                    else f"... Healing (Attempt {current_healing_attempt}) ..."
                )
                self.emit_event("debug_thought", content=f"{log_label} Step {step} ...")

                def on_usage_update(usage_data: dict):
                    self.emit_event("token_usage", data=usage_data, is_internal=True)

                llm_config = self.llm.config
                if llm_config.get("provider") == "openrouter":
                    content = [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                    messages = [
                        {"role": "system", "content": content},
                        {"role": "user", "content": temp_context},
                    ]
                    response_text = await protected_llm_generate(
                        prompt=messages,
                        system=None,
                        stream=False,
                        stop=["</invoke>", "<|endoftext|>"],
                        on_usage=on_usage_update,
                    )
                else:
                    response_text = await protected_llm_generate(
                        prompt=temp_context,
                        system=system_prompt,
                        stream=False,
                        stop=["</invoke>", "<|endoftext|>"],
                        on_usage=on_usage_update,
                    )

                if self.stop_requested or self.global_stop_event.is_set():
                    self.stop_requested = True
                    break

                # Introspective Healing
                rlm_ctx = RLMInterface(
                    self,
                    session_id=session_id,
                    root_session_id=self.session_cache.get(
                        "root_session_id", session_id
                    ),
                )
                correction = await intelli_synth.introspective_probe(
                    response_text,
                    rlm=rlm_ctx,
                    context_scratchpad="",
                    agent=self,
                    session_id=session_id,
                    root_session_id=self.session_cache.get(
                        "root_session_id", session_id
                    ),
                    step_id=step,
                    turn_id=self.current_turn,
                )

                if not correction:
                    break

                current_healing_attempt += 1
                if current_healing_attempt > max_healing_retries:
                    break

                self.emit_event(
                    "thinking",
                    content=f"⚠️ **[Healing]** {correction['type']} detected. Patching...",
                    tag="SYSTEM",
                )

                # Record the healing event as a reflexion node for graph visibility
                await self._create_system_node(
                    logical_id=f"{session_id}:T{self.current_turn}:S{step}:Healing:{correction['type']}",
                    summary=f"Healing: {correction['type']}",
                    result=f"### Healing Correction\n**Issue**: {correction['message']}\n**Hint**: {correction['hint']}",
                    status="reflexion",
                    session_id=session_id,
                    root_session_id=self.session_cache.get(
                        "root_session_id", session_id
                    ),
                    turn_id=self.current_turn,
                    step_id=step,
                )

                temp_context += f"\n\n[SYSTEM ERROR]: {correction['message']}\nHINT: {correction['hint']}\nFix and retry."
                response_text = ""

            except (
                AttributeError,
                RuntimeError,
                KeyError,
                ValueError,
                httpx.RequestError,
                asyncio.TimeoutError,
            ) as e:
                logger.error("Error in generate_thought: %s", e)
                if not response_text:
                    response_text = f"System Error: {e}"
                break

        trace_action("AGENT", "THOUGHT", result=response_text, tag="AGENT")
        if self.navigator:
            self.navigator.update_history(response_text)

        return response_text

    async def _process_response(
        self, response_text: str, rlm_ctx: Optional[RLMInterface]
    ) -> Tuple[Optional[str], Optional[ThimacIntention]]:
        """
        Parses the LLM response for code and intentions.
        """
        # LIDA Intention Parsing
        distal = re.search(
            r"<distal_intention>(.*?)</distal_intention>", response_text, re.DOTALL
        )
        proximal = re.search(
            r"<proximal_intention>(.*?)</proximal_intention>", response_text, re.DOTALL
        )
        motor = re.search(
            r"<motor_intention>(.*?)</motor_intention>", response_text, re.DOTALL
        )

        current_intent = None
        if motor:
            current_intent = ThimacIntention.MOTOR
            if rlm_ctx:
                rlm_ctx.proximal_intention = motor.group(1).strip()
        elif proximal:
            current_intent = ThimacIntention.PROXIMAL
            if rlm_ctx:
                rlm_ctx.proximal_intention = proximal.group(1).strip()
        elif distal:
            current_intent = ThimacIntention.DISTAL
            if rlm_ctx:
                rlm_ctx.distal_intention = distal.group(1).strip()

        code = self._extract_code(response_text)
        if code:
            trace_action("AGENT", "CODE_BLOCK", result=code, tag="REPL")

        return code, current_intent

    async def _execute_action(
        self,
        code: str,
        thought_id: str,
        session_id: str,
        root_id: str,
        prompt: str,
        turn: int,
        step: int,
    ) -> Tuple[str, bool, List[str], Optional[str]]:
        """
        Executes the extracted code using the kernel.
        """
        if (
            self.global_stop_event.is_set()
            or self.stop_requested
            or getattr(self.runtime, "stopping", False)
        ):
            self.stop_requested = True
            return "", False, [], None

        output, failed, _, c_hash = await self._execute_code(
            code,
            thought_id,
            session_id,
            root_session_id=root_id,
            task_input=prompt,
            turn_id=turn,
            step_id=step,
        )

        tool_calls = self.execution_logs.get(session_id, [])
        self._check_verification(code, tool_calls)

        if tool_calls:
            self.execution_logs[session_id] = []

        return output, failed, tool_calls, c_hash

    async def _validate_and_finalize(
        self,
        response_text: str,
        context_scratchpad: str,
        prompt: str,
        session_id: str,
        root_id: str,
        step: int,
        round_id: str,
        repl_id: str,
        code: bool,
        code_hash: Optional[str] = None,
    ) -> bool:
        """
        Runs validation gates (Epistemic, Axiomatic, Dreamer) and emits final output if valid.
        Returns True if the turn should exit.
        """
        # Linearized exit gate
        has_final_marker = any(t in response_text for t in ["RLM_FINAL_OUTPUT"])
        is_implicit_final = self.awaiting_validation and any(
            p in response_text.lower()
            for p in ["conclusion", "summary", "final result", "task complete"]
        )

        # FINAL() and FINAL_VAR() support (Pro Technique)
        explicit_final_match = re.search(r"FINAL\((.*?)\)", response_text)
        if explicit_final_match:
            self.final_result = explicit_final_match.group(1).strip("'\"")
            has_final_marker = True

        if not (has_final_marker or self.awaiting_validation or is_implicit_final):
            return False

        exec_state = agent_state.get()
        if exec_state:
            exec_state.phase = "VALIDATING"

        # 1. Epistemic Integrity
        integrity = self._verify_epistemic_integrity(
            response_text, prompt, self.execution_logs.get(session_id, [])
        )
        if integrity["status"] == "RETRY":
            self.final_result = None
            msg = f"SYSTEM WARNING: Epistemic integrity check failed. Flags: {', '.join(integrity['flags'])}"
            self.current_thought_id = await self._create_system_node(
                logical_id=f"{session_id}:T{self.current_turn}:S{step}:EpistemicWarning",
                summary=msg,
                parent_id=self.current_thought_id,
                status="reflexion",
                session_id=session_id,
                root_session_id=root_id,
                round_id=round_id,
                turn_id=self.current_turn,
                step_id=step,
                repl_id=repl_id,
                analysis={"code_hash": code_hash},
            )
            self.emit_event(
                "warning", content=f"Epistemic Failure: {', '.join(integrity['flags'])}"
            )
            return False

        # 2. Forced Synthesis
        if (
            not getattr(self, "synthesis_triggered", False)
            and code
            and not self.awaiting_validation
            and not explicit_final_match
        ):
            self.synthesis_triggered = True
            self.final_result = None
            msg = "SYSTEM: You provided code and results. You MUST now provide a COMPREHENSIVE Final Answer summarizing your findings."
            self.current_thought_id = await self._create_system_node(
                logical_id=f"{session_id}:T{self.current_turn}:S{step}:SynthesisRequired",
                summary=msg,
                parent_id=self.current_thought_id,
                status="reflexion",
                session_id=session_id,
                root_session_id=root_id,
                round_id=round_id,
                turn_id=self.current_turn,
                step_id=step,
                repl_id=repl_id,
                analysis={"code_hash": code_hash},
            )
            return False

        if not self.final_result:
            self.final_result = response_text

        # 3. Axiomatic Consistency
        axiom_diag = await sheaf.check_axiomatic_consistency(
            proposed_code=self.final_result or "",
            task_tags=["final_synthesis"],
            depth=self.current_depth,
            metadata=exec_state.metadata if exec_state else {},
        )
        if axiom_diag.get("status") == "AXIOMATIC_VIOLATION":
            self.final_result = None
            critique = axiom_diag.get("critique")
            self.current_thought_id = await self._create_system_node(
                logical_id=f"{session_id}:T{self.current_turn}:S{step}:AxiomViolation",
                summary=f"AXIOM VIOLATION: {critique}",
                parent_id=self.current_thought_id,
                status="reflexion",
                session_id=session_id,
                root_session_id=root_id,
                round_id=round_id,
                turn_id=self.current_turn,
                step_id=step,
                repl_id=repl_id,
                analysis={"code_hash": code_hash},
            )
            self.emit_event("warning", content=f"Axiom Violation: {critique}")
            return False

        # 4. Dreamer Validation
        validation = await dreamer.validate_response(
            candidate=self.final_result or "",
            context=context_scratchpad,
            session_id=session_id,
            current_step=step,
            goal_embedding=self.session_cache.get("task_embedding"),
            turn_id=self.current_turn,
            root_session_id=root_id,
            memory_trajectory=self.morph_memory.all_events,
        )

        if validation.get("status") in ["valid", "forced_valid"]:
            self.emit_event(
                "thinking",
                content="✨ **[Dreamer]** Validation successful! Synthesizing final report...",
                tag="DREAMER",
            )
            self.final_result = await self._generate_validated_response(
                root_session_id=root_id, original_task=prompt
            )
            self.emit_event("RLM_FINAL_OUTPUT", content=self.final_result)
            self._final_output_emitted = True
            self.current_thought_id = await self._create_system_node(
                logical_id=f"{session_id}:T{self.current_turn}:S{step}:VALIDATED",
                summary=f"DREAMER VALIDATED: {validation.get('message', 'Passed')}",
                parent_id=self.current_thought_id,
                status="validated",
                session_id=session_id,
                root_session_id=root_id,
                round_id=round_id,
                turn_id=self.current_turn,
                step_id=step,
                repl_id=repl_id,
                result=self.final_result,
                analysis={"code_hash": code_hash},
            )
            self.eval_success_count += 1

            try:
                # 5. Persistent Memory Flush (Phase 4 Consolidation)
                # Flush the successful thought chain from RAM to FalkorDB
                await self._flush_memory_chain(self.current_thought_id)

                # 6. Archive the completed round to the graph for history persistence
                self.db.save_round(
                    round_id=round_id,
                    root_session_id=root_id,
                    user_prompt=prompt,
                    repl_ids=list(self.active_repls.values()),
                    final_response=(
                        str(self.final_result) if self.final_result is not None else ""
                    ),
                    full_scratchpad=context_scratchpad,
                    started_at=getattr(self, "round_started_at", int(time.time())),
                    ended_at=int(time.time()),
                )
            except (AttributeError, RuntimeError, KeyError, ValueError) as db_err:
                logger.error(
                    "Failed to archive round %s (DB error): %s", round_id, db_err
                )

            try:
                await dreamer.dream_cycle(
                    emit_callback=self.emit_event,
                    session_id=session_id,
                    final_response_candidate=self.final_result,
                    context=context_scratchpad,
                    turn_id=self.current_turn,
                    root_session_id=root_id,
                )
            except (AttributeError, RuntimeError) as e:
                logger.warning("Post-success dream cycle failed: %s", e)
            return True
        else:
            self.last_rejected_result = self.final_result
            self.final_result = None
            instruction = validation.get("instruction", "Review validation failure.")
            reasons = ", ".join(validation.get("reasons", []))
            feedback = f"DREAMER REJECTION: {instruction}\nREASONS: {reasons}"

            if exec_state:
                exec_state.phase = "EXPLORING"
                exec_state.last_dreamer_critique = instruction[:200]
                exec_state.intervention_count += 1

            self.current_thought_id = await self._create_system_node(
                logical_id=f"{session_id}:T{self.current_turn}:S{step}:DreamerRejection",
                summary=feedback,
                parent_id=self.current_thought_id,
                status="reflexion",
                session_id=session_id,
                root_session_id=root_id,
                round_id=round_id,
                turn_id=self.current_turn,
                step_id=step,
                repl_id=repl_id,
                analysis={"code_hash": code_hash},
            )
            self.last_dream_insight = instruction
            self.emit_event("warning", content=f"Dreamer Rejected: {instruction}")
            return False

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
        Modularized core execution loop.
        """
        exec_ctx = await self._initialize_turn(
            prompt,
            parent_id,
            session_id,
            depth,
            root_session_id,
            turn_id,
            recursion_stack,
            metadata,
        )

        while exec_ctx["step"] < exec_ctx["max_steps"] and not self.stop_requested:
            exec_ctx["step"] += 1
            step = exec_ctx["step"]

            await self._initialize_step(step, session_id, exec_ctx)

            # 1. Thought Generation
            response_text = await self._generate_thought(
                exec_ctx["system_prompt"], exec_ctx["pad"], session_id, step
            )
            if not response_text:
                break

            # 2. Response Processing
            rlm_ctx = RLMInterface(
                self, session_id=session_id, root_session_id=exec_ctx["root_id"]
            )
            code, _ = await self._process_response(response_text, rlm_ctx)

            # 3. Action Execution
            c_hash = None
            if code and self.current_thought_id:
                # Capture failed state for reflexion
                _output, execution_failed, execution_summary, c_hash = (
                    await self._execute_action(
                        code,
                        self.current_thought_id,
                        session_id,
                        exec_ctx["root_id"],
                        prompt,
                        turn_id,
                        step,
                    )
                )

                # [REFLEXION] Trigger immediate self-healing if code failed
                if execution_failed:
                    logger.warning(
                        "🚨 [Execution Failure] Triggering Dreamer Reflexion..."
                    )
                    from .dream import Dreamer

                    reflexion_dreamer = Dreamer()

                    # Passing current pad as context for analysis
                    reflexion_res = await reflexion_dreamer.dream_cycle(
                        session_id=session_id,
                        context=exec_ctx["pad"],
                        turn_id=turn_id,
                        root_session_id=exec_ctx["root_id"],
                        reflexion_context={"error": execution_summary, "code": code},
                    )
                    insight = reflexion_res.get("insight", "")
                    if insight:
                        self.last_dream_insight = insight
                        logger.info("🛡️ [Reflexion] Captured healing insight.")

            # 4. Validation & Finalization
            if await self._validate_and_finalize(
                response_text,
                exec_ctx["pad"],
                prompt,
                session_id,
                exec_ctx["root_id"],
                step,
                exec_ctx["round_id"],
                exec_ctx["repl_id"],
                bool(code),
                code_hash=c_hash,
            ):
                break

            # 5. Refresh State
            exec_ctx["pad"] = await self._refresh_scratchpad(
                session_id,
                exec_ctx["root_id"],
                prompt,
                step,
                exec_ctx["max_steps"],
                exec_ctx["round_id"],
                execution_state=agent_state.get(),
            )
            await asyncio.sleep(0.01)

        # FINAL REPORTING (Ghost Error Prevention)
        if not self._final_output_emitted:
            if exec_ctx["step"] >= exec_ctx["max_steps"]:
                self._emit_terminal_report(
                    "MAX_STEPS_REACHED",
                    f"Agent reached maximum execution limits ({exec_ctx['max_steps']} steps) without a successful validation.",
                )
            elif self.stop_requested:
                self._emit_terminal_report(
                    "STOP_REQUESTED",
                    "Task processing was halted explicitly (system or user request).",
                )
            else:
                self._emit_terminal_report(
                    "UNKNOWN_TERMINATION",
                    "The agent loop exited unexpectedly without emitting a final result.",
                )

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

    def _check_verification(self, code: str, tool_calls: Optional[List[str]] = None):
        """Scans code and tools for explicit verification patterns (Rule 5)."""
        state = agent_state.get()
        if not state or not state.pending_side_effects:
            return

        # Verification patterns
        verification_patterns = [
            r"os\.path\.exists",
            r"os\.path\.isfile",
            r"os\.path\.isdir",
            r"Path\.exists",
            r"os\.stat",
            r"os\.path\.getsize",
            r"json\.load",
            r"\.read\(",
            r"view_file",
            r"ls ",
            r"list_dir",
            r"grep_search",
        ]

        code_verified = any(re.search(p, code) for p in verification_patterns)
        tools_verified = False
        if tool_calls:
            verification_tools = [
                "view_file",
                "list_dir",
                "grep_search",
                "find_by_name",
            ]
            tools_verified = any(
                any(vt in t.lower() for vt in verification_tools) for t in tool_calls
            )

        if code_verified or tools_verified:
            logger.info(
                "[Rule 5] Verification detected. Clearing pending side-effects: %s",
                state.pending_side_effects,
            )
            state.pending_side_effects.clear()

    def _extract_code(self, text: str) -> str:
        """Extracts python code blocks from LLM response text."""
        from .guardrails import extract_python_code

        return extract_python_code(text)

    async def _execute_code(
        self,
        code: str,
        thought_id: str,
        session_id: str,
        root_session_id: Optional[str] = None,
        task_input: str = "",
        turn_id: Optional[int] = None,
        step_id: int = 0,
    ) -> Tuple[str, bool, Optional[str], Optional[str]]:
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
            # [Standardized Code Hash] (Phase 3)
            current_code_hash = hashlib.md5(
                code.encode(), usedforsecurity=False
            ).hexdigest()[:7]

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
                        f"Use 'await rlm.recall_node(\"{thought_id}\")' or read the log file if you need details."
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
                "code_output",
                content=output,
                code=code,
                data={"repl_id": repl_id, "code_hash": current_code_hash},
            )

            return output, execution_failed, execution_summary, current_code_hash

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

                entry = (
                    f"Turn {i + 1} [{step_type}] (REPL: {repl_id}, Status: {status}):\n"
                )
                entry += f"Content: {content}\n"
                if exec_summary and exec_summary != content:
                    entry += f"Summary: {exec_summary}\n"
                if result:
                    # Provide the full result for grounding, truncated if extreme
                    res_val = str(result)
                    if len(res_val) > 4000:
                        res_val = res_val[:3997] + "..."
                    entry += f"Result: {res_val}\n"

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
