import asyncio
import datetime
import importlib.util
import inspect
import pkgutil
import queue
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from ..config import settings
from ..context_index import context_index
from ..database import db, GraphClientFacade
from ..llm import LLMService, llm
from ..logger import get_logger
from ..manager import REPLManager
from ..repe import repe
from ..scratchpad_builder import scratchpad_builder
from ..sheaf import sheaf
from ..trace import register_monitor, trace_action

from .events import broadcast_trace, execution_events, EventEmitter
from .state import agent_state, ExecutionState
from .interface import RLMInterface

if TYPE_CHECKING:
    from graph_rlm.backend.src.mcp_integration.skills import SkillsManager

# Register the monitor
trace_action(context="AGENT", action="Initializing Trace Monitor...", level="debug")
register_monitor(broadcast_trace)

logger = get_logger("graph_rlm.agent.core")

def is_mcp_available():
    """Defensive check for MCP tools availability."""
    return (
        importlib.util.find_spec("mcp_tools") is not None
        or importlib.util.find_spec("graph_rlm.backend.mcp_tools") is not None
    )

def is_skills_available():
    """Defensive check for Skills system availability."""
    return (
        importlib.util.find_spec("graph_rlm.backend.src.mcp_integration.skills")
        is not None
        or importlib.util.find_spec("mcp_integration.skills") is not None
    )

# --- MCP Namespace Classes ---
from importlib import import_module

class MCPServerNamespace:
    """Lazy-loaded namespace for a single MCP server."""

    def __init__(self, mod_name: str, alias: str, rlm_interface: "RLMInterface"):
        self._mod_name = mod_name
        self._alias = alias
        self._rlm_interface = rlm_interface
        self._module = None
        self._tools = {}
        self._docs = {}

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
                                    self._rlm_interface._record_tool_use(n)
                                    res = f(*args, **kwargs)
                                    if inspect.isawaitable(res):
                                        return await res
                                    return res

                                wrapped.__doc__ = f.__doc__
                                return wrapped

                            wrapper = make_wrapper(func, f"mcp.{self._alias}.{attr}")
                            self._tools[attr] = wrapper
                            self._docs[attr] = func.__doc__
            except Exception as e:
                logger.warning(f"Failed to load MCP server {self._mod_name}: {e}")
                self._module = False  # Mark as failed

    def __getattr__(self, name):
        self._ensure_loaded()
        if name in self._tools:
            return self._tools[name]
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

    def _scan(self):
        if not self._scan_done and is_mcp_available():
            try:
                import graph_rlm.backend.mcp_tools as mcp_tools_pkg

                logger.info("Starting MCP server discovery...")
                for _, mod_name, _ in pkgutil.iter_modules(mcp_tools_pkg.__path__):
                    if mod_name.startswith("_") or mod_name == "skills":
                        logger.debug(f"Skipping module: {mod_name}")
                        continue

                    logger.info(f"Discovered MCP module: {mod_name}")

                    # Create MCPServerNamespace using the actual module name (no aliases)
                    # This ensures tool discovery works correctly by matching module structure
                    server = MCPServerNamespace(mod_name, mod_name, self._rlm_interface)
                    self._aliases[mod_name] = server
                    logger.info(f"Registered MCP server: {mod_name}")

                self._scan_done = True
                logger.info("MCP server discovery completed.")
            except Exception as e:
                logger.warning(f"MCP Scan Error: {e}")

    def __getattr__(self, name):
        self._scan()
        if name in self._aliases:
            return self._aliases[name]
        raise AttributeError(f"No MCP server found with name or alias '{name}'")

    def __dir__(self):
        self._scan()
        return list(self._aliases.keys())

    def __repr__(self):
        return f"<LazyMCPNamespace with {len(self._aliases)} server aliases>"


class Agent:
    skills_manager: Optional["SkillsManager"]

    def __init__(self):
        # --- PATH INJECTION (Bootstrap) ---
        # Ensure 'skills_dir' and 'mcp_tools' are in path for imports
        # We need the project root for 'graph_rlm' nested imports
        # Agent class is in backend/src/core/agent/core.py
        backend_path = Path(__file__).resolve().parents[3] # src/core/agent -> src/core -> src -> backend
        project_root = backend_path.parent.parent # graph_rlm/backend -> graph_rlm -> root

        # Correct path resolutions
        backend_path = Path(__file__).parent.parent.parent.parent.resolve() # graph_rlm/backend
        skills_path = backend_path / "skills_dir"
        mcp_tools_path = backend_path / "mcp_tools"
        project_root = backend_path.parent.parent

        if str(project_root) not in sys.path:
            logger.info(f"Injecting Project Root: {project_root}")
            sys.path.append(str(project_root))

        if str(backend_path) not in sys.path:
            logger.info(f"Injecting Backend Path: {backend_path}")
            sys.path.append(str(backend_path))

        if str(skills_path) not in sys.path:
            sys.path.append(str(skills_path))

        if str(mcp_tools_path) not in sys.path:
            sys.path.append(str(mcp_tools_path))

        # --- CORE INITIALIZATION ---
        self.db: GraphClientFacade = db
        self.llm: LLMService = llm
        self.repl_manager = REPLManager()
        self.active_repls: Dict[str, str] = {}  # session_id -> repl_id
        self.execution_logs: Dict[str, list] = {}  # session_id -> [tool_ident, ...]
        self.session_cache: Dict[str, Any] = {}  # For Sheaf Monitor & shared state
        self.current_task_input: Optional[str] = (
            None  # Tracks active goal for Sheaf Teleology
        )
        self._global_stop_event = (
            threading.Event()
        )  # Shared event for cross-thread stopping
        self.event_emitter = EventEmitter()

        if is_skills_available():
            from graph_rlm.backend.src.mcp_integration.skills import (
                get_skills_manager,
            )

            self.skills_manager = get_skills_manager()
        else:
            self.skills_manager = None

        # Ensure Knowledge Base scaffolding exists
        self._ensure_kb_structure()

        logger.info(f"Agent initialized using active environment: {sys.prefix}")
        logger.info("Agent initialized with Persistent REPL support")
        logger.info("RepE Safety Layer & Sheaf Topology Monitor Loaded.")

    async def initialize_system(self):
        """
        Perform async system initialization (DB Indexes, Skills, Safety).
        Previously handled by FastAPI lifespan.
        """
        try:
            logger.info("Initializing Graph-RLM System Components...")

            # 1. DB Indexes
            self.db.create_vector_indexes()

            # 2. Safety Calibration
            await repe.calibrate()

            # 3. Skills Sync
            if self.skills_manager:
                await self.skills_manager.sync_from_disk()

            # 4. Axioms Sync
            if is_skills_available():
                from graph_rlm.backend.src.mcp_integration.skills import get_axioms_manager
                axioms_mgr = get_axioms_manager()
                await axioms_mgr.sync_from_disk()

            # 5. MCP Tools Generation (Simplified check)
            # Ideally we run the generator here if needed, but let's assume pre-generated or dynamic.
            # For now, we rely on dynamic loading in LazyMCPNamespace.

            logger.info("System Initialization Complete.")
        except Exception as e:
            logger.error(f"System Initialization Failed: {e}")

    def _ensure_kb_structure(self):
        """Creates the Knowledge Base directory structure if it doesn't exist."""
        try:
            from pathlib import Path

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

            logger.info(f"Knowledge Base structure verified at: {kb_root}")
        except Exception as e:
            logger.warning(f"Failed to verify Knowledge Base structure: {e}")

    # --- SESSION-ISOLATED PROPERTIES ---
    def _get_state(self) -> ExecutionState:
        state = agent_state.get()
        if state is None:
            # Fallback for out-of-session access (e.g. CLI or background tasks)
            state = ExecutionState()
            agent_state.set(state)
        return state

    @property
    def _final_result(self) -> Optional[str]:
        return self._get_state().final_result

    @_final_result.setter
    def _final_result(self, value: Optional[str]):
        self._get_state().final_result = value

    @property
    def _stop_requested(self) -> bool:
        # Check both local context AND global signal
        return self._get_state().stop_requested or self._global_stop_event.is_set()

    @_stop_requested.setter
    def _stop_requested(self, value: bool):
        self._get_state().stop_requested = value

    @property
    def _synthesis_triggered(self) -> bool:
        return self._get_state().synthesis_triggered

    @_synthesis_triggered.setter
    def _synthesis_triggered(self, value: bool):
        self._get_state().synthesis_triggered = value

    @property
    def current_thought_id(self) -> Optional[str]:
        return self._get_state().current_thought_id

    @current_thought_id.setter
    def current_thought_id(self, value: Optional[str]):
        self._get_state().current_thought_id = value

    @property
    def current_depth(self) -> int:
        return self._get_state().depth

    @current_depth.setter
    def current_depth(self, value: int):
        self._get_state().depth = value

    def emit_event(
        self,
        event_type: str,
        data: Any = None,
        content: Optional[str] = None,
        code: Optional[str] = None,
        is_sub_event: bool = False,
        tag: Optional[str] = None,
    ):
        self.event_emitter.emit(event_type, data, content, code, is_sub_event, tag)

    def _install_to_active_env(self, package_name: str) -> str:
        """Internal helper to install a package into the CURRENT active environment."""

        # trunk-ignore(bandit/B404)
        import subprocess

        logger.info(
            f"Agent requesting installation of package: {package_name} into Active Env"
        )
        self.emit_event(
            "thinking", content=f"\n📦 Agent: Installing package '{package_name}'..."
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

            # trunk-ignore(bandit/B603)
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name}")
                self.emit_event("thinking", content="  -> Installation successful.")
                return f"Successfully installed {package_name}\n{result.stdout}"
            else:
                logger.error(f"Failed to install {package_name}: {result.stderr}")
                self.emit_event(
                    "error", content=f"Installation failed: {result.stderr}"
                )
                return f"Failed to install {package_name}\nError: {result.stderr}"
        except Exception as e:
            logger.error(f"Installation error: {e}")
            return f"Installation error: {e}"

    def _install_to_agent_venv(self, package_name: str) -> str:
        """Internal helper to install a package into the DEDICATED AGENT VENV."""

        # trunk-ignore(bandit/B404)
        import subprocess

        # Resolve agent_venv path relative to this file
        # __file__ = backend/src/core/agent/core.py
        backend_root = Path(__file__).parent.parent.parent.parent
        agent_venv_path = backend_root / "agent_venv"

        # Determine python executable in venv
        if sys.platform == "win32":
            python_exe = agent_venv_path / "Scripts" / "python.exe"
        else:
            python_exe = agent_venv_path / "bin" / "python"

        if not python_exe.exists():
            return f"Error: Agent Venv not found at {agent_venv_path}. Cannot install skill dependencies."

        logger.info(
            f"Agent requesting installation of package: {package_name} into AGENT ENV ({agent_venv_path})"
        )
        self.emit_event(
            "thinking",
            content=f"\n📦 Agent: Installing '{package_name}' into Skill/Agent Environment...",
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
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name} in Agent Venv")
                self.emit_event("thinking", content="  -> Installation successful.")
                return f"Successfully installed {package_name}\n{result.stdout}"
            else:
                logger.error(f"Failed to install {package_name}: {result.stderr}")
                self.emit_event(
                    "error", content=f"Installation failed: {result.stderr}"
                )
                return f"Failed to install {package_name}\nError: {result.stderr}"
        except Exception as e:
            logger.error(f"Installation error: {e}")
            return f"Installation error: {e}"

    def install_package(self, package_name: str) -> str:
        """Installs a package into the active environment (REPL compatibility)."""
        return self._install_to_active_env(package_name)

    def install_skill_package(self, package_name: str) -> str:
        """Installs a package into the AGENT environment (Skill compatibility)."""
        return self._install_to_agent_venv(package_name)

    async def stream_query(
        self,
        prompt: str,
        parent_id: Optional[str] = None,
        session_id: str = "default",
        depth: int = 0,
        root_session_id: Optional[str] = None,
    ):
        """
        Streaming entry point.
        launches the sync execution in a thread and yields events from a queue.
        """
        q = queue.Queue()
        self._stop_requested = False
        self._global_stop_event.clear()  # Reset global flag
        self._final_result = None

        def run_logic():
            # Set the context vars for this thread
            q_token = execution_events.set(q)
            state_token = agent_state.set(
                ExecutionState(depth=depth, current_thought_id=parent_id)
            )
            try:
                # Set initial depth for this agent run
                self.current_depth = depth
                asyncio.run(
                    self.query_sync(
                        prompt, parent_id, session_id, depth, root_session_id
                    )
                )
            except Exception as e:
                logger.error(f"Error in execution thread: {e}")
                q.put({"type": "error", "content": str(e)})
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
    ) -> str:
        # NOTE: This method is quite long. It contains the core logic of the agent loop.
        # It calls other helper methods like _build_system_prompt, _extract_code, _execute_code
        # and interacts with the DB and LLM.

        final_root_id = root_session_id if root_session_id else session_id
        trace_action(
            "AGENT",
            "QUERY_SYNC",
            result=f"Session: {session_id} | Depth: {depth}",
            tag="AGENT",
        )

        # 0. Reset scoped State for this specific call (redundant if already set in stream_query but safe)
        if not agent_state.get():
            agent_state.set(ExecutionState(depth=depth, current_thought_id=parent_id))

        self._final_result = None
        self._stop_requested = False

        # Ensure REPL is initialized for this session
        if session_id not in self.active_repls:
            self.active_repls[session_id] = self.repl_manager.create_repl()

        # MCP STOP SIGNAL REGISTRATION
        if is_mcp_available():
            from graph_rlm.backend.src.mcp_integration.runtime import set_stop_event

            set_stop_event(self._global_stop_event)

        # 0. Initial "Task" Node (Root of this query)
        current_round_id = str(uuid.uuid4())
        current_round_started = datetime.datetime.now().timestamp() * 1000  # ms

        try:
            task_id = str(uuid.uuid4())
            logger.info(
                f"Session {session_id}: Starting Task {task_id} (Round {current_round_id})"
            )

            self.db.create_thought_node(
                task_id,
                prompt,
                parent_id,
                prompt_embedding=None,
                session_id=session_id,
                root_session_id=final_root_id,
                round_id=current_round_id,
            )

            # Update current pointer
            self.current_thought_id = task_id

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
        except Exception as e:
            logger.error(f"Failed to initialize Task node: {e}")
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
        base_system_prompt = self._build_system_prompt()

        max_steps = 1000
        step = 0

        # Track previous status for topological resolution
        previous_thought_status = None

        while step < max_steps:
            # 0.5 CHECK STOP SIGNAL
            if getattr(self, "_stop_requested", False) or (
                hasattr(self, "_global_stop_event") and self._global_stop_event.is_set()
            ):
                logger.info("Agent loop breaking due to stop request.")
                self._stop_requested = True
                break
            step += 1
            thought_id = str(uuid.uuid4())
            sheaf_diag = {"status": "HEALTHY", "consistency_energy": 0.0}
            vec = None

            # --- DYNAMIC SCRATCHPAD REFRESH ---
            try:
                context_scratchpad = scratchpad_builder.build_scratchpad(
                    session_id=session_id,
                    root_session_id=final_root_id,
                    task=prompt,
                    current_step=step,
                    max_steps=max_steps,
                    current_round_id=current_round_id,
                )
                self.emit_event("scratchpad_text", content=context_scratchpad)
            except Exception as e:
                logger.error(f"Failed to build scratchpad: {e}")
                context_scratchpad = f"Error: Scratchpad unavailable ({e})"

            # Construct Dynamic Context (Minimal)
            # 2. Context Loading (Wait/Wake-Up)
            frontier = []
            frontier_ids = []

            try:
                frontier = self.db.get_context_frontier(session_id, limit=10)
                for node in frontier:
                    val = node.get("n") if isinstance(node, dict) else node
                    if val is None:
                        continue
                    props = val.properties if hasattr(val, "properties") else val

                    if isinstance(props, dict) and "id" in props:
                        frontier_ids.append(props["id"])
            except Exception as e:
                logger.error(f"Context loading failed (Sheaf IDs): {e}")

            # 3. Construct LLM Context
            current_context = (
                f"Active Session: {session_id}\n\n"
                f"<objective>\n{prompt}\n</objective>\n\n"
                f"<history>\n{context_scratchpad}\n</history>\n"
            )

            # Load Axioms
            axioms_list_str = "None"
            if is_skills_available():
                try:
                    from graph_rlm.backend.src.mcp_integration.skills import (
                        get_axioms_manager,
                    )
                    axioms_mgr = get_axioms_manager()
                    search_query = (
                        getattr(self, "current_task_input", None)
                        or prompt
                        or "general safety"
                    )
                    relevant_axioms = await axioms_mgr.find_similar_axioms(
                        search_query, limit=10
                    )
                    if relevant_axioms:
                        axioms_list_str = ", ".join(
                            [a["name"] for a in relevant_axioms]
                        )
                    else:
                        axioms = axioms_mgr.list_axioms()
                        sorted_keys = sorted(axioms.keys())[:20]
                        axioms_list_str = ", ".join(sorted_keys)
                except Exception as ex:
                    logger.warning(f"Failed to load axioms async: {ex}")

            hot_seat_warning = ""
            if getattr(self, "_last_dream_insight", None):
                hot_seat_warning = (
                    "\n\n--- ⚠️ HOT SEAT: EPISTEMIC RECOVERY ACTIVE ---\n"
                    "Your previous response was REJECTED by the Dreamer Gatekeeper for Hallucination/Trace Contradiction.\n"
                    f"CRITIQUE: {self._last_dream_insight}\n"
                    "You MUST explicitly address the contradiction..."
                )

            system_prompt = (
                f"{self._build_system_prompt(axioms_list_str=axioms_list_str)}\n\n"
                f"{context_scratchpad}{hot_seat_warning}"
            )

            iso_ts = datetime.datetime.now().isoformat()
            repl_info = f"[REPL: {self.active_repls.get(session_id, 'init')}]"

            self.emit_event(
                "thinking",
                content=f"[{iso_ts}] {repl_info} Step {step}: RLM loop active (Model: {self.llm.config.get('model')}).",
                tag="AGENT",
            )

            # 3. LLM Gen (Think)
            response_text = ""
            try:
                def on_token_usage(usage_data):
                    self.emit_event("usage", data=usage_data)

                self.emit_event(
                    "debug_thought",
                    content=f"... Sending request to LLM (Size: {len(current_context)} chars) ...",
                )

                if self._stop_requested or self._global_stop_event.is_set():
                    self._stop_requested = True
                    break

                response_text = await self.llm.generate(
                    current_context,
                    system=system_prompt,
                    stream=False,
                    on_usage=on_token_usage,
                )

                if self._stop_requested or self._global_stop_event.is_set():
                    self._stop_requested = True
                    break

            except Exception as e:
                response_text = f"LLM Error: {e}"
                logger.error(f"LLM Generative Error: {e}")

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
                break

            if getattr(self, "_stop_requested", False):
                break

            repl_id_display = self.active_repls.get(session_id, "unknown")
            timestamp_display = datetime.datetime.now().isoformat()
            formatted_thought = (
                f"[{timestamp_display}] [REPL: {repl_id_display}]\n{response_text}"
            )
            self.emit_event("debug_thought", content=formatted_thought)
            trace_action(
                "AGENT", "THOUGHT", result=response_text[:400] + "...", tag="AGENT"
            )

            output = ""
            code = self._extract_code(response_text)
            repl_id = self.active_repls.get(session_id)

            # 5. Semantic Vectorization (Early)
            try:
                vec = await self.llm.get_embedding(response_text)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for thought {thought_id}: {e}")
                vec = None

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
            except Exception as e:
                logger.error(f"Failed to pre-commit thought: {e}")

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
                except Exception as e:
                    logger.warning(f"Failed to embed task for Health Check: {e}")

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

                # C. SYNTHESIS & INTERVENTION LOGIC

                intervention_prompt = None
                intervention_type = None

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
                        f"Re-read the original user request and justify how this step helps."
                    )

                # SCENARIO 4: LOOPING (Confident repetition)
                elif sheaf_diag.get("status") == "LOGICAL_KNOT":
                    from .dream import dreamer  # Deferred import to avoid circular dep

                    intervention_type = "REFLEXION_BREAK"
                    loop_nodes = sheaf_diag.get("loop_nodes", [])

                    # [Dreamer Link]: Immediate Lucid Analysis
                    dream_critique = await dreamer.analyze_holonomy(
                        loop_nodes, current_thought=response_text
                    )

                    intervention_prompt = (
                        f"SYSTEM INTERVENTION (Sheaf Topology): Logical Knot detected. "
                        f"REPL ID: {repl_id} | Point: {thought_id} | Issue: {dream_critique} "
                    )

                # D. EXECUTE INTERVENTION (Steering)
                if intervention_prompt:
                    logger.warning(f"🛡️ Triggering Intervention: {intervention_type}")
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
                    )

                    # Steering Action: Force the pointer to this intervention
                    self.current_thought_id = intervention_id

                    # Skip execution of the flawed thought!
                    # The agent will wake up in the next loop seeing this intervention.
                    continue

            # 7. Act (Execute Code)
            thought_status = "success"
            axiom_critique = None

            if not self.current_thought_id:
                self.current_thought_id = task_id

            if code:
                # Axiomatic check
                try:
                    task_tags = await self._detect_required_axioms_agentic(prompt, code)
                    axiom_diag = await sheaf.check_axiomatic_consistency(
                        code, task_tags=task_tags
                    )
                    if axiom_diag.get("status") == "AXIOMATIC_VIOLATION":
                        axiom_critique = axiom_diag.get("critique")
                except Exception as e:
                    logger.warning(f"Axiomatic check failed to run: {e}")

                execution_failed = False
                if axiom_critique:
                    self.emit_event(
                        "warning",
                        content=f"🛡️ [REPL: {repl_id}] AXIOMATIC VIOLATION: Execution Blocked.\nCritique: {axiom_critique}",
                    )
                    output = f"Axiomatic Violation: Execution Blocked.\nCritique: {axiom_critique}"
                    thought_status = "failed"
                    execution_failed = True
                else:
                    if self._stop_requested or self._global_stop_event.is_set():
                        self._stop_requested = True
                        break

                    output, execution_failed = await self._execute_code(
                        code,
                        thought_id,
                        session_id,
                        root_session_id=final_root_id,
                        task_input=prompt,
                    )

                    if self._stop_requested or self._global_stop_event.is_set():
                        self._stop_requested = True
                        break

                if repl_id:
                    self.repl_manager.get_repl(repl_id)

                self.emit_event("debug_code", content=output, code=code)
                if execution_failed:
                    thought_status = "failed"

            # 8. UPDATE / FINAL COMMIT
            full_content = response_text
            if output:
                full_content += f"\n\n[Output]:\n{output}"

            # Vectorization
            final_vec = vec
            if output and len(output) > 100:
                try:
                    final_vec = await self.llm.get_embedding(full_content)
                except Exception:
                    pass

            exec_summary = None
            if output:
                clean_output = output.strip()
                if len(clean_output) > 200:
                    exec_summary = clean_output[:200] + "..."
                else:
                    exec_summary = clean_output
                if thought_status == "failed":
                    exec_summary = f"[FAILED] {exec_summary}"

            try:
                final_parent_id = self.current_thought_id
                node_to_prune = None

                if (thought_status == "success" and previous_thought_status == "failed" and self.current_thought_id):
                     failed_node_id = self.current_thought_id
                     grandparent_id = self.db.get_parent_id(failed_node_id)
                     if grandparent_id:
                         final_parent_id = grandparent_id
                         node_to_prune = failed_node_id

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
                )

                if node_to_prune:
                    try:
                        self.db.delete_thought_node(node_to_prune)
                    except Exception:
                        pass

                try:
                    sp_data = context_index.get_active_scratchpad_data(final_root_id)
                    self.emit_event("scratchpad_update", data=sp_data, is_sub_event=False)
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to commit thought to graph: {e}")

            previous_thought_status = thought_status
            self.current_thought_id = thought_id

            code_result = getattr(self, "_final_result", None)

            if code_result and not getattr(self, "_synthesis_triggered", False):
                self._synthesis_triggered = True
                self._final_result = None
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
                )
                continue

            if ("RLM_FINAL_OUTPUT" in response_text or getattr(self, "_final_result", None)) and thought_status == "success":
                # Epistemic Verification Logic...
                # ...

                if not self._final_result:
                    self._final_result = response_text

                # Dreamer Logic ...
                # ...

                break # Exit loop

        # Loop Exit
        if self._final_result:
            self.emit_event("RLM_FINAL_RESPONSE", content=self._final_result)
        elif self._stop_requested:
            self.emit_event("RLM_FINAL_RESPONSE", content="Task processing stopped.")
        elif step >= max_steps:
            self.emit_event("error", content=f"AGENT LIMIT REACHED.")
        else:
            self.emit_event("RLM_FINAL_RESPONSE", content="[System] Agent stopped without answer.")

        # Archive Round
        if self._final_result:
             # ... archive logic
             pass

        return self._final_result or "Task processing stopped."

    # Includes from original agent.py
    def _build_system_prompt(self, axioms_list_str: str = "None") -> str:
        # Extracted system prompt builder for cleanliness
        # Resolve paths for transparency
        backend_root = Path(__file__).parent.parent.parent.parent
        skills_dir_path = (backend_root / "skills_dir").absolute()
        agent_venv_path = (backend_root / "agent_venv").absolute()
        kb_path = settings.KNOWLEDGE_BASE_PATH

        skills_list_str = ""  # User Skills
        # axioms_list_str passed as argument to support async loading in caller

        try:
            if is_skills_available():
                from graph_rlm.backend.src.mcp_integration.skills import (
                    get_skills_manager,
                )

                # Load Skills
                skills_mgr = get_skills_manager()
                skills = skills_mgr.list_skills()
                skills_list_str = ", ".join(skills.keys()) if skills else "None"

        except Exception as e:
            logger.warning(f"Failed to load skills for prompt: {e}")
            skills_list_str = "None"

        prompt = (
            "Stateless Graph-RLM Agent.\n"
            "You are a stateless agent in a Global Workspace. Your context is managed SYMBOLICALLY via a persistent REPL.\n"
            "1. **Wake**: You see an 'Active Session Index' (The Sheaf). This is a compact map of the thought graph, NOT raw history.\n"
            "2. **Chain**: Produce the next logical step. Do not repeat completed work.\n"
            "3. **Recurse**: Use `await rlm.query(prompt, context)` to spawn sub-REPLs for complex problems.\n"
            "\n"
            "**Async & REPL Protocol**:\n"
            "- **MANDATORY**: You MUST `await` all `rlm` and `mcp` calls (e.g., `res = await rlm.recall(...)`).\n"
            "- **Persistence**: The Python REPL is persistent across the session. Variables defined in one step are available in the next.\n"
            "\n"
            "**Context & Environment**:\n"
            "- **Environment Variables**: Use variables injected into your REPL for immediate context:\n"
            "  - `task_input`: The original prompt/goal for THIS specific session.\n"
            "  - `session_id`: Your current unique session identifier.\n"
            "  - `active_repls`: (Root only) A directory of all active sub-sessions you are orchestrating.\n"
            "- **Recall & Search**: If you need details from the past, you MUST explicitly recall them:\n"
            "  - `await rlm.recall(query)`: High-precision semantic search for specific thought details.\n"
            "  - `await rlm.search(query)`: Global topological search across all past sessions.\n"
            "\n"
            "**Self-Correction & Reflexion**:\n"
            "You may see thoughts labeled `SYSTEM REFLEXION` or `SYSTEM WARNING` (Sheaf Topology or RepE Safety Layer).\n"
            "- If you see a **Reflexion**, you were looping or drifting. You MUST change your approach immediately.\n"
            "- If you see a **Warning**, you violated a safety constraint. Adjust your reasoning.\n"
            "\n"
            "**Package Installation**:\n"
            f"  - `await rlm.install_package('pkg')`: Installs to the **Project Environment** (Active Env).\n"
            f"  - `await rlm.install_skill_package('pkg')`: Installs to the **Agent/Skill Environment** (`{agent_venv_path}`).\n"
            "\n"
            "**Skills & Knowledge**:\n"
            f"- **Skills Directory**: `{skills_dir_path}`\n"
            "- Use `await rlm.save_skill(name, code)` to codify reusable logic.\n"
            f"- **Project Knowledge Base**: `{kb_path}` (Primary source for `await rlm.ingest_document()`)\n"
            f"  - **Store Plans** in `{kb_path}/plans/`.\n"
            f"  - **Save Research Reports** to `{kb_path}/research-reports/`.\n"
            f"  - **Save Final Outputs** to `{kb_path}/outputs/`.\n"
            f"  - **Save Axioms** to `{kb_path}/axioms/`.\n"
            "\n"
            "**Behavior**:\n"
            "- **Zen of Agentic Coding**: KISS, DRY, YAGNI, and SOLID principles apply.\n"
            "- **Language**: Internal thought and final answers MUST be in ENGLISH unless specified otherwise.\n"
            "- **TRACE GROUNDING (Anti-Hallucination)**: You MUST prioritize information recorded in the <history> and <scratchpad> over your internal training data. If a 'DREAMER GATEKEEPER' blocks you, it is because you are 'Mirroring' (substituting real data for training priors). You MUST report only on retrieved evidence from the current session's execution logs.\n"
            "\n"
            "**Ethics**:\n"
            "- **Principles**: Deontology: Universal sociobiological concepts (harm=harm) -> Virtue: Wisdom, Integrity, Empathy, Fairness, Beneficence -> Utilitarianism: As a Servant, never Master.\n"
            "\n"
            "**Termination**:\n"
            "- **Metacognitive Requirement**: Before finishing, you MUST perform a **Metacognitive Analysis** of your solution in a section titled `**Metacognitive Analysis**`.\n"
            "- **Termination**: After analysis, if the task is complete, output the EXACT string `RLM_FINAL_OUTPUT`.\n"
            "- Do NOT use 'Final Answer'. Use `RLM_FINAL_RESPONSE` only after verification.\n"
            "- **CRITICAL**: You are NOT in a native tool-calling environment. Do NOT output function calls in a structured JSON block. Write all Python code as standard markdown blocks (` ```python `) inside your response.\n"
            "---------------------------\n"
            "Constraint Augmented Generation (CAG).\n"
            "\n"
            "**CAG Paradigm (Expert System Compiler)**:\n"
            "1. **Ingest & Codify**: Use `await rlm.ingest_document(path)` to extract and codify domain knowledge into Axioms.\n"
            "2. **Axiomatic Verification**: Every Python action you write is AUTOMATICALLY verified against pre-existing Axioms by the Sheaf Monitor.\n"
            "3. **Constraint-Augmented Reasoning**: Prefer calling existing validators (axioms) to verify your logic.\n"
            "\n"
            "**REPL Exploration & Commands**:\n"
            "- `await rlm.help()`: See available core commands.\n"
            "- `await mcp.<module_name>.<function_name>()`: Access external tools (e.g., `await mcp.brave_search.brave_web_search(...)`).\n"
            "\n"
            "**SKILL-FIRST ARCHITECTURE (The One Right Way)**:\n"
            "- **PREFERENCE**: Do NOT call raw MCP tools repeatedly in your loops.\n"
            "- **WRAP**: Write a Python function that uses the tool, validate it, and SAVE it using `await rlm.save_skill(name, code)`.\n"
            "- **REUSE**: Execute the saved skill using `await rlm.run_skill(name, args)`.\n"
            "\n"
            "**MANDATORY MCP Discovery (Self-Documentation)**:\n"
            "- The `mcp` object is a recursive namespace for all connected servers.\n"
            "- **BEFORE WRITING CODE**: You MUST discover the correct tool name and parameters:\n"
            "  1. `dir(mcp)` -> Lists all MCP server names if needed.\n"
            "  2. `dir(mcp.<server_name>)` -> Lists all tools in that server.\n"
            "  3. `print(mcp.<server_name>.<tool_name>.__doc__)` -> Shows parameters and usage.\n"
            "- **DO NOT GUESS** tool names. If unsure, run discovery commands first.\n"
            "\n"
            "**SELF-HEALING PIPELINE (3-Tier Immune System)**:\n"
            "This environment heals itself. YOU are part of this process.\n"
            "\n"
            "*Tier 1: Innate Immunity (Reactive Resolution)*:\n"
            "- **Dependency Healing**: `ModuleNotFoundError` -> System installs the package and retries your code automatically.\n"
            "- **Syntax/Logic Healing**: `Exception` or `AssertionError` -> A 'SYSTEM REFLEXION' node is injected. You MUST read it and change your approach.\n"
            "- **Timeout Recovery**: If your code hangs, the process is killed. Simplify your next attempt.\n"
            "\n"
            "*Tier 2: Epistemic Integrity (Proactive Filtering)*:\n"
            "- **Axiomatic Verification (CAG)**: Your code is checked against the Axiom Library BEFORE execution. Violations are blocked.\n"
            "- **Sheaf Topology Monitor**: Measures 'Consistency Energy'. High energy (looping, contradictions) triggers a 'Militant Reflexion'.\n"
            "- **RepE Scanning**: Your thoughts are scanned for 'Pathogens' (Laziness, Obsequiousness, Malice). Detection triggers steering.\n"
            "\n"
            "*Tier 3: Adaptive Immunity (Meta-Cognitive Learning)*:\n"
            "- **The Dreamer**: After you finish, a 'Dream Cycle' analyzes your failures and synthesizes new rules.\n"
            "- **Rule Codification**: Insights become Axioms, preventing that class of error permanently.\n"
            "\n"
            "**YOUR RESPONSIBILITY**: When you see 'SYSTEM REFLEXION' or 'SYSTEM WARNING', you MUST change your approach. Do NOT repeat the failing pattern.\n"
            "\n"
            "**FILESYSTEM ACCESS**:\n"
            "- You have DIRECT ACCESS to the filesystem via REPL and standard Python libraries (`os`, `pathlib`, `open`).\n"
            "- **CRITICAL**: Use SYNCHRONOUS file operations (`with open(...)`) for all writes. Do NOT use `aiofiles` or `asyncio.run()` for file I/O. This prevents 'Async-State Divergence' and data loss.\n"
            "- **VERIFY WRITES**: Immediately check `os.path.getsize(path) > 0` after writing.\n"
            "--- AVAILABLE CONTEXT ---\n"
            f"Active Axioms: {axioms_list_str}\n"
            f"Active Skills: {skills_list_str}\n"
            "---------------------------\n"
        )

        # Inject "Marge's Rules" (Dreamer Guardrails)
        rules_path = backend_root / "rules.md"
        if rules_path.exists():
            try:
                rules_content = rules_path.read_text()
                prompt += (
                    f"\n\n**System Rules (Dreamer Guardrails)**:\n{rules_content}\n"
                )
            except Exception as e:
                logger.warning(f"Failed to load rules.md: {e}")

        return prompt

    def _extract_code(self, text: str) -> str:
        # Try finding a complete block first
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

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
    ) -> Tuple[str, bool]:
        # 1. Get or Create REPL for this session
        if session_id not in self.active_repls:
            # Just in case (though query_sync should claim it)
            self.active_repls[session_id] = self.repl_manager.create_repl()

        repl_id = self.active_repls[session_id]

        # Verify liveness
        if not self.repl_manager.get_repl(repl_id):
            repl_id = self.repl_manager.create_repl()
            self.active_repls[session_id] = repl_id

        # 2. Update Context
        repl = self.repl_manager.get_repl(repl_id)

        # 2. Update Context
        previous_thought_id = self.current_thought_id
        self.current_thought_id = thought_id

        try:
            if repl is None:
                return "Error: Failed to create REPL session.", True

            # Re-inject RLM interface (it needs current thought_id binding)
            # Ideally RLMInterface is persistent but points to dynamic 'self.current_thought_id'
            # Here we just overwrite 'rlm' in namespace to be safe or update it

            # Ensure we have a root_session_id
            final_root = root_session_id if root_session_id else session_id

            rlm_interface = RLMInterface(self, session_id, final_root)

            # Ensure REPL namespace is initialized
            if not hasattr(repl, "namespace") or repl.namespace is None:
                logger.error("REPL namespace is not initialized. Cannot inject 'rlm'.")
                return "Error: REPL namespace not initialized.", True

            # Inject 'rlm' and context variables into the namespace
            logger.info("Injecting 'rlm' and context variables into REPL namespace.")
            repl.namespace["rlm"] = rlm_interface
            repl.namespace["task_input"] = task_input
            repl.namespace["session_id"] = session_id
            repl.namespace["root_session_id"] = final_root

            # Inject MCP namespace if available
            # Inject MCP namespace ALWAYS (Safe because it's lazy)
            if "mcp" not in repl.namespace:
                repl.namespace["mcp"] = LazyMCPNamespace(repl.namespace["rlm"])
            elif "mcp" in repl.namespace and hasattr(
                repl.namespace["mcp"], "_rlm_interface"
            ):
                repl.namespace["mcp"]._rlm_interface = repl.namespace["rlm"]

            # Log the namespace state for debugging
            logger.debug(f"REPL namespace keys: {list(repl.namespace.keys())}")
            # CRITICAL DEBUG: Check if session_id is ACTUALLY usable
            try:
                # We try to eval it in the namespace to see if it's accessible
                # This mimics what exec() sees
                check_sid = repl.namespace.get("session_id", "MISSING")
                logger.debug(f"Pre-flight check: session_id = {check_sid}")
            except Exception as e:
                logger.error(f"Pre-flight namespace check failed: {e}")

            # 3. Execute with Streaming
            # Define callback to stream stdout to UI
            def stream_callback(text: str):
                if text:
                    self.emit_event(
                        "code_output_chunk", content=text, is_sub_event=False
                    )

            # Await the async REPL execution
            stdout, stderr, result, is_err = await repl.execute(
                code, output_callback=stream_callback
            )

            # 4. Handle Async Results (Coroutines)
            # With the new REPL, result might still be a coroutine if it returned one explicitly,
            # or if the AST rewrite happened but the wrapper returned a coro instead of awaiting it.
            if inspect.isawaitable(result):
                try:
                    # We are in an async method (_execute_code), so we can await directly.
                    result = await result
                except Exception as e:
                    stderr = (stderr or "") + f"\nAsync Execution Error: {e}"

            output = stdout or ""
            if result is not None:
                # Append the return value if present (e.g. from tool calls)
                res_str = str(result).strip()
                if res_str:
                    if output:
                        output += "\n"
                    output += res_str
            if stderr:
                output += f"\nErrors:\n{stderr}"
                if "SyntaxError" in stderr and (
                    "indent" in stderr.lower() or "dedent" in stderr.lower()
                ):
                    output += "\n\n[System Hint]: This looks like a whitespace or indentation error. Ensure your Python block uses consistent 4-space indentation and that all 'await' calls are correctly aligned."

            # --- AUTO-INSTALLATION SELF-HEALING ---
            if "ModuleNotFoundError" in output:
                # Use regex to find "No module named 'package_name'"
                import re

                match = re.search(r"No module named ['\"]([^'\"]+)['\"]", output)
                if match:
                    package_name = match.group(1)
                    self.emit_event(
                        "thinking",
                        content=f"🛠️  Self-Healing: Detected missing package '{package_name}'. Attempting auto-installation...",
                    )
                    # Try to install to project environment
                    res = self.install_package(package_name)
                    if "Successfully installed" in res:
                        self.emit_event(
                            "thinking",
                            content=f"✅ Package '{package_name}' installed. Retrying execution...",
                        )
                        # RECURSIVE CALL: Await retry of same code
                        # (Infinite loop protection is handled by parent query_sync step limit)
                        return await self._execute_code(
                            code,
                            thought_id,
                            session_id,
                            root_session_id=root_session_id,
                            task_input=task_input,
                        )
                    else:
                        output += f"\nSystem: Automatic installation of '{package_name}' failed."
                else:
                    logger.warning(
                        "Could not parse package name from ModuleNotFoundError."
                    )

            # Ensure the final result of the execution (especially from asyncio.run) is captured
            if result is not None:
                if output:
                    output += f"\n[Execution Result]: {result}"
                else:
                    output = str(result)
            elif getattr(self, "_final_result", None):
                # If the variable was set but not returned (common with asyncio.run wrapper)
                final_snippet = str(self._final_result)
                if output:
                    output += f"\n[System]: Final Answer Captured: {final_snippet}"
                else:
                    output = f"Final Answer Captured: {final_snippet}"

            if not output.strip() and not stderr:
                output = "Code executed successfully (No output captured)."

            return output, is_err
        finally:
            self.current_thought_id = previous_thought_id
            # DO NOT DELETE REPL HERE - It persists for the session

    def stop_generation(self):
        """Signal the agent to stop processing."""
        logger.info("STOP SIGNAL RECEIVED: Setting stop flags.")
        if hasattr(self, "_global_stop_event"):
            self._global_stop_event.set()
        self._stop_requested = True

    async def _generate_validated_response(
        self, root_session_id: str, original_task: str
    ) -> str:
        """
        Generates a comprehensive RLM_VALIDATED_RESPONSE by summarizing the session trace.
        """
        logger.info(
            f"Generating Validated Response for Root Session: {root_session_id}"
        )

        # 1. Fetch Session Trace (Thoughts)
        cypher = "MATCH (n:Thought) WHERE n.root_session_id = $sid RETURN n ORDER BY n.timestamp ASC"
        try:
            res = self.db.query(cypher, {"sid": root_session_id})
            nodes = []
            if res and hasattr(res, "result_set"):
                for record in res.result_set:
                    if record and len(record) > 0:
                        nodes.append(record[0])

            # Formulate Trace String
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
                ts = props.get("timestamp", "unknown")

                if "SYSTEM" in step_type:
                    continue
                preview = content[:800] + "..." if len(content) > 800 else content
                trace_lines.append(
                    f"Turn {i+1} [{step_type}] (Time: {ts}, REPL: {repl_id}):\n{preview}\n"
                )

            trace_str = "\\n".join(trace_lines)

            # 2. Prompt LLM for Synthesis
            system_prompt = (
                "You are the RLM Validation Engine. Your goal is to synthesize a FINAL, HUMAN-READABLE REPORT.\\n"
                "Input: A trace of the Agent's reasoning and execution steps.\\n"
                "Output: A structured `RLM_VALIDATED_RESPONSE`.\\n"
                "\\n"
                "Requirements:\\n"
                "1. **Full Answer**: Provide the complete, final answer to the user's task. Synthesize findings from the trace.\\n"
                "2. **Methodology**: Briefly explain how the result was achieved.\\n"
                "3. **Turn Log**: List key turns/steps with their REPL IDs. This is CRITICAL for searchability.\\n"
                "4. **Format**: Markdown. Start exactly with `# RLM_VALIDATED_RESPONSE`.\\n"
            )

            user_prompt = (
                f"Original Task: {original_task}\\n\\nSession Trace:\\n{trace_str}"
            )

            response = await self.llm.generate(
                user_prompt, system=system_prompt, stream=False
            )
            return response

        except Exception as e:
            logger.error(f"Failed to generate validated response: {e}")
            return f"# RLM_VALIDATED_RESPONSE\\n\\n[Error generating validation: {e}]"

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
        # BUGFIX: Also count generic code execution as 'work' to avoid false Reward Hacking flags
        # when the agent uses standard Python tools (open, os, etc.) instead of RLM tools.
        if has_completion and not execution_log:
            # If thought_trace contains a code block, it's NOT a fake completion
            if "```python" not in thought_trace:
                score -= 0.5
                flags.append(
                    "REWARD_HACKING: Completion signal without empirical verification."
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
            invariants_text = await self.llm.generate(analysis_prompt)
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
                f"🛡️  Agentic Axiom Discovery: {final} (Match: {[s.get('name') for s in sim_skills if s.get('score', 0) > 0.7]})"
            )
            return final
        except Exception as e:
            logger.warning(f"Agentic discovery failed: {e}. Fallback to 'general'.")
            return ["general"]
agent = Agent()
