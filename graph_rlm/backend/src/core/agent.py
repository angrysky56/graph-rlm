"""
Recursive Logic Machine (RLM) Agent.
Handles the core execution loop, recursive querying, and tool integration.
"""

import asyncio
import contextvars
import datetime
import importlib.util
import inspect
import pkgutil
import queue
import re
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .config import settings
from .context_index import context_index
from .db import GraphClient, db
from .llm import LLMService, llm
from .logger import get_logger
from .manager import REPLManager
from .repe import repe
from .scratchpad_builder import scratchpad_builder
from .sheaf import sheaf
from .trace import register_monitor, trace_action

if TYPE_CHECKING:
    from graph_rlm.backend.src.mcp_integration.skills import SkillsManager

# ... existing imports ...


# MCP Integration
def is_mcp_available():
    """Defensive check for MCP tools availability."""
    return (
        importlib.util.find_spec("mcp_tools") is not None
        or importlib.util.find_spec("graph_rlm.backend.mcp_tools") is not None
    )


# Skills System
def is_skills_available():
    """Defensive check for Skills system availability."""
    return (
        importlib.util.find_spec("graph_rlm.backend.src.mcp_integration.skills")
        is not None
        or importlib.util.find_spec("mcp_integration.skills") is not None
    )


logger = get_logger("graph_rlm.agent")

# Context Variable to hold the event queue for the current execution thread/chain
execution_events: contextvars.ContextVar[Optional[queue.Queue]] = (
    contextvars.ContextVar("execution_events", default=None)
)


@dataclass
class ExecutionState:
    """Thread-local state for the agent's execution loop."""

    final_result: Optional[str] = None
    stop_requested: bool = False
    synthesis_triggered: bool = False
    current_thought_id: Optional[str] = None
    depth: int = 0
    turn_id: int = 1


def broadcast_trace(msg: str):
    """Monitor callback to push trace logs to the active event loop."""
    try:
        q = execution_events.get()
        if q:
            # Clean ANSI codes for UI (optional, but UI handles raw text better usually)
            # For now sending raw, UI can render or strip if needed.
            # Actually, let's keep ANSI for terminal but strip for UI?
            # Or send as is and let UI ignore colors or parse them.
            # Simple approach: Strip ANSI for cleaner UI text
            clean_msg = re.sub(r"\x1b\[[0-9;]*m", "", msg)
            q.put_nowait({"type": "trace", "content": clean_msg})
    except LookupError:
        pass
    except Exception as e:
        # Fallback log to avoid recursion loop if logging fails
        sys.stderr.write(f"Failed to broadcast trace: {e}\n")


# Register the monitor
trace_action(context="AGENT", action="Initializing Trace Monitor...", level="debug")

register_monitor(broadcast_trace)


# Session-specific state isolated by thread/context
agent_state: contextvars.ContextVar[Optional[ExecutionState]] = contextvars.ContextVar(
    "agent_state", default=None
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

                                wrapped.__doc__ = f.__doc__
                                return wrapped

                            wrapper = make_wrapper(func, f"mcp.{self._alias}.{attr}")
                            self._tools[attr] = wrapper
                            self._docs[attr] = func.__doc__
            except Exception as e:
                logger.warning("Failed to load MCP server %s: %s", self._mod_name, e)
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

    def set_rlm_interface(self, rlm_interface: "RLMInterface"):
        """Update the RLM interface binding and propagate to children."""
        self._rlm_interface = rlm_interface
        for server in self._aliases.values():
            if hasattr(server, "set_rlm_interface"):
                server.set_rlm_interface(rlm_interface)

    def _scan(self):
        if not self._scan_done and is_mcp_available():
            try:
                import graph_rlm.backend.mcp_tools as mcp_tools_pkg

                logger.info("Starting MCP server discovery...")
                for _, mod_name, _ in pkgutil.iter_modules(mcp_tools_pkg.__path__):
                    if mod_name.startswith("_") or mod_name == "skills":
                        logger.debug("Skipping module: %s", mod_name)
                        continue

                    logger.info("Discovered MCP module: %s", mod_name)

                    # Create MCPServerNamespace using the actual module name (no aliases)
                    # This ensures tool discovery works correctly by matching module structure
                    server = MCPServerNamespace(mod_name, mod_name, self._rlm_interface)
                    self._aliases[mod_name] = server
                    logger.info("Registered MCP server: %s", mod_name)

                self._scan_done = True
                logger.info("MCP server discovery completed.")
            except Exception as e:
                logger.warning("MCP Scan Error: %s", e)

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


class RLMInterface:
    """
    The object exposed to the REPL as 'rlm'.
    Allows recursive queries and memory recall.
    """

    def __init__(self, agent_instance: "Agent", session_id: str, root_session_id: str):
        self.agent = agent_instance
        self.session_id = session_id
        self.root_session_id = root_session_id

    def record_tool_use(self, name: str):
        # FAST STOP CHECK: If the user hit stop, we must abort immediately.
        if getattr(self.agent, "stop_requested", False) or (
            hasattr(self.agent, "global_stop_event")
            and self.agent.global_stop_event.is_set()
        ):
            logger.warning("Stop signal detected. Aborting tool call: %s", name)
            raise InterruptedError(f"Execution stopped by user (Attempted: {name})")

        if self.session_id not in self.agent.execution_logs:
            self.agent.execution_logs[self.session_id] = []
        self.agent.execution_logs[self.session_id].append(name)

    @property
    def history(self):
        """
        Exposes the full thought trace of the current session as a list of dictionaries.
        Allows the agent to programmatically inspect its own reasoning history.
        """
        self.record_tool_use("rlm.history")
        try:
            return self.agent.db.get_session_trace(self.root_session_id)
        except Exception as e:
            logger.error("Failed to fetch history for rlm.history: %s", e)
            return []

    async def query(
        self,
        prompt: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
        depth: Optional[int] = None,
    ):
        """
        Recursive Primitive: Spawns a recursive child agent.
        """
        self.record_tool_use("rlm.query")
        # CRITICAL: Each thought gets a FRESH session_id (Atomic REPL)
        new_session_id = session_id or str(uuid.uuid4())

        # Turn tracking: Inherit from parent turn
        current_state = agent_state.get()
        current_turn = current_state.turn_id if current_state else 1

        # Depth tracking: Increment depth for children
        current_depth = (
            depth if depth is not None else getattr(self.agent, "current_depth", 0)
        )
        new_depth = current_depth + 1

        trace_action(
            "RLM",
            "RECURSE",
            result=f"Spawning child session (Depth: {new_depth}) for: {prompt}",
            tag="AGENT",
        )
        self.agent.emit_event(
            "thinking",
            content=f"\n⚡ RLM: Spawning Recursive Agent (Depth: {new_depth}) for: '{prompt}'",
        )

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask: {prompt}"

        # stream_query is async, so we await it
        results = ""
        async for event in self.agent.stream_query(
            full_prompt,
            parent_id=self.agent.current_thought_id,
            session_id=new_session_id,
            depth=new_depth,
            turn_id=current_turn,
        ):
            # Pipe child events up to parent's queue
            # We prefix the type or content to show it's a sub-agent
            if event["type"] == "done":
                results = event["content"]
                # DO NOT pipe "done" to parent's UI, it kills the stream
                continue

            if event["type"] == "error":
                self.agent.emit_event(
                    "thinking",
                    content=f"⚠️ [RLM Child Error]: {event.get('content')}",
                    is_sub_event=True,
                )
                continue

            # Prefix content for better visibility
            content = event.get("content", "")
            if event["type"] == "code_output_chunk":
                content = f"[RLM Child Output] {content}"
            elif event["type"] == "thinking" and content:
                content = f"⚡ [RLM Child]: {content}"

            self.agent.emit_event(
                event["type"],
                data=event.get("data"),
                content=content,
                code=event.get("code"),
                is_sub_event=True,
            )
        if not results:
            msg = "Recursion completed without a final answer."
            print(f"\n[RLM] {msg}")
            raise RuntimeError(msg)

        print(f"\n[RLM] Recursion completed with results: {results}")
        return results

    async def recall(self, query: str, limit: int = 5):
        """
        Active Recall: Search the Graph for similar past thoughts.
        """
        self.record_tool_use("rlm.recall")
        logger.info("Thought %s: Recalling '%s'", self.agent.current_thought_id, query)
        self.agent.emit_event(
            "thinking", content=f"\n🧠 RLM: Recalling memories for '{query}'..."
        )
        try:
            vec = await self.agent.llm.get_embedding(query)
            if not vec:
                trace_action(
                    "RLM",
                    "RECALL_FAIL",
                    result="Failed to generate embedding",
                    tag="AGENT",
                    level="error",
                )
                return "Failed to generate embedding for recall query."

            # Use db.find_similar_thoughts
            results = self.agent.db.find_similar_thoughts(vec, limit=limit)

            trace_action(
                "RLM",
                "RECALL",
                result=f"Found {len(results)} past thoughts for query '{query}'",
                tag="AGENT",
                level="info" if results else "warning",
            )

            # Format results for the LLM
            formatted = []
            for row in results:
                # Results from find_similar_thoughts: {"id": row[0], "prompt": row[1], "result": row[2], "score": row[3]}

                tid = row.get("id", "Unknown")
                prompt = row.get("prompt", "No prompt")
                result = row.get("result", "No result")
                score = float(row.get("score", 0.0))

                formatted.append(
                    f"- [Similarity: {score:.2f}] (ID: {tid}) Thought: {prompt} -> Result: {result}"
                )

            if not formatted:
                self.agent.emit_event(
                    "thinking", content="\n🧠 RLM: No matching memories found."
                )
                return "No semantically similar thoughts found in memory."

            output = f"Recall found {len(formatted)} relevant entries."
            print(f"\n[RLM] {output}")
            return (
                "\n\n".join(formatted)
                if formatted
                else "No relevant past thoughts found."
            )

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Recall Error: %s", e)
            return f"Error during memory recall: {e}"

    async def search(self, query: str, limit: int = 10):
        """Topological search across the graph (alias for graph_search)."""
        self.record_tool_use("rlm.search")
        vec = await self.agent.llm.get_embedding(query)
        if vec:
            results = self.agent.db.find_similar_thoughts(vec, limit)
            if not results:
                return "No results found."
            return results
        return "Failed to generate embedding."

    async def ingest_document(self, path: str, domain: str = "general"):
        """Ingests a document and codifies its knowledge into Axioms (CAG)."""
        self.record_tool_use("rlm.ingest_document")
        from .dream import dreamer

        res = await dreamer.ingest_document(path, domain)
        if res.get("status") == "success":
            return f"Successfully codified {len(res.get('codified_axioms', []))} axioms: {res.get('codified_axioms')}"
        return f"Ingestion failed: {res.get('message', 'Unknown error')}"

    async def save_skill(self, name: str, code: str, description: Optional[str] = None):
        """Saves a code snippet as a persistent skill."""
        self.record_tool_use("rlm.save_skill")
        if not is_skills_available():
            return "Skills system not available."
        from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager

        mgr = get_skills_manager()
        await mgr.save_skill(name, code, description)
        return f"Skill '{name}' saved successfully."

    async def run_skill(self, name: str = "", args: Optional[dict] = None, **kwargs):
        """Executes a registered skill."""
        self.record_tool_use("rlm.run_skill")
        # Handle 'title' as an alias for 'name' if the agent hallucinates it
        skill_name = name or kwargs.get("title") or ""
        if not skill_name:
            return "Error: No skill name or title provided."

        if not is_skills_available():
            return "Skills system not available."
        from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill

        return await execute_skill(skill_name, args or {})

    async def get_axiom(self, name: str):
        """Retrieves an axiom's code and metadata by name."""
        self.record_tool_use("rlm.get_axiom")
        from graph_rlm.backend.src.mcp_integration.skills import get_axioms_manager

        mgr = get_axioms_manager()
        axiom = mgr.get_axiom(name)
        if not axiom:
            return f"Axiom '{name}' not found."
        return axiom

    async def recall_axioms(self, query: str, limit: int = 5):
        """High-precision semantic search for domain rules and axioms."""
        self.record_tool_use("rlm.recall_axioms")
        from graph_rlm.backend.src.mcp_integration.skills import get_axioms_manager

        mgr = get_axioms_manager()
        results = await mgr.find_similar_axioms(query, limit)
        if not results:
            return "No relevant axioms found."

        formatted = []
        for a in results:
            type_tag = f" ({a.get('axiom_type')})" if a.get("axiom_type") else ""
            formatted.append(
                f"Axiom: {a.get('name')}{type_tag}\n"
                f"Description: {a.get('description')}\n"
                f"Code: {a.get('code')}"
            )
        return "\n\n---\n\n".join(formatted)

    async def execute_axiom(self, name: str, args: Optional[dict] = None):
        """Executes a 'solver' or 'heuristic' axiom directly."""
        self.record_tool_use("rlm.execute_axiom")
        from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill
        from graph_rlm.backend.src.mcp_integration.skills import get_axioms_manager

        mgr = get_axioms_manager()
        axiom = mgr.get_axiom(name)
        if not axiom:
            return f"Axiom '{name}' not found."

        if axiom.get("axiom_type") == "validator":
            return (
                "Warning: This is a 'validator' axiom. It should be used via rlm.verify_axiom "
                "or by the Sheaf monitor. Running as a skill might not have the intended effect."
            )

        return await execute_skill(name, args or {})

    async def install_package(self, package_name: str):
        """Install a Python package into the agent's REPL environment."""
        self.record_tool_use("rlm.install_package")
        return self.agent.install_package(package_name)

    async def install_skill_package(self, package_name: str):
        """Install a package specifically for the AGENT skills (agent_venv) environment."""
        self.record_tool_use("rlm.install_skill_package")
        return self.agent.install_skill_package(package_name)

    async def done(self, final_answer: str = ""):
        """Signifies the agent has reached a final conclusion."""
        self.record_tool_use("rlm.done")
        self.agent.stop_requested = True
        if final_answer:
            self.agent.final_result = final_answer

        # Log a summary to console, but return full confirmation
        summary = final_answer
        msg = f"Task Marked Complete. Summary: {summary}"
        print(f"\n[RLM] {msg}")

        # Emit final answer to UI
        self.agent.emit_event("answer", content=final_answer)

        return "Task completed successfully."

    async def stop(self, final_answer: str = ""):
        """Alias for done()."""
        self.record_tool_use("rlm.stop")
        return await self.done(final_answer)

    async def help(self):
        """Broad discovery of available commands within the 'rlm' namespace."""
        self.record_tool_use("rlm.help")

        # Core RLM Commands
        help_dict = {
            "query(prompt, context)": "Spawn a recursive child agent.",
            "recall(query, limit)": "Semantic search through memory.",
            "search(query, limit)": "Graph search (alias for recall).",
            "ingest_document(path, domain)": "CAG: Codify docs into Axioms.",
            "save_skill(name, code, desc)": "Persist a code block.",
            "run_skill(name, args)": "Run a saved code block.",
            "get_axiom(name)": "Retrieve axiom code and metadata.",
            "recall_axioms(query, limit)": "Semantic search for domain rules.",
            "execute_axiom(name, args)": "Execute a solver or healing axiom.",
            "install_package(name)": "Install Python dependencies.",
        }

        # Dynamic MCP Tool Discovery
        try:
            # Resolve backend root to find mcp_tools
            backend_root = Path(__file__).parent.parent.parent
            tools_dir = backend_root / "mcp_tools"
            ignored = {"list_servers", "call_tool", "run_skill"}

            for f in tools_dir.glob("*.py"):
                if f.name.startswith("_") or f.stem in ignored:
                    continue

                module_name = f.stem
                try:
                    mod = importlib.import_module(
                        f"graph_rlm.backend.mcp_tools.{module_name}"
                    )
                    for name, obj in inspect.getmembers(mod, inspect.isfunction):
                        if not name.startswith("_") and name != "list_tools":
                            # Format: mcp.server.tool(args)
                            sig = str(inspect.signature(obj))
                            key = f"mcp.{module_name}.{name}{sig}"
                            doc = inspect.getdoc(obj) or "No description."
                            help_dict[key] = doc.split("\n", maxsplit=1)[0]  # Brief doc
                except Exception as e:
                    logger.warning(
                        f"Failed to load module '{module_name}' for help: {e}"
                    )
                    continue
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Error discovering MCP tools for help(): %s", e)

        return help_dict

    def __repr__(self):
        return f"<RLMInterface [Session: {self.session_id[:8]}...] Type 'rlm.help()' for tools>"


class Agent:
    """
    The core Recursive Logic Machine (RLM) agent.
    Handles the main execution loop, epistemic health checks, and dreamer integration.
    """

    skills_manager: Optional["SkillsManager"]

    def __init__(self):
        # NOTE: nest_asyncio removed - incompatible with uvloop
        # Code flow is properly async (async def + await) so not needed.

        # --- PATH INJECTION (Bootstrap) ---
        # Ensure 'skills_dir' and 'mcp_tools' are in path for imports
        # We need the project root for 'graph_rlm' nested imports
        project_root = Path(__file__).resolve().parents[4]
        backend_path = Path(__file__).resolve().parents[2]
        skills_path = backend_path / "skills_dir"
        mcp_tools_path = backend_path / "mcp_tools"

        if str(project_root) not in sys.path:
            logger.info("Injecting Project Root: %s", project_root)
            sys.path.append(str(project_root))

        if str(backend_path) not in sys.path:
            logger.info("Injecting Backend Path: %s", backend_path)
            sys.path.append(str(backend_path))

        if str(skills_path) not in sys.path:
            sys.path.append(str(skills_path))

        if str(mcp_tools_path) not in sys.path:
            sys.path.append(str(mcp_tools_path))

        # --- CORE INITIALIZATION ---
        self.db: GraphClient = db
        self.llm: LLMService = llm
        self.repl_manager = REPLManager()
        self.active_repls: Dict[str, str] = {}  # session_id -> repl_id
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
        self.stop_requested: bool = False
        self.final_result: Optional[str] = None
        self.synthesis_triggered: bool = False
        self.step_id: int = 0
        self.current_turn: int = 1
        self.current_thought_id: Optional[str] = None

        if is_skills_available():
            from graph_rlm.backend.src.mcp_integration.skills import (
                get_skills_manager,
            )

            self.skills_manager = get_skills_manager()
        else:
            self.skills_manager = None

        # Ensure Knowledge Base scaffolding exists
        self._ensure_kb_structure()

        # Environment Strategy:
        # 1. Core Agent / REPL: Runs in the active project environment (sys.prefix).
        # 2. Skills / Tools: Run in the dedicated 'agent_venv' for isolation.
        # 3. MCP Servers: Run in their own independent environments (managed by uv or configured venvs).
        logger.info("Agent initialized using active environment: %s", sys.prefix)
        logger.info("Agent initialized with Persistent REPL support")
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
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to verify Knowledge Base structure: %s", e)

    # --- SESSION-ISOLATED PROPERTIES ---
    def _get_state(self) -> ExecutionState:
        state = agent_state.get()
        if state is None:
            # Fallback for out-of-session access (e.g. CLI or background tasks)
            state = ExecutionState()
            agent_state.set(state)
        return state

    @property
    def final_result(self) -> Optional[str]:
        return self._get_state().final_result

    @final_result.setter
    def final_result(self, value: Optional[str]):
        self._get_state().final_result = value

    @property
    def stop_requested(self) -> bool:
        # Check both local context AND global signal
        return self._get_state().stop_requested or self.global_stop_event.is_set()

    @stop_requested.setter
    def stop_requested(self, value: bool):
        self._get_state().stop_requested = value

    @property
    def synthesis_triggered(self) -> bool:
        return self._get_state().synthesis_triggered

    @synthesis_triggered.setter
    def synthesis_triggered(self, value: bool):
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
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                logger.info("Successfully installed %s", package_name)
                self.emit_event("thinking", content="  -> Installation successful.")
                return f"Successfully installed {package_name}\n{result.stdout}"
            else:
                logger.error("Failed to install %s: %s", package_name, result.stderr)
                self.emit_event(
                    "error", content=f"Installation failed: {result.stderr}"
                )
                return f"Failed to install {package_name}\nError: {result.stderr}"
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Installation error: %s", e)
            return f"Installation error: {e}"

    def _install_to_agent_venv(self, package_name: str) -> str:
        """Internal helper to install a package into the DEDICATED AGENT VENV."""

        # trunk-ignore(bandit/B404)
        import subprocess

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
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Installation error: %s", e)
            return f"Installation error: {e}"

    def install_package(self, package_name: str) -> str:
        """Installs a package into the active environment (REPL compatibility)."""
        return self._install_to_active_env(package_name)

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
            from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager

            mgr = get_skills_manager()
            skill = mgr.get_skill(name)
            if not skill:
                self.emit_event("error", content=f"Skill '{name}' not found.")
                return f"Error: Skill '{name}' not found."
            return skill["code"]
        except Exception as e:  # pylint: disable=broad-except
            self.emit_event("error", content=f"Error reading skill: {e}")
            return f"Error reading skill: {e}"

    def emit_event(
        self,
        event_type: str,
        data: Any = None,
        content: Optional[str] = None,
        code: Optional[str] = None,
        is_sub_event: bool = False,
        tag: Optional[str] = None,
    ):
        """
        Helper to emit events to the current context's queue if it exists.
        Also mirrors key events to the server logs (terminal) for visibility.
        """
        # Determine logical REPL ID for this event tracking
        # We try to get it from the active_repls mapping using session_id if possible
        # but since this is called from anywhere, we rely on content prefixing for UI visibility.

        prefix = "↳ " if is_sub_event else ""

        # Mirror to Terminal/Logs
        if event_type == "thinking" and content:
            # Use tag if available for better log mirroring
            log_prefix = f"[THINKING] [{tag}]" if tag else "[THINKING]"
            logger.info("%s%s %s", prefix, log_prefix, content.strip())
        elif event_type == "code_output" and content:
            logger.info("%s[REPL OUTPUT] >>\n%s", prefix, content)
        elif event_type == "error" and content:
            logger.error("%s[AGENT ERROR] %s", prefix, content)

        q = execution_events.get()
        if q:
            payload = {"type": event_type, "is_sub_event": is_sub_event}
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
        session_id: str = "default",
        depth: int = 0,
        root_session_id: Optional[str] = None,
        turn_id: int = 1,
    ):
        """
        Streaming entry point.
        launches the sync execution in a thread and yields events from a queue.
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
                    depth=depth, current_thought_id=parent_id, turn_id=turn_id
                )
            )
            try:
                # Set initial depth for this agent run
                self.current_depth = depth
                asyncio.run(
                    self.query_sync(
                        prompt, parent_id, session_id, depth, root_session_id, turn_id
                    )
                )
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error in execution thread: %s", e)
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
        turn_id: int = 1,
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

        # 0. Reset scoped State for this specific call (redundant if already set in stream_query but safe)
        if not agent_state.get():
            agent_state.set(
                ExecutionState(
                    depth=depth, current_thought_id=parent_id, turn_id=turn_id
                )
            )

        state = agent_state.get()
        if state:
            self.current_turn = state.turn_id

        self.final_result = None
        self.stop_requested = False

        # Ensure REPL is initialized for this session
        if session_id not in self.active_repls:
            self.active_repls[session_id] = self.repl_manager.create_repl()

        # MCP STOP SIGNAL REGISTRATION
        if is_mcp_available():
            from graph_rlm.backend.src.mcp_integration.runtime import set_stop_event

            set_stop_event(self.global_stop_event)

        # 0. Initial "Task" Node (Root of this query)
        # Wrap everything in try/except to prevent DB crashes from killing the agent
        # Generate Round ID for this execution cycle (compress context)
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
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to initialize Task node: %s", e)
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
        base_system_prompt = await self._build_system_prompt()

        max_steps = 1000
        step = 0

        # Track previous status for topological resolution
        previous_thought_status = None

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

            # --- DYNAMIC SCRATCHPAD REFRESH ---
            try:
                context_scratchpad = scratchpad_builder.build_scratchpad(
                    session_id=session_id,
                    root_session_id=final_root_id,
                    task=prompt,
                    # current_step=step matches the loop counter (1-based now)
                    current_step=step,
                    max_steps=max_steps,
                    current_round_id=current_round_id,
                )
                self.emit_event("scratchpad_text", content=context_scratchpad)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Failed to build scratchpad: %s", e)
                context_scratchpad = f"Error: Scratchpad unavailable ({e})"

            system_prompt = f"{base_system_prompt}\n\n{context_scratchpad}"

            # Construct Dynamic Context (Minimal)
            # No longer pre-loading raw Frontier content into the prompt.
            # History is accessible via context_scratchpad (Index) and graph_search.
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
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Context loading failed (Sheaf IDs): %s", e)

            # 3. Construct LLM Context using XML isolation for Gemini safety
            # [STABILITY] Explicitly prefix all paths and wrap in XML to avoid command hallucination
            current_context = (
                f"Active Session: {session_id}\n\n"
                f"<objective>\n{prompt}\n</objective>\n\n"
                f"<history>\n{context_scratchpad}\n</history>\n"
            )

            # 2b. Language Guard: Check if frontier is primarily non-English
            if frontier:
                # Heuristic...
                pass

            # Load Axioms (Semantic Retrieval)
            axioms_list_str = "None"
            if is_skills_available():
                try:
                    from graph_rlm.backend.src.mcp_integration.skills import (
                        get_axioms_manager,
                    )

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
                            f"Loaded {len(relevant_axioms)} semantic axioms: {axioms_list_str[:100]}..."
                        )
                    else:
                        # Fallback
                        axioms = axioms_mgr.list_axioms()
                        sorted_keys = sorted(axioms.keys())[:20]
                        axioms_list_str = ", ".join(sorted_keys)
                except Exception as ex:  # pylint: disable=broad-except
                    logger.warning("Failed to load axioms async: %s", ex)

            # --- HOT SEAT INJECTION ---
            hot_seat_warning = ""
            if getattr(self, "last_dream_insight", None):
                hot_seat_warning = (
                    "\n\n--- ⚠️ HOT SEAT: EPISTEMIC RECOVERY ACTIVE ---\n"
                    "Your previous response was REJECTED by the Dreamer Gatekeeper for Hallucination/Trace Contradiction.\n"
                    f"CRITIQUE: {self.last_dream_insight}\n"
                    "You MUST explicitly address the contradiction, explain why you failed, "
                    "and provide a GROUNDED response based strictly on the execution trace.\n"
                    "Failure to align will result in a recursive block.\n---"
                )

            system_prompt = (
                f"{await self._build_system_prompt(axioms_list_str=axioms_list_str)}\n\n"
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

                # Define usage callback
                def on_token_usage(usage_data):
                    self.emit_event("usage", data=usage_data)

                # [DIAGNOSTIC] Log start of network request
                self.emit_event(
                    "debug_thought",
                    content=f"... Sending request to LLM (Size: {len(current_context)} chars) ...",
                )

                # Pre-gen stop check
                if self.stop_requested or self.global_stop_event.is_set():
                    self.stop_requested = True
                    break

                response_text = await self.llm.generate(
                    current_context,
                    system=system_prompt,
                    stream=False,
                    on_usage=on_token_usage,
                )

                # Post-gen stop check
                if self.stop_requested or self.global_stop_event.is_set():
                    self.stop_requested = True
                    break
                # self.emit_event("token", content=response_text) # Redundant with thinking output
            except Exception as e:  # pylint: disable=broad-except
                response_text = f"LLM Error: {e}"
                # Log exception to standard logger
                logger.error("LLM Generative Error: %s", e)

            # Raw response logging removed for stability

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
                    )
                except Exception as db_err:  # pylint: disable=broad-except
                    logger.error("Failed to commit error node: %s", db_err)
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

            self.emit_event("debug_thought", content=formatted_thought)

            # 4. Step Initialization
            # We create the ID early so it can be used in tool execution

            trace_action(
                "AGENT", "THOUGHT", result=response_text[:400] + "...", tag="AGENT"
            )

            output = ""
            code = self._extract_code(response_text)

            # --- PRE-EXECUTION DIAGNOSTICS ---
            repl_id = self.active_repls.get(session_id)

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
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Failed to pre-commit thought: %s", e)

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
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("Failed to embed task for Health Check: %s", e)

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
                        "Re-read the original user request and justify how this step helps."
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
                    logger.warning("🛡️ Triggering Intervention: %s", intervention_type)
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

                # 6b. [CAG Pivot] Axiomatic Consistency Check
                # Before executing, run the proposed code through all verified Axiom Skills
                try:
                    # Agentic Axiom Discovery
                    task_tags = await self._detect_required_axioms_agentic(prompt, code)

                    axiom_diag = await sheaf.check_axiomatic_consistency(
                        code, task_tags=task_tags
                    )
                    if axiom_diag.get("status") == "AXIOMATIC_VIOLATION":
                        axiom_critique = axiom_diag.get("critique")
                        logger.warning("🛡️  CAG Blocked execution: %s", axiom_critique)
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("Axiomatic check failed to run: %s", e)

                execution_failed = False
                if axiom_critique:
                    # Self-Healing: Inject Reflexion Node instead of executing
                    self.emit_event(
                        "warning",
                        content=f"🛡️ [REPL: {repl_id}] AXIOMATIC VIOLATION: Execution Blocked.\nCritique: {axiom_critique}",
                    )
                    # We skip execution but we MUST update current_thought_id to ensure chaining
                    output = f"Axiomatic Violation: Execution Blocked.\nCritique: {axiom_critique}"
                    thought_status = "failed"
                    execution_failed = True
                else:
                    # Pre-execution stop check
                    if self.global_stop_event.is_set() or self.stop_requested:
                        self.stop_requested = True
                        break

                    # Check code safety?
                    output, execution_failed = await self._execute_code(
                        code,
                        thought_id,
                        session_id,
                        root_session_id=final_root_id,
                        task_input=prompt,
                    )

                    # Post-execution stop check
                    if self.stop_requested or self.global_stop_event.is_set():
                        self.stop_requested = True
                        break

                if repl_id:
                    self.repl_manager.get_repl(repl_id)

                self.emit_event("debug_code", content=output, code=code)

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
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("Failed to generate final embedding: %s", e)

            # Generate execution summary for scratchpad display
            # Summary is brief (first ~200 chars), full result stored in 'result' field
            exec_summary = None
            if output:
                # Truncate to first meaningful line(s), max 200 chars
                clean_output = output.strip()
                if len(clean_output) > 200:
                    exec_summary = clean_output[:200] + "..."
                else:
                    exec_summary = clean_output
                # Mark success/failure in summary
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
                )

                # Execute Pruning
                if node_to_prune:
                    try:
                        self.db.delete_thought_node(node_to_prune)
                    except Exception as prune_err:
                        logger.error(
                            "Failed to prune node %s: %s", node_to_prune, prune_err
                        )

                # Emit immediate scratchpad update for UI responsiveness
                try:
                    sp_data = context_index.get_active_scratchpad_data(final_root_id)
                    self.emit_event(
                        "scratchpad_update", data=sp_data, is_sub_event=False
                    )
                except Exception as ex:  # pylint: disable=broad-except
                    logger.warning("Failed to emit scratchpad update: %s", ex)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Failed to commit thought to graph: %s", e)

            # Update Frontier Pointer
            # Update previous status for next iteration
            previous_thought_status = thought_status
            self.current_thought_id = thought_id

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
                )
                # Skip the rest of the loop to let the Agent generate the synthesis
                continue

            if (
                "RLM_FINAL_OUTPUT" in response_text
                or getattr(self, "final_result", None)
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
                        + "\nI MUST show my work, avoid being overly obsequious, and verify results with tools."
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
                    )
                    continue  # Keep the loop running
                # --- END VERIFICATION ---

                if not self.final_result:
                    self.final_result = response_text

                # 1. Generate Candidate Validated Response (Draft)
                # We do this BEFORE Dreamer so Dreamer can validate whether this response resolves prior failures.
                final_response_candidate = None
                if self.final_result:
                    try:
                        final_response_candidate = (
                            await self._generate_validated_response(
                                final_root_id, prompt
                            )
                        )
                    except Exception as e:  # pylint: disable=broad-except
                        logger.warning("Failed to generate candidate response: %s", e)

                # 2. Dreamer Trigger (Auto-Consolidate before exit)
                try:
                    # Lazy import to avoid circular dependency at top level if any
                    from .dream import dreamer

                    logger.info("💤 Triggering Pre-Exit Dream Cycle (No timeout)...")
                    try:
                        # Pass emit_event so Dreamer can emit progress to UI
                        def dreamer_emit(event_type, content):
                            self.emit_event(event_type, content=content, tag="DREAMER")

                        # Re-generate scratchpad for Dreamer context if needed
                        # Or use the last known context. The prompt variable might contain it,
                        # but scratchpad_content is cleaner
                        scratchpad_content = scratchpad_builder.build_scratchpad(
                            session_id=session_id,
                            root_session_id=final_root_id,
                            task=prompt,
                            current_step=getattr(self, "step_count", 0),
                            max_steps=getattr(self, "max_steps", 10),
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
                    except Exception as e:  # pylint: disable=broad-except
                        logger.warning("Dream cycle failed during execution: %s", e)
                        dream_res = {}

                    # GATEKEEPER LOGIC: Check Dreamer Status
                    if dream_res.get("status") == "lucid":
                        insight = dream_res.get("insight") or ""

                        # 1. Emit Insight
                        if insight:
                            dreamer_msg = f"💤 [Dreamer Gatekeeper]: Systemic Issue Detected. Blocking Exit.\n\n{insight}"
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
                                    Exception
                                ) as db_err:  # pylint: disable=broad-except
                                    logger.error(
                                        "Failed to persist Dreamer msg: %s", db_err
                                    )

                        # 2. Check if we've already tried to fix this exact issue to prevent infinite loops
                        last_insight = getattr(self, "last_dream_insight", None)
                        if last_insight != insight:
                            # 3. REJECT EXIT. Force Self-Healing.
                            logger.info(
                                "🛡️ Dreamer Gatekeeper REJECTED exit. Injecting insight for self-healing."
                            )
                            self.last_dream_insight = insight
                            self.final_result = None  # Cancel valid result
                            self.stop_requested = False
                            self.synthesis_triggered = False

                            # Inject the Insight as a High-Priority Thought
                            rejection_msg = (
                                f"DREAMER GATEKEEPER: I cannot accept this Result yet.\n"
                                f"I detected a systemic failure pattern in your recent actions:\n{insight}\n\n"
                                "MANDATE: You must explicitly address this failure pattern before finishing.\n"
                                "If you have already fixed it, prove it with a final verification step."
                            )

                            self.db.create_thought_node(
                                str(uuid.uuid4()),
                                rejection_msg,
                                session_id=session_id,
                                root_session_id=final_root_id,
                                dreamer_analysis=insight,
                                round_id=current_round_id,
                                status="rejected",
                                turn_id=self.current_turn,
                                step_id=step,
                            )
                            # Set internal state to force recovery behavior
                            self.synthesis_triggered = False
                            continue  # Loop back
                        else:
                            logger.warning(
                                "Dreamer loop detected (insight repeated). allowing force exit."
                            )

                        # If status is 'peaceful', we proceed to exit.

                        self.db.create_thought_node(
                            str(uuid.uuid4()),
                            f"SYSTEM DREAM: I have consolidated recent failures into new Insights: {insight}...",
                            session_id=session_id,
                            root_session_id=final_root_id,
                            dreamer_analysis=insight,  # Store the full insight
                            round_id=current_round_id,
                            turn_id=self.current_turn,
                            step_id=step,
                        )
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("Dream cycle failed on exit: %s", e)

                # 3. Store the final response (If we passed the Gatekeeper)
                if final_response_candidate:
                    try:
                        # CRITICAL FIX: Adopt the Dreamer-validated response as the Final Result
                        self.final_result = final_response_candidate

                        # EMIT IT SO USER SEES IT IMMEDIATELY
                        self.emit_event(
                            "thinking",
                            content=final_response_candidate,
                            tag="VALIDATOR",
                        )

                        self.db.create_thought_node(
                            str(uuid.uuid4()),
                            final_response_candidate,  # Store the RICH response as content
                            session_id=session_id,
                            root_session_id=final_root_id,
                            status="success",
                            final_response=self.final_result,
                            round_id=current_round_id,
                            turn_id=self.current_turn,
                            step_id=step,
                        )

                    except Exception as e:  # pylint: disable=broad-except
                        logger.warning("Failed to store final response: %s", e)

                break

            # 2. Sheaf-based Stall/Loop Detection (Self-Healing)
            # If the Sheaf Monitor detected a high energy knot (repetition or contradiction),
            # we do NOT terminate. We inject a "Reflexion" to break the loop.
            energy = float(sheaf_diag.get("consistency_energy", 0.0))
            if energy > 0.9:
                logger.warning(
                    "Sheaf detected logical knot (Loop/Contradiction). Initiating Reflexion."
                )

                # Overwrite the 'thought' with a Meta-Cognitive critique
                reflexion_content = (
                    f"SYSTEM REFLEXION: I have detected a High-Energy Logical Knot (Energy: {sheaf_diag.get('consistency_energy'):.2f}). "
                    "I am repeating myself or contradicting recent history. "
                    "I MUST now change my approach completely. What variable am I missing?"
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
                )

                # Update pointer
                self.current_thought_id = reflexion_id

                # Do NOT break. Let the loop continue.
                continue

        # 8. Loop Exit: Emit Final Answer if available
        # 8. Loop Exit: Emit Final Answer if available
        if self.final_result:
            self.emit_event("RLM_FINAL_RESPONSE", content=self.final_result)
        elif self.stop_requested:
            # Stop requested by user or tool
            self.emit_event(
                "RLM_FINAL_RESPONSE",
                content="Task processing stopped (Done/Stop signal received).",
            )
        elif step >= max_steps:
            self.emit_event(
                "error",
                content=f"AGENT LIMIT REACHED: Reached max_steps ({max_steps}). Stopping execution.",
            )
            logger.warning(
                "Session %s reached max steps (%s) and aborted.", session_id, max_steps
            )
        else:
            # Fallback: Emit a system notice if the loop exits without a result (e.g. error/circuit breaker)
            # This prevents the UI from hanging fastidiously waiting for an event that never comes.
            logger.warning(
                "Agent loop exited without a final result. Emitting fallback."
            )
            self.emit_event(
                "RLM_FINAL_RESPONSE",
                content="[System] The agent stopped without generating a final answer. Please check the logs for errors.",
            )

        # 9. ARCHIVE ROUND (If we have a result or just to save state)
        if self.final_result:
            try:
                # Reconstruct full scratchpad for archive
                final_scratchpad = scratchpad_builder.build_scratchpad(
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
                        "MATCH (n:Thought) WHERE n.round_id = $rid AND n.repl_id IS NOT NULL RETURN DISTINCT n.repl_id",
                        {"rid": current_round_id},
                    )
                    repl_ids = (
                        [row.get("n.repl_id") or row["n.repl_id"] for row in r_res]
                        if r_res
                        else []
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch REPL IDs for round {current_round_id}: {e}"
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
            except Exception as e:
                logger.error("Failed to archive round: %s", e)

        return self.final_result or "Task processing stopped."

    async def _build_system_prompt(self, axioms_list_str: str = "None") -> str:
        # Extracted system prompt builder for cleanliness
        # Resolve paths for transparency
        backend_root = Path(__file__).parent.parent.parent
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
            logger.warning("Failed to load skills for prompt: %s", e)
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
                logger.warning("Failed to load rules.md: %s", e)

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
                repl.namespace["mcp"], "set_rlm_interface"
            ):
                repl.namespace["mcp"].set_rlm_interface(repl.namespace["rlm"])

            # Log the namespace state for debugging
            logger.debug("REPL namespace keys: %s", list(repl.namespace.keys()))
            # CRITICAL DEBUG: Check if session_id is ACTUALLY usable
            try:
                # We try to eval it in the namespace to see if it's accessible
                # This mimics what exec() sees
                check_sid = repl.namespace.get("session_id", "MISSING")
                logger.debug("Pre-flight check: session_id = %s", check_sid)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Pre-flight namespace check failed: %s", e)

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
            elif getattr(self, "final_result", None):
                # If the variable was set but not returned (common with asyncio.run wrapper)
                final_snippet = str(self.final_result)
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
        if hasattr(self, "global_stop_event"):
            self.global_stop_event.set()
        self.stop_requested = True

    async def _generate_validated_response(
        self, root_session_id: str, original_task: str
    ) -> str:
        """
        Generates a comprehensive RLM_VALIDATED_RESPONSE by summarizing the session trace.
        """
        logger.info(
            "Generating Validated Response for Root Session: %s", root_session_id
        )

        # 1. Fetch Session Trace (Thoughts)
        cypher = "MATCH (n:Thought) WHERE n.root_session_id = $sid RETURN n ORDER BY n.timestamp ASC"
        try:
            res = self.db.query(cypher, {"sid": root_session_id})
            nodes = (
                [r["n"] for r in res if isinstance(r, dict) and "n" in r] if res else []
            )

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

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to generate validated response: %s", e)
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
                "🛡️  Agentic Axiom Discovery: %s (Match: %s)",
                final,
                [s.get("name") for s in sim_skills if s.get("score", 0) > 0.7],
            )
            return final
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Agentic discovery failed: %s. Fallback to 'general'.", e)
            return ["general"]


agent = Agent()
