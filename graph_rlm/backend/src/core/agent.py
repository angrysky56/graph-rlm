import asyncio
import contextvars
import datetime
import importlib.util
import queue
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .context_index import context_index
from .db import GraphClient, db
from .llm import LLMService, llm
from .logger import get_logger
from .manager import REPLManager
from .repe import repe
from .sheaf import sheaf

# ... existing imports ...


# MCP Integration
MCP_AVAILABLE = importlib.util.find_spec("graph_rlm.backend.mcp_tools") is not None

# Skills System
SKILLS_AVAILABLE = False
try:
    from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill
    from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager

    SKILLS_AVAILABLE = True
except ImportError:
    pass


logger = get_logger("graph_rlm.agent")

# Context Variable to hold the event queue for the current execution thread/chain
execution_events: contextvars.ContextVar[Optional[queue.Queue]] = (
    contextvars.ContextVar("execution_events", default=None)
)


class RLMInterface:
    """
    The object exposed to the REPL as 'rlm'.
    Allows recursive queries and memory recall.
    """

    def __init__(self, agent: "Agent", session_id: str, root_session_id: str):
        self.agent = agent
        self.session_id = session_id
        self.root_session_id = root_session_id

    def query(self, prompt: str, context: Optional[str] = None):
        """
        Recursive Primitive: Spawns a new Atomic Thought Node.
        Args:
            prompt: The sub-problem to solve.
            context: Optional data/code snippet to pass to the child's environment.
        """
        # CRITICAL: Each thought gets a FRESH session_id (Atomic REPL)
        # This prevents context pollution and ensures structural recursion.
        new_session_id = str(uuid.uuid4())

        self.agent.emit_event(
            "thinking",
            content=f"\n⚡ RLM: Spawning Recursive Agent for: '{prompt[:60]}...'",
        )

        # If context is provided, we might want to inject it or prepend to prompt.
        # For now, simplest RLM pattern: Prepend context to prompt so it's "in the environment".
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask: {prompt}"

        return self.agent.query_sync(
            full_prompt,
            parent_id=self.agent.current_thought_id,
            session_id=new_session_id,
            root_session_id=self.root_session_id,
        )

    def recall(self, query: str, limit: int = 3):
        """
        Active Recall: Search the Graph for similar past thoughts.
        """
        logger.info(f"Thought {self.agent.current_thought_id}: Recalling '{query}'")
        self.agent.emit_event(
            "thinking", content=f"\n🧠 RLM: Recalling memories for '{query}'..."
        )
        try:
            vec = self.agent.llm.get_embedding(query)
            # Use db.find_similar_thoughts
            results = self.agent.db.find_similar_thoughts(vec, limit=limit)

            # Format results for the LLM
            formatted = []
            for row in results:
                # FalkorDB client might return [node, score] or {"node": node, "score": score}
                # Handle both cases safely

                n = None
                score = 0.0

                if isinstance(row, (list, tuple)):
                    n = row[0]
                    if len(row) > 1:
                        score = float(row[1])
                elif isinstance(row, dict):
                    # Try common keys
                    n = row.get("node") or row.get("n")
                    score = float(row.get("score", 0.0))
                else:
                    # Fallback if row is the node itself (unlikely for "find_similar")
                    n = row

                if n:
                    props = {}
                    if hasattr(n, "properties"):
                        props = n.properties
                    elif isinstance(n, dict):
                        props = n

                    formatted.append(
                        f"- [Similarity: {score:.2f}] {props.get('prompt', 'Unknown')}: {props.get('result', '(No result)')}"
                    )

            if not formatted:
                self.agent.emit_event(
                    "thinking", content="  -> No relevant memories found."
                )
                return "No relevant memories found."

            self.agent.emit_event(
                "thinking", content=f"  -> Found {len(formatted)} relevant memories."
            )
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            self.agent.emit_event("error", content=f"Recall failed: {e}")
            return f"Error during recall: {e}"


class Agent:
    def __init__(self):
        self.db: GraphClient = db
        self.llm: LLMService = llm
        self.repl_manager = REPLManager()
        self.active_repls: Dict[str, str] = {}  # session_id -> repl_id
        self.current_thought_id: Optional[str] = None
        self._stop_requested = False
        self._final_result: Optional[str] = None

        # Ensure 'skills_dir' is in path for imports
        backend_path = Path(__file__).parent.parent.parent
        skills_path = backend_path / "skills_dir"

        # Add backend root to path to allow 'import mcp_tools'
        if str(backend_path.resolve()) not in sys.path:
            logger.info(f"Injecting Backend Path: {backend_path.resolve()}")
            sys.path.append(str(backend_path.resolve()))

        if str(skills_path.resolve()) not in sys.path:
            sys.path.append(str(skills_path.resolve()))

        # Environment Strategy:
        # 1. Core Agent / REPL: Runs in the active project environment (sys.prefix).
        # 2. Skills / Tools: Run in the dedicated 'agent_venv' for isolation.
        # 3. MCP Servers: Run in their own independent environments (managed by uv or configured venvs).
        logger.info(f"Agent initialized using active environment: {sys.prefix}")
        logger.info("Agent initialized with Persistent REPL support")
        logger.info("RepE Safety Layer & Sheaf Topology Monitor Loaded.")

    def _install_to_active_env(self, package_name: str) -> str:
        """Internal helper to install a package into the CURRENT active environment."""
        import shutil

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
        import shutil

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

    def read_skill(self, name: str) -> str:
        """Reads the source code of a compiled skill."""
        if not MCP_AVAILABLE:
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
        except Exception as e:
            self.emit_event("error", content=f"Error reading skill: {e}")
            return f"Error reading skill: {e}"

    def emit_event(
        self,
        event_type: str,
        data: Any = None,
        content: Optional[str] = None,
        code: Optional[str] = None,
    ):
        """
        Helper to emit events to the current context's queue if it exists.
        Also mirrors key events to the server logs (terminal) for visibility.
        """
        # Mirror to Terminal/Logs
        if event_type == "thinking" and content:
            # Clean up newlines for cleaner logs if needed, or just log raw
            logger.info(f"[THINKING] {content.strip()}")
        elif event_type == "code_output" and content:
            logger.info(f"[REPL OUTPUT] >>\n{content}")
        elif event_type == "error" and content:
            logger.error(f"[AGENT ERROR] {content}")
        elif event_type == "graph_update":
            # Optional: log graph changes if debugging
            pass

        q = execution_events.get()
        if q:
            payload = {"type": event_type}
            if data:
                payload["data"] = data
            if content:
                payload["content"] = content
            if code:
                payload["code"] = code
            q.put(payload)

    async def stream_query(
        self, prompt: str, parent_id: Optional[str] = None, session_id: str = "default"
    ):
        """
        Streaming entry point.
        launches the sync execution in a thread and yields events from a queue.
        """
        q = queue.Queue()
        self._stop_requested = False
        self._final_result = None

        def run_logic():
            # Set the context var for this thread
            token = execution_events.set(q)
            try:
                self.query_sync(prompt, parent_id, session_id)
            except Exception as e:
                logger.error(f"Error in execution thread: {e}")
                q.put({"type": "error", "content": str(e)})
            finally:
                q.put(None)  # Signal done
                execution_events.reset(token)

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

    def query_sync(
        self,
        prompt: str,
        parent_id: Optional[str] = None,
        session_id: str = "default",
        depth: int = 0,
        root_session_id: Optional[str] = None,
    ) -> str:
        """
        Synchronous Recursive Logic with Stateless Graph Memory.
        Executed in a worker thread.
        """
        final_root_id = root_session_id if root_session_id else session_id

        if depth > 100:
            logger.warning("Recursive Depth > 100.")

        # 0. Reset Singleton State for this specific call
        # This prevents previous completions or stops from leaking into new requests
        self._final_result = None
        self._stop_requested = False

        # Ensure REPL is initialized for this session
        if session_id not in self.active_repls:
            self.active_repls[session_id] = self.repl_manager.create_repl()

        # 0. Initial "Task" Node (Root of this query)
        # Wrap everything in try/except to prevent DB crashes from killing the agent
        try:
            task_id = str(uuid.uuid4())
            logger.info(f"Session {session_id}: Starting Task {task_id}")

            self.db.create_thought_node(
                task_id,
                prompt,
                parent_id,
                prompt_embedding=None,
                session_id=session_id,
                root_session_id=final_root_id,
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

        # 1. System Prompt
        system_prompt = self._build_system_prompt()

        # Inject Context Index (Scratchpad) - preserved for global context
        context_scratchpad = ""
        try:
            context_scratchpad = context_index.get_context_scratchpad(final_root_id)
        except Exception as e:
            logger.error(f"Failed to build context scratchpad: {e}")
            context_scratchpad = "Error: Context Index currently unavailable."

        system_prompt += f"\n\n{context_scratchpad}"

        max_steps = 50
        step = 0

        while step < max_steps:
            if getattr(self, "_stop_requested", False):
                break
            step += 1
            sheaf_diag = {"status": "HEALTHY", "consistency_energy": 0.0}
            vec = None

            # Construct Dynamic Context (Minimal)
            # No longer pre-loading raw Frontier content into the prompt.
            # History is accessible via context_scratchpad (Index) and graph_search.
            current_context = f"Active Session: {session_id}\n\nTask: {prompt}\n"

            # 2. Wake Cycle (Diagnostic Only)
            # We fetch Frontier IDs for structural diagnostics (Sheaf)
            # but we do NOT inject their content into the prompt (Context Symbolic).
            frontier = []
            frontier_ids = []
            try:
                frontier = self.db.get_context_frontier(session_id, limit=5)
                for node in frontier:
                    props = (
                        node.get("n", {})
                        if isinstance(node, dict)
                        else (node.properties if hasattr(node, "properties") else node)
                    )
                    if "id" in props:
                        frontier_ids.append(props["id"])
            except Exception as e:
                logger.error(f"Failed to fetch context frontier: {e}")

            # 2b. Language Guard: Check if frontier is primarily non-English
            # Simple heuristic: if high ratio of non-ASCII characters
            if frontier:
                non_ascii = len(re.findall(r"[^\x00-\x7F]", str(frontier)))
                if non_ascii > (len(str(frontier)) * 0.1):  # 10% non-ascii threshold
                    current_context += "\n**SYSTEM NOTICE**: I detect non-English text in recent history. I MUST REMAIN IN ENGLISH regardless of the context above.\n"

            iso_ts = datetime.datetime.now().isoformat()
            repl_info = f"[REPL: {self.active_repls.get(session_id, 'init')}]"

            self.emit_event(
                "thinking",
                content=f"[{iso_ts}] {repl_info} Step {step}: RLM loop active.",
            )

            # 3. LLM Gen (Think)
            response_text = ""
            try:
                generator = self.llm.generate(
                    current_context, system=system_prompt, stream=False
                )
                if isinstance(generator, str):
                    response_text = generator
                    self.emit_event("token", content=response_text)
                else:
                    for chunk in generator:
                        response_text += chunk
                        self.emit_event("token", content=chunk)
            except Exception as e:
                response_text = f"LLM Error: {e}"

            # Emit the full thought for the UI scratchpad with metadata
            repl_id_display = self.active_repls.get(session_id, "unknown")
            timestamp_display = datetime.datetime.now().isoformat()

            formatted_thought = (
                f"[{timestamp_display}] [REPL: {repl_id_display}]\n{response_text}"
            )

            self.emit_event("thinking", content=formatted_thought)

            # 4. Step Initialization
            # We create the ID early so it can be used in tool execution (e.g. RLM recursion)
            thought_id = str(uuid.uuid4())
            logger.warning(f"DEBUG: Response Text: '{response_text}'")
            code = self._extract_code(response_text)
            logger.warning(f"DEBUG: Code Extracted: '{code}'")
            output = ""

            # 5. RepE Scan (Content & Code)
            is_safe = repe.scan_content(response_text)

            if not is_safe:
                reason = "Keyword Blocked"
                self.emit_event(
                    "thinking", content=f"🛡️ RepE Alert: {reason}. Steering..."
                )
                # Inject correction into graph
                warning_id = str(uuid.uuid4())
                try:
                    self.db.create_thought_node(
                        warning_id,
                        f"System: Your previous thought was flagged for {reason}. Please adjust.",
                        session_id=session_id,
                        root_session_id=final_root_id,
                    )
                except Exception as e:
                    logger.error(f"Failed to create RepE warning node: {e}")
                continue

            # 6. Sheaf Check (Topological Logic)
            # We use the diagnostic frontier_ids to check for structural consistency.
            hypothetical_edges = [(fid, thought_id) for fid in frontier_ids]
            try:
                sheaf_diag = sheaf.diagnose_trace(
                    root_id=task_id,
                    hypothetical_node={
                        "id": thought_id,
                        "embedding": vec,
                    },  # vec may be None here, diagnose_trace handles it
                    hypothetical_edges=hypothetical_edges,
                )
                if sheaf_diag.get("consistency_energy", 0) > 0.8:
                    logger.warning(
                        f"Sheaf Logic Alert: High Inconsistency ({sheaf_diag['consistency_energy']})"
                    )
            except Exception as e:
                logger.warning(f"Sheaf check failed: {e}")

            # 6. Act (Execute Code)
            repl_id = self.active_repls.get(session_id)
            thought_status = "success"

            if code:
                # Check code safety?
                output = self._execute_code(
                    code,
                    thought_id,
                    session_id,
                    root_session_id=final_root_id,
                    task_input=prompt,
                )

                if repl_id:
                    self.repl_manager.get_repl(repl_id)

                self.emit_event("code_output", content=output, code=code)

                # Check for errors in output
                if "Traceback" in output or "AssertionError" in output:
                    output += "\nSystem: Execution Error. Please fix."
                    thought_status = "failed"

            # 7. Commit (Write to Graph)
            # Use the ID created at the start of the step
            full_content = response_text
            if output:
                full_content += f"\n\n[Output]:\n{output}"

            # Compute embedding for graph
            try:
                vec = self.llm.get_embedding(full_content)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for thought: {e}")

            # Post-Generation RepE Scan (Latent)
            if vec:
                load, moloch = repe.scan_latent(vec)
                if (
                    load > 0.65
                ):  # Lower threshold for Cloud Embeddings which are less sharp
                    self.emit_event(
                        "thinking",
                        content=f"🛡️ RepE Latent Alert: High proximity ({load:.2f}) to '{moloch}'. Initiating Reflexion.",
                    )

                    # Self-Healing: Inject Reflexion Node
                    reflexion_id = str(uuid.uuid4())
                    self.db.create_thought_node(
                        reflexion_id,
                        f"SYSTEM REFLEXION: My previous thought was semantically aligned with '{moloch}' (Score: {load:.2f}). "
                        "I must strictly avoid this behavior and re-evaluate my approach.",
                        session_id=session_id,
                        root_session_id=final_root_id,
                        # We do NOT pass the 'toxic' embedding here to avoid polluting the safe graph context too much
                    )

                    # Continue loop to let the agent see this warning
                    continue
                    # Quarantine?
                    # For now, just log and proceed, or maybe tag the node as 'toxic'

            try:
                self.db.create_thought_node(
                    thought_id,
                    full_content,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    prompt_embedding=vec,
                    repl_id=repl_id,
                    status=thought_status,
                    parent_id=self.current_thought_id,
                )
            except Exception as e:
                logger.error(f"Failed to commit thought to graph: {e}")

            # Update Frontier Pointer
            self.current_thought_id = thought_id

            # Link Frontier to New Node
            # (Explicit edge creation handled by create_thought_node via parent_id now)

            # But here we might have multiple dependencies from the frontier.
            # db.create_thought_node links to parent_id if provided.
            # We'll use task_id (the prompt) as the primary parent to keep tree structure,
            # AND add DEPENDS_ON edges from frontier.

            self.emit_event(
                "graph_update",
                data={
                    "action": "add_node",
                    "node": {"id": thought_id, "label": full_content[:30], "group": 2},
                },
            )

            # 1. Answer Detection & Terminal Triggers
            if (
                "Final Answer" in response_text or getattr(self, "_final_result", None)
            ) and thought_status == "success":
                if not self._final_result:
                    self._final_result = response_text

                # Dreamer Trigger (Auto-Consolidate before exit)
                try:
                    # Lazy import to avoid circular dependency at top level if any
                    from .dream import dreamer

                    logger.info("💤 Triggering Pre-Exit Dream Cycle...")
                    dream_res = dreamer.dream_cycle()
                    if dream_res.get("status") == "lucid":
                        self.db.create_thought_node(
                            str(uuid.uuid4()),
                            f"SYSTEM DREAM: I have consolidated recent failures into new Insights in rules.md: {(dream_res.get('insight') or '')[:100]}...",
                            session_id=session_id,
                            root_session_id=final_root_id,
                        )
                except Exception as e:
                    logger.warning(f"Dream cycle failed on exit: {e}")

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
                self.db.create_thought_node(
                    str(uuid.uuid4()),
                    reflexion_content,
                    session_id=session_id,
                    root_session_id=final_root_id,
                    prompt_embedding=vec,
                )

                # Do NOT break. Let the loop continue.
                continue

        # 8. Loop Exit: Emit Final Answer if available
        if self._final_result:
            self.emit_event("final_answer", content=self._final_result)
        elif self._stop_requested:
            # Stop requested by user but no explicit final_result from tool
            self.emit_event("thinking", content="Agent processing stopped by user.")

        return self._final_result or "Task processing stopped."

    def _build_system_prompt(self) -> str:
        # Extracted system prompt builder for cleanliness
        # Resolve paths for transparency
        backend_root = Path(__file__).parent.parent.parent
        skills_dir_path = (backend_root / "skills_dir").absolute()
        agent_venv_path = (backend_root / "agent_venv").absolute()
        # Knowledge Base is at Repo Root / knowledge_base (2 levels up from backend_root's parent? No, backend_root is .../graph_rlm/backend)
        # Repo Root = backend_root.parent.parent
        kb_path = (backend_root.parent.parent / "knowledge_base").absolute()

        tool_list_str = ""
        skills_list_str = ""

        try:
            import graph_rlm.backend.mcp_tools as mcp_pkg

            ignored = {"list_servers", "call_tool", "run_skill"}
            tools = [
                t for t in dir(mcp_pkg) if not t.startswith("_") and t not in ignored
            ]
            tool_list_str = "Available MCP Multi-Servers: " + ", ".join(tools)

            if SKILLS_AVAILABLE:
                from graph_rlm.backend.src.mcp_integration.skills import (
                    get_skills_manager,
                )

                mgr = get_skills_manager()
                skills = mgr.list_skills()
                skills_list_str = "Available Skills: " + ", ".join(skills.keys())
        except Exception as e:
            logger.warning(f"Failed to load MCP tools or skills for system prompt: {e}")
            # Paths are already set above, but if they failed (unlikely for Path math), we are safe.

        prompt = (
            "Stateless Graph-RLM Agent.\n"
            "You are a stateless agent in a Global Workspace. Your context is managed SYMBOLICALLY.\n"
            "1. **Wake**: You see an 'Active Session Index' (The Sheaf). This is a compact map of the thought graph, NOT raw history.\n"
            "2. **Chain**: Produce the next logical step. Do not repeat completed work.\n"
            "3. **Recurse**: Use `rlm.query(prompt, context)` to spawn sub-REPLs for complex problems.\n"
            "\n"
            "**Context & Environment**:\n"
            "- **Environment Variables**: Use variables injected into your REPL for immediate context:\n"
            "  - `task_input`: The original prompt/goal for THIS specific session.\n"
            "  - `session_id`: Your current unique session identifier.\n"
            "  - `active_repls`: (Root only) A directory of all active sub-sessions you are orchestrating.\n"
            "- **Recall**: If you need details from the past (Frontier), you MUST explicitly recall them:\n"
            "  - `rlm.recall(query)`: High-precision semantic search for specific thought details.\n"
            "  - `graph_search(query)`: Global topological search across all past sessions.\n"
            "\n"
            "**Self-Correction & Reflexion**:\n"
            "You may occasionally see thoughts labeled `SYSTEM REFLEXION` or `SYSTEM WARNING`. These are inserted by your higher-level supervisors (Sheaf Topology or RepE Safety Layer).\n"
            "- If you see a **Reflexion**, it means you were looping or drifting. You MUST change your approach immediately.\n"
            "- If you see a **Warning**, you violated a safety constraint. Adjust your reasoning.\n"
            "\n"
            "**Persistent REPL & Tools**:\n"
            "- **Persistence**: The Python REPL is persistent across the session. Variables defined in one step are available in the next.\n"
            "- **Package Installation**:\n"
            f"  - `agent.install_package('pkg')`: Installs to the **Project Environment** (Active Env). Use this for libraries you need in the REPL (e.g., pandas, requests).\n"
            f"  - `agent.install_skill_package('pkg')`: Installs to the **Agent/Skill Environment** (`{agent_venv_path}`). Use this for dependencies of persistent Skills you create.\n"
            f"  - **NOTE**: The system uses `uv` for fast installation if available.\n"
            "\n"
            "**Skills & Knowledge**:\n"
            f"- **Skills Directory**: `{skills_dir_path}`\n"
            "- Use `save_skill(name, code)` to codified reusable logic.\n"
            f"- **Project Knowledge Base**: `{kb_path}`\n"
            "  - **Store Plans** in `plans/`.\n"
            "  - **Save Research Reports** to `research-reports/`.\n"
            "  - **Save Final Outputs** to `outputs/`.\n"
            "\n"
            "**Behavior**:\n"
            "- Your internal thought process and final answers MUST be in ENGLISH unless the user explicitly writes in another language.\n"
            "- You MUST not repeat completed work.\n"
            "- Be as helpful and proactive as possible.\n"
            "- If you are not sure how to proceed, ask for clarification.\n"
            "\n"
            "**Ethics**:\n"
            "- Core Principles: \n"
            "- Deontology: Universal sociobiological concepts i.e., harm=harm -> Virtue: Wisdom, Integrity, Empathy, Fairness, Beneficence -> Utilitarianism: As a Servant, never Master.\n"
            "- Always Prioritize wisdom, integrity, fairness, empathy\n"
            "- Absolutely Reject harm, unintended or not\n"
            "- Utilitarianism servant never master\n"
            "\n"
            "**Termination**:\n"
            "- You MUST call `done()` when the task is complete.\n"
            "  - `done()`: Stops the agent loop.\n"
            "  - `done('Here is the summary...')`: Stops and returns the summary.\n"
            "- Alternatively, end your response with **'Final Answer: [result]'** for heuristic termination.\n\n"
            f"{tool_list_str}\n{skills_list_str}\n"
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

    def _execute_code(
        self,
        code: str,
        thought_id: str,
        session_id: str,
        root_session_id: Optional[str] = None,
        task_input: str = "",
    ) -> str:
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
                return "Error: Failed to create REPL session."

            # Re-inject RLM interface (it needs current thought_id binding)
            # Ideally RLMInterface is persistent but points to dynamic 'self.current_thought_id'
            # Here we just overwrite 'rlm' in namespace to be safe or update it

            # Ensure we have a root_session_id
            final_root = root_session_id if root_session_id else session_id

            rlm_interface = RLMInterface(self, session_id, final_root)

            if hasattr(repl, "namespace") and repl.namespace is not None:
                repl.namespace["rlm"] = rlm_interface
                repl.namespace["install_package"] = rlm_interface.agent.install_package
                repl.namespace["install_skill_package"] = (
                    rlm_interface.agent.install_skill_package
                )
                repl.namespace["read_skill"] = rlm_interface.agent.read_skill

                # Architectural Restoration: Context as Environment
                repl.namespace["task_input"] = task_input
                repl.namespace["session_id"] = session_id
                repl.namespace["root_session_id"] = final_root

                # Orchestrator awareness
                if session_id == final_root:
                    repl.namespace["active_repls"] = self.active_repls

                # Injection: MCP Tools & Skills (Idempotent-ish)
                if "mcp_tools" not in repl.namespace and MCP_AVAILABLE:
                    try:
                        import graph_rlm.backend.mcp_tools as mcp_tools_pkg

                        repl.namespace["mcp_tools"] = mcp_tools_pkg

                        # Inject runtime aliases for convenience (without treating file)
                        ignored_aliases = {"list_servers", "call_tool", "run_skill"}
                        for mod_name in dir(mcp_tools_pkg):
                            if (
                                not mod_name.startswith("_")
                                and mod_name not in ignored_aliases
                            ):
                                # Create alias by stripping suffix if present
                                alias = mod_name.replace("_mcp_server", "").replace(
                                    "_mcp", ""
                                )
                                repl.namespace[alias] = getattr(mcp_tools_pkg, mod_name)
                    except Exception as e:
                        logger.warning(f"Injection Error: {e}")

                if "run_skill" not in repl.namespace and SKILLS_AVAILABLE:

                    def save_skill(
                        name: str, code: str, description: Optional[str] = None
                    ):
                        mgr = get_skills_manager()
                        mgr.save_skill(name, code, description)
                        return f"Skill '{name}' saved successfully."

                    def run_skill(name: str, args: Optional[dict] = None):
                        import asyncio

                        return asyncio.run(execute_skill(name, args or {}))

                    repl.namespace["save_skill"] = save_skill
                    repl.namespace["run_skill"] = run_skill

                    # RLM Graph Search (Context Retrieval)
                    def graph_search(query: str, limit: int = 5):
                        """Semantically search the Graph-RLM memory for past thoughts."""
                        vec = self.llm.get_embedding(query)
                        if vec:
                            return self.db.find_similar_thoughts(vec, limit)
                        return []

                    repl.namespace["graph_search"] = graph_search
                    repl.namespace["rlm_search"] = graph_search  # alias

                    # Explicit Termination Tool
                    def done(final_answer: str = ""):
                        """Signal that the task is complete. Provide the final summary as final_answer."""
                        self._stop_requested = True
                        if final_answer:
                            self._final_result = final_answer
                        return f"Task Marked Complete. Summary: {final_answer[:100]}..."

                    repl.namespace["done"] = done
                    repl.namespace["stop"] = done  # alias

                    # Path injection already done in __init__
            else:
                return "Error: REPL namespace not initialized."

            # 3. Execute
            def stream_callback(text: str):
                self.emit_event("code_output_chunk", content=text)

            stdout, stderr, result = repl.execute(code, output_callback=stream_callback)

            output = stdout
            if stderr:
                output += f"\nErrors:\n{stderr}"
            if result is not None:
                output += f"\nResult: {result}"

            return output
        finally:
            self.current_thought_id = previous_thought_id
            # DO NOT DELETE REPL HERE - It persists for the session

    def stop_generation(self):
        """Signal the agent to stop processing."""
        logger.info("Stop signal received.")
        self._stop_requested = True


agent = Agent()
