import uuid
import re
import asyncio
import threading
import queue
import contextvars
from typing import Optional, Dict, Any, Generator
from .db import db, GraphClient
from .llm import llm, LLMService
from .logger import get_logger
from .manager import REPLManager
from pathlib import Path
import sys

# ... existing imports ...
from termcolor import colored

# MCP Integration
MCP_AVAILABLE = False # Default
try:
    from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager
    from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill
    MCP_AVAILABLE = True
except ImportError:
    pass


logger = get_logger("graph_rlm.agent")

# Context Variable to hold the event queue for the current execution thread/chain
execution_events: contextvars.ContextVar[Optional[queue.Queue]] = contextvars.ContextVar('execution_events', default=None)

class RLMInterface:
    """
    The object exposed to the REPL as 'rlm'.
    Allows recursive queries and memory recall.
    """
    def __init__(self, agent: 'Agent', session_id: str, root_session_id: str):
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

        # If context is provided, we might want to inject it or prepend to prompt.
        # For now, simplest RLM pattern: Prepend context to prompt so it's "in the environment".
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask: {prompt}"

        return self.agent.query_sync(
            full_prompt,
            parent_id=self.agent.current_thought_id,
            session_id=new_session_id,
            root_session_id=self.root_session_id
        )

    def recall(self, query: str, limit: int = 3):
        """
        Active Recall: Search the Graph for similar past thoughts.
        """
        logger.info(f"Thought {self.agent.current_thought_id}: Recalling '{query}'")
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
                    n = row.get('node') or row.get('n')
                    score = float(row.get('score', 0.0))
                else:
                    # Fallback if row is the node itself (unlikely for "find_similar")
                    n = row

                if n:
                    props = {}
                    if hasattr(n, 'properties'): props = n.properties
                    elif isinstance(n, dict): props = n

                    formatted.append(f"- [Similarity: {score:.2f}] {props.get('prompt', 'Unknown')}: {props.get('result', '(No result)')}")

            if not formatted:
                return "No relevant memories found."
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return f"Error during recall: {e}"

class Agent:
    def __init__(self):
        self.db: GraphClient = db
        self.llm: LLMService = llm
        self.repl_manager = REPLManager()
        self.active_repls: Dict[str, str] = {} # session_id -> repl_id
        self.current_thought_id: Optional[str] = None
        self._stop_requested = False

        # Ensure 'skills' is in path for imports
        # Ensure 'skills_dir' is in path for imports
        backend_path = Path(__file__).parent.parent.parent
        skills_path = backend_path / "skills_dir"

        # Add backend root to path to allow 'import mcp_tools'
        if str(backend_path.resolve()) not in sys.path:
            logger.info(f"Injecting Backend Path: {backend_path.resolve()}")
            sys.path.append(str(backend_path.resolve()))

        if str(skills_path.resolve()) not in sys.path:
            sys.path.append(str(skills_path.resolve()))

        # Inject Agent Venv site-packages
        self.agent_venv_path = backend_path / "agent_venv"
        self.skills_venv_path = backend_path / "skills_venv"
        if self.agent_venv_path.exists():
            # Find site-packages (e.g., lib/python3.x/site-packages)
            # This is a robust way to find it across python versions
            for site_packages in self.agent_venv_path.glob("lib/python*/site-packages"):
                if str(site_packages.resolve()) not in sys.path:
                    logger.info(f"Injecting Agent Venv: {site_packages}")
                    sys.path.append(str(site_packages.resolve()))

        logger.info("Agent initialized with Persistent REPL support")

    def _install_to_venv(self, venv_path: Path, package_name: str) -> str:
        """Internal helper to install a package into a specific venv."""
        import subprocess
        import shutil
        import os

        logger.info(f"Agent requesting installation of package: {package_name} into {venv_path.name}")

        python_exe = venv_path / "bin" / "python"
        if not python_exe.exists():
            return f"Error: Venv not found at {python_exe}. Ensure it is created."

        try:
            cmd = [str(python_exe), "-m", "pip", "install", package_name]
            if shutil.which("uv"):
                cmd = ["uv", "pip", "install", "--python", str(python_exe), package_name]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name} into {venv_path.name}")
                return f"Successfully installed {package_name} into {venv_path.name}\n{result.stdout}"
            else:
                logger.error(f"Failed to install {package_name} into {venv_path.name}: {result.stderr}")
                return f"Failed to install {package_name} into {venv_path.name}\nError: {result.stderr}"
        except Exception as e:
            logger.error(f"Installation error in {venv_path.name}: {e}")
            return f"Installation error in {venv_path.name}: {e}"

    def install_package(self, package_name: str) -> str:
        """Installs a package into the agent_venv (REPL environment)."""
        return self._install_to_venv(self.agent_venv_path, package_name)

    def install_skill_package(self, package_name: str) -> str:
        """Installs a package into the skills_venv (Background skills)."""
        return self._install_to_venv(self.skills_venv_path, package_name)

    def read_skill(self, name: str) -> str:
        """Reads the source code of a compiled skill."""
        if not MCP_AVAILABLE:
            return "Error: MCP/Skills system not available."
        try:
            from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager
            mgr = get_skills_manager()
            skill = mgr.get_skill(name)
            if not skill:
                return f"Error: Skill '{name}' not found."
            return skill["code"]
        except Exception as e:
            return f"Error reading skill: {e}"

    def emit_event(self, event_type: str, data: Any = None, content: Optional[str] = None, code: Optional[str] = None):
        """
        Helper to emit events to the current context's queue if it exists.
        """
        q = execution_events.get()
        if q:
            payload = {"type": event_type}
            if data: payload["data"] = data
            if content: payload["content"] = content
            if code: payload["code"] = code
            q.put(payload)

    async def stream_query(self, prompt: str, parent_id: Optional[str] = None, session_id: str = "default"):
        """
        Streaming entry point.
        launches the sync execution in a thread and yields events from a queue.
        """
        q = queue.Queue()
        self._stop_requested = False

        def run_logic():
            # Set the context var for this thread
            token = execution_events.set(q)
            try:
                self.query_sync(prompt, parent_id, session_id)
            except Exception as e:
                logger.error(f"Error in execution thread: {e}")
                q.put({"type": "error", "content": str(e)})
            finally:
                q.put(None) # Signal done
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

    def query_sync(self, prompt: str, parent_id: Optional[str] = None, session_id: str = "default", depth: int = 0, root_session_id: Optional[str] = None) -> str:
        """
        Synchronous Recursive Logic with Self-Healing Loop.
        Executed in a worker thread.
        """
        # If root_session_id is missing, default to session_id (implies we are the root)
        final_root_id = root_session_id if root_session_id else session_id

        # if depth > 10:
        #      return "Error: Maximum recursion depth (10) reached."
        if depth > 100:
             # Just a very high sanity check to prevent stack overflow crashing Python
             logger.warning("Recursive Depth > 100. Continuing but be careful.")

        thought_id = str(uuid.uuid4())
        logger.info(f"Thought {thought_id} (Session {session_id}, Root {final_root_id}, Depth {depth}): Processing.")

        # 0. Embedding
        try:
            prompt_vec = self.llm.get_embedding(prompt)
        except:
            prompt_vec = None

        # 1. Graph Node
        self.db.create_thought_node(thought_id, prompt, parent_id, prompt_vec, session_id=session_id, root_session_id=final_root_id)

        self.emit_event("graph_update", data={
            "action": "add_node",
            "node": {
                "id": thought_id,
                "label": prompt[:40] + "..." if len(prompt) > 40 else prompt,
                "group": 2 if parent_id else 1,
                "val": 10 if not parent_id else 5,
                "status": "processing"
            }
        })

        if parent_id:
            self.emit_event("graph_update", data={
                "action": "add_link",
                "link": {"source": parent_id, "target": thought_id}
            })

        # 2. System Prompt
        tool_list_str = ""
        skills_list_str = ""
        try:
            import graph_rlm.backend.mcp_tools as mcp_pkg
            # List all modules in mcp_tools package that are not internal/utility
            ignored = {'list_servers', 'call_tool', 'run_skill'}
            tools = [t for t in dir(mcp_pkg) if not t.startswith('_') and t not in ignored]
            tool_list_str = "Available MCP Multi-Servers: " + ", ".join(tools)

            # Fetch Skills
            if MCP_AVAILABLE:
                from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager
                mgr = get_skills_manager()
                skills = mgr.list_skills()
                skills_list_str = "Available Compiled Skills: " + ", ".join(skills.keys())
        except Exception as e:
            tool_list_str = f"Tools Error: {e}"

        system_prompt = (
            "You are a Self-Healing Recursive Language Model (RLM). "
            "You operate in a persistent Python REPL, but you should treat each large step as a new node.\n"
            "## Recursive Philosophy (CRITICAL)\n"
            "You are not a chatbot; you are a Graph of Thoughts. Do not solve complex problems in one linear loop.\n"
            "1. **Decompose**: Break the problem into sub-units.\n"
            "2. **Recurse**: Call `rlm.query(sub_prompt)` immediately. This spawns a **fresh, atomic REPL** for that thought.\n"
            "3. **Synthesize**: Use the return value of `rlm.query` to build your answer.\n"
            "4. **Atomic**: Do not pollute your current REPL with unrelated tasks. Spawn a child.\n"
            "\n"
            "## Core Loop\n"
            "1. **Think**: Plan your next action. Is this task complex? If yes, `rlm.query()`.\n"
            "2. **Act**: Execute code with ```python ... ```.\n"
            "3. **Observe**: Review output. If error, **Reflect** and retry.\n"
            "\n"
            "## Environment Access\n"
            "### MCP Multi-Servers\n"
            "Access specialized servers via `mcp_tools.<server_name>`. \n"
            f"{tool_list_str}\n"
            "Common aliases are pre-injected: `arxiv`, `google_search`, etc.\n"
            "\n"
            "### High-Level Skills\n"
            f"{skills_list_str}\n"
            "Execute a skill via `run_skill(\"skill_name\", args_dict)`. You SHOULD prefer skills for complex repeatable tasks.\n"
            "You have full agency over your skill library. You can read, edit, and create skills.\n"
            "\n"
            "### Helper Functions\n"
            "- `read_skill(name)`: Returns the source code of a compiled skill for inspection or refactoring.\n"
            "- `install_package(name)`: Installs a pip package into your REPL environment (agent_venv).\n"
            "- `install_skill_package(name)`: Installs a pip package into the background skill environment (skills_venv). Use this if `run_skill` fails with `ModuleNotFoundError`.\n"
            "- `save_skill(name, code, description)`: Persists a successful snippet as a reusable skill in your library.\n"
            "- `rlm.recall(query)`: Search your graph memory for relevant past thoughts.\n"
            "\n"
            "### Skill Development Workflow\n"
            "1. **Inspect**: Use `read_skill(name)` to understand a skill's logic before running or if it fails.\n"
            "2. **Iterate**: Test logic in the REPL. Use `install_package` for dependencies.\n"
            "3. **Persist**: Use `save_skill` to update an existing skill or create a new one. Skills are stored physically in `graph_rlm/backend/skills_dir/`.\n"
            "NOTE: Check tool functions with `dir()` or `help()` if unsure. Do not hallucinate methods.\n"
        )

        current_context = prompt
        final_response = ""
        max_steps = 100
        step = 0

        while step < max_steps:
            # Check for cancellation
            try:
                 # Check if the queue loop in stream_query has stopped listening or flagged stop
                 # We can't easily check the queue consumer status from here without a flag.
                 # But we can check a threading Event if passed.
                 # Let's assume we can check a context variable or just the queue's health if needed.
                 # Better approach: stream_query sets a stop_event.
                 if getattr(self, '_stop_requested', False):
                     logger.info("Execution stopped by user.")
                     self.emit_event("error", content="Stopped by user")
                     break
            except:
                pass

            step += 1
            self.emit_event("thinking", content=f"Step {step}/{max_steps}..." if step > 1 else f"Analyzing: {prompt[:5000]}...")

            # 3. LLM Gen
            try:
                # If this is step > 1, current_context contains previous thoughts + code output
                response_text = self.llm.generate(current_context, system=system_prompt)
            except Exception as e:
                response_text = f"LLM Error: {e}"

            self.emit_event("token", content=response_text)

            # 4. Code Exec
            executed_result = ""
            code_block_found = "```python" in response_text

            if code_block_found:
                code = self._extract_code(response_text)
                self.emit_event("thinking", content="\nExecuting Code...")
                executed_result = self._execute_code(code, thought_id, session_id, root_session_id=final_root_id)
                self.emit_event("code_output", content=executed_result, code=code)

                # Append result to context for next step
                step_record = f"\n\n--- Step {step} ---\nThought: {response_text}\n[REPL Output]:\n{executed_result}\n"
                current_context += step_record

            # 5. Graph Update (Intermediate)
            try:
                self.db.update_thought_result(thought_id, response_text, embedding=None)
            except:
                pass

            # 6. Diagnosis / Self-Healing
            error_keywords = ["Traceback", "Error:", "Exception:", "AttributeError"]
            has_error = any(k in executed_result for k in error_keywords)

            critique = ""
            diagnosis = "HEALTHY"

            if has_error:
                diagnosis = "ERROR"
                critique = f"Code execution failed. Analyze the Traceback in [REPL Output] and fix the code."

            # Sheaf Check (Logic)
            try:
                from .sheaf import sheaf
                sheaf_diag = sheaf.diagnose_trace(thought_id)
                if sheaf_diag["status"] != "HEALTHY":
                    diagnosis = "LOGICAL_KNOT"
                    critique += f" {sheaf_diag['critique']}"
            except:
                pass

            # DECISION: Continue or Stop?

            if diagnosis != "HEALTHY":
                # Must fix error on next step
                self.emit_event("thinking", content=f"\n[Self-Healing] {critique}. Retrying...")
                current_context += f"\n\nSYSTEM ALERT: {critique}\nFix the logic and try again."
                continue

            # If no error, check if we are done
            # Heuristic: If model says "Final Answer:" or didn't run code and just talked?
            # Or if it ran code, we usually assume it wants to see the result.
            # If it didn't run code, it might be answering.

            if not code_block_found:
                # Pure text response -> likely the answer
                final_response = response_text
                break

            # If code ran successfully, loop to allow next thought (e.g. "Now I will...")
            # We append output to context (already done above) and continue.

        if not final_response and step >= max_steps:
             final_response = "Error: Maximum reasoning steps reached without final answer."

        self.emit_event("graph_update", data={"action": "update_node", "node": {"id": thought_id, "status": "completed", "group": 3}})
        return final_response

    def _extract_code(self, text: str) -> str:
        # Try finding a complete block first
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # Fallback: check for unclosed block at the end (common with truncation)
        match_open = re.search(r"```python\s*(.*)", text, re.DOTALL)
        if match_open:
            logger.warning("Found unclosed code block, extracting tail.")
            return match_open.group(1)

        return ""

    def _execute_code(self, code: str, thought_id: str, session_id: str, root_session_id: Optional[str] = None) -> str:
        # 1. Get or Create REPL for this session
        if session_id in self.active_repls:
            repl_id = self.active_repls[session_id]
            # Verify existence
            if not self.repl_manager.get_repl(repl_id):
                # Stale ID, recreate
                repl_id = self.repl_manager.create_repl()
                self.active_repls[session_id] = repl_id
        else:
            repl_id = self.repl_manager.create_repl()
            self.active_repls[session_id] = repl_id

        repl = self.repl_manager.get_repl(repl_id)

        # 2. Update Context
        previous_thought_id = self.current_thought_id
        self.current_thought_id = thought_id

        try:
            if repl is None:
                 return "Error: Failed to create REPL session."

            # Re-inject RLM interface (it needs current thought_id binding)
            # Ideally RLMInterface is persistent but points to dynamic 'agent.current_thought'
            # Here we just overwrite 'rlm' in namespace to be safe or update it

            # Ensure we have a root_session_id
            final_root = root_session_id if root_session_id else session_id

            rlm_interface = RLMInterface(self, session_id, final_root)

            if hasattr(repl, 'namespace') and repl.namespace is not None:
                repl.namespace['rlm'] = rlm_interface
                repl.namespace['install_package'] = rlm_interface.agent.install_package
                repl.namespace['install_skill_package'] = rlm_interface.agent.install_skill_package
                repl.namespace['read_skill'] = rlm_interface.agent.read_skill

                # Injection: MCP Tools & Skills (Idempotent-ish)
                if 'mcp_tools' not in repl.namespace and MCP_AVAILABLE:
                    try:
                        import graph_rlm.backend.mcp_tools as mcp_tools_pkg
                        repl.namespace['mcp_tools'] = mcp_tools_pkg

                        # Inject runtime aliases for convenience (without treating file)
                        ignored_aliases = {'list_servers', 'call_tool', 'run_skill'}
                        for mod_name in dir(mcp_tools_pkg):
                            if not mod_name.startswith('_') and mod_name not in ignored_aliases:
                                # Create alias by stripping suffix if present
                                alias = mod_name.replace('_mcp_server', '').replace('_mcp', '')
                                repl.namespace[alias] = getattr(mcp_tools_pkg, mod_name)
                    except Exception as e:
                        logger.warning(f"Injection Error: {e}")

                if 'run_skill' not in repl.namespace and MCP_AVAILABLE:
                    def save_skill(name: str, code: str, description: Optional[str] = None):
                        mgr = get_skills_manager()
                        mgr.save_skill(name, code, description)
                        return f"Skill '{name}' saved successfully."

                    def run_skill(name: str, args: Optional[dict] = None):
                         import asyncio
                         return asyncio.run(execute_skill(name, args or {}))

                    repl.namespace['save_skill'] = save_skill
                    repl.namespace['run_skill'] = run_skill

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
