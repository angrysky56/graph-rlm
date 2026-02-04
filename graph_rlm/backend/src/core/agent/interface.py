from typing import Any, Dict, List, Optional
import uuid

from ..logger import get_logger

logger = get_logger("graph_rlm.agent.interface")

class RLMInterface:
    """
    The object exposed to the REPL as 'rlm'.
    Allows recursive queries and memory recall.
    """

    def __init__(self, agent: "Agent", session_id: str, root_session_id: str):
        self.agent = agent
        self.session_id = session_id
        self.root_session_id = root_session_id

    def _record_tool_use(self, name: str):
        # FAST STOP CHECK: If the user hit stop, we must abort immediately.
        if getattr(self.agent, "_stop_requested", False) or (
            hasattr(self.agent, "_global_stop_event")
            and self.agent._global_stop_event.is_set()
        ):
            logger.warning(f"Stop signal detected. Aborting tool call: {name}")
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
        self._record_tool_use("rlm.history")
        try:
            return self.agent.db.get_session_trace(self.root_session_id)
        except Exception as e:
            logger.error(f"Failed to fetch history for rlm.history: {e}")
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
        from ..trace import trace_action
        self._record_tool_use("rlm.query")
        # CRITICAL: Each thought gets a FRESH session_id (Atomic REPL)
        new_session_id = session_id or str(uuid.uuid4())

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
        from ..trace import trace_action
        self._record_tool_use("rlm.recall")
        logger.info(f"Thought {self.agent.current_thought_id}: Recalling '{query}'")
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

        except Exception as e:
            logger.error(f"Recall Error: {e}")
            return f"Error during memory recall: {e}"

    async def search(self, query: str, limit: int = 10):
        """Topological search across the graph (alias for graph_search)."""
        self._record_tool_use("rlm.search")
        vec = await self.agent.llm.get_embedding(query)
        if vec:
            results = self.agent.db.find_similar_thoughts(vec, limit)
            if not results:
                return "No results found."
            return results
        return "Failed to generate embedding."

    async def ingest_document(self, path: str, domain: str = "general"):
        """Ingests a document and codifies its knowledge into Axioms (CAG)."""
        self._record_tool_use("rlm.ingest_document")
        from ..dream import dreamer

        res = await dreamer.ingest_document(path, domain)
        if res.get("status") == "success":
            return f"Successfully codified {len(res.get('codified_axioms', []))} axioms: {res.get('codified_axioms')}"
        return f"Ingestion failed: {res.get('message', 'Unknown error')}"

    async def save_skill(self, name: str, code: str, description: Optional[str] = None):
        """Saves a code snippet as a persistent skill."""
        from .core import is_skills_available
        self._record_tool_use("rlm.save_skill")
        if not is_skills_available():
            return "Skills system not available."
        from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager

        mgr = get_skills_manager()
        await mgr.save_skill(name, code, description)
        return f"Skill '{name}' saved successfully."

    async def run_skill(self, name: str = "", args: Optional[dict] = None, **kwargs):
        """Executes a registered skill."""
        from .core import is_skills_available
        self._record_tool_use("rlm.run_skill")
        # Handle 'title' as an alias for 'name' if the agent hallucinates it
        skill_name = name or kwargs.get("title") or ""
        if not skill_name:
            return "Error: No skill name or title provided."

        if not is_skills_available():
            return "Skills system not available."
        from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill

        return await execute_skill(skill_name, args or {})

    async def install_package(self, package_name: str):
        """Install a Python package into the agent's REPL environment."""
        self._record_tool_use("rlm.install_package")
        return self.agent.install_package(package_name)

    async def install_skill_package(self, package_name: str):
        """Install a package specifically for the AGENT skills (agent_venv) environment."""
        self._record_tool_use("rlm.install_skill_package")
        return self.agent.install_skill_package(package_name)

    async def done(self, final_answer: str = ""):
        """Signal that the task is complete."""
        self._record_tool_use("rlm.done")
        self.agent._stop_requested = True
        if final_answer:
            self.agent._final_result = final_answer

        # Log a summary to console, but return full confirmation
        summary = final_answer
        msg = f"Task Marked Complete. Summary: {summary}"
        print(f"\n[RLM] {msg}")

        # Emit final answer to UI
        self.agent.emit_event("answer", content=final_answer)

        return "Task completed successfully."

    async def stop(self, final_answer: str = ""):
        """Alias for done()."""
        self._record_tool_use("rlm.stop")
        return await self.done(final_answer)

    async def help(self):
        """Broad discovery of available commands within the 'rlm' namespace."""
        self._record_tool_use("rlm.help")

        # Core RLM Commands
        help_dict = {
            "query(prompt, context)": "Spawn a recursive child agent.",
            "recall(query, limit)": "Semantic search through memory.",
            "search(query, limit)": "Graph search (alias for recall).",
            "ingest_document(path, domain)": "CAG: Codify docs into Axioms.",
            "save_skill(name, code, desc)": "Persist a code block.",
            "run_skill(name, args)": "Run a saved code block.",
            "install_package(name)": "Install Python dependencies.",
        }

        # Dynamic MCP Tool Discovery
        try:
            import importlib
            import inspect
            from pathlib import Path

            # Resolve backend root to find mcp_tools
            # This file is in graph_rlm/backend/src/core/agent/
            backend_root = Path(__file__).parent.parent.parent.parent
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
                            help_dict[key] = doc.split("\n")[0]  # Brief doc
                except Exception as e:
                    logger.warning(
                        f"Failed to load module '{module_name}' for help: {e}"
                    )
                    continue
        except Exception as e:
            logger.warning(f"Error discovering MCP tools for help(): {e}")

        return help_dict

    def __repr__(self):
        return f"<RLMInterface [Session: {self.session_id[:8]}...] Type 'rlm.help()' for tools>"
