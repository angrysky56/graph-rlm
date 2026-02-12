"""
The RLM interface exposed to the agent REPL as 'rlm'.
"""

import importlib
import inspect
import json
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

# MCP Integration imports moved to top-level to resolve linting issues
from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill
from graph_rlm.backend.src.mcp_integration.skill_storage import (
    get_axioms_manager,
    get_skills_manager,
)

from .config import settings
from .core import KnowledgeBaseStructure
from .dream import dreamer
from .logger import get_logger
from .mcp_runtime import LazyMCPNamespace, is_skills_available
from .meta_agents import Fragment, meta_agents
from .state import agent_state
from .trace import trace_action

if TYPE_CHECKING:
    from .agent import Agent

logger = get_logger("graph_rlm.rlm_interface")


class RLMInterface:
    """
    The object exposed to the REPL as 'rlm'.
    Allows recursive queries and memory recall.
    """

    def __init__(self, agent_instance: "Agent", session_id: str, root_session_id: str):
        self.agent = agent_instance
        self.session_id = session_id
        self.root_session_id = root_session_id
        # Attach lazy MCP tools to the interpreter namespace
        self.mcp = LazyMCPNamespace(self)

    @property
    def kb(self) -> "KnowledgeBaseStructure":
        """Provides semantic access to the Project Knowledge Base folders."""

        return KnowledgeBaseStructure(settings.KNOWLEDGE_BASE_PATH)

    def get_mcp_config(self, server_name: str) -> dict:
        """Retrieves raw configuration for a specific MCP server from mcp_servers.json."""
        self.record_tool_use("rlm.get_mcp_config")
        # Agent has project_root attribute (defaulting to cwd)
        config_path = (
            Path(getattr(self.agent, "project_root", ".")) / "mcp_servers.json"
        )
        if not config_path.exists():
            # Fallback to current working directory
            config_path = Path.cwd() / "mcp_servers.json"

        if not config_path.exists():
            return {"error": "Config file mcp_servers.json not found in project root."}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            servers = data.get("mcpServers", {})
            if server_name in servers:
                return servers[server_name]
            return {"error": f"Server '{server_name}' not found in configuration."}
        except Exception as e:  # pylint: disable=broad-except
            return {"error": f"Failed to read MCP config: {e}"}

    def record_tool_use(self, name: str):
        """Records the use of a tool in the current session."""
        # Log to backend firehose (mirrors terminal)
        logger.info("[TOOL] %s", name)

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

    async def history(self, limit: int = 20):
        """Retrieve recent thought history for the current session."""
        self.record_tool_use("rlm.history")
        try:
            return self.agent.db.get_session_trace(self.root_session_id, limit=limit)
        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.error("Failed to fetch history for rlm.history: %s", e)
            return []

    async def query(
        self,
        prompt: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
        depth: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

        # ENFORCE MAX_RECURSION_DEPTH
        if new_depth > settings.MAX_RECURSION_DEPTH:
            msg = (
                f"Recursion Limit Reached (Depth: {new_depth}). "
                "Aborting sub-query to prevent infinite loop. "
                "Please resolve the task directly or terminate the chain."
            )
            logger.warning("[RLM] %s", msg)
            self.agent.emit_event(
                "thinking",
                content=f"⚠️ [RLM]: Recursion Limit Reached (Depth: {new_depth}). "
                "Aborting child agent.",
                is_internal=True,
            )
            return (
                f"Error: Recursion depth limit {settings.MAX_RECURSION_DEPTH} exceeded."
            )

        # LOOP DETECTION
        stack = []
        if current_state:
            stack = list(current_state.recursion_stack)

        # We normalize prompt to avoid simple variant bypasses
        normalized_prompt = prompt.strip().lower()
        if normalized_prompt in [p.lower() for p in stack]:
            msg = (
                f"Recursive Loop Detected! Prompt '{prompt}' was already "
                "seen in the current stack. Breaking loop."
            )
            logger.error("[RLM] %s", msg)
            self.agent.emit_event(
                "thinking",
                content=f"🚫 [RLM]: Recursive Loop Detected for prompt: '{prompt}'. "
                "Aborting child agent.",
                is_internal=True,
            )
            return (
                "Error: Recursive loop detected. You are repeating a task that "
                "already failed or is stuck. Break the loop and try a different approach."
            )

        stack.append(prompt)

        trace_action(
            "RLM",
            "RECURSE",
            result=f"Spawning child session (Depth: {new_depth}) for: {prompt}",
            tag="AGENT",
        )
        self.agent.emit_event(
            "thinking",
            content=f"\n⚡ RLM: Spawning Recursive Agent (Depth: {new_depth}) for: '{prompt}'",
            is_internal=True,
        )

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask: {prompt}"

        # --- META-AGENT: Inject Breaker Protocol ---
        # Sub-REPLs receive Breaker instructions to extract/summarize
        breaker_instructions = meta_agents.get_breaker_instructions(
            subtask=prompt, fragment_index=new_depth
        )
        full_prompt = f"{breaker_instructions}\n\n{full_prompt}"

        # stream_query is async, so we await it
        results = ""
        async for event in self.agent.stream_query(
            full_prompt,
            parent_id=self.agent.current_thought_id,
            session_id=new_session_id,
            depth=new_depth,
            turn_id=current_turn,
            root_session_id=self.root_session_id,
            recursion_stack=stack,
            metadata=metadata,
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
            logger.info("[RLM] %s", msg)
            raise RuntimeError(msg)

        logger.info("[RLM] Recursion completed with results: %s", results)

        # --- META-AGENT: Register Fragment for Synthesizer ---
        # Register this Sub-REPL's output as a Fragment for the parent to synthesize
        fragment = Fragment(
            session_id=new_session_id,
            summary=results[:500] if results else "",
            subtopics=[],  # Could be parsed from structured output
            confidence=0.7,  # Could be derived from oMCD evaluation
            raw_output=results,
        )
        meta_agents.register_fragment(self.root_session_id, fragment)

        return results

    async def recall(self, query: str, limit: int = 5):
        """
        Active Recall: Semantic search for past thoughts using core database methods.
        Now supports direct UUID lookups for precise grounding.
        """
        self.record_tool_use("rlm.recall")
        query = query.strip()
        logger.info("Thought %s: Recalling '%s'", self.agent.current_thought_id, query)
        self.agent.emit_event(
            "thinking", content=f"\n🧠 RLM: Recalling memories for '{query}'..."
        )
        try:
            # 1. Check if the query is a Direct Node ID (UUID or Partial)
            uuid_pattern = r"^[0-9a-f\-]{4,36}$"  # Allow partial hex/uuid strings
            is_hex_query = re.match(uuid_pattern, query.lower())

            if is_hex_query:
                # Direct Match or Prefix Search
                logger.info("Direct/Partial ID check in recall: %s", query)
                cypher = (
                    "MATCH (n:Thought) WHERE n.id STARTS WITH $id "
                    "RETURN n.id as id, n.prompt as prompt, n.result as result "
                    "LIMIT 5"
                )
                res = self.agent.db.query(cypher, {"id": query.lower()})
                if res:
                    formatted = []
                    for row in res:
                        formatted.append(
                            f"- [DIRECT RECALL] (ID: {row['id']}) "
                            f"Thought: {row['prompt']} -> Result: {row['result']}"
                        )
                    logger.info("[RLM] %d direct matches found for %s", len(res), query)
                    return "\n\n".join(formatted)

                # Fallback: Check if it's a repl_id
                repl_matches = self.agent.db.find_thought_by_repl_id(query, limit=5)
                if repl_matches:
                    formatted = []
                    for row in repl_matches:
                        formatted.append(
                            f"- [REPL RECALL] (ID: {row['id']}) "
                            f"Thought: {row['prompt']} -> Result: {row['result']}"
                        )
                    logger.info(
                        "[RLM] %d repl_id matches found for %s",
                        len(repl_matches),
                        query,
                    )
                    return "\n\n".join(formatted)

                logger.warning(
                    "Recall query looks like ID/REPL but nothing found: %s", query
                )
                # Fall through to semantic search

            # 2. Semantic Search
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
                # Results from find_similar_thoughts:
                # {"id": row[0], "prompt": row[1], "result": row[2], "score": row[3]}

                tid = row.get("id", "Unknown")
                prompt = row.get("prompt", "No prompt")
                result = row.get("result", "No result")
                score = float(row.get("score", 0.0))

                formatted.append(
                    f"- [Similarity: {score:.2f}] (ID: {tid}) "
                    f"Thought: {prompt} -> Result: {result}"
                )

            if not formatted:
                self.agent.emit_event(
                    "thinking", content="\n🧠 RLM: No matching memories found."
                )
                return "No semantically similar thoughts found in memory."

            output = f"Recall found {len(formatted)} relevant entries."
            logger.info("[RLM] %s", output)
            return (
                "\n\n".join(formatted)
                if formatted
                else "No relevant past thoughts found."
            )

        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
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

    async def topological_search(self, query: str, limit: int = 10):
        """Search across the graph using both structure and content embeddings."""
        self.record_tool_use("rlm.topological_search")
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

        res = await dreamer.ingest_document(path, domain)
        if res.get("status") == "success":
            return (
                f"Successfully codified {len(res.get('codified_axioms', []))} axioms: "
                f"{res.get('codified_axioms')}"
            )
        return f"Ingestion failed: {res.get('message', 'Unknown error')}"

    async def save_skill(self, name: str, code: str, description: Optional[str] = None):
        """Saves a code snippet as a persistent skill."""
        self.record_tool_use("rlm.save_skill")
        if not is_skills_available():
            return "Skills system not available."

        mgr = get_skills_manager()
        await mgr.save_skill(name, code, description)
        return f"Skill '{name}' saved successfully."

    async def save_instructional_skill(
        self,
        name: str,
        instructions: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Saves an instructional (folder-based) skill with SKILL.md."""
        self.record_tool_use("rlm.save_instructional_skill")
        if not is_skills_available():
            return "Skills system not available."

        mgr = get_skills_manager()
        return await mgr.save_instructional_skill(name, instructions, description, tags)

    async def read_skill(self, name: str) -> str:
        """Reads the source code or instructions of a skill."""
        self.record_tool_use("rlm.read_skill")
        return self.agent.read_skill(name)

    async def run_skill(self, name: str = "", args: Optional[dict] = None, **kwargs):
        """Executes a registered skill."""
        self.record_tool_use("rlm.run_skill")
        # Handle 'title' as an alias for 'name' if the agent hallucinates it
        skill_name = name or kwargs.get("title") or ""
        if not skill_name:
            return "Error: No skill name or title provided."

        if not is_skills_available():
            return "Skills system not available."

        return await execute_skill(skill_name, args or {})

    async def get_axiom(self, name: str):
        """Retrieves an axiom's code and metadata by name."""
        self.record_tool_use("rlm.get_axiom")

        mgr = get_axioms_manager()
        axiom = mgr.get_axiom(name)
        if not axiom:
            return f"Axiom '{name}' not found."
        return axiom

    async def recall_axioms(self, query: str, limit: int = 5):
        """High-precision semantic search for domain rules and axioms."""
        self.record_tool_use("rlm.recall_axioms")

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
        mgr = get_axioms_manager()
        axiom = mgr.get_axiom(name)
        if not axiom:
            return f"Axiom '{name}' not found."

        if axiom.get("axiom_type") == "validator":
            return (
                "Warning: This is a 'validator' axiom. It should be used via "
                "rlm.verify_axiom or by the Sheaf monitor. Running as a skill "
                "might not have the intended effect."
            )

        return await execute_skill(name, args or {})

    async def install_package(self, package_name: str):
        """Install a package into the project environment."""
        self.record_tool_use("rlm.install_package")
        return await self.agent.install_package(package_name)

    async def install_skill_package(self, package_name: str):
        """Install a package specifically for the AGENT skills context."""
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
            "get_kernel_results()": "Retrieve kernel computation data (sheaf_score, spectral_energy, h0_rank) from DB.",
            "generate_report_data(title)": "Generate complete report data from DB for template population.",
            "ingest_document(path, domain)": "CAG: Codify docs into Axioms.",
            "save_skill(name, code, desc)": "Persist a code block.",
            "save_instructional_skill(name, inst, desc)": (
                "Persist an instructional skill (SKILL.md)."
            ),
            "read_skill(name)": "Read source code or instructions of a skill.",
            "run_skill(name, args)": "Run a saved code block.",
            "get_axiom(name)": "Retrieve axiom code and metadata.",
            "recall_axioms(query, limit)": "Semantic search for domain rules.",
            "execute_axiom(name, args)": "Execute a solver or healing axiom.",
            "install_package(name)": "Install Python dependencies.",
            "get_mcp_config(server)": "View raw MCP server configuration.",
            "kb": "KnowledgeBase object with standard paths.",
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
                            help_dict[key] = doc.split("\n", maxsplit=1)[0]
                except (ImportError, AttributeError, ValueError, TypeError) as e:
                    logger.warning(
                        "Failed to load module '%s' for help: %s", module_name, e
                    )
                    continue
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Error discovering MCP tools for help(): %s", e)

        return help_dict

    async def query_sync(self, prompt: str, **kwargs):
        """Helper to run a full stream_query to completion and return the final text."""
        final_answer = ""
        async for event in self.agent.stream_query(prompt, **kwargs):
            if event["type"] == "done":
                final_answer = event["content"]
            elif event["type"] == "error":
                logger.error("Error in query_sync: %s", event.get("content"))
                raise RuntimeError(event.get("content", "Unknown error"))
        return final_answer

    async def get_kernel_results(self) -> Dict[str, Any]:
        """
        Retrieves kernel computation results (sheaf_score, spectral_energy, h0_rank)
        for the current session from the database.

        Returns:
            Dictionary with kernel computation data that can be used for report generation.
        """
        self.record_tool_use("rlm.get_kernel_results")
        try:
            # Use root_session_id to get all session data
            kernel_data = self.agent.db.get_kernel_results(self.root_session_id)
            logger.info(
                "Retrieved kernel results for session %s: %s",
                self.root_session_id[:8],
                kernel_data.get("status"),
            )
            return kernel_data
        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.error("Failed to retrieve kernel results: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "sheaf_scores": [],
                "spectral_energies": [],
                "h0_ranks": [],
                "avg_sheaf_score": 0.0,
                "avg_spectral_energy": 0.0,
                "avg_h0_rank": 0,
            }

    async def generate_report_data(self, title: str = None) -> Dict[str, Any]:
        """
        Generates comprehensive report data for the current session.
        Combines kernel results with session metadata for complete report generation.

        Args:
            title: Optional custom title for the report

        Returns:
            Complete report data dictionary ready for template population
        """
        self.record_tool_use("rlm.generate_report_data")
        try:
            report_data = self.agent.db.get_session_report_data(self.root_session_id)
            if title:
                report_data["paper_title"] = title
            logger.info(
                "Generated report data for session %s: %d thoughts, %s",
                self.root_session_id[:8],
                report_data.get("thought_count", 0),
                report_data.get("kernel_results", {}).get("status"),
            )
            return report_data
        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.error("Failed to generate report data: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "session_id": self.root_session_id,
                "paper_title": title or f"Error Report - {self.root_session_id[:8]}",
                "kernel_results": {
                    "status": "error",
                    "avg_sheaf_score": 0.0,
                    "avg_spectral_energy": 0.0,
                    "avg_h0_rank": 0,
                },
            }

    def __repr__(self):
        return (
            f"<RLMInterface [Session: {self.session_id[:8]}...] "
            "Type 'rlm.help()' for tools>"
        )
