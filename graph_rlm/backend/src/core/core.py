"""
Core types and isolated REPL for Graph-RLM.
"""

import json
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

from .logger import get_logger
from .trace import trace_action

logger = get_logger("graph_rlm.core")


class KnowledgeBaseStructure:
    """Helper to provide semantic access to the KB and System folders."""

    def __init__(self, base_path: Optional[str] = None):
        self.root = Path(base_path).absolute() if base_path else Path.cwd()

        # Calculate Project and Backend roots
        # core.py is in graph_rlm/backend/src/core/
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.backend_root = self.project_root / "graph_rlm" / "backend"

        # Knowledge Folders
        self.plans_dir = self.root / "plans"
        self.reports_dir = self.root / "reports"
        self.outputs_dir = self.root / "outputs"

        # System Folders (Direct pointers to the code/config)
        self.axioms_dir = self.backend_root / "axioms_dir"
        self.skills_dir = self.backend_root / "skills"
        self.mcp_tools_dir = self.backend_root / "mcp_tools"
        self.src_dir = self.backend_root / "src"

    def ensure_exists(self):
        """Creates the KB structure if missing (only for data folders)."""
        for d in [self.plans_dir, self.reports_dir, self.outputs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # We do NOT ensure_exists for system folders as they should be pre-populated
        # or managed by specific tools.


class PythonREPL:
    """
    Stateful Python REPL for the agent, executing in an isolated AgentRuntime.
    """

    def __init__(self, repl_id: Optional[str] = None):
        self.repl_id = repl_id or "default"
        self.runtime = self._get_runtime()
        self.namespace = {}  # Local cache of injected variables (for preamble)

    def _get_runtime(self):
        """Lazy loader for AgentRuntime to avoid circular imports."""
        from ..mcp_integration.runtime import AgentRuntime

        project_root = Path(__file__).parent.parent.parent.parent.parent
        return AgentRuntime(project_root)

    def _serialize_namespace(self) -> str:
        """Converts local namespace to a python preamble."""
        preamble = []
        for k, v in self.namespace.items():
            if k.startswith("__"):
                continue
            if isinstance(v, (str, int, float, bool, list, dict)):
                try:
                    val_repr = json.dumps(v)
                    preamble.append(f"{k} = {val_repr}")
                except (TypeError, ValueError):
                    continue
        return "\n".join(preamble) + "\n"

    async def execute(
        self,
        code: str,
        output_callback=None,
        silent: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[str, str, Any, int]:
        """
        Execute Python code in the isolated AgentRuntime.
        """
        if not isinstance(code, str):
            return ("", "Error: Code must be a string", None, 1)

        if not code.strip():
            return ("", "", None, 0)

        if not silent:
            trace_action("REPL", "EXECUTE", result=code, tag="REPL")

        # 1. Serialize Namespace
        preamble = self._serialize_namespace()

        full_script = preamble + "\n" + code

        # 3. Execute in Subprocess
        try:
            # We don't have thought_id/session_id here usually, use defaults
            stdout, stderr, exec_result, exit_code = await self.runtime.execute(
                full_script,
                context={"thought_id": "core_repl", "session_id": "core_repl"},
                timeout=timeout,
            )

            is_failed = exit_code != 0
            if is_failed and not stderr:
                # If failed but no stderr, treat output as potentially containing error info
                stderr = stdout

            if output_callback:
                output_callback(stdout)

            if not silent:
                if stdout:
                    trace_action("REPL", "STDOUT", result=stdout, tag="REPL")
                if is_failed:
                    trace_action(
                        "REPL", "ERROR", result=stdout, tag="REPL", level="error"
                    )

            return (stdout, stderr, exec_result, exit_code)

        except (RuntimeError, AttributeError, ValueError):
            err = f"REPL Execution Error:\n{traceback.format_exc()}"
            logger.error("REPL Isolation Error: %s", err)
            return ("", err, None, 1)
