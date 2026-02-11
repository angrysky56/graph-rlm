"""
The Python REPL implementation that maintains state between executions.
Ported from local-repl-mcp with minimal changes.
"""

import json
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from graph_rlm.backend.src.mcp_integration.runtime import AgentRuntime

from .logger import get_logger
from .trace import trace_action

logger = get_logger("graph_rlm.repl.core")


class KnowledgeBaseStructure:
    """Helper to provide semantic access to the KB folders."""

    def __init__(self, base_path: str):
        self.root = Path(base_path).absolute()
        self.plans_dir = self.root / "plans"
        self.reports_dir = self.root / "research-reports"
        self.outputs_dir = self.root / "outputs"
        self.axioms_dir = self.root / "axioms"

        # Ensure directories exist
        for d in [self.plans_dir, self.reports_dir, self.outputs_dir, self.axioms_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"<KnowledgeBase root='{self.root}'>"


class PythonREPL:
    """
    A stateful Python REPL implementation using isolated AgentRuntime processes.
    Replaces the legacy 'exec' based implementation for security.
    """

    def __init__(self, repl_id: Optional[str] = None):
        self.repl_id = repl_id or str(uuid.uuid4())
        logger.debug("Initializing Isolated REPL %s", self.repl_id)

        # Calculate project root (assumes backend/src/core/core.py structure)
        current_file = Path(__file__).resolve()
        # graph-rlm/graph_rlm/backend/src/core/core.py -> graph-rlm/
        self.project_root = current_file.parent.parent.parent.parent.parent

        self.runtime = AgentRuntime(self.project_root)

        # We maintain a local namespace dict primarily to track what *would* be
        # injected. We'll verify what makes sense to serialize.
        self.namespace: Dict[str, Any] = {}

    def _serialize_namespace(self) -> str:
        """
        Generates Python code to reconstruct the namespace in the isolated subprocess.
        Handles:
        1. UniversalAsyncMock (for Dreamer/Sheaf)
        2. Primitives (int, str, bool, dict, list)
        3. KnowledgeBaseStructure (re-instantiated)
        """
        preamble = []

        # 1. Helper classes definition if needed
        has_mock = any(
            "Mock" in str(type(v)) or "UniversalAsyncMock" in str(type(v))
            for v in self.namespace.values()
        )

        if has_mock:
            preamble.append("""
class UniversalAsyncMock:
    def __getattr__(self, name):
        return UniversalAsyncMock()
    def __call__(self, *args, **kwargs):
        async def _dummy():
            return "MOCK_RESULT"
        return _dummy()
    def __repr__(self):
        return "<UniversalAsyncMock>"
    def __await__(self):
        async def _coro(): return "MOCK_RESULT"
        return _coro().__await__()
""")

        # 2. Serialize variables
        for name, value in self.namespace.items():
            if name.startswith("__"):
                continue

            # Handle Mocks
            if "UniversalAsyncMock" in str(type(value)) or "MockREPLInterface" in str(
                type(value)
            ):
                preamble.append(f"{name} = UniversalAsyncMock()")
                continue

            # Handle KnowledgeBaseStructure
            if isinstance(value, KnowledgeBaseStructure):
                path_str = str(value.root)
                preamble.append(
                    "from graph_rlm.backend.src.core.core import KnowledgeBaseStructure"
                )
                preamble.append(f"{name} = KnowledgeBaseStructure('{path_str}')")
                continue

            # Handle Primitives
            try:
                # Basic JSON serialization for primitives
                json_val = json.dumps(value)
                preamble.append(f"{name} = {json_val}")
            except (TypeError, OverflowError):
                # Skip complex objects we can't serialize
                # logger.warning(f"Skipping serialization of complex object: {name} ({type(value)})")
                pass

        return "\n".join(preamble) + "\n"

    async def execute(
        self, code: str, output_callback=None, silent: bool = False
    ) -> Tuple[str, str, Any, bool]:
        """
        Execute Python code in the isolated AgentRuntime.
        """
        if not isinstance(code, str):
            return ("", "Error: Code must be a string", None, True)

        if not code.strip():
            return ("", "", None, False)

        if not silent:
            trace_action("REPL", "EXECUTE", result=code, tag="REPL")

        # 1. Serialize Namespace
        preamble = self._serialize_namespace()

        # 2. Add explicit print for the last expression if it's an expression
        # AgentRuntime wraps user code in _main(), so top-level await is supported.
        # But AgentRuntime doesn't automatically return the last expression value unless printed or returned.
        # For compatibility with legacy REPL, we rely on stdout capture mostly.
        # However, legacy returned (stdout, stderr, result, exception_occurred)
        # We can't easily get the python object 'result' back from subprocess across IPC boundry
        # without pickling, which is unsafe.
        # For now, 'result' will be None or a string representation if we parse stdout.
        # The legacy tests check for strings in stdout, so that matches.

        full_script = preamble + "\n" + code

        # 3. Execute in Subprocess
        try:
            # We don't have thought_id/session_id here usually, use defaults
            # AgentRuntime.execute return signature: (output, is_failed)
            # Wait, verify signature:
            # async def execute(self, code: str, context: Dict[str, str]) -> Tuple[str, bool]:

            output, stderr_output, exit_code = await self.runtime.execute(
                full_script,
                context={"thought_id": "core_repl", "session_id": "core_repl"},
            )

            is_failed = exit_code != 0

            # 4. Parse Output
            stdout = output
            stderr = stderr_output
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
                        "REPL", "ERROR", result=output, tag="REPL", level="error"
                    )

            return (stdout, stderr, None, is_failed)

        except Exception:
            err = traceback.format_exc()
            logger.error("REPL Isolation Error: %s", err)
            return ("", err, None, True)
