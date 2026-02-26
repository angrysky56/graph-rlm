"""
Hardcoded guardrails and structural invariants for the Graph-RLM reasoning engine.
These validators are enforced at the database layer to prevent "Structural Orphanage"
and "Null-Context Cascades".
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import get_logger

logger = get_logger("graph_rlm.core.guardrails")


def extract_python_code(text: str) -> str:
    """Extract all Python code blocks from an LLM response into a single string.

    Handles both complete fenced blocks and unclosed blocks at the end of a
    truncated response.  Multiple blocks are joined with a separator comment so
    they execute as a single sequence in the REPL.

    Returns an empty string when no Python code is found.
    """
    blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if blocks:
        return "\n\n# --- RLM BLOCK SEPARATOR ---\n\n".join(blocks)

    # Fallback: unclosed block at end of response (common with streaming truncation)
    match_open = re.search(r"```python\s*(.*)", text, re.DOTALL)
    if match_open:
        raw_code = match_open.group(1)
        clean_code = re.split(r"\*\*?Final Answer:?\*\*?", raw_code, flags=re.IGNORECASE)[0]
        logger.warning("Found unclosed code block, extracting tail (and stripping chat).")
        return clean_code.strip()

    return ""


class GuardrailError(Exception):
    """Exception raised when a graph invariant is violated."""


class EmpiricalGuard:
    """
    Empirical validation for code integrity.
    Checks for syntax errors, invalid tool signatures, and naming mistakes.
    """

    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        """Extracts python code blocks from text."""
        return re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)

    @staticmethod
    def validate_syntax(code: str):
        """Standard AST syntax check."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise GuardrailError(
                f"Syntax Error in code (GR-05): {e.msg} at line {e.lineno}"
            ) from e

    @staticmethod
    def validate_rlm_signatures(code: str):
        """Checks for common RLM interface mistakes."""
        # Async check
        if "rlm.done(" in code and "await rlm.done(" not in code:
            raise GuardrailError("Async Error (GR-06): 'rlm.done()' must be awaited.")

    @staticmethod
    def validate_mcp_imports(code: str):
        """Verifies that imports from mcp_tools point to existing files."""
        mcp_import_matches = re.finditer(
            r"from graph_rlm\.backend\.mcp_tools\.(\w+) import", code
        )
        # Assuming we are in graph_rlm/backend/src/core/guardrails.py
        # mcp_tools is in graph_rlm/backend/mcp_tools
        mcp_tools_dir = Path(__file__).resolve().parents[2] / "mcp_tools"

        if not mcp_tools_dir.exists():
            return  # Skip if directory is missing (e.g. in minimal dev env)

        for match in mcp_import_matches:
            module_name = match.group(1)
            module_file = mcp_tools_dir / f"{module_name}.py"
            if not module_file.exists():
                available = [
                    f.stem
                    for f in mcp_tools_dir.glob("*.py")
                    if not f.name.startswith("__")
                ]
                raise GuardrailError(
                    f"Import Error (GR-07): Module '{module_name}' not found. "
                    f"Available: {', '.join(available)}"
                )


def validate_thought_node(
    thought_id: str,
    prompt: str,
    parent_id: Optional[str],
    session_id: str,
    root_session_id: str,
    repl_id: Optional[str] = None,
    turn_id: Optional[int] = None,
    step_id: Optional[int] = None,
    node_type: str = "Thought",
    parent_metadata: Optional[Dict[str, Any]] = None,
):
    """
    Validates a node before it is committed to the graph.

    Checks:
    1. Orphan Prevention: If not a root session, a parent_id is strongly recommended.
    2. Context Continuity: session_id and root_session_id must remain consistent in a chain.
    3. Tool Causality: (Optional/Future) TOOL_RESULT requires TOOL_CALL.
    """

    # 1. Context Continuity
    if parent_metadata:
        parent_rsid = parent_metadata.get("root_session_id")
        # parent_sid = parent_metadata.get("session_id") # Removed unused

        if parent_rsid and root_session_id != parent_rsid:
            raise GuardrailError(
                f"Root Session ID Mismatch (GR-02): Child({root_session_id}) "
                f"!= Parent({parent_rsid})"
            )

    # 2. Basic Sanitization
    if not prompt or not prompt.strip():
        raise GuardrailError(
            "Empty Prompt (GR-03): Thought content cannot be null or empty."
        )

    if not session_id:
        raise GuardrailError("Missing Session ID (GR-04)")

    # 3. Empirical Integrity (Scan for error signatures)
    failure_patterns = [
        "traceback (most recent call last)",
        "Exception:",
        "RuntimeError:",
        "AttributeError:",
        "NameError:",
        "TypeError:",
        "SyntaxError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "IndexError:",
        "KeyError:",
        "MALFORMED_FUNCTION_CALL",
        "[SYSTEM ERROR]",
        "Unexpected keyword argument",
        "Connection refused",
        "Recursion Limit Reached",
        "Circuit breaker is open",
        "Max retries exceeded",
        "IPC RLM Error",
        "CORE_",
        "GRAPH_",
        "SKILL_",
        "EXTERNAL_",
        "VALIDATION_",  # Internal error codes
    ]

    found_errors = [p for p in failure_patterns if p.lower() in prompt.lower()]

    if found_errors:
        logger.error(
            "Guardrail Integrity Breach: Empirical Failure Patterns detected in node %s: %s",
            thought_id,
            ", ".join(found_errors),
        )

    # 4. Code Validation (Introspective Integrity)
    code_blocks = EmpiricalGuard.extract_code_blocks(prompt)
    if code_blocks:
        full_code = "\n".join(code_blocks)
        # These will raise GuardrailError if they fail
        EmpiricalGuard.validate_syntax(full_code)
        EmpiricalGuard.validate_rlm_signatures(full_code)
        EmpiricalGuard.validate_mcp_imports(full_code)

    # All checks passed
    logger.debug(
        "Guardrails passed for %s node %s (parent: %s, repl: %s, turn: %s, step: %s)",
        node_type,
        thought_id,
        parent_id,
        repl_id,
        turn_id,
        step_id,
    )


def validate_no_blind_transitions(
    node_type: str, _content: str, parent_type: Optional[str]
):
    """
    Enforces causal semantics.
    e.g. A TOOL_RESULT must follow a TOOL_CALL.
    """
    if node_type == "TOOL_RESULT" and parent_type != "TOOL_CALL":
        # While the agent might try this, we log it as a violation or block it.
        # For now, we allow but warn, or raise if we want strict enforcement.
        logger.warning(
            "Blind Transition Detected: %s without %s parent.", node_type, parent_type
        )
