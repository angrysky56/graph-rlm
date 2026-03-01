"""
Hardcoded guardrails and structural invariants for the Graph-RLM reasoning engine.
These validators are enforced at the database layer to prevent "Structural Orphanage"
and "Null-Context Cascades".
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import ErrorCode, ValidationError
from .logger import get_logger

logger = get_logger("graph_rlm.core.guardrails")


def extract_python_code(text: str) -> str:
    """
    Extracts all Python code blocks from an LLM response into a single sequence.

    Handles:
    1. Standard ```python ... ``` blocks.
    2. Unclosed blocks at the end of truncated streams.
    3. Joins multiple blocks with a separator to maintain execution order.

    @param text The raw LLM response text.
    @returns A unified string of Python code ready for AST parsing or REPL execution.
    """
    if not text:
        return ""

    # 1. Extraction of closed blocks
    blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if blocks:
        return "\n\n# --- RLM BLOCK SEPARATOR ---\n\n".join(blocks)

    # 2. Heuristic for unclosed block at end of response
    # (Common in streaming scenarios where the LLM hits a token limit)
    match_open = re.search(r"```python\s*(.*)", text, re.DOTALL)
    if match_open:
        raw_code = match_open.group(1)
        # Strip trailing "Final Answer" markers if the LLM hallucinated them inside the block
        clean_code = re.split(
            r"\*\*?Final Answer:?\*\*?", raw_code, flags=re.IGNORECASE
        )[0]
        logger.warning("Guardrails: Extracting tail of unclosed code block.")
        return clean_code.strip()

    return ""


class EmpiricalGuard:
    """
    Psychologically grounded code integrity checks.
    Ensures that generated code adheres to syntax, interface, and environment constraints.
    """

    @staticmethod
    def validate_syntax(code: str) -> None:
        """
        Performs static analysis of the code using Python's AST.

        @param code The Python source string to validate.
        @throws ValidationError if the code is syntactically invalid (GR-05).
        """
        if not code.strip():
            return

        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValidationError(
                message=f"Syntax Error in RLM code: {e.msg} at line {e.lineno}",
                error_code=ErrorCode.VALIDATION_CONSTRAINT_FAILED,
            ).with_field("line", e.lineno).with_constraint("AST_PARSE") from e

    @staticmethod
    def validate_rlm_signatures(code: str) -> None:
        """
        Verifies correct usage of the RLM control interface (e.g. async/await).

        @param code The code string.
        @throws ValidationError for signature or async mismatches (GR-06).
        """
        # Async Enforcement: rlm.done() is an async call in the Graph-RLM bridge
        if "rlm.done(" in code and "await rlm.done(" not in code:
            raise ValidationError(
                message="Graph-RLM Protocol Error: 'rlm.done()' must be awaited.",
                error_code=ErrorCode.VALIDATION_CONSTRAINT_FAILED,
            ).with_field("method", "rlm.done").with_constraint("ASYNC_AWAIT")

    @staticmethod
    def validate_mcp_imports(code: str) -> None:
        """
        Verifies that dynamically generated tool imports are valid.

        @param code The code string.
        @throws ValidationError if an MCP tool module is hallucinated (GR-07).
        """
        # Pattern matches: from graph_rlm.backend.mcp_tools.<tool> import <member>
        mcp_import_matches = re.finditer(
            r"from graph_rlm\.backend\.mcp_tools\.(\w+) import", code
        )

        # Resolve path relative to this file
        mcp_tools_dir = Path(__file__).resolve().parents[2] / "mcp_tools"

        if not mcp_tools_dir.exists():
            return

        for match in mcp_import_matches:
            module_name = match.group(1)
            module_file = mcp_tools_dir / f"{module_name}.py"

            if not module_file.exists():
                available = [
                    f.stem
                    for f in mcp_tools_dir.glob("*.py")
                    if not f.name.startswith("__")
                ]
                raise ValidationError(
                    message=f"Hallucinated MCP Module: '{module_name}' not found.",
                    error_code=ErrorCode.VALIDATION_FIELD_INVALID,
                ).with_field("import_path", module_name).with_context(
                    available_tools=available
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
) -> None:
    """
    Validates a node before it is committed to the graph.

    Checks:
    1. Structural Orphanage: Prevents detached nodes in non-root sessions.
    2. Context Continuity: Ensures root_session_id consistency.
    3. Empirical Integrity: Scans for failure signatures in reasoning/output.
    4. Introspective Code Integrity: AST and Proto-code validation.
    """

    # 1. Context Continuity (Graph Integrity)
    if parent_metadata:
        parent_rsid = parent_metadata.get("root_session_id")

        if parent_rsid and root_session_id != parent_rsid:
            raise ValidationError(
                message=f"Structural Orphanage: Child({root_session_id}) detached from Parent({parent_rsid})",
                error_code=ErrorCode.GRAPH_INVALID_STRUCTURE,
            ).with_field("root_session_id", root_session_id).with_context(
                parent_id=parent_id
            )

    # 2. Basic Sanitization
    if not prompt or not prompt.strip():
        raise ValidationError(
            message="Null-Context Cascade: Thought content cannot be empty.",
            error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
        ).with_field("prompt", "None")

    if not session_id:
        raise ValidationError(
            message="Ghost Thread: Missing session_id.",
            error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
        ).with_field("session_id", "None")

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
        logger.warning(
            "Guardrails: Empirical failure signature detected in node %s: %s",
            thought_id,
            ", ".join(found_errors),
        )

    # 4. Introspective Code Integrity (AST + Protocol)
    code = extract_python_code(prompt)
    if code:
        # These raise ValidationError if constraints are breached
        EmpiricalGuard.validate_syntax(code)
        EmpiricalGuard.validate_rlm_signatures(code)
        EmpiricalGuard.validate_mcp_imports(code)

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
) -> None:
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
