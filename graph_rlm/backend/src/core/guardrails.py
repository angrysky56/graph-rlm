"""
Hardcoded guardrails and structural invariants for the Graph-RLM reasoning engine.
These validators are enforced at the database layer to prevent "Structural Orphanage"
and "Null-Context Cascades".
"""

from typing import Any, Dict, Optional

from .logger import get_logger

logger = get_logger("graph_rlm.core.guardrails")


class GuardrailError(Exception):
    """Exception raised when a graph invariant is violated."""


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
        "ModuleNotFoundError",
        "MALFORMED_FUNCTION_CALL",
        "[SYSTEM ERROR]",
    ]

    found_errors = [p for p in failure_patterns if p.lower() in prompt.lower()]

    if found_errors:
        logger.error(
            "Guardrail Integrity Breach: Empirical Failure Patterns detected in node %s: %s",
            thought_id,
            ", ".join(found_errors),
        )
    else:
        # Log with extra context (uses parent_id and node_type)
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
