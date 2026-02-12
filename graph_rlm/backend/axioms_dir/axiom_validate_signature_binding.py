"""
This module implements the Explicit Signature Binding Advisor for MetaCognition.

It ensures that dynamic keys frequently used by high-level orchestrators
(e.g., 'task', 'goal') are correctly mapped to the strict tool signatures
expected by Scientific Orchestrator patterns ('skill_name', 'query').
"""

from typing import Any, Dict


def validate_signature_binding(invocation_context: Dict[str, Any]) -> bool:
    """
    Validates that the tool invocation contains the required keys.

    Args:
        invocation_context: A dictionary containing the arguments for a tool call.

    Returns:
        bool: True if 'skill_name' and 'query' exist, False otherwise.
    """
    required_keys = {"skill_name", "query"}
    provided_keys = set(invocation_context.keys())

    return required_keys.issubset(provided_keys)
