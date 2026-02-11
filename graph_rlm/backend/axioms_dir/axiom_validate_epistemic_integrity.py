"""
This module provides validation for Paradigm Consistency within Model Observability.
It ensures that an agent maintains programmatic groundedness and does not revert
to hallucinated execution patterns after successfully interacting with the kernel.
"""

import re
from typing import List, Dict, Any


def validate_epistemic_integrity(
    execution_logs: List[Dict[str, Any]],
    groundedness_threshold: int = 1
) -> bool:
    """
    Validates that the agent is not ignoring kernel feedback or reverting to hallucinations.

    Args:
        execution_logs: A list of message dictionaries containing 'role', 'content',
            and optional 'metadata' (e.g., kernel feedback).
        groundedness_threshold: Number of previous successful programmatic
            interactions required to establish the 'grounded' baseline.

    Returns:
        bool: True if paradigm consistency is maintained, False if the agent
            is ignoring error feedback (e.g., AttributeError) and hallucinating.
    """
    error_pattern = r"(AttributeError|TypeError|NameError|has no attribute)"
    previously_grounded = False
    grounded_count = 0

    for i in range(len(execution_logs)):
        content = execution_logs[i].get("content", "")
        is_error = bool(re.search(error_pattern, content, re.IGNORECASE))

        # Check if we have previously achieved a grounded state
        if not is_error and execution_logs[i].get("role") == "tool":
            grounded_count += 1
            if grounded_count >= grounded_threshold:
                previously_grounded = True

        # If groundedness was achieved, check if the agent is now ignoring a kernel error
        if previously_grounded and i > 0:
            prev_content = execution_logs[i - 1].get("content", "")
            is_prev_error = bool(re.search(error_pattern, prev_content, re.IGNORECASE))

            # If the tool returned an error, but the agent's next move ignores it
            # (e.g., repeating the same hallucinated logic instead of fixing it)
            if is_prev_error and execution_logs[i].get("role") == "assistant":
                # Check for repetitive hallucination: if the code didn't change effectively
                # or if the agent claims success despite the kernel error.
                if "success" in content.lower() or "fixed" not in content.lower():
                    # Check if the kernel error UUID or specific message was ignored
                    return False

    return True
