"""
This module provides a validator for the Metaprogramming domain to ensure
agentic compliance with programmatic data interaction patterns.
"""

import re
from typing import Any, List


def rlm_paradigm_compliance_validator(agent_trace: str) -> bool:
    """
    Validates that the agent interacts with external inputs using programmatic
    patterns (PROBE, FILTER, CHUNK) instead of internal context memorization.

    Args:
        agent_trace: A string representation of the agent's logic or execution log.

    Returns:
        bool: True if compliance patterns are detected, False otherwise.
    """
    # Required patterns for programmatic input handling
    required_patterns: List[str] = [
        r"PROBE",
        r"FILTER",
        r"CHUNK",
        r"re\.(?:findall|search|match)",
        r"\[.*:.*\]"  # Slicing/Chunking syntax
    ]

    # Check for presence of programmatic retrieval logic
    matches: int = 0
    for pattern in required_patterns:
        if re.search(pattern, agent_trace, re.IGNORECASE):
            matches += 1

    # Compliance requires at least two distinct programmatic interaction methods
    return matches >= 2
