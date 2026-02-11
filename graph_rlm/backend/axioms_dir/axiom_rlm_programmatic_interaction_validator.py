"""
This module provides validation for the RLM_Programmatic_Interaction_Validator.
It ensures that agents interact with the environment using functional patterns
such as PROBE, FILTER, and CHUNK rather than treats code blocks as static
containers for pre-computed text strings.
"""

import re


def rlm_programmatic_interaction_validator(response_text: str) -> bool:
    """
    Validates that the response utilizes programmatic logic (PROBE, FILTER, CHUNK)
    instead of treating code blocks as raw text buffers or pre-computed dumps.

    Args:
        response_text: The full string response from the agent.

    Returns:
        bool: True if programmatic patterns are detected, False if the response
              is detected as a static text buffer.
    """
    # Pattern to detect if the code block is just a raw text dump
    static_buffer_pattern = r"\[Code\]\n(?!.*(?:rlm\.recall|filter\(|\[.*for.*in.*\])).*?\n\[/Code\]"

    # Pattern to detect programmatic engagement
    programmatic_patterns = [
        r"rlm\.recall\(",
        r"filter\(",
        r"\[.*?for.*?in.*?\]",
        r"yield",
        r"\.chunk\(",
        r"PROBE"
    ]

    has_static_buffer = bool(re.search(static_buffer_pattern, response_text, re.DOTALL))
    has_programmatic_logic = any(re.search(p, response_text) for p in programmatic_patterns)

    # Valid if logic exists and we aren't just dumping raw text markers
    return has_programmatic_logic or not has_static_buffer
