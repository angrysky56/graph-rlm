"""
This module provides a validator to ensure MetaCognitive paradigm consistency.
It evaluates the PROBE/FILTER/EXECUTE/PERSIST sequence by verifying logic
flow and environmental introspection.
"""

import inspect
import re
from typing import List, Callable, Any


def paradigm_consistency_checker(
    execution_log: str, 
    agent_functions: List[Callable[..., Any]]
) -> bool:
    """
    Evaluates if the agent follows the PROBE/FILTER/EXECUTE/PERSIST sequence.

    Args:
        execution_log: A string representing the trace of agent actions.
        agent_functions: A list of callables available to the agent.

    Returns:
        bool: True if the sequence and introspection checks pass, False otherwise.
    """
    # Define sequence patterns
    sequence_pattern = r"PROBE.*FILTER.*EXECUTE.*PERSIST"
    
    # Check 1: Verify Sequence order in logs (Regex case-insensitive)
    if not re.search(sequence_pattern, execution_log, re.DOTALL | re.IGNORECASE):
        return False

    # Check 2: Verify Trace Evidence of Introspection (The 'PROBE' requirement)
    # We look for usage of introspection tools within the agent's logic
    introspection_tools_found = False
    introspection_keywords = [
        "inspect.getfullargspec",
        "os.environ",
        "sys.modules",
        "getattr",
        "dir("
    ]

    # Inspect function source code for introspection signatures
    for func in agent_functions:
        try:
            source = inspect.getsource(func)
            if any(keyword in source for keyword in introspection_keywords):
                introspection_tools_found = True
                break
        except (TypeError, OSError):
            continue

    return introspection_tools_found
