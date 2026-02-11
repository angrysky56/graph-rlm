"""
This module provides a rate-limit resilient solver for metaprogramming tasks.
It implements exponential backoff to handle HTTP 429 status codes during
final state persistence or response generation.
"""

import time
import random
from typing import Any, Callable, Dict


def rate_limit_resilience_solver(
    action_payload: Dict[str, Any],
    submission_func: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Executes a metaprogramming task and submits the final response with
    resilience against rate limiting (HTTP 429).

    Args:
        action_payload: The data representing the agent's internal state.
        submission_func: The function responsible for the final API call.

    Returns:
        The API response after successful submission or state persistence.

    Raises:
        RuntimeError: If maximum retry attempts are exhausted.
    """
    max_retries = 5
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            # Attempt the final response generation or state submission
            response = submission_func(action_payload)
            return response
        except Exception as exc:
            # Check if the error indicates a 429 Too Many Requests
            if "429" in str(exc) and attempt < max_retries - 1:
                # Calculate exponential backoff with jitter
                delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Failed to submit after {max_retries} retries: {exc}") from exc

    return {"status": "failure", "reason": "exhausted_retries"}
