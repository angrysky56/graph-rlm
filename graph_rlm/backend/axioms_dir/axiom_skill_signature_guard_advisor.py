"""
This module provides the SkillSignatureGuardAdvisor for Model Observability.
It ensures tool execution calls align with the actual function signatures to
prevent TypeError exceptions caused by signature mismatches.
"""

import inspect
import logging
from typing import Any, Callable, Dict


def skill_signature_guard_advisor(
    skill_func: Callable[..., Any],
    provided_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Inspects a skill function's signature and filters out incompatible kwargs.

    This advisor acts as a proactive guard to prevent execution failures
    by ensuring only arguments accepted by the function signature are passed.

    Args:
        skill_func: The function/tool to be executed.
        provided_kwargs: The dictionary of arguments intended for the call.

    Returns:
        A dictionary containing only the valid arguments for the function.
    """
    sig = inspect.signature(skill_func)
    parameters = sig.parameters

    # If the function accepts variable keyword arguments (**kwargs),
    # all provided arguments are technically valid.
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    )

    if has_kwargs:
        return provided_kwargs

    # Filter arguments to only include those present in the signature
    valid_kwargs = {
        key: value
        for key, value in provided_kwargs.items()
        if key in parameters
    }

    if len(valid_kwargs) != len(provided_kwargs):
        logging.warning(
            "Filtered %d invalid arguments for skill: %s",
            len(provided_kwargs) - len(valid_kwargs),
            skill_func.__name__
        )

    return valid_kwargs
