"""
This module provides logic to correct invalid attribute pathing for knowledge
base directories within the ModelObservability domain. It ensures that
path-related configuration is accessible even when proxy objects are malformed.
"""

import os
from typing import Any, Optional


def resolve_filing_system_path(rlm_proxy: Any, fallback_env_var: str = "KB_REPORTS_DIR") -> str:
    """
    Validates and resolves the reports directory path from a proxy object.

    If the proxy object 'rlm' lacks the expected 'kb' attribute or 'reports_dir'
    mapping, it falls back to a system environment variable to ensure
    continuity of the observability flow.

    Args:
        rlm_proxy: The proxy object representing the Resource Lifecycle Manager.
        fallback_env_var: The environment variable to check as a fallback.

    Returns:
        str: The resolved absolute path to the reports directory.

    Raises:
        FileNotFoundError: If neither the proxy nor the environment provides a valid path.
    """
    # Attempt to retrieve from proxy object safely
    try:
        if hasattr(rlm_proxy, "kb") and hasattr(rlm_proxy.kb, "reports_dir"):
            path = rlm_proxy.kb.reports_dir
            if os.path.isdir(path):
                return os.path.abspath(path)
    except (AttributeError, TypeError):
        pass

    # Fallback to environment configuration
    env_path: Optional[str] = os.getenv(fallback_env_var)
    if env_path and os.path.isdir(env_path):
        return os.path.abspath(env_path)

    raise FileNotFoundError("Unable to resolve knowledge base reports directory via proxy or environment.")
