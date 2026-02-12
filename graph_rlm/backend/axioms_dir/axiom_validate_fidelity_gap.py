"""
This module provides a Fidelity Gap Validator for the MetaCognition domain.
It cross-references an Agent's internal 'Proposal' state against actual
Python kernel 'Trace Evidence' to detect success hallucinations.
"""

from typing import Dict, Any, List
import os


def validate_fidelity_gap(
    proposal_state: Dict[str, Any],
    kernel_trace_evidence: Dict[str, Any]
) -> bool:
    """
    Validates that the Agent's reported success matches the filesystem reality.

    Args:
        proposal_state: A dictionary representing the agent's internal
            belief of the outcome (e.g., {"status": "success", "files": [...]}).
        kernel_trace_evidence: A dictionary containing actual kernel logs
            and filesystem snapshots (e.g., {"files_created": [...], "errors": [...]}).

    Returns:
        bool: True if the internal state matches kernel evidence,
            False if a success hallucination (fidelity gap) is detected.
    """
    claimed_success: bool = proposal_state.get("status") == "success"
    claimed_files: List[str] = proposal_state.get("created_files", [])

    actual_errors: List[str] = kernel_trace_evidence.get("errors", [])
    actual_files: List[str] = kernel_trace_evidence.get("files_created", [])

    # If the agent claims success but the kernel logged errors, fail validation
    if claimed_success and actual_errors:
        return False

    # Check for file persistence success hallucinations
    for file_path in claimed_files:
        if file_path not in actual_files:
            return False

        # Physical verification if the path is accessible in the current context
        if not os.path.exists(file_path):
            return False

    return True
