"""
MetaCognition Domain: Verified Path Persistence Solver.

This module provides a robust mechanism for writing data to the filesystem
by ensuring path normalization via Path objects and post-write integrity
checks through file size verification.
"""

import os
from pathlib import Path


def verified_path_persistence_solver(destination: str, content: str) -> bool:
    """
    Persists content to a destination path with grounded I/O verification.

    Args:
        destination: The target file path as a string.
        content: The string content to be written to the file.

    Returns:
        bool: True if the file was written and verified to be non-empty.

    Raises:
        OSError: If directory creation or file writing fails.
    """
    # Ground the environment string to a Path object
    target_path = Path(destination)

    # Ensure the parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Perform the write operation
    with open(target_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(content)

    # Post-write size verification (>0 bytes)
    if target_path.exists() and target_path.stat().st_size > 0:
        return True

    return False
