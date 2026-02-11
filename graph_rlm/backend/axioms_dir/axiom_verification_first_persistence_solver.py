"""
This module provides a solver for the ModelObservability domain that ensures
data persistence is always followed by a mandatory filesystem verification step.
"""

import os
from typing import Any


def verification_first_persistence_solver(
    file_path: str, data: Any, write_mode: str = "w"
) -> bool:
    """
    Writes data to a file and verifies its physical persistence on disk.

    The verification step confirms the file exists and contains content
    (non-zero size) immediately after the I/O operation to ensure observability.

    Args:
        file_path: The target filesystem path for the data.
        data: The content to be written to the file.
        write_mode: The mode in which the file is opened (default 'w').

    Returns:
        bool: True if the file was written and verified successfully.

    Raises:
        IOError: If the persistence verification fails after the write attempt.
    """
    # Perform the I/O operation
    with open(file_path, write_mode, encoding="utf-8") as target_file:
        target_file.write(str(data))

    # Mandatory Verification Step
    if os.path.exists(file_path):
        # Check for non-zero file size
        if os.path.getsize(file_path) > 0:
            return True

    raise IOError(
        f"ModelObservability Error: Persistence verification failed for {file_path}. "
        "File may be missing or empty."
    )
