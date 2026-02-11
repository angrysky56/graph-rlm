"""
Diagnostic Skill: Docker PS Raw.

Simple utility to execute a raw 'docker ps' command and return all output for debugging.
"""

import shutil
import subprocess


def docker_ps_raw() -> str:
    """
    Executes 'docker ps' and returns stdout, stderr, and the return code.

    Returns:
        A formatted string with command results.
    """
    docker_executable = shutil.which("docker")
    if not docker_executable:
        return "Error: Docker executable not found in PATH"

    try:
        # trunk-ignore(bandit/B603)
        result = subprocess.run(
            [docker_executable, "ps"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,  # Explicitly False since we want the return code
        )
        return (
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}\nRC: {result.returncode}"
        )
    except subprocess.TimeoutExpired:
        return "Error: Docker command timed out"
    except Exception as e:  # noqa: BLE001
        return f"Error: Unexpected failure: {e}"
