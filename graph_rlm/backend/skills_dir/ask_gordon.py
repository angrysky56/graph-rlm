import os
import subprocess


def ask_gordon(query: str, work_dir: str | None = None) -> str:
    """
    Executes the 'docker ai' CLI command to interact with the Gordon agent.

    Args:
        query: The question or instruction for Gordon.
        work_dir: Optional directory to execute the command in (defaults to current directory).

    Returns:
        The standard output from the 'docker ai' command.
    """
    cmd = ["docker", "ai", query]

    if work_dir and not os.path.isdir(work_dir):
        raise ValueError(f"The directory {work_dir} does not exist.")

    try:
        result = subprocess.run(
            cmd, cwd=work_dir, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Gordon failed with exit code {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        raise RuntimeError("The 'docker' command was not found.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")
