"""
Skill execution harness.

Allows executing skills from the database via CLI, similar to the mcp-code-execution-enhanced pattern.
Usage: python -m graph_rlm.backend.src.mcp_integration.skill_harness <skill_name> [args...]

Enforces execution in a dedicated 'skills_venv' for safety and dependency isolation.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil

# trunk-ignore(bandit/B404)
import subprocess
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("skill_harness")

BACKEND_ROOT = Path(__file__).parent.parent.parent.resolve()
SKILLS_VENV_PATH = BACKEND_ROOT / "skills_venv"


def ensure_skills_venv() -> Path:
    """Ensure the skills virtual environment exists."""
    if not SKILLS_VENV_PATH.exists():
        logger.info(f"Creating isolated skills environment at {SKILLS_VENV_PATH}...")
        try:
            uv_path = shutil.which("uv")
            if not uv_path:
                raise RuntimeError("uv executable not found in PATH")

            # trunk-ignore(bandit/B603)
            subprocess.run(
                [uv_path, "venv", str(SKILLS_VENV_PATH)],
                check=True,
                capture_output=True,
            )
            logger.info("Created skills_venv.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create venv: {e.stderr.decode()}")
            raise RuntimeError("Could not create skills virtual environment.") from e

    return SKILLS_VENV_PATH


def get_venv_python() -> Path:
    """Get path to the venv python executable."""
    if sys.platform == "win32":
        return SKILLS_VENV_PATH / "Scripts" / "python.exe"
    return SKILLS_VENV_PATH / "bin" / "python"


async def execute_skill_in_venv(skill_name: str, kwargs: dict[str, Any]) -> Any:
    """Spawn a subprocess to run the skill in the isolated venv."""
    ensure_skills_venv()
    venv_python = get_venv_python()

    if not venv_python.exists():
        raise RuntimeError(f"Venv python not found at {venv_python}")

    # Prepare command
    # We call this same module as a script
    module_path = "graph_rlm.backend.src.mcp_integration.skill_harness"

    cmd = [
        str(venv_python),
        "-m",
        module_path,
        skill_name,
        "--args",
        json.dumps(kwargs),
        "--internal-run",  # Flag to signal we are inside the venv
    ]

    # Ensure PYTHONPATH includes the project root so we can import the harness
    # project root is 3 levels up from backend: graph-rlm -> graph_rlm -> backend
    # Actually, we need 'graph-rlm' (the repo root) in pythonpath to import graph_rlm.*
    # BACKEND_ROOT is .../graph_rlm/backend
    # REPO_ROOT is .../graph_rlm (the outer one, containing graph_rlm package)

    # Current file: .../graph_rlm/backend/src/mcp_integration/skill_harness.py
    # Root of package 'graph_rlm' is .../graph-rlm/ (the folder containing graph_rlm dir)
    repo_root = (
        BACKEND_ROOT.parent.parent
    )  # /home/ty/Repositories/ai_workspace/graph-rlm

    env = os.environ.copy()
    # We need repo_root for graph_rlm package and BACKEND_ROOT for skills_dir module
    env["PYTHONPATH"] = f"{repo_root}:{BACKEND_ROOT}:{env.get('PYTHONPATH', '')}"
    # Also ensure unbuffered output
    env["PYTHONUNBUFFERED"] = "1"

    logger.info(f"Spawning skill '{skill_name}' in isolated venv...")

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode()
        logger.error(f"Skill execution failed (RC {process.returncode}): {error_msg}")
        raise RuntimeError(f"Skill subprocess failed: {error_msg}")

    # Parse result from stdout (last line should be the JSON result)
    output = stdout.decode().strip()
    try:
        # We look for the last line, assuming it contains the JSON result
        lines = output.splitlines()
        # Filter out log lines if any (though we configured logging to stderr mostly?)
        # For now, blindly try to parse the whole output or the last line
        if not lines:
            return None
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        logger.error(f"Failed to parse skill output: {output}")
        raise RuntimeError(f"Skill returned invalid JSON: {output}") from None


async def execute_skill_internal(skill_name: str, kwargs: dict[str, Any]) -> Any:
    """
    Internal execution logic (runs INSIDE the venv).
    Imports and runs the skill function.
    """
    from .client import cleanup_global_client
    from .skills import get_skills_manager

    # Get skill code
    manager = get_skills_manager()
    skill = manager.get_skill(skill_name)

    if skill:
        # DB path - ensure file exists
        manager.get_import_statement(skill_name)
        module_name = f"skills_dir.{skill_name}"
        function_name = skill["function_name"]
    else:
        # File fallback path
        skill_file = Path("skills_dir") / f"{skill_name}.py"
        if not skill_file.exists():
            raise ValueError(
                f"Skill '{skill_name}' not found in DB or skills_dir/ directory"
            )

        logger.info(f"Skill found in file: {skill_file}")
        module_name = f"skills_dir.{skill_name}"
        function_name = None

    # Import the module
    try:
        # Ensure current directory is in path (for local imports if any)
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))

        module = __import__(module_name, fromlist=["*"])

        if not function_name:
            # Try to resolve function name
            if hasattr(module, skill_name):
                function_name = skill_name
            elif hasattr(module, "main"):
                function_name = "main"
            elif hasattr(module, "research_topic") and skill_name == "research":
                function_name = "research_topic"
            else:
                import inspect

                funcs = [
                    n
                    for n, o in inspect.getmembers(module, inspect.isfunction)
                    if not n.startswith("_") and o.__module__ == module.__name__
                ]
                if len(funcs) == 1:
                    function_name = funcs[0]
                else:
                    raise ValueError(
                        f"Could not determine entry point function for skill '{skill_name}'"
                    )

        func = getattr(module, function_name)

        # Execute
        if asyncio.iscoroutinefunction(func):
            result = await func(**kwargs)
        else:
            result = func(**kwargs)

        return result

    except ImportError as e:
        raise RuntimeError(f"Failed to import skill {skill_name}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Skill execution failed: {e}") from e
    finally:
        cleanup_global_client()


async def execute_skill(skill_name: str, kwargs: dict[str, Any]) -> Any:
    """
    Public entry point.
    Decides whether to spawn venv or run directly (if we are already in internal mode).
    """
    # If we are already running as the script with internal flag, we shouldn't be here calling execute_skill usually,
    # but for safety:
    # Actually, execute_skill is called by mcp_tools wrappers.
    # Those wrappers run in the MAIN process.
    # So execute_skill MUST spawn the venv.

    return await execute_skill_in_venv(skill_name, kwargs)


async def main():
    parser = argparse.ArgumentParser(description="Execute an MCP skill")
    parser.add_argument("skill_name", help="Name of the skill to execute")
    parser.add_argument("--args", help="JSON string of arguments", default="{}")
    parser.add_argument(
        "--internal-run", action="store_true", help="Internal flag: running inside venv"
    )
    parser.add_argument("extra_args", nargs="*", help="Key=value arguments")

    args = parser.parse_args()

    # Parse arguments
    kwargs = {}
    if args.args:
        try:
            kwargs = json.loads(args.args)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in --args: {args.args}")
            sys.exit(1)

    for arg in args.extra_args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            kwargs[k] = v

    try:
        if args.internal_run:
            # We are inside the venv. Execute logic directly.
            result = await execute_skill_internal(args.skill_name, kwargs)
            # Print ONLY the result JSON to stdout for capture
            print(json.dumps(result, default=str))
        else:
            # We are the CLI wrapper. Spawn the venv.
            result = await execute_skill_in_venv(args.skill_name, kwargs)
            print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        # Print error to stderr so as not to pollute stdout JSON
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
