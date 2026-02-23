"""
Skill execution harness.

Allows executing skills from the database via CLI,
similar to the mcp-code-execution-enhanced pattern.

Usage: python -m graph_rlm.backend.src.mcp_integration.skill_harness <skill_name> [args...]

Enforces execution in a dedicated 'skills_venv' for safety and dependency isolation.
"""

import argparse
import asyncio
import importlib.util
import inspect
import json
import logging
import os
import re
import shutil

# trunk-ignore(bandit/B404)
import subprocess
import sys
from pathlib import Path
from typing import Any

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.llm import llm

from .client import call_mcp_tool, cleanup_global_client_async
from .skill_storage import get_axioms_manager, get_skills_manager


def _spec_name(raw: str) -> str:
    """Sanitize a raw name to Agent Skills spec format."""
    name = re.sub(r"[^a-z0-9-]", "-", raw.lower()).strip("-")
    name = re.sub(r"-{2,}", "-", name)  # collapse consecutive hyphens
    return name[:64] or "unnamed-skill"


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("skill_harness")

BACKEND_ROOT = Path(__file__).parent.parent.parent.resolve()
SKILLS_VENV_PATH = BACKEND_ROOT / "agent_venv"


def verify_skill_importable(skill_name: str, directory: str = "skills") -> bool:
    """
    Checks if a newly created RLM skill is importable.

    Args:
        skill_name: The name of the .py file or module.
        directory: The relative path (or absolute path) to the skill directory.
    """
    skill_name = _spec_name(skill_name)
    repo_root = BACKEND_ROOT.parent.parent

    # Handle both relative and absolute paths for directory
    if Path(directory).is_absolute():
        skill_dir = Path(directory)
    else:
        skill_dir = repo_root / directory

    # Check flat .py first
    skill_path = skill_dir / f"{skill_name}.py"
    if not skill_path.exists():
        # Check for spec-compliant directory with scripts/ subdir
        module_safe = skill_name.replace("-", "_")
        spec_script = skill_dir / skill_name / "scripts" / f"{module_safe}.py"
        if spec_script.exists():
            skill_path = spec_script
        else:
            logger.warning(
                "Skill file %s missing (might be instructional).", skill_path
            )
            return False

    # Ensure directory is in sys.path
    abs_dir = str(skill_dir.resolve())
    if abs_dir not in sys.path:
        sys.path.append(abs_dir)

    try:
        spec = importlib.util.spec_from_file_location(skill_name, str(skill_path))
        if spec and spec.loader:
            return True
    except (ImportError, AttributeError, ValueError) as e:
        logger.error("Failed to verify import for %s: %s", skill_name, e)

    return False


def ensure_skills_venv() -> Path:
    """Ensure the skills virtual environment exists."""
    if not SKILLS_VENV_PATH.exists():
        logger.info("Creating agent_venv at %s...", SKILLS_VENV_PATH)
        try:
            uv_path = shutil.which("uv")
            if not uv_path:
                raise RuntimeError("uv executable not found in PATH")

            # 1. Create venv
            # trunk-ignore(bandit/B603)
            subprocess.run(
                [uv_path, "venv", str(SKILLS_VENV_PATH)],
                check=True,
                capture_output=True,
            )
            print("Created agent_venv.")

            # 2. Install dependencies
            # We use the pyproject.toml from the repo root
            repo_root = Path(__file__).parent.parent.parent.parent.parent
            pyproject_path = repo_root / "pyproject.toml"

            logger.info("Installing dependencies into agent_venv...")
            if pyproject_path.exists():
                # trunk-ignore(bandit/B603)
                subprocess.run(
                    [uv_path, "pip", "install", "-r", "pyproject.toml"],
                    cwd=str(repo_root),
                    env={**os.environ, "VIRTUAL_ENV": str(SKILLS_VENV_PATH)},
                    check=True,
                    capture_output=True,
                )
            else:
                # Fallback: install essential deps if pyproject.toml missing
                # trunk-ignore(bandit/B603)
                subprocess.run(
                    [
                        uv_path,
                        "pip",
                        "install",
                        "structlog",
                        "mcp",
                        "pydantic",
                        "fastapi",
                    ],
                    env={**os.environ, "VIRTUAL_ENV": str(SKILLS_VENV_PATH)},
                    check=True,
                    capture_output=True,
                )
            print("Dependencies installed in agent_venv.")

        except subprocess.CalledProcessError as e:
            logger.error("Failed to setup venv: %s", e.stderr.decode())
            raise RuntimeError("Could not setup skills virtual environment.") from e

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
    # Repo root contains 'skills' directory
    repo_root = BACKEND_ROOT.parent.parent

    env = os.environ.copy()
    # We need repo_root for graph_rlm package and skills package
    # skills are in backend/skills, so we need BACKEND_ROOT in path to import 'skills'
    env["PYTHONPATH"] = f"{repo_root}:{str(BACKEND_ROOT)}:{env.get('PYTHONPATH', '')}"
    # Also ensure unbuffered output
    env["PYTHONUNBUFFERED"] = "1"

    logger.info("Spawning skill '%s' in isolated venv...", skill_name)

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode()
        if "429" in error_msg:
            error_msg = f"[TERMINAL ERROR: RATE LIMITED] {error_msg}"
        logger.error(
            "Skill execution failed (RC %d): %s", process.returncode, error_msg
        )
        raise RuntimeError(f"Skill subprocess failed: {error_msg}")

    # Parse result from stdout (last line should be the JSON result)
    output = stdout.decode().strip()
    try:
        # We look for the last line, assuming it contains the JSON result
        lines = output.splitlines()

        if not lines:
            logger.warning("Skill '%s' produced no output.", skill_name)
            return None

        # Attempt to parse the last line as the JSON result
        return json.loads(lines[-1])

    except json.JSONDecodeError:
        logger.error("Failed to parse skill output: %s", output)
        raise RuntimeError(f"Skill returned invalid JSON: {output}") from None


class MCPServerProxy:
    """Proxy for a specific MCP server's tools."""

    def __init__(self, server_name: str):
        self._server_name = server_name

    def __getattr__(self, tool_name: str):

        async def _call_proxy(*_args, **kwargs):
            # Positional args not supported by MCP, but we handle them gracefully if possible
            # Usually agent code uses kwargs
            return await call_mcp_tool(
                server_name=self._server_name, tool_name=tool_name, arguments=kwargs
            )

        return _call_proxy


class MCPProxy:
    """Proxy for the 'mcp' namespace in skills."""

    def __getattr__(self, server_name: str):
        return MCPServerProxy(server_name)


class MinimalAgent:
    """Mock agent for RLMInterface in isolated environments."""

    def __init__(self):
        self.db = db
        self.llm = llm
        self.current_thought_id = "SKILL_RUN"
        self.execution_logs = {}

    def emit_event(self, *args, **kwargs):
        """No-op event emission."""


class RLMProxy:
    """Proxy for 'rlm' interface in skills (subset of functionality)."""

    async def recall(self, *args, **kwargs):
        """High-precision semantic search for domain rules and axioms."""
        from graph_rlm.backend.src.core.rlm_interface import RLMInterface

        # Note: This is a hack because the harness doesn't have the full agent session context.
        # It relies on the global 'db' being configured.
        # We use Any to bypass the strict type check for our lightweight mock agent.
        agent_mock: Any = MinimalAgent()
        interface = RLMInterface(
            agent_instance=agent_mock,
            session_id="SKILL_RUN",
            root_session_id="SKILL_RUN_ROOT",
        )
        return await interface.recall(*args, **kwargs)

    async def run_skill(self, skill_name: str, **kwargs):
        """Recursive skill execution."""
        return await execute_skill_internal(skill_name, kwargs)


async def execute_skill_internal(skill_name: str, kwargs: dict[str, Any]) -> Any:
    """
    Internal execution logic (runs INSIDE the venv).
    Imports and runs the skill/axiom function.
    """
    skill_name = _spec_name(skill_name)
    # 1. Try to find in Skills DB
    skills_mgr = get_skills_manager()
    skill = skills_mgr.get_skill(skill_name)

    # 2. Try to find in Axioms DB
    axioms_mgr = get_axioms_manager()
    if not skill:
        skill = axioms_mgr.get_axiom(skill_name)
        if skill:
            logger.info("Axiom found in DB: %s", skill_name)

    if skill:
        # DB path - ensure file exists
        if skill.get("type", "python") == "python":
            # Just verify it's registered
            pass

        # Distinguish between skills and axioms in module path
        if "axiom_" in skill_name or skill.get("axiom_type"):
            module_name = f"axioms_dir.{skill_name}"
        else:
            module_name = f"skills.{skill_name}"

        function_name = skill["function_name"]
    else:
        # 3. File fallback path - check skill/axiom directories
        skills_root = BACKEND_ROOT / "skills"
        axioms_root = BACKEND_ROOT / "axioms_dir"

        skill_file = skills_root / f"{skill_name}.py"
        axiom_file = axioms_root / f"{skill_name}.py"

        # Also check spec-compliant directories with scripts/ subdir
        module_safe = skill_name.replace("-", "_")
        spec_skill_script = skills_root / skill_name / "scripts" / f"{module_safe}.py"
        spec_axiom_script = axioms_root / skill_name / "scripts" / f"{module_safe}.py"

        if skill_file.exists():
            logger.info("Skill found in file: %s", skill_file)
            module_name = f"skills.{skill_name}"
            function_name = None
        elif spec_skill_script.exists():
            logger.info("Spec skill found: %s", spec_skill_script)
            module_name = f"skills.{skill_name}"
            function_name = None
        elif axiom_file.exists():
            logger.info("Axiom found in file: %s", axiom_file)
            module_name = f"axioms_dir.{skill_name}"
            function_name = None
        elif spec_axiom_script.exists():
            logger.info("Spec axiom found: %s", spec_axiom_script)
            module_name = f"axioms_dir.{skill_name}"
            function_name = None
        else:
            # Check for package (OpenCode style)
            skill_dir = skills_root / skill_name
            if skill_dir.exists():
                module_name = f"skills.{skill_name}"
                function_name = None
            else:
                raise ValueError(
                    f"Skill/Axiom '{skill_name}' not found in DB or local directories"
                )

    # Verify importability before attempting import
    # Determine the search root based on the module name prefix
    search_dir = BACKEND_ROOT / module_name.split(".")[0]

    if not verify_skill_importable(skill_name, str(search_dir)):
        # Check if it is an instructional skill (SKILL.md only)
        # Check if it is an instructional skill (SKILL.md only)
        # We check the directory for SKILL.md
        skills_root = BACKEND_ROOT / "skills"
        skill_dir = skills_root / skill_name
        skill_md = skill_dir / "SKILL.md"

        if skill_dir.exists() and skill_md.exists():
            logger.info("Skill '%s' is instructional. Returning content.", skill_name)
            try:
                content = skill_md.read_text(encoding="utf-8")
                return f"Instructional Skill ({skill_name}):\n\n{content}"
            except (OSError, UnicodeDecodeError) as e:
                raise RuntimeError(
                    f"Failed to read instructional skill {skill_name}: {e}"
                ) from e

        logger.warning(
            "Skill '%s' verification failed. Attempting import anyway as fallback.",
            skill_name,
        )

    # Import the module
    try:
        # Ensure current directory is in path (for local imports if any)
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))

        module = __import__(module_name, fromlist=["*"])

        # Inject proxies into module namespace
        module.__dict__["rlm"] = RLMProxy()
        module.__dict__["mcp"] = MCPProxy()

        if not function_name:
            # Try to resolve function name
            if hasattr(module, skill_name):
                function_name = skill_name
            elif hasattr(module, "main"):
                function_name = "main"
            elif hasattr(module, "research_topic") and skill_name == "research":
                function_name = "research_topic"
            else:

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
    except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
        raise RuntimeError(f"Skill execution failed: {e}") from e
    finally:
        await cleanup_global_client_async()


async def execute_skill(skill_name: str, kwargs: dict[str, Any]) -> Any:
    """
    Public entry point.
    Decides whether to spawn venv or run directly (if we are already in internal mode).
    """
    # Skills are authored by the agent (client) and executed in the isolated `agent_venv`.
    # Skills may call MCP tools using the `mcp` proxy, which routes through client.py
    # to external MCP server processes (stdio/sse). MCP servers use their own environments.
    return await execute_skill_in_venv(skill_name, kwargs)


async def main():
    """CLI entry point for skill execution."""
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
            logger.error("Invalid JSON in --args: %s", args.args)
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

    except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
        # Print error to stderr so as not to pollute stdout JSON
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
