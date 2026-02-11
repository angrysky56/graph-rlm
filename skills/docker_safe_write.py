"""
Docker-Safe File Writer Skill.

Writes files to the knowledge base, automatically detecting if running
inside a Docker container or local environment to adjust paths accordingly.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("graph_rlm.skills.docker_safe_write")


def docker_safe_write(
    filename: str, content: str, subdir: str = "outputs"
) -> Dict[str, Any]:
    """
    Write a file to the knowledge base using Docker-aware paths.

    Args:
        filename: Name of the file to write (e.g., "report.md").
        content: Content to write to the file.
        subdir: Subdirectory within knowledge_base (default: "outputs").

    Returns:
        A dictionary containing success status, resolved paths, and environment info.
    """
    try:
        # Detect if we're in Docker or local environment
        # Specific check for the standard mount point in our Docker images
        if Path("/knowledge_base").exists():
            base_path = Path("/knowledge_base")
            env_name = "Docker"
        else:
            # Local execution - use project-relative path or environment override
            project_root = Path(__file__).parent.parent
            default_kb = project_root / "knowledge_base"
            kb_dir = os.getenv("MCP_KNOWLEDGE_BASE_DIR", str(default_kb))
            base_path = Path(kb_dir)
            env_name = "Local"

        # Create the target directory
        target_dir = base_path / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Write the file
        file_path = target_dir / filename
        file_path.write_text(content, encoding="utf-8")

        # Return the path as it appears in the container (for logging)
        display_path = (
            f"/{subdir}/{filename}" if env_name == "Docker" else str(file_path)
        )

        return {
            "success": True,
            "path": str(file_path),
            "display_path": display_path,
            "environment": env_name,
            "message": f"File written successfully to {display_path} ({env_name})",
        }

    except PermissionError as e:
        logger.error("Permission denied writing to %s: %s", subdir, e)
        return {"success": False, "message": f"Permission denied: {e}"}
    except OSError as e:
        logger.error("OS error writing file %s: %s", filename, e)
        return {"success": False, "message": f"File system error: {e}"}
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error in docker_safe_write: %s", e)
        return {"success": False, "message": f"Unexpected error: {str(e)}"}
