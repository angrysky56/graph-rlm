"""
Docker Management Skill.

Provides utilities for listing and monitoring running MCP-related Docker containers.
"""

import json
import shutil
import subprocess
from typing import Any

# trunk-ignore(bandit/B404)


def list_mcp_containers(
    filter_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    List running Docker containers, optionally filtering by keyword.

    Args:
        filter_keywords: List of keywords to filter by (matches Name or Image).
                         If None or empty, returns ALL running containers.

    Returns:
        List of dictionaries containing container details (name, image, status, ports, id).
        Returns a list containing a single error dict if operation fails.
    """
    # Check for docker executable
    docker_executable = shutil.which("docker")
    if not docker_executable:
        return [{"error": "Docker executable not found in PATH"}]

    try:
        # Run docker ps with JSON formatting
        # Explicitly resolved the docker path above using shutil.which
        # trunk-ignore(bandit/B603)
        result = subprocess.run(
            [docker_executable, "ps", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

        containers = []
        output = result.stdout.strip()

        # dynamic splitting might be safer if we just iterate lines
        if not output:
            return []

        for line in output.split("\n"):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                name = data.get("Names", "")
                image = data.get("Image", "")

                # Check match if filters provided, otherwise include all
                match = True
                if filter_keywords:
                    # Case-insensitive match against Name OR Image
                    match = any(
                        k.lower() in name.lower() or k.lower() in image.lower()
                        for k in filter_keywords
                    )

                if match:
                    containers.append(
                        {
                            "name": name,
                            "image": image,
                            "status": data.get("Status", ""),
                            "ports": data.get("Ports", ""),
                            "id": data.get("ID", ""),
                        }
                    )
            except json.JSONDecodeError:
                continue

        return containers

    except subprocess.TimeoutExpired:
        return [{"error": "Docker command timed out"}]
    except subprocess.CalledProcessError as e:
        return [{"error": f"Docker command failed: {e.stderr.strip()}"}]
    except Exception as e:  # noqa: BLE001
        return [{"error": f"Unexpected error: {str(e)}"}]
