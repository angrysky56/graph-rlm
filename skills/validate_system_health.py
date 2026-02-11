"""
Skill: validate_system_health
Description: Verifies the integrity of the agent's runtime environment, including MCP server availability, critical paths, and REPL interface presence.
"""

import os


async def validate_system_health():
    """
    Checks the health of the system by verifying:
    1. MCP server availability (via mcp object).
    2. Critical directory paths exist.
    3. RLM interface is available in the global scope.

    Returns:
        dict: A report containing server count, path status, and overall health.
    """
    report = {
        "mcp_servers": [],
        "mcp_server_count": 0,
        "critical_paths": {},
        "rlm_available": False,
        "status": "healthy",
    }

    # 1. Check MCP Servers
    if "mcp" in globals():
        try:
            # mcp object exposes servers via dir()
            servers = dir(globals()["mcp"])
            # Filter out private attributes
            servers = [s for s in servers if not s.startswith("_")]
            report["mcp_servers"] = servers
            report["mcp_server_count"] = len(servers)
        except Exception as e:  # noqa: BLE001
            report["mcp_error"] = str(e)
            report["status"] = "degraded"
    else:
        report["mcp_error"] = "'mcp' object not found in globals"
        report["status"] = "critical"

    # 2. Check Critical Paths
    # Adjust paths based on known structure
    # We are in backend/skills_dir, so root is ../../..
    # Expected paths relative to project root
    base_path = os.getcwd()  # adherence to agent runtime CWD

    # Common paths to check
    paths_to_check = [
        "graph_rlm/backend/skills_dir",
        "graph_rlm/backend/mcp_tools",
        "graph_rlm/backend/src",
    ]

    for p in paths_to_check:
        full_path = os.path.join(base_path, p)
        exists = os.path.exists(full_path)
        report["critical_paths"][p] = exists
        if not exists:
            report["status"] = "degraded"

    # 3. Check RLM Interface
    if "rlm" in globals():
        report["rlm_available"] = True
    else:
        report["rlm_available"] = False
        report["status"] = "critical"

    return report
