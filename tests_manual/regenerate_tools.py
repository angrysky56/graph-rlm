from pathlib import Path

from graph_rlm.backend.src.mcp_integration.generator import generate_from_config

if __name__ == "__main__":
    repo_root = Path(".").resolve()
    config_path = repo_root / "mcp_servers.json"
    output_dir = repo_root / "graph_rlm/backend/mcp_tools"

    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")

    generate_from_config(config_path, output_dir)
