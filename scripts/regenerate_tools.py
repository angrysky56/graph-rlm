from pathlib import Path

from graph_rlm.backend.src.mcp_integration.generator import generate_from_config


def regenerate():
    project_root = Path(__file__).parent.parent.resolve()
    config_path = project_root / "mcp_servers.json"
    output_dir = project_root / "graph_rlm" / "backend" / "mcp_tools"

    print(f"Project ROOT: {project_root}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")

    if config_path.exists():
        generate_from_config(config_path, output_dir)
        print("Regeneration successful!")
    else:
        print("mcp_servers.json NOT FOUND")


if __name__ == "__main__":
    import sys

    # Add root to path
    sys.path.append(str(Path(__file__).parent.parent.resolve()))
    regenerate()
