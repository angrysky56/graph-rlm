
import re
from pathlib import Path


def verify_mcp_counts():
    project_root = Path("/home/ty/Repositories/ai_workspace/graph-rlm").resolve()
    output_dir = project_root / "graph_rlm" / "backend" / "mcp_tools"

    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist.")
        return

    server_files = list(output_dir.glob("*.py"))
    server_count = 0
    tool_count = 0

    print(f"Checking {len(server_files)} files in {output_dir}")

    for f in server_files:
        if f.stem in ["__init__", "skills"]:
            continue
        server_count += 1
        try:
            content = f.read_text()
            match = re.search(r"return\s+\[(.*?)\]", content)
            if match:
                tools = [t.strip().strip("'").strip('"') for t in match.group(1).split(",") if t.strip()]
                tool_count += len(tools)
                # print(f"  - {f.name}: {len(tools)} tools")
        except Exception as e:
            print(f"  - Error reading {f.name}: {e}")

    print(f"\nFinal Result: Found {server_count} servers and {tool_count} tools.")

if __name__ == "__main__":
    verify_mcp_counts()
