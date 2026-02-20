import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "graph_rlm" / "backend" / "src"))

from mcp_integration.kernel import KBProxy


def test_kb_proxy():
    kb = KBProxy()
    print(f"KB Root: {kb.root}")
    print(f"Reports: {kb.reports_dir}")
    print(f"Plans: {kb.plans_dir}")
    print(f"Outputs: {kb.outputs_dir}")
    print(f"Axioms: {kb.axioms_dir}")
    print(f"MCP Tools: {kb.mcp_tools_dir}")
    print(f"Skills: {kb.skills_dir}")
    print(f"Src: {kb.src_dir}")
    print(f"Workspace: {kb.workspace_dir}")

    # Test __dir__
    print(f"Dir: {dir(kb)}")

    # Test __getitem__
    assert kb["mcp_tools_dir"] == kb.mcp_tools_dir
    assert kb["skills_dir"] == kb.skills_dir
    print("All checks passed!")

if __name__ == "__main__":
    test_kb_proxy()
