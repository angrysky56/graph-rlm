
import os
import sys
from pathlib import Path

# Add src to path to import kernel
repo_root = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root / "graph_rlm" / "backend" / "src"))

from mcp_integration.kernel import KBProxy


def test_kb_proxy():
    print("Testing KBProxy...")
    kb = KBProxy()

    # Test property access
    print(f"kb.root: {kb.root}")
    assert "knowledge_base" in kb.root

    # Test subscriptable access
    print(f"kb['root']: {kb['root']}")
    assert kb['root'] == kb.root

    # Test new src_dir property
    print(f"kb.src_dir: {kb.src_dir}")
    assert "/graph_rlm/backend/src" in kb.src_dir

    # Test subscriptable src_dir
    print(f"kb['src_dir']: {kb['src_dir']}")
    assert kb['src_dir'] == kb.src_dir

    # Test workspace_dir
    print(f"kb['workspace_dir']: {kb['workspace_dir']}")
    assert "workspace" in kb['workspace_dir']

    # Test dynamic __dir__
    print(f"dir(kb): {dir(kb)}")
    assert "src_dir" in dir(kb)
    assert "__getitem__" in dir(kb)

    # Test Invalid Key
    try:
        kb['invalid_key']
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"Expected error caught: {e}")

    print("\n✅ All KBProxy tests passed!")

if __name__ == "__main__":
    test_kb_proxy()
