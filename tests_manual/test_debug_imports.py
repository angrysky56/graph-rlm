
import sys
import os

def test_print_path():
    print("\n--- SYS PATH ---")
    for p in sys.path:
        print(p)
    print("----------------")
    
    import graph_rlm.backend.src.mcp_integration as mcp_pkg
    print(f"mcp_pkg file: {mcp_pkg.__file__}")
    print(f"mcp_pkg path: {mcp_pkg.__path__}")
    
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    print("Successfully imported normalize_mcp_result")
