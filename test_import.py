import sys
from pathlib import Path
root = Path('.').resolve()
if str(root) not in sys.path:
    sys.path.append(str(root))
try:
    from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager
    from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill
    print("IMPORT_SUCCESS")
except Exception as e:
    print(f"IMPORT_FAILED: {e}")
