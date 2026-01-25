import sys
from pathlib import Path
root = Path('.').resolve()
if str(root) not in sys.path:
    sys.path.append(str(root))
try:
    from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager
    mgr = get_skills_manager()
    skills = mgr.list_skills()
    print("SKILLS_FOUND:", ", ".join(skills.keys()))
except Exception as e:
    print(f"SKILLS_FAILED: {e}")
