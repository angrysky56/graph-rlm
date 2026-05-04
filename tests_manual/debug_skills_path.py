import sys
from pathlib import Path

# Add backend to path
repo_root = Path("/home/ty/Repositories/ai_workspace/graph-rlm")
sys.path.append(str(repo_root))

from graph_rlm.backend.src.mcp_integration.skill_storage import get_skills_manager

manager = get_skills_manager()
print(f"Skills Directory: {manager.skills_dir}")
print(f"Exists: {manager.skills_dir.exists()}")
print(f"Contents: {[x.name for x in manager.skills_dir.iterdir()]}")
