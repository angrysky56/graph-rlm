import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.absolute()
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from graph_rlm.backend.src.mcp_integration.skill_storage import (
    get_axioms_manager,
    get_skills_manager,
)


def verify_paths():
    skills_mgr = get_skills_manager()
    axioms_mgr = get_axioms_manager()

    print(f"Skills Dir: {skills_mgr.skills_dir}")
    print(f"Axioms Dir: {axioms_mgr.axioms_dir}")

    repo_root = Path(__file__).parent.parent.parent.parent.absolute()
    expected_skills = repo_root / "graph_rlm" / "backend" / "skills"
    expected_axioms = repo_root / "graph_rlm" / "backend" / "axioms_dir"

    if str(skills_mgr.skills_dir) == str(expected_skills):
        print("✅ Skills Dir matches expected path.")
    else:
        print(f"❌ Skills Dir mismatch!\nExpected: {expected_skills}\nActual:   {skills_mgr.skills_dir}")

    if str(axioms_mgr.axioms_dir) == str(expected_axioms):
        print("✅ Axioms Dir matches expected path.")
    else:
        print(f"❌ Axioms Dir mismatch!\nExpected: {expected_axioms}\nActual:   {axioms_mgr.axioms_dir}")

if __name__ == "__main__":
    verify_paths()
