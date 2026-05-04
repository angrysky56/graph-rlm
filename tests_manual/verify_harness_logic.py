
import asyncio
import os
import sys
from pathlib import Path

# Ensure repo root is in path to import backend
repo_root = Path(os.getcwd())
sys.path.append(str(repo_root))

from graph_rlm.backend.src.mcp_integration.skill_harness import verify_skill_importable


def verify_harness_logic():
    skills_dir = repo_root / "graph_rlm" / "backend" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy skill
    skill_file = skills_dir / "test_harness_import.py"
    skill_file.write_text("def ping(): return 'pong'")

    try:
        # Verify importable
        print(f"Verifying {skill_file}...")
        is_valid = verify_skill_importable("test_harness_import", "graph_rlm/backend/skills")

        if is_valid:
            print("SUCCESS: verify_skill_importable returned True.")
        else:
            print("FAILURE: verify_skill_importable returned False.")

        # Verify execution logic?
        # Ideally we'd call execute_skill_internal but that requires more mocking.
        # The unit test above confirms the helper logic works.

    finally:
        if skill_file.exists():
            skill_file.unlink()

if __name__ == "__main__":
    verify_harness_logic()
