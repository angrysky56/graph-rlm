import sys
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

print(f"Checking imports in {repo_root}...")

try:
    from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill
    print("✓ Successfully imported execute_skill from skill_harness")
except ImportError as e:
    print(f"✗ Failed to import execute_skill from skill_harness: {e}")
    sys.exit(1)

try:
    # This might fail for other reasons (dependencies), but we want to check syntax and basic imports
    import graph_rlm.backend.src.core.agent as agent
    print("✓ Successfully imported agent module")
except ImportError as e:
    print(f"− Agent import failed (as expected if dependencies missing), but let's check the specific error: {e}")
    if "execute_skill" in str(e):
         print(f"✗ Agent still has an execute_skill import error: {e}")
         sys.exit(1)
except Exception as e:
    print(f"✓ Agent module loaded (but threw {type(e).__name__} during init, which is fine)")

print("Verification complete.")
