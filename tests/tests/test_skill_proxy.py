import asyncio
import sys
from pathlib import Path

# Add project root and backend to sys.path
root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(root))
backend = root / "graph_rlm" / "backend"
sys.path.append(str(backend / "src"))

from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill_internal


async def main():
    print("Testing skill namespace injection...")
    try:
        # execute_skill_internal imports proxy_test.py from skills_dir
        # and injects 'mcp' and 'rlm'
        result = await execute_skill_internal("proxy_test", {})
        print(f"Result: {result}")
        if result and result.get("status") == "success":
            print("✓ Namespace injection test passed!")
        else:
            print(f"FAIL: {result}")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
