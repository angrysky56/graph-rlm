import asyncio
import sys

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.sheaf import sheaf


async def verify_emergency_fixes():
    print("🚀 Verifying Emergency Fixes...")

    # 1. Verify SheafMonitor methods
    print("  Checking SheafMonitor methods...")
    try:
        surprise = sheaf.compute_sheaf_surprise_score(limit=1)
        print(f"    -> compute_sheaf_surprise_score: OK (Found {len(surprise)})")
    except Exception as e:
        print(f"    ❌ ERROR: compute_sheaf_surprise_score failed: {e}")
        return False

    # 2. Verify REPL done() injection
    print("  Checking REPL done() injection...")
    agent = Agent()
    code = """
import asyncio
async def test():
    return "OK"
res = asyncio.run(test())
done(res)
"""
    output = await agent._execute_code(code, "test_thought", "test_session")
    print(f"    REPL Output:\n{output}")

    if "Task Marked Complete" in output and "NameError" not in output:
        print("    -> done() injection: OK")
    else:
        print("    ❌ ERROR: done() injection failed or threw NameError")
        return False

    # 3. Verify Axiomatic warning emission (internal check)
    print("  Checking Axiomatic filtering robustness...")
    # This should be HEALTHY now even without tags if it doesn't match general
    diag = sheaf.check_axiomatic_consistency("x=1", task_tags=["math"])
    print(f"    -> check_axiomatic_consistency: {diag['status']}")
    if diag["status"] != "HEALTHY":
        print(
            f"    ❌ ERROR: Expected HEALTHY for unrelated task, got {diag['status']}"
        )
        return False

    print("✅ All emergency fixes verified!")
    return True


if __name__ == "__main__":
    success = asyncio.run(verify_emergency_fixes())
    sys.exit(0 if success else 1)
