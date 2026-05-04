import asyncio
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root.resolve()) not in sys.path:
    sys.path.append(str(repo_root.resolve()))

from graph_rlm.backend.src.core.sheaf import sheaf


async def test_honest_fail():
    print("Testing Honest Sheaf Failure (Non-existent tool)...")
    code = """
import asyncio

async def fail_save():
    # mcp.filesystem does not exist in the real servers list
    await mcp.filesystem.write_file(path="test.md", content="hello")

asyncio.run(fail_save())
"""
    # Run check
    res = sheaf.check_axiomatic_consistency(code, task_tags=["general"])
    print(f"Status: {res['status']}")
    if res["status"] == "AXIOMATIC_VIOLATION":
        print(f"Critique: {res['critique']}")
    else:
        print("FAIL: Verification should have failed with AttributeError.")


async def test_brave_success():
    print("\nTesting Honest Sheaf Success (Existing tool)...")
    code = """
import asyncio

async def do_search():
    # mcp.brave_search.brave_web_search exists
    return await mcp.brave_search.brave_web_search(query="Bardo")

asyncio.run(do_search())
"""
    res = sheaf.check_axiomatic_consistency(code, task_tags=["general"])
    print(f"Status: {res['status']}")
    if res["status"] == "HEALTHY":
        print("SUCCESS: Existing tool passed validation.")
    else:
        print(f"FAIL: Verification should have passed. Critique: {res.get('critique')}")


if __name__ == "__main__":
    asyncio.run(test_honest_fail())
    asyncio.run(test_brave_success())
