import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(root))

from graph_rlm.backend.src.core.core import PythonREPL


async def test_auto_await():
    repl = PythonREPL()

    # Mocking a coroutine function in the namespace
    async def mock_coro():
        return "SUCCESS"

    repl.namespace["mock_tool"] = mock_coro

    print("Testing auto-await for mock_tool()...")
    # Execute without 'await'
    stdout, stderr, result, error = await repl.execute("mock_tool()")

    print(f"Result: {result}")
    assert result == "SUCCESS", f"Expected 'SUCCESS', got {result}"
    print("✓ Auto-await test passed!")

if __name__ == "__main__":
    asyncio.run(test_auto_await())
