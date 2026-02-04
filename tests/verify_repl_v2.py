
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
backend_root = Path(__file__).parent.parent
sys.path.append(str(backend_root))

from graph_rlm.backend.src.core.config import settings
from graph_rlm.backend.src.core.core import PythonREPL


async def test_repl_timeout():
    print("\n--- Testing REPL Timeout ---")
    repl = PythonREPL()
    settings.REPL_TIMEOUT = 5  # Set short timeout for testing

    code = "import time\nwhile True: time.sleep(1)"
    print(f"Executing infinite loop with {settings.REPL_TIMEOUT}s timeout...")

    try:
        stdout, stderr, result = await repl.execute(code)
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        print(f"RESULT: {result}")
    except Exception as e:
        print(f"Caught Expected Timeout/Error: {e}")

async def test_top_level_await():
    print("\n--- Testing Top-Level Await ---")
    repl = PythonREPL()

    # Define an async function in namespace
    async def hello():
        await asyncio.sleep(0.1)
        return "Hello from Async!"

    repl.namespace["hello"] = hello

    code = "result = await hello()\nprint(f'Captured: {result}')\nresult"
    print("Executing code with top-level await...")

    stdout, stderr, result = await repl.execute(code)
    print(f"STDOUT: {stdout}")
    print(f"STDERR: {stderr}")
    print(f"RESULT: {result}")

    if result == "Hello from Async!":
        print("✅ SUCCESS: Top-level await works.")
    else:
        print("❌ FAILURE: Top-level await did not return expected result.")

if __name__ == "__main__":
    asyncio.run(test_repl_timeout())
    asyncio.run(test_top_level_await())
