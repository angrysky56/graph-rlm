import asyncio
import os
import sys
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_rlm.backend.src.core.core import PythonREPL


async def mock_skill():
    """A mock skill that returns a value but prints nothing."""
    await asyncio.sleep(0.1)
    return "Skill Result: 42"


async def mock_skill_print():
    """A mock skill that prints and returns nothing."""
    print("Skill Output: Hello World")
    await asyncio.sleep(0.1)


async def main():
    print("--- Verifying PythonREPL Output Capture ---")
    repl = PythonREPL()

    # 1. Inject Mock Skill
    # We simulate mcp structure: mcp.skill()
    mcp_mock = MagicMock()
    mcp_mock.skill_return = mock_skill
    mcp_mock.skill_print = mock_skill_print

    repl.namespace["mcp"] = mcp_mock

    print("\nTest 1: Skill that RETURNS value (no print)")
    code1 = """
import asyncio
async def test_func():
    await asyncio.sleep(0.01)
    return "Skill Result: 42"
await test_func()
"""
    stdout, stderr, result, is_err = await repl.execute(code1)

    print(f"Code: {code1}")
    print(f"Result: '{result}'")

    if result == "Skill Result: 42":
        print("✅ PASS: Return value captured.")
    else:
        print(f"❌ FAIL: Return value NOT captured. Got: {result}")

    print("\nTest 2: Skill that PRINTS (no return)")
    code2 = """
import asyncio
async def test_print():
    print("Skill Output: Hello World")
    await asyncio.sleep(0.01)
await test_print()
"""
    stdout, stderr, result, is_err = await repl.execute(code2)

    print(f"Code: {code2}")
    print(f"Stdout: '{stdout}'")
    print(f"Result: '{result}'")

    if "Skill Output: Hello World" in stdout:
        print("✅ PASS: Stdout captured.")
    else:
        print(f"❌ FAIL: Stdout NOT captured. Got: {stdout}")

    print("\nTest 3: Expression (1+1)")
    code3 = "1 + 1"
    stdout, stderr, result, is_err = await repl.execute(code3)
    print(f"Code: {code3}")
    print(f"Result: '{result}'")

    if result == 2:
        print("✅ PASS: Expression result captured.")
    else:
        print("❌ FAIL: Expression result NOT captured.")


if __name__ == "__main__":
    asyncio.run(main())
