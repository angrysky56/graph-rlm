import asyncio
import json
import sys
from unittest.mock import MagicMock

# Mock modules
sys.modules["falkordb"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()

import builtins

from graph_rlm.backend.src.mcp_integration.kernel import execute_code, mcp, rlm


async def test_kb_access():
    print("Testing KB Access in Kernel globals...")

    # Mock Globals similar to kernel_loop
    user_globals = globals().copy()
    user_globals.update({
        "mcp": mcp,
        "rlm": rlm,
        "kb": rlm.kb, # This is what we added
        "print": builtins.print,
        "asyncio": asyncio,
        "json": json,
        "sys": sys,
    })

    # Test Code 1: Access via rlm.kb (Old way)
    code1 = "x = rlm.kb.root"
    await execute_code(code1, user_globals)
    if "x" in user_globals:
        print(f"[PASS] rlm.kb.root accessed: {user_globals['x']}")
    else:
        print("[FAIL] rlm.kb.root failed")

    # Test Code 2: Access via kb directly (New way, previously failed)
    code2 = "y = kb.reports_dir"
    await execute_code(code2, user_globals)
    if "y" in user_globals:
        print(f"[PASS] kb.reports_dir accessed: {user_globals['y']}")
    else:
        print("[FAIL] kb.reports_dir failed")

if __name__ == "__main__":
    asyncio.run(test_kb_access())
