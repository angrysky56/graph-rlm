
import asyncio
import json
import logging
import sys
import unittest.mock
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules["graph_rlm.backend.src.core.agent"] = MagicMock()
sys.modules["graph_rlm.backend.src.mcp_integration.discovery"] = MagicMock()

# Import Kernel (need to make sure we don't start the loop)
from graph_rlm.backend.src.mcp_integration import kernel


async def test_kernel_robustness():
    print("Testing Kernel Robustness...")

    # 1. Test execute_code error reporting (simulating 'status' KeyError)
    print("\n1. Testing Execution Error Logging...")

    # We want to verify that if user code raises an exception, the kernel logs it with traceback
    # We'll mock the logger to check calls
    kernel.logger = MagicMock()

    code_that_raises = "raise KeyError('status')"

    # We execute this inside execute_code
    # execute_code catches nothing, so it propagates to kernel_loop
    # But we can call execute_code directly and assert it raises
    try:
        await kernel.execute_code(code_that_raises, {})
    except KeyError as e:
        print(f"✅ execute_code propagated KeyError: {e}")
    except Exception as e:
        print(f"❌ execute_code raised unexpected exception: {type(e)} {e}")

    # 2. Test Kernel Loop Error Handling (Mocking stdin/stdout)
    print("\n2. Testing Kernel Loop Error Catching...")

    # We need to simulate the kernel loop receiving a packet that causes execution failure
    # and verify it logs "Kernel Loop Error: KeyError: 'status'"

    # Mock sys.stdin.readline to return:
    # 1. A packet causing KeyError
    # 2. Empty string (EOF) to exit loop

    packet_bad = json.dumps({
        "command": "EXECUTE",
        "code": "raise KeyError('status')",
        "context": {}
    })

    mock_stdin = MagicMock()
    mock_stdin.readline.side_effect = [packet_bad, ""]

    with patch("sys.stdin", mock_stdin), \
         patch("sys.stderr", new_callable=MagicMock) as mock_stderr:

        # Run kernel loop
        await kernel.kernel_loop()

        # Check logger calls
        # We expect logger.error("Kernel Loop Error: %s: %s", "KeyError", "'status'")
        found_log = False
        for call in kernel.logger.error.call_args_list:
            args = call.args
            if "Kernel Loop Error" in args[0] and "KeyError" in str(args[1]):
                found_log = True
                print(f"✅ Kernel captured and logged error: {args}")
                break

        if not found_log:
            print("❌ Kernel failed to log error correctly.")
            print(f"Log calls: {kernel.logger.error.call_args_list}")

if __name__ == "__main__":
    asyncio.run(test_kernel_robustness())
