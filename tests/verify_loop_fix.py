
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Adjust path to find backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_rlm.backend.src.mcp_integration.runtime import AgentRuntime

async def main():
    print("--- Starting Loop Fix Verification ---")

    # 1. Setup Runtime
    project_root = Path("/home/ty/Repositories/ai_workspace/graph-rlm")
    runtime = AgentRuntime(project_root)

    # 2. Start a session in the CURRENT loop
    print("Step 1: Starting session 1...")
    proc1 = await runtime._ensure_session("test_session")
    pid1 = proc1.pid
    print(f"Session 1 started with PID: {pid1}")

    # 3. Simulate "Old Loop" scenario manually
    # We will modify the stored session data to pretend it was created in a different loop
    # In reality, to test this fully we'd need to actually restart the event loop,
    # but that's hard in a single script.
    # Instead, we mock the loop object stored in the session data.

    print("Step 2: Tampering with session loop data...")
    fake_loop = MagicMock()
    fake_loop._thread_id = "fake_thread_id"

    # Access the stored session dict
    if "test_session" in runtime.sessions:
        runtime.sessions["test_session"]["loop"] = fake_loop
        print("Tampering successful: Session loop replaced with fake loop.")
    else:
        print("ERROR: Session data not found!")
        return

    # 4. Request session again - should trigger RESTART
    print("Step 3: Requesting session 1 again (should trigger restart)...")
    proc2 = await runtime._ensure_session("test_session")
    pid2 = proc2.pid
    print(f"Session 1 re-acquired with PID: {pid2}")

    # 5. Assertions
    if pid1 == pid2:
        print("FAILURE: Process ID did not change! detection failed.")
        sys.exit(1)
    else:
        print("SUCCESS: Process ID changed! Loop mismatch detection worked.")

    # 6. Test 'is_failed' capability via Agent (Mocked)
    # We can't easily run the full Agent here, but we can verify runtime.execute
    # returns valid output on the new process.
    print("Step 4: Executing code on new process...")
    stdout, stderr, code = await runtime.execute("print('Hello World')", {"session_id": "test_session"})

    print(f"Execution Result: code={code}, stdout={stdout.strip()}, stderr={stderr.strip()}")

    if code == 0 and "Hello World" in stdout:
        print("SUCCESS: Execution worked on restarted process.")
    else:
        print("FAILURE: Execution failed on restarted process.")
        sys.exit(1)

    # Cleanup
    try:
        proc2.kill()
    except:
        pass

if __name__ == "__main__":
    asyncio.run(main())
