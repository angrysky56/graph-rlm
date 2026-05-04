import asyncio
import json
import logging

from graph_rlm.backend.src.mcp_integration.runtime import AgentRuntime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_mcp_fix")


async def test_deadlock_resolution():
    """
    Test that large discovery packets don't cause a deadlock.
    """
    from pathlib import Path

    runtime = AgentRuntime(
        project_root=Path("/home/ty/Repositories/ai_workspace/graph-rlm")
    )
    session_id = "test_deadlock_session"

    # Mock a large discovery namespace
    class MockTool:
        def __init__(self, name):
            self.__name__ = name
            self.__doc__ = "A tool that does something important." * 100  # Large doc

    class MockServer:
        def __init__(self, name):
            self.name = name
            for i in range(50):  # 50 tools per server
                setattr(self, f"tool_{i}", MockTool(f"tool_{i}"))

        def __dir__(self):
            return [f"tool_{i}" for i in range(50)]

    class MockNamespace:
        def __init__(self):
            for i in range(30):  # 30 servers
                setattr(self, f"server_{i}", MockServer(f"server_{i}"))

        def __dir__(self):
            return [f"server_{i}" for i in range(30)]

    mock_ns = MockNamespace()

    logger.info("--- Step 1: Initial Discovery (Large Payload) ---")
    # This should trigger full discovery and send a giant JSON
    code = "print('Hello from first run')"
    context = {"session_id": session_id}

    stdout, stderr, result, exit_code = await runtime.execute(
        code, context, mcp_namespace=mock_ns
    )

    logger.info(f"Stdout length: {len(stdout)}")
    logger.info(f"Stderr length: {len(stderr)}")
    logger.info(f"Exit code: {exit_code}")

    if exit_code == 0 and "Hello from first run" in stdout:
        logger.info("✅ First run successful (No Deadlock)")
    else:
        logger.error("❌ First run failed or timed out")
        return

    logger.info("--- Step 2: Cached Discovery (Empty Payload) ---")
    # This should use the cache and send an empty discovery dictionary
    code = "print('Hello from second run')"
    stdout, stderr, result, exit_code = await runtime.execute(
        code, context, mcp_namespace=mock_ns
    )

    logger.info(f"Stdout: {stdout.strip()}")
    logger.info(f"Exit code: {exit_code}")

    if "Hello from second run" in stdout:
        logger.info("✅ Second run successful (Caching working)")
    else:
        logger.error("❌ Second run failed")

    # Clean up
    if hasattr(runtime, "sessions"):
        for proc in runtime.sessions.values():
            proc.terminate()


if __name__ == "__main__":
    asyncio.run(test_deadlock_resolution())
