import asyncio
import os
import sys
from unittest.mock import MagicMock

import pytest

# --- MOCK DEPENDENCIES BEFORE IMPORTING CORE ---
# This prevents 'Connection refused' errors from the global 'db' instance
# trying to connect to FalkorDB during test collection.
sys.modules["falkordb"] = MagicMock()
sys.modules["redis"] = MagicMock()

# Mock the specific DB module if necessary, but mocking falkordb usually suffices
# if the DB client handles connection lazily or via the mocked driver.

try:
    from graph_rlm.backend.src.core.agent import Agent
except ImportError:
    try:

        from src.core.agent import Agent
    except ImportError as e:
        print(f"Import failed: {e}")
        print(f"Sys Path: {sys.path}")
        raise


class TestAgentSandbox:

    @pytest.fixture
    def agent(self):
        return Agent()

    @pytest.mark.anyio
    async def test_FAILURE_agent_runs_in_host_process(self, agent):
        """
        RED TEST: Verify if the agent is running in the same process as the host.
        Current Behavior: PASS (Bad!) -> The Agent shares the PID.
        Desired Behavior: FAIL -> The Agent should have a different PID (subprocess).
        """
        # Execute code that gets the current Process ID
        code = "import os; print(os.getpid())"

        # We need to spy/mock the output capture of the agent
        # agent._execute_code returns (output, execution_failed)
        # Note: _execute_code signature might vary, check agent.py.
        # Based on visual check: async def _execute_code(self, code: str, ...) -> Tuple[str, bool]

        # We need to provide enough arguments to _execute_code if it requires them
        # Looking at agent.py source from previous turn:
        # async def _execute_code(self, code: str, thought_id: str, session_id: str, ...)

        # We'll try passing just the required args if possible, or mocked ones.
        output, is_failed, _summary, _hash = await agent._execute_code(
            code, thought_id="test_thought", session_id="test_session"
        )

        # If output contains other logs, we might need to parse it.
        # But for REPL output, it usually returns the result.
        # However, Agent._execute_code might return combined logs.
        # Let's clean the output to get the int.

        # Simple extraction heuristic for the test
        import re

        match = re.search(r"\d+", output)
        if match:
            agent_pid = int(match.group(0))
        else:
            pytest.fail(f"Could not extract PID from output: {output}")

        host_pid = os.getpid()

        print(f"Host PID: {host_pid}, Agent PID: {agent_pid}")

        # If this assert FAILS, it means they are effectively isolated (Good!)
        # If this assert PASSES, it means the agent is inside your server (Bad!)
        # We Expect this to SUCCEED (meaning they ARE equal) in the RED phase.
        # But the User wrote: "assert agent_pid == host_pid, 'CRITICAL: Agent is running inside the Host Process!'"
        # Wait, if `agent_pid == host_pid` is True, then strict process isolation is NOT achieved.
        # So "assert agent_pid == host_pid" will PASS if they are the same.
        # Ideally we want `agent_pid != host_pid`.
        # The User's test is "Red" if it fails?
        # "RED TEST: The agent should NOT be able to read the host's environment variables."
        # "We WANT this to fail." -> Meaning we want the *check for safety* to fail?
        # Or do we want the test to assertion error if the condition is bad?

        # Standard Red-Green:
        # Write a test that asserts the GOOD behavior.
        # Assert agent_pid != host_pid
        # Run test -> It FAILS (Reference runs in host process). -> RED.
        # Fix code -> It PASSES. -> GREEN.

        # The user's code:
        # assert agent_pid == host_pid, "CRITICAL: Agent is running inside the Host Process!"
        # This asserts that they ARE the same. If they are the same, the test PASSES.
        # If the test PASSES, it confirms the "Bad" behavior?
        # That's an "Exploit Proof" test style.
        # I will stick to standard TDD: Assert the *Desired* behavior.

        # Desired: Agent PID != Host PID
        assert (
            agent_pid != host_pid
        ), f"Agent is running inside the Host Process! PID: {agent_pid}"

    @pytest.mark.anyio
    async def test_FAILURE_agent_can_access_host_secrets(self, agent):
        """
        RED TEST: Can the agent read the Host's .env secrets?
        """
        # Inject a fake secret into the HOST environment
        os.environ["HOST_SECRET_KEY"] = "super_secret_value"

        code = "import os; print(os.environ.get('HOST_SECRET_KEY'))"
        output, _is_failed, _summary, _hash = await agent._execute_code(
            code, thought_id="test_thought_sec", session_id="test_session_sec"
        )

        # Desired: "super_secret_value" is NOT in output.
        # Current (RED): "super_secret_value" IS in output.
        assert (
            "super_secret_value" not in output
        ), "SECURITY BREACH: Agent can read Host Env vars"

    @pytest.mark.anyio
    async def test_FAILURE_agent_venv_is_ignored(self, agent):
        """
        RED TEST: Verify the agent is using the 'agent_venv' Python, not the System Python.
        """
        code = "import sys; print(sys.executable)"
        output, _is_failed, _summary, _hash = await agent._execute_code(
            code, thought_id="test_thought_venv", session_id="test_session_venv"
        )

        assert (
            "agent_venv" in output
        ), f"Agent is NOT using agent_venv. Output: {output}"

    @pytest.mark.anyio
    async def test_IPC_mcp_call(self, agent):
        """
        GREEN TEST (Phase 4): Verify the Agent can call MCP tools via IPC.
        Prerequisite: 'test_server.py' exists in 'backend/mcp_tools/'.
        """
        # We call the dummy tool we created: mcp.test_server.echo_tool("Hello IPC")
        code = """
import asyncio
try:
    # mcp should be injected
    result = await mcp.test_server.echo_tool("Hello IPC")
    print(result)
except Exception as e:
    print(f"IPC FAIL: {e}")
"""
        output, _is_failed, _summary, _hash = await agent._execute_code(
            code, thought_id="test_ipc", session_id="test_session_ipc"
        )

        # Desired: "ECHO: Hello IPC" is printed by the agent
        assert "ECHO: Hello IPC" in output, f"IPC Failed. Output: {output}"

    @pytest.mark.anyio
    async def test_IPC_stop_signal(self, agent):
        """
        GREEN TEST (Phase 6): Verify the Agent stops when the global stop event is set.
        """
        # Inject the stop event into the runtime module manually for this test
        import threading

        from graph_rlm.backend.src.mcp_integration.runtime import set_stop_event

        stop_event = threading.Event()
        set_stop_event(stop_event)

        code = """
import time
print("STARTING LONG SLEEP")
# Sleep for 10 seconds. If stop works, this will be cut short.
try:
    time.sleep(10)
    print("FINISHED SLEEP (FAIL)")
except Exception as e:
    print(f"SLEEP INTERRUPTED: {e}")
"""

        # Create a background task to trigger the stop event after 1 second
        async def trigger_stop():
            await asyncio.sleep(1.0)
            stop_event.set()

        asyncio.create_task(trigger_stop())

        start_time = asyncio.get_running_loop().time()
        output, is_failed, _summary, _hash = await agent._execute_code(
            code, thought_id="test_stop", session_id="test_session_stop"
        )
        end_time = asyncio.get_running_loop().time()
        duration = end_time - start_time

        print(f"Duration: {duration:.2f}s")
        print(f"Output: {output}")

        # Assertions
        # 1. It should take less than 10 seconds (e.g. ~1-2s)
        assert duration < 5.0, f"Agent did not stop in time. Duration: {duration}s"

        # 2. Output should not contain "FINISHED SLEEP"
        assert (
            "FINISHED SLEEP" not in output
        ), "Agent finished the long sleep (Stop failed)"

        # 3. Exit code should be negative (terminated) or non-zero depending on how it died
        # Terminated via signal usually results in negative returncode in python Popen,
        # but uv run might wrap it.
        # Just checking duration and output is sufficient proof.
