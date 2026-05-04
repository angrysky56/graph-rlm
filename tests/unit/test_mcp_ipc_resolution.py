import json

# Mock DB dependencies to prevent connection errors
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.modules["falkordb"] = MagicMock()
sys.modules["redis"] = MagicMock()

try:
    from graph_rlm.backend.src.core.mcp_runtime import LazyMCPNamespace
    from graph_rlm.backend.src.mcp_integration.runtime import AgentRuntime
except ImportError:
    try:
        from src.core.mcp_runtime import LazyMCPNamespace
        from src.mcp_integration.runtime import AgentRuntime
    except ImportError:
        pass


class TestMCPDiscoveryIPC:

    @pytest.fixture
    def mock_mcp_namespace(self):
        """Creates a mock LazyMCPNamespace with predictable tools."""
        mock_ns = MagicMock(spec=LazyMCPNamespace)

        # Mocking the discovery structure
        # The runtime iterates dir(ns) then dir(server)

        # 1. Server Object
        mock_server = MagicMock()
        mock_server.__doc__ = "Mock Server Doc"

        # 2. Tool Object
        mock_tool = MagicMock()
        mock_tool.__doc__ = "Returns the input string."
        mock_tool.return_value = "ECHO: hello world"

        # 3. Structure
        setattr(mock_server, "echo", mock_tool)
        setattr(mock_ns, "mock_server", mock_server)

        # 4. Mock __dir__ for discovery
        mock_ns.__dir__ = MagicMock(return_value=["mock_server"])
        mock_server.__dir__ = MagicMock(return_value=["echo"])

        return mock_ns

    @pytest.mark.anyio
    async def test_ipc_discovery_injection(self, mock_mcp_namespace):
        """
        GREEN TEST: Verify that dir(mcp) and docstrings work inside the agent.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        runtime = AgentRuntime(project_root)

        code = """
import json
try:
    # 1. Check Server List
    servers = dir(mcp)
    print(f"SERVERS: {servers}")

    # 2. Check Tool Docstring
    if "mock_server" in servers:
        doc = mcp.mock_server.echo.__doc__
        print(f"DOC: {doc}")
    else:
        print("Mock server not found in dir(mcp)")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {e}")
"""
        context = {"session_id": "test", "thought_id": "test"}

        stdout, stderr, _result, exit_code = await runtime.execute(
            code, context=context, mcp_namespace=mock_mcp_namespace
        )

        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")

        assert exit_code == 0, f"Execution failed: {stderr}"
        assert "'mock_server'" in stdout or "['mock_server']" in stdout
        assert "Returns the input string." in stdout

    @pytest.mark.anyio
    async def test_ipc_resolution_via_namespace(self, mock_mcp_namespace):
        """
        Test that IPC requests are routed through the LazyMCPNamespace on the host.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        runtime = AgentRuntime(project_root)

        code = """
try:
    res = await mcp.mock_server.echo("hello world")
    print(f"RESULT: {res}")
except Exception as e:
    print(f"ERROR: {e}")
"""

        # Configure the mock to return "ECHO: hello world"
        mock_tool = mock_mcp_namespace.mock_server.echo
        # Ensure it behaves like a sync function for simplicity or async if needed
        # The runtime handles both.

        stdout, stderr, _result, exit_code = await runtime.execute(
            code, context={}, mcp_namespace=mock_mcp_namespace
        )

        print(f"STDOUT: {stdout}")

        assert "RESULT: ECHO: hello world" in stdout

        # Verify the mock was called on the HOST side
        mock_tool.assert_called_once_with("hello world")
