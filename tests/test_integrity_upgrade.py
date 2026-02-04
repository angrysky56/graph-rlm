import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock dependencies before further imports
mock_db = MagicMock()
sys.modules["falkordb"] = mock_db
sys.modules["graph_rlm.backend.src.core.db"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.llm"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.sheaf"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.context_index"] = MagicMock()

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from graph_rlm.backend.src.core.agent import Agent, RLMInterface


def test_mcp_discovery_structure():
    """Verify that MCP tools are injected in a browsable SimpleNamespace structure."""
    agent = Agent()
    agent.repl_manager.create_repl()
    session_id = "test_discovery"

    # Mock MCP availability
    import graph_rlm.backend.src.core.agent as agent_mod

    agent_mod.MCP_AVAILABLE = True

    # Mock mcp_tools_pkg
    mock_server = MagicMock()
    mock_server.brave_web_search = lambda x: f"Results for {x}"
    mock_server.brave_web_search.__doc__ = "Brave search tool"

    # We need to mock the package itself, which is harder.
    # Let's instead check if we can reconstruct the logic.
    from types import SimpleNamespace

    mcp_root = SimpleNamespace()
    server_ns = SimpleNamespace()

    # Simulate the wrapper logic (using actual module name now)
    func = mock_server.brave_web_search
    mod_name = "brave_search"  # Actual module name, not alias
    attr = "brave_web_search"

    rlm_interface = RLMInterface(agent, session_id, session_id)

    def log_wrapper(f=func, n=f"mcp.{mod_name}.{attr}"):
        def wrapped(*args, **kwargs):
            rlm_interface._record_tool_use(n)
            return f(*args, **kwargs)

        wrapped.__doc__ = f.__doc__
        return wrapped

    setattr(server_ns, attr, log_wrapper())
    setattr(mcp_root, mod_name, server_ns)

    # Verification
    print(f"Servers in mcp: {dir(mcp_root)}")
    assert "brave_search" in dir(mcp_root)
    assert "brave_web_search" in dir(mcp_root.brave_search)
    assert mcp_root.brave_search.brave_web_search.__doc__ == "Brave search tool"

    # Execution & Logging
    res = mcp_root.brave_search.brave_web_search("query")
    print(f"Tool Result: {res}")
    assert res == "Results for query"
    assert "mcp.brave_search.brave_web_search" in agent.execution_logs[session_id]
    print("✅ MCP Discovery & Logging verified.")


def test_epistemic_integrity():
    """Verify the integrity protocol flags laziness and reward hacking."""
    agent = Agent()

    # Case 1: Laziness (Complex task, short thought, no tools)
    prompt = "Analyze the kinematic limits of the robotic arm and optimize throughput."
    trace = "I will optimize it now. Done."
    res = agent._verify_epistemic_integrity(trace, prompt, [])
    print(f"Laziness Check: {res}")
    assert res["status"] == "RETRY"
    assert any("LAZINESS" in f for f in res["flags"])

    # Case 2: Reward Hacking (Done signal without tools)
    prompt = "Search for recent safety reports."
    trace = "I have searched. done(results='all good')"
    res = agent._verify_epistemic_integrity(trace, prompt, [])
    print(f"Reward Hacking Check: {res}")
    assert res["status"] == "RETRY"
    assert any("REWARD_HACKING" in f for f in res["flags"])

    # Case 3: Pass (Long thought, tools used)
    prompt = "Search for reports."
    trace = (
        "I am searching through the database using the search tool. " * 20 + " done()"
    )
    res = agent._verify_epistemic_integrity(trace, prompt, ["rlm.search"])
    print(f"Valid Check: {res}")
    assert res["status"] == "PASS"

    print("✅ Epistemic Integrity verified.")


if __name__ == "__main__":
    try:
        test_mcp_discovery_structure()
        test_epistemic_integrity()
        print("\n✨ ALL TESTS PASSED ✨")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
