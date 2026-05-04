import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rlm.backend.src.core.agent import Agent


async def test_tool_recovery():
    print("🚀 Starting Tool Recovery Verification...")

    # Mock dependencies
    with patch("graph_rlm.backend.src.core.agent.db"), patch(
        "graph_rlm.backend.src.core.agent.llm"
    ), patch("graph_rlm.backend.src.core.agent.sheaf"), patch(
        "graph_rlm.backend.src.core.agent.context_index"
    ):

        agent = Agent()

        # 1. Test RLMInterface Async Conversion
        print("\n📝 Testing rlm.recall (async)...")
        # Mock embedding return
        agent.llm.get_embedding.return_value = [0.1] * 1536
        agent.db.find_similar_thoughts.return_value = [
            {"id": "t1", "prompt": "p1", "result": "r1", "score": 0.9}
        ]

        code = """
import asyncio
async def run():
    res = await rlm.recall('test query')
    return res
print(asyncio.run(run()))
"""
        # Note: We use asyncio.run(run()) in the agent's REPL, but the agent's execute code
        # might need to handle the top-level await if we want to be modern.
        # Actually, our agent._execute_code wraps the code.

        output = await agent._execute_code(code, "test_thought", "test_session")
        print(f"Recall Output: {output}")
        assert "Similarity: 0.90" in output, "Recall result mismatch"

        # 2. Test MCP Tool Access (via actual module name, not alias)
        print("\n📝 Testing mcp.brave_search.brave_web_search() (correct pattern)...")
        # Mock an MCP tool
        mock_tool = MagicMock()
        mock_tool.return_value = asyncio.Future()
        mock_tool.return_value.set_result({"results": ["brave result"]})

        # Simulate browser search tool
        with patch("pkgutil.iter_modules") as mock_pkg:
            mock_pkg.return_value = [(None, "brave_search", False)]
            with patch("importlib.import_module") as mock_import:
                mock_mod = MagicMock()
                mock_mod.brave_web_search = mock_tool
                mock_import.return_value = mock_mod

                # Re-run _execute_code to trigger discovery in a fresh REPL
                code_mcp = """
import asyncio
async def run():
    return await mcp.brave_search.brave_web_search(query='news')
asyncio.run(run())
"""
                output_mcp = await agent._execute_code(
                    code_mcp, "mcp_thought", "mcp_session"
                )
                print(f"MCP Output: {output_mcp}")
                # The result variable will be in the namespace if we don't return it
                # Our _execute_code returns the captured stdout or the result of eval.
                # Since result = ... is an assignment, it doesn't return anything.
                # We should check if the tool was called.
                assert mock_tool.called, "MCP tool was not called through alias"

    print("\n✅ Tool Recovery Verified Successfully!")


if __name__ == "__main__":
    asyncio.run(test_tool_recovery())
