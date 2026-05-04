import asyncio
import sys

from graph_rlm.backend.src.core.agent import Agent


async def test_mcp_reflection():
    print("🚀 Starting MCP Reflection Verification...")

    agent = Agent()
    session_id = "test_reflection"

    # We'll use a simple mock-like check via the REPL
    # We need to ensure the mcp object exists and has the servers as attributes
    code = """
import asyncio

async def verify():
    # 1. Check dir(mcp)
    servers = dir(mcp)
    print(f"SERVERS: {servers}")

    # 2. Check a specific server if it exists (e.g., 'brave_search' - actual module name, not alias)
    if 'brave_search' in servers:
        tools = dir(mcp.brave_search)
        print(f"BRAVE_SEARCH TOOLS: {tools}")
        # The actual function is brave_web_search or search (alias)
        if hasattr(mcp.brave_search, 'search'):
            doc = mcp.brave_search.search.__doc__
            print(f"BRAVE_SEARCH SEARCH DOC: {doc[:50]}..." if doc else "No doc")
        elif hasattr(mcp.brave_search, 'brave_web_search'):
            doc = mcp.brave_search.brave_web_search.__doc__
            print(f"BRAVE_SEARCH BRAVE_WEB_SEARCH DOC: {doc[:50]}..." if doc else "No doc")

    return "SUCCESS"

result = await verify()
print(f"Result: {result}")
"""

    output_text, failed = await agent._execute_code(code, "thought_1", session_id)
    print(f"Output:\n{output_text}")

    if not failed and "SERVERS:" in output_text and "Result: SUCCESS" in output_text:
        print("✅ MCP Reflection verification passed!")
        return True
    else:
        print("❌ MCP Reflection verification failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mcp_reflection())
    sys.exit(0 if success else 1)
