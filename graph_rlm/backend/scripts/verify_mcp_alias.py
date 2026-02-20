import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.absolute()
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from graph_rlm.backend.src.core.mcp_runtime import LazyMCPNamespace


async def verify_alias():
    # We need a mock RLMInterface or just enough to satisfy the lazy loader
    class MockRLM:
        def record_tool_use(self, name):
            print(f"Tool used: {name}")

    rlm = MockRLM()
    mcp = LazyMCPNamespace(rlm)

    print("Listing servers...")
    servers = mcp.list_servers()
    print(f"Servers: {servers}")

    if "advanced_reasoning" in servers:
        print("\nChecking advanced_reasoning server...")
        server = mcp.advanced_reasoning
        tools = dir(server)
        print(f"Tools in advanced_reasoning: {tools}")

        print("\nChecking for 'analyze' alias...")
        try:
            analyze_tool = server.analyze
            print("✅ 'analyze' alias found!")
            # Check if it's the same as advanced_reasoning (internally they should both be wrappers)
            # We can't easily compare the functions directly but we successfully retrieved it
        except AttributeError as e:
            print(f"❌ 'analyze' alias NOT found: {e}")
    else:
        print("\n❌ advanced_reasoning server not found in discovered tools.")

if __name__ == "__main__":
    asyncio.run(verify_alias())
