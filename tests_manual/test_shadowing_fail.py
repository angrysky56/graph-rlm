import asyncio

from graph_rlm.backend.mcp_tools.neo4j_mcp import read_cypher


async def test_shadowing_bug():
    print("Testing neo4j_mcp.read_cypher shadowing bug...")

    query = "MATCH (n {id: $id}) RETURN n"
    parameters = {"id": "test-id"}

    try:
        # This will call call_mcp_tool under the hood.
        # Since it uses asyncio.run() or direct call, we need to mock call_mcp_tool or just observe behavior.
        # But wait, if I run this, I don't have the MCP server running.
        # I can check the code instead, or use a mock.

        from unittest.mock import patch
        with patch('graph_rlm.backend.src.mcp_integration.runtime.call_mcp_tool') as mock_call:
            mock_call.return_value = {"content": "mocked"}

            # Use asyncio.run if sync, or await if async.
            # The generated wrapper handles both.
            read_cypher(query, parameters)

            # Check what was actually sent
            args, kwargs = mock_call.call_args
            sent_args = kwargs.get('arguments', {})
            print(f"Sent Arguments: {sent_args}")

            if "params" in sent_args and sent_args["params"] == parameters:
                print("SUCCESS: Parameters preserved.")
            else:
                print(f"FAILED: Parameters lost or malformed. Found: {sent_args.get('params')}")

    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    asyncio.run(test_shadowing_bug())
