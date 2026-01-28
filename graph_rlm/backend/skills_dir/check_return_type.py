
async def check_return_type():
    from graph_rlm.backend.mcp_tools import verifier_graph
    res = await verifier_graph.propose_thought(type="PREMISE", content="Test")
    return f"Type: {type(res)}, Content: {res}"
