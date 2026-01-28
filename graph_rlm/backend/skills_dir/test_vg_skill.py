
async def test_vg_skill():
    from graph_rlm.backend.mcp_tools import verifier_graph
    return await verifier_graph.get_graph_state()
