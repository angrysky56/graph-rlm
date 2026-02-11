"""
Diagnostic Skill: Check Return Type.

Simple utility to verify the return type and content of the verifier_graph tool response.
"""


async def check_return_type() -> str:
    """
    Executes a test thought proposal and returns the type and content of the response.

    Returns:
        A string describing the type and raw content of the tool response.
    """
    from graph_rlm.backend.mcp_tools import verifier_graph

    # Propose a simple test premise
    res = await verifier_graph.propose_thought(
        type="PREMISE", content="Diagnostic Test"
    )

    return f"Type: {type(res)}, Content: {res}"
