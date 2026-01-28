from graph_rlm.backend.mcp_tools.chatdag import search_knowledge
from graph_rlm.backend.mcp_tools.reflective_agent_architecture import inspect_graph


async def hybrid_cognitive_search(query: str, project_id: str = None) -> dict:
    """
    Performs a hybrid search across the Semantic Layer (ChatDAG)
    and the Structural Layer (Neo4j/RAA).

    Args:
        query: The search term or concept to investigate.
        project_id: Optional project context to narrow the search.

    Returns:
        A dictionary containing fused results from both memory systems.
    """
    results = {"semantic_voxels": [], "structural_nodes": [], "synthesis": ""}

    # 1. Semantic Search (ChatDAG)
    # We search for voxels that resonate with the query
    try:
        chatdag_res = await search_knowledge(query=query)
        # ChatDAG returns a list of TextContent or similar; we extract the text
        results["semantic_voxels"] = [str(v) for v in chatdag_res]
    except Exception as e:
        results["semantic_voxels"] = [f"ChatDAG search failed: {e}"]

    # 2. Structural Search (Neo4j via RAA)
    # We look for nodes in the thought graph with matching labels or properties
    try:
        # Search for nodes that match the query in their content or labels
        graph_res = await inspect_graph(
            mode="nodes",
            filters={"content": query},  # Simplified filter for demonstration
        )
        results["structural_nodes"] = graph_res
    except Exception as e:
        results["structural_nodes"] = [f"Neo4j search failed: {e}"]

    return results
