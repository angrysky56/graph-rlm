"""
Hybrid Cognitive Search Skill.

Performs a fused search across semantic and structural memory layers.
"""

from typing import Any

from graph_rlm.backend.mcp_tools.chatdag import search_knowledge
from graph_rlm.backend.mcp_tools.reflective_agent_architecture import inspect_graph


async def hybrid_cognitive_search(query: str, project_id: Any = None) -> dict:
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
        # If project_id is provided, we incorporate it into the search query or context
        chatdag_query = f"{query} [context: {project_id}]" if project_id else query
        chatdag_res = await search_knowledge(query=chatdag_query)
        # ChatDAG returns a list of fragments; we extract strings
        results["semantic_voxels"] = [str(v) for v in chatdag_res]
    except RuntimeError as e:
        results["semantic_voxels"] = [f"ChatDAG search failed: {e}"]
    except Exception as e:  # noqa: BLE001
        results["semantic_voxels"] = [f"Unexpected semantic error: {e}"]

    # 2. Structural Search (Neo4j via RAA)
    # We look for nodes in the thought graph with matching labels or properties
    try:
        filters = {"content": query}
        if project_id:
            filters["project_id"] = project_id

        # Search for nodes that match the query in their content or labels
        graph_res = await inspect_graph(
            mode="nodes",
            filters=filters,
        )
        results["structural_nodes"] = graph_res
    except RuntimeError as e:
        results["structural_nodes"] = [f"Neo4j search failed: {e}"]
    except Exception as e:  # noqa: BLE001
        results["structural_nodes"] = [f"Unexpected structural error: {e}"]

    return results
