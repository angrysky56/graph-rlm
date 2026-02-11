"""
Commit Epistemic Victory Skill.

The 'Commit Protocol' for Epistemic Victories. Serializes verified logic to
ChatDAG, Vector Memory, and the Reflective Agent Architecture.
"""

import logging

from graph_rlm.backend.mcp_tools import call_tool

logger = logging.getLogger("graph_rlm.skills.commit_epistemic_victory")


async def commit_epistemic_victory_skill(node_id: str, verification_trace: str) -> str:
    """
    Executes the 'Commit Protocol' to crystallize a verified epistemic victory.

    Args:
        node_id: The ID of the node being verified.
        verification_trace: The trace or proof of the verification.

    Returns:
        A status string indicating success.
    """
    print(f"--- EXECUTING COMMIT PROTOCOL: {node_id} ---")
    verified_logic = f"CAUSAL_MODEL_VERIFIED: {node_id}\nTrace: {verification_trace}"

    # 1. ChatDAG
    print("Action: Serializing verified logic to ChatDAG...")
    try:
        await call_tool(
            "chatdag",
            "feed_data",
            {
                "content": verified_logic,
                "source_id": f"golden_graph/{node_id}",
                "metadata": {"tags": ["verified", "epistemic_victory"]},
            },
        )
    except RuntimeError as e:
        logger.error("Failed to feed data to ChatDAG for %s: %s", node_id, e)

    # 2. Vector Memory
    print("Action: Committing to Vector Memory...")
    try:
        await call_tool(
            "memory",
            "save_memory",
            {
                "text": f"The component '{node_id}' is verified. Logic: {verification_trace}",
                "metadata": {"type": "GOLDEN_ASSET", "node_id": node_id},
            },
        )
    except RuntimeError as e:
        logger.error("Failed to save memory for %s: %s", node_id, e)

    # 3. RAA
    print("Action: Tagging node in RAA...")
    try:
        await call_tool(
            "reflective-agent-architecture",
            "teach_cognitive_state",
            {"label": f"GOLDEN_ASSET_{node_id}"},
        )
    except RuntimeError as e:
        logger.error("Failed to teach cognitive state for %s: %s", node_id, e)

    print(f"--- COMMIT COMPLETE: {node_id} is now processed ---")
    return "ASSET_CRYSTALLIZED"
