async def crystallize_epistemic_victory(node_id, verification_trace, project_id=None):
    """
    Crystallizes a research breakthrough (Epistemic Victory) into the system's long-term memory.

    Args:
        node_id: Unique identifier for the breakthrough (e.g., 'fractal_bottleneck_v1')
        verification_trace: Detailed summary of the logic and evidence.
        project_id: Optional project to add a note to.
    """
    from graph_rlm.backend.mcp_tools import save_memory, call_tool, update_project

    print(f"--- CRYSTALLIZING EPISTEMIC VICTORY: {node_id} ---")

    # 1. Save to Vector Memory
    print("Action: Committing to Vector Memory...")
    await save_memory(
        text=f"EPISTEMIC VICTORY: {node_id}\nLogic: {verification_trace}",
        metadata={"type": "GOLDEN_ASSET", "node_id": node_id, "category": "research_breakthrough"}
    )

    # 2. Feed to ChatDAG Knowledge Graph
    print("Action: Serializing to ChatDAG...")
    await call_tool("chatdag", "feed_data", {
        "content": f"VERIFIED_RESEARCH_NODE: {node_id}\nTrace: {verification_trace}",
        "filename": f"golden_assets/{node_id}.md",
        "metadata": {"tags": ["verified", "epistemic_victory", "research"]}
    })

    # 3. Update Project Notes if project_id is provided
    if project_id:
        print(f"Action: Updating Project {project_id} notes...")
        await update_project(
            project_id=project_id,
            add_note=f"EPISTEMIC VICTORY [{node_id}]: {verification_trace[:200]}..."
        )

    # 4. Tag in RAA if available
    try:
        print("Action: Notifying Reflective Agent Architecture...")
        await call_tool("reflective-agent-architecture", "teach_cognitive_state", {
            "label": f"GOLDEN_ASSET_{node_id}"
        })
    except Exception as e:
        print(f"RAA notification skipped: {e}")

    print(f"--- CRYSTALLIZATION COMPLETE: {node_id} ---")
    return f"Asset {node_id} successfully crystallized."
