async def commit_epistemic_victory_skill(node_id, verification_trace):
    """
    REPAIRED (v5): The 'Commit Protocol' for Epistemic Victories.

    This version is completely self-contained and uses 'call_mcp_tool'
    from the runtime to avoid any global scope issues.
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    print(f"--- EXECUTING REPAIRED COMMIT PROTOCOL: {node_id} ---")
    verified_logic = f"CAUSAL_MODEL_VERIFIED: {node_id}\nTrace: {verification_trace}"

    # 1. ChatDAG
    print("Action: Serializing verified logic to ChatDAG...")
    await call_mcp_tool("chatdag", "feed_data", {
        "content": verified_logic,
        "source_id": f"golden_graph/{node_id}",
        "metadata": {"tags": ["verified", "epistemic_victory"]}
    })

    # 2. Vector Memory
    print("Action: Committing to Vector Memory...")
    await call_mcp_tool("memory", "save_memory", {
        "text": f"The component '{node_id}' is verified. Logic: {verification_trace}",
        "metadata": {"type": "GOLDEN_ASSET", "node_id": node_id}
    })

    # 3. RAA
    print("Action: Tagging node in RAA...")
    await call_mcp_tool("reflective-agent-architecture", "teach_cognitive_state", {
        "label": f"GOLDEN_ASSET_{node_id}"
    })

    print(f"--- COMMIT COMPLETE: {node_id} is now processed ---")
    return "ASSET_CRYSTALLIZED"
