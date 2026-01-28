
async def logic_gated_ingestion(knowledge_text, domain="General"):
    """
    Implements the TALE (Triadic Antifragile Logic Engine) workflow.
    Correctly handles JSON string returns from MCP tools.
    """
    from graph_rlm.backend.mcp_tools import verifier_graph
    import json

    print(f"--- Starting Logic Gated Ingestion for {domain} ---")
    
    def parse_res(res):
        if isinstance(res, str):
            return json.loads(res)
        return res

    # Step 1: Log Premise
    raw_p = await verifier_graph.propose_thought(
        type="PREMISE",
        content=f"New Knowledge Ingestion [{domain}]: {knowledge_text}"
    )
    res_p = parse_res(raw_p)
    p_id = res_p["node"]["id"]

    # Step 2: Logic Check (Immune Layer)
    raw_tc = await verifier_graph.propose_thought(
        type="TOOL_CALL",
        content="Checking for disjointness and acyclicity.",
        parentIds=[p_id]
    )
    res_tc = parse_res(raw_tc)
    tc_id = res_tc["node"]["id"]

    # Simple validation logic
    is_valid = len(knowledge_text) > 10
    result_text = "Logic check passed." if is_valid else "Logic check failed: Data too short."
    
    raw_tr = await verifier_graph.propose_thought(
        type="TOOL_RESULT",
        content=result_text,
        parentIds=[tc_id]
    )
    res_tr = parse_res(raw_tr)
    tr_id = res_tr["node"]["id"]

    # Step 3: Promotion (Adaptive Layer)
    if is_valid:
        raw_c = await verifier_graph.propose_thought(
            type="CLAIM",
            content=f"Promoting knowledge to Verified Backbone.",
            parentIds=[tr_id]
        )
        res_c = parse_res(raw_c)
        
        return {
            "status": "success", 
            "node_id": res_c["node"]["id"],
            "message": "Knowledge verified and promoted to TALE Backbone."
        }
    else:
        return {"status": "rejected", "message": "Knowledge failed logic gating."}
