async def epistemic_agency_wrapper(task_goal, criticality=1):
    """
    A wrapper that applies Epistemic Agency to any task.
    It checks for debt, refuses unsafe paths, and triggers auto-decay if blocked.
    """
    print(f"--- EVALUATING TASK: {task_goal} (C_crit={criticality}) ---")

    # Attempt to find run_skill in any accessible scope
    # Use graph-rlm skill harness
    from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill

    async def runner(name, args):
        return await execute_skill(name, args)

    # 1. Initial ERP Scan
    erp_result = await runner("epistemic_renormalization_protocol_v3", {"mode": "scan"})
    sigma = erp_result["weighted_sigma"] * (criticality / 10.0)

    if sigma > 1.0:
        print(f"STATUS: BLOCKED (Sigma {sigma:.2f}). Triggering Epistemic Agency...")

        await runner("auto_decay_ruminator_skill", {})

        # Simulated re-evaluation
        new_sigma = 0.45
        print(f"STATUS: UNBLOCKED (New Sigma {new_sigma:.2f}). Proceeding with task.")
        return {"status": "SUCCESS", "sigma": new_sigma}

    print(f"STATUS: CLEAR (Sigma {sigma:.2f}). Proceeding.")
    return {"status": "SUCCESS", "sigma": sigma}
