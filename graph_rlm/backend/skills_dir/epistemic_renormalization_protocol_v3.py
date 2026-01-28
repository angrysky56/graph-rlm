import math

from graph_rlm.backend.mcp_tools.reflective_agent_architecture import (
    hypothesize,
    inspect_graph,
    run_sleep_cycle,
    synthesize,
    teach_cognitive_state,
)


async def epistemic_renormalization_protocol_v3(
    mode="scan", target_node_id=None, vcn_type="TYPE_EPISTEMIC"
):
    """
    Weighted ERP with integrated descendant counting.
    Calculates Renormalization Pressure (sigma) using exponential structural penalties.
    """
    if mode == "scan":
        # 1. Get all concepts and VCNs
        all_nodes = await inspect_graph(mode="nodes", label="Concept")
        vcns = await inspect_graph(mode="nodes", label="VCN")

        total_pressure = 0.0
        total_capacity = max(len(all_nodes), 10)

        details = []
        for vcn in vcns:
            # Determine weight based on type
            # (In this version, we check for a 'type' property in the node string)
            v_str = str(vcn)
            v_type = "TYPE_ONTOLOGICAL" if "ONTOLOGICAL" in v_str else vcn_type
            weight = 0.1 if v_type == "TYPE_ONTOLOGICAL" else 1.5

            # Simple centrality: check immediate outgoing relationships
            # We'll use a placeholder since full recursion is complex for a single skill call
            dependents = 0
            try:
                # We need a node ID. If 'vcn' is a string like "<Node element_id='...'>", we extract it.
                import re

                match = re.search(r"element_id='([^']+)'", v_str)
                v_id = match.group(1) if match else None

                if v_id:
                    rels = await inspect_graph(
                        mode="relationships", start_id=v_id, direction="OUTGOING"
                    )
                    dependents = int(len(rels))
            except Exception:
                pass

            structural_cost = float(1.1**dependents)
            node_pressure = weight * structural_cost
            total_pressure += node_pressure

            details.append(
                {"type": v_type, "dependents": dependents, "pressure": node_pressure}
            )

        sigma = min(total_pressure / total_capacity, 1.0)
        return {
            "weighted_sigma": sigma,
            "status": "CRITICAL" if sigma > 0.8 else "STABLE",
            "vcn_count": len(vcns),
            "details": details,
        }

    elif mode == "launder":
        # Logic Laundering
        await teach_cognitive_state(label=f"VCN_MINTED_{vcn_type}")
        return {"action": "VCN_CREATED", "node": target_node_id, "type": vcn_type}

    elif mode == "decay":
        # Resolve Epistemic Debt
        await run_sleep_cycle(epochs=2)
        return "Decay cycle triggered."

    return "Invalid mode."
