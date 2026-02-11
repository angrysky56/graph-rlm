"""
Epistemic Renormalization Protocol v3 Skill.

Calculates Renormalization Pressure (sigma) within the knowledge graph
using exponential structural penalties based on node centrality and type.
"""

import logging
import re
from typing import Any, Optional

from graph_rlm.backend.mcp_tools import call_tool

logger = logging.getLogger("graph_rlm.skills.erp_v3")


async def epistemic_renormalization_protocol_v3(
    mode: str = "scan",
    target_node_id: Optional[str] = None,
    vcn_type: str = "TYPE_EPISTEMIC",
) -> Any:
    """
    Weighted ERP with integrated descendant counting.
    Calculates Renormalization Pressure (sigma) using exponential structural penalties.

    Args:
        mode: Operation mode ('scan', 'launder', 'decay').
        target_node_id: Specific node to target for laundering/decay.
        vcn_type: The category of the Vacuum Causal Node.

    Returns:
        Analysis results for 'scan', or action confirmation for other modes.
    """
    if mode == "scan":
        # 1. Get all concepts and VCNs
        try:
            all_nodes = await call_tool(
                "reflective-agent-architecture",
                "inspect_graph",
                {"mode": "nodes", "label": "Concept"},
            )
            vcns = await call_tool(
                "reflective-agent-architecture",
                "inspect_graph",
                {"mode": "nodes", "label": "VCN"},
            )
        except RuntimeError as e:
            logger.error("Failed to inspect graph for ERP scan: %s", e)
            return {"error": str(e), "weighted_sigma": 1.0, "status": "UNKNOWN"}

        total_pressure = 0.0
        total_capacity = max(len(all_nodes), 10)

        details = []
        for vcn in vcns:
            v_str = str(vcn)
            # Determine weight based on vcn_type or internal logic
            # TYPE_ONTOLOGICAL is a structural debt, TYPE_EPISTEMIC is a logical gap.
            current_type = "TYPE_ONTOLOGICAL" if "ONTOLOGICAL" in v_str else vcn_type
            weight = 0.1 if current_type == "TYPE_ONTOLOGICAL" else 1.5

            # Calculate centrality (outgoing connections)
            dependents = 0
            try:
                match = re.search(r"element_id='([^']+)'", v_str)
                v_id = match.group(1) if match else None

                if v_id:
                    rels = await call_tool(
                        "reflective-agent-architecture",
                        "inspect_graph",
                        {
                            "mode": "relationships",
                            "start_id": v_id,
                            "direction": "OUTGOING",
                        },
                    )
                    dependents = len(rels)
            except RuntimeError:
                dependents = 0
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Unexpected error checking dependents for %s: %s", vcn, e
                )

            # ERP formula: sigma = weight * 1.1^dependents
            structural_cost = float(1.1**dependents)
            node_pressure = weight * structural_cost
            total_pressure += node_pressure

            details.append(
                {
                    "type": current_type,
                    "dependents": dependents,
                    "pressure": node_pressure,
                }
            )

        sigma = min(total_pressure / total_capacity, 1.0)
        return {
            "weighted_sigma": sigma,
            "status": "CRITICAL" if sigma > 0.8 else "STABLE",
            "vcn_count": len(vcns),
            "details": details,
        }

    if mode == "launder":
        # Logic Laundering: Persistence of a cognitive intervention
        try:
            await call_tool(
                "reflective-agent-architecture",
                "teach_cognitive_state",
                {"label": f"VCN_MINTED_{vcn_type}"},
            )
            return {"action": "VCN_CREATED", "node": target_node_id, "type": vcn_type}
        except RuntimeError as e:
            return {"error": f"Laundering failed: {e}"}

    if mode == "decay":
        # Resolve Epistemic Debt through verification cycles
        try:
            await call_tool(
                "reflective-agent-architecture", "run_sleep_cycle", {"epochs": 2}
            )
            return "Decay cycle triggered successfully."
        except RuntimeError as e:
            return {"error": f"Decay trigger failed: {e}"}

    return "Invalid mode. Use 'scan', 'launder', or 'decay'."
