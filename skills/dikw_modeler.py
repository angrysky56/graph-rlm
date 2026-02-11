"""
DIKW Modeler Skill.

Structures raw data into Data, Information, Knowledge, and Wisdom
layers and generates a visual flowchart.
"""

import json
import logging
from typing import Any, Dict

from graph_rlm.backend.mcp_tools import call_tool

logger = logging.getLogger("graph_rlm.skills.dikw_modeler")


async def dikw_modeler(data_input: str) -> Dict[str, Any]:
    """
    Takes raw data, uses verifier-graph to structure it into DIKW layers,
    and then generates a visual diagram using diagram-server.

    Args:
        data_input: The raw content to analyze and model.

    Returns:
        A dictionary containing the logical analysis and the diagram output.
    """
    # 1. Verifier-Graph - Propose DIKW Structure
    vg_prompt = (
        f"Analyze the following input and categorize it into "
        f"DIKW (Data, Information, Knowledge, Wisdom) layers: {data_input}"
    )

    try:
        # Note: Using 'content' for the thought text and 'type' for logical category
        thought_res = await call_tool(
            "verifier-graph",
            "propose_thought",
            {"content": vg_prompt, "type": "PREMISE"},
        )
    except RuntimeError as e:
        logger.error("Failed to propose DIKW thought: %s", e)
        thought_res = {"error": str(e)}

    # 2. Transformation
    # Creating a standardized DIKW flowchart structure
    dikw_structure = {
        "nodes": [
            {"id": "D", "label": "Data"},
            {"id": "I", "label": "Information"},
            {"id": "K", "label": "Knowledge"},
            {"id": "W", "label": "Wisdom"},
        ],
        "edges": [
            {"from": "D", "to": "I"},
            {"from": "I", "to": "K"},
            {"from": "K", "to": "W"},
        ],
    }

    # 3. Diagram-Server - Create Visual
    try:
        diagram_res = await call_tool(
            "diagram-server",
            "create_diagram",
            {
                "name": "DIKW_Model",
                "content": json.dumps(dikw_structure),
                "diagram_type": "flowchart",
            },
        )
    except RuntimeError as e:
        logger.error("Failed to create DIKW diagram: %s", e)
        diagram_res = {"error": str(e)}

    return {"analysis": thought_res, "diagram": diagram_res}
