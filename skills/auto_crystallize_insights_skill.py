"""
Auto Crystallize Project Insights Skill.

Automates the 'Crystallization' of recent chat history and tool outputs
into structured project knowledge files.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from graph_rlm.backend.mcp_tools.chatdag import search_knowledge
from graph_rlm.backend.mcp_tools.desktop_commander import write_file

logger = logging.getLogger("graph_rlm.skills.auto_crystallize_insights")


async def auto_crystallize_project_insights(
    project_id: str, query: str = "key insights"
) -> Dict[str, Any]:
    """
    Automates the 'Crystallization' of recent chat history and tool outputs
    into a structured project knowledge file.

    1. Searches ChatDAG for recent voxels related to the project.
    2. Saves the raw synthesis of these voxels to a project knowledge file.

    Args:
        project_id: The identifier for the project.
        query: Additional search query to filter insights.

    Returns:
        A dictionary containing the status and the path to the saved file.
    """
    # 1. Gather recent voxels
    try:
        voxels = await search_knowledge(query=f"project: {project_id} {query}")
        context = "\n".join([str(v) for v in voxels])
    except RuntimeError as e:
        logger.error("Failed to gather voxels for crystallization: %s", e)
        return {"status": "error", "message": f"Search failed: {e}"}

    # 2. Save to Project Knowledge Base
    # Path construction - using relative logic or standardized base
    filename = f"crystallized_insights_{project_id}.md"

    # Standard knowledge base location in the user's workspace
    # Based on user_information: /home/ty/Repositories/ai_workspace/graph-rlm
    kb_base = Path(
        "/home/ty/Repositories/ai_workspace/graph-rlm/knowledge_base/projects"
    )
    project_dir = kb_base / project_id
    full_path = project_dir / filename

    # Ensure project_dir is handled - write_file tool usually handles parent creation
    content = (
        f"# Crystallized Insights: {project_id}\n\n"
        f"Generated: {Path(__file__).name}\n\n"
        f"## Recent Voxels\n{context}"
    )

    try:
        await write_file(path=str(full_path), content=content)
    except RuntimeError as e:
        logger.error("Failed to write crystallized insights: %s", e)
        return {"status": "error", "message": f"Write failed: {e}"}

    return {
        "status": "success",
        "file_path": str(full_path),
        "insight_summary": "Insights crystallized and saved to project folder.",
    }
