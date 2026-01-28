from graph_rlm.backend.mcp_tools.chatdag import search_knowledge
from graph_rlm.backend.mcp_tools.desktop_commander import write_file


async def auto_crystallize_project_insights(
    project_id: str, query: str = "key insights"
) -> dict:
    """
    Automates the 'Crystallization' of recent chat history and tool outputs
    into a structured project knowledge file.

    1. Searches ChatDAG for recent voxels related to the project.
    2. Saves the raw synthesis of these voxels to a project knowledge file.
    """
    # 1. Gather recent voxels
    voxels = await search_knowledge(query=f"project: {project_id} {query}")
    context = "\n".join([str(v) for v in voxels])

    # 2. Save to Project Knowledge Base
    filename = f"crystallized_insights_{project_id}.md"
    path = f"/home/ty/Repositories/ai_workspace/mcp_coordinator/knowledge_base/projects/{project_id}/{filename}"

    # We use a simple markdown wrapper since we are doing direct synthesis here
    content = f"# Crystallized Insights: {project_id}\n\n## Recent Voxels\n{context}"

    await write_file(path=path, content=content)

    return {
        "status": "success",
        "file_path": path,
        "insight_summary": "Insights crystallized and saved to project folder.",
    }
