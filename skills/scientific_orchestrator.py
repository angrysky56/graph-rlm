from typing import Optional
import os


async def scientific_orchestrator(
    skill_name: Optional[str] = None,
    query: str = "",
    task: Optional[str] = None,
    goal: Optional[str] = None,
):
    """
    A pointer-based orchestrator for 140+ scientific skills.
    Logic: Probes metadata (SKILL.md) from the source repository.

    Supports both legacy signature (skill_name, query) and new signature (task, goal).
    task and goal map to skill_name and query respectively for backward compatibility.
    """
    # Support task->skill_name and goal->query mapping for callers using new signature
    if task is not None and skill_name is None:
        skill_name = task
    if goal is not None and query == "":
        query = goal

    if not skill_name:
        return {"error": "Skill name or task parameter is required"}

    base_repo_path = "/home/ty/Repositories/claude-scientific-skills/scientific-skills"
    target_path = os.path.join(base_repo_path, skill_name, "SKILL.md")
    if not os.path.exists(target_path):
        return {"error": f"Skill {skill_name} not found."}
    try:
        with open(target_path, "r") as f:
            return {"skill_id": skill_name, "metadata": f.read(), "status": "Ready"}
    except Exception as e:
        return {"error": str(e)}
