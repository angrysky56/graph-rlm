"""
Internal Skills Tool Wrapper.

Exposes the SkillsManager functionality as a pseudo-MCP tool module.
This allows the agent to call 'skills' tools just like any other MCP tool.
"""

from typing import Any, List
from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager
from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill
import asyncio

async def save_new_skill(name: str, code: str, description: str | None = None) -> str:
    """
    Save a new skill (Python function).

    Args:
        name: Name of the skill (snake_case)
        code: Complete Python code for the skill
        description: Description of what the skill does

    Returns:
        Confirmation message
    """
    manager = get_skills_manager()
    manager.save_skill(name, code, description)
    return f"Skill '{name}' saved successfully."

async def list_available_skills() -> List[dict]:
    """
    List all available skills.

    Returns:
        List of skill metadata objects
    """
    manager = get_skills_manager()
    skills = manager.list_skills()

    # Convert dict to list for easier consumption
    result = []
    for name, metadata in skills.items():
        meta = metadata.copy()
        meta["name"] = name
        result.append(meta)
    return result

async def read_skill_code(name: str) -> str:
    """
    Read the code of an existing skill.

    Args:
        name: Name of the skill

    Returns:
        The Python code of the skill
    """
    manager = get_skills_manager()
    skill = manager.get_skill(name)
    if not skill:
        raise ValueError(f"Skill '{name}' not found")
    return skill["code"]

async def run_skill_by_name(skill_name: str, args: dict[str, Any] | None = None) -> Any:
    """
    Execute a skill by name.

    Args:
        skill_name: Name of the skill
        args: Dictionary of arguments to pass to the function

    Returns:
        Result of the skill execution
    """
    return await execute_skill(skill_name, args or {})

# Tool Definitions for explicit listing
TOOLS = [
    {
        "name": "save_new_skill",
        "description": "Save a new Python skill for future use.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name (snake_case)"},
                "code": {"type": "string", "description": "Python code"},
                "description": {"type": "string", "description": "Description"}
            },
            "required": ["name", "code"]
        }
    },
    {
        "name": "list_available_skills",
        "description": "List all saved skills.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "read_skill_code",
        "description": "Get the source code of a skill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "run_skill_by_name",
        "description": "Execute a saved skill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string"},
                "args": {"type": "object"}
            },
            "required": ["skill_name"]
        }
    }
]
