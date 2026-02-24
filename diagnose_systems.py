import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path("/home/ty/Repositories/ai_workspace/graph-rlm")
sys.path.append(str(project_root))

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.mcp_runtime import is_mcp_available, is_skills_available
from graph_rlm.backend.src.mcp_integration.skill_storage import (
    get_axioms_manager,
    get_skills_manager,
)


async def diagnose():
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")

    mcp_ok = is_mcp_available()
    skills_ok = is_skills_available()
    print(f"is_mcp_available: {mcp_ok}")
    print(f"is_skills_available: {skills_ok}")

    try:
        thought_count = db.query("MATCH (n:Thought) RETURN count(n) as count")
        round_count = db.query("MATCH (n:Round) RETURN count(n) as count")
        skill_count = db.query("MATCH (n:Skill) RETURN count(n) as count")
        axiom_count = db.query("MATCH (n:Axiom) RETURN count(n) as count")

        print(f"Thought count: {thought_count}")
        print(f"Round count: {round_count}")
        print(f"Skill count: {skill_count}")
        print(f"Axiom count: {axiom_count}")
    except Exception as e:
        print(f"Database query error: {e}")

    if skills_ok:
        try:
            sm = get_skills_manager()
            print(f"Skills Manager initialized: {sm}")
            print(f"Skills Dir: {sm.skills_dir}")
            skills_list = sm.list_skills()
            print(f"Skills list size: {len(skills_list)}")

            am = get_axioms_manager()
            print(f"Axioms Manager initialized: {am}")
            print(f"Axioms Dir: {am.axioms_dir}")
            axioms_list = am.list_axioms()
            print(f"Axioms list size: {len(axioms_list)}")
        except Exception as e:
            print(f"Skills/Axioms error: {e}")


if __name__ == "__main__":
    asyncio.run(diagnose())
