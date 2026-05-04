import sys
from pathlib import Path

# Add backend to path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root.resolve()))

from graph_rlm.backend.src.core.db import db


def list_all_skills():
    print("Listing all Skill nodes in FalkorDB...")
    cypher = "MATCH (s:Skill) RETURN s.name, s.description, s.tags"
    results = db.query(cypher)

    if not results:
        print("No Skill nodes found.")
        return

    for row in results:
        # result_set format depends on the wrapper logic
        # Based on db.py: results.append(dict(zip(column_names, row, strict=True)))
        name = row.get("s.name")
        desc = row.get("s.description")
        tags = row.get("s.tags")
        print(f"Skill: {name}")
        print(f"  Tags: {tags}")
        print(f"  Description: {desc[:100]}..." if desc else "  No description.")
        print("-" * 20)

if __name__ == "__main__":
    list_all_skills()
