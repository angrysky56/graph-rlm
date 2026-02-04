import sys
from pathlib import Path

# Add backend to path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root.resolve()))

from graph_rlm.backend.src.core.db import db


def get_axiom_code(name):
    print(f"Fetching code for {name}...")
    cypher = "MATCH (s:Skill {name: $name}) RETURN s.code"
    results = db.query(cypher, {"name": name})

    if not results:
        print(f"Skill {name} not found.")
        return

    code = results[0].get("s.code")
    print("CODE START")
    print(code)
    print("CODE END")

if __name__ == "__main__":
    get_axiom_code("axiom_test_coding_5130")
