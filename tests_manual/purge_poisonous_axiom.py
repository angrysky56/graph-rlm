import sys
from pathlib import Path

# Add backend to path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root.resolve()))

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.logger import get_logger

logger = get_logger("purge_axiom")

def purge_axiom(name):
    print(f"Purging axiom {name} from FalkorDB...")
    # 1. Delete the node and its relationships
    cypher = "MATCH (s:Skill {name: $name}) DETACH DELETE s"
    db.query(cypher, {"name": name})

    # 2. Check if it's really gone
    check_cypher = "MATCH (s:Skill {name: $name}) RETURN s"
    results = db.query(check_cypher, {"name": name})

    if not results:
        print(f"SUCCESS: Axiom {name} has been purged.")
    else:
        print(f"FAILURE: Axiom {name} still exists in the database.")

if __name__ == "__main__":
    purge_axiom("axiom_test_coding_5130")
