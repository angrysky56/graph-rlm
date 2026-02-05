import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import json

from graph_rlm.backend.src.core.db import db

if __name__ == "__main__":
    node_id = "12afe77f-e0f9-4399-8076-e6a67a6007bd"
    cypher = "MATCH (n) WHERE n.id = $id RETURN properties(n) as props"
    results = db.query(cypher, {"id": node_id})

    if results:
        print(f"Properties for node {node_id}:")
        print(json.dumps(results[0]["props"], indent=2))
    else:
        print(f"Node {node_id} not found.")
