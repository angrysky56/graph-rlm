import sys
import os
sys.path.append(os.getcwd())
try:
    from core.db import db
except ImportError:
    # Fallback if run from backend subdir
    sys.path.append(os.path.join(os.getcwd(), ".."))
    from core.db import db

def probe():
    print("Attempt 1: CREATE VECTOR INDEX syntax")
    cypher = "CREATE VECTOR INDEX FOR (t:Probe) ON (t.embedding) OPTIONS {dimension: 768, similarityFunction: 'cosine'}"
    # OR: CREATE VECTOR INDEX ON :Probe(embedding) ...
    # FalkorDB documentation is key.

    # Try the likely modern syntax
    # "CREATE VECTOR INDEX idx_probe FOR (n:Probe) ON (n.embedding) OPTIONS {dimension:768, similarityFunction:'cosine'}"
    cypher_modern = "CREATE VECTOR INDEX idx_probe FOR (n:Probe) ON (n.embedding) OPTIONS {dimension:768, similarityFunction:'cosine'}"

    try:
        db.query(cypher_modern)
        print("Success! Modern syntax works.")
        return
    except Exception as e:
        print(f"Failed Modern: {e}")

    print("Attempt 2: Old Procedure Syntax")
    cypher_proc = "CALL db.idx.vector.createNodeIndex('Probe', 'embedding', 'FLOAT32', 'COSINE', 768)"
    try:
        db.query(cypher_proc)
        print("Success! Procedure syntax works.")
        return
    except Exception as e:
        print(f"Failed Procedure: {e}")

if __name__ == "__main__":
    probe()
