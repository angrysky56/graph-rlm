import sys
import os
import time
sys.path.append(os.getcwd())
try:
    from core.db import db
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), ".."))
    from core.db import db

def probe():
    print("Attempt 1: Create Index")
    cypher_idx = "CREATE VECTOR INDEX FOR (n:Probe) ON (n.embedding) OPTIONS {dimension:768, similarityFunction:'cosine'}"
    try:
        db.query(cypher_idx)
        print("Index creation command sent.")
    except Exception as e:
        print(f"Index creation note: {e}")

    # Wait for index
    print("Waiting for index...")
    for i in range(10):
        try:
            res = db.query("CALL db.indexes()")
            # Check if operational?
            # In python client res is a list of [label, status...] maybe?
            # Let's just print it
            # print(res.result_set)
            pass
        except:
            pass
        time.sleep(0.5)

    print("Attempt 2: Insert Data")
    db.query("MERGE (p:Probe {id: 'similarity-test'}) SET p.embedding = vecf32($vec)", {"vec": [0.1]*768})

    print("Attempt 3: Query Nodes")
    cypher_proc = "CALL db.idx.vector.queryNodes('Probe', 'embedding', 5, vecf32($vec)) YIELD node, score RETURN node, score"
    try:
        res = db.query(cypher_proc, {"vec": [0.1]*768})
        print(f"Success! queryNodes works. Results: {len(res.result_set)}")
        for row in res.result_set:
            print(f"Score: {row[1]}")
    except Exception as e:
        print(f"Failed queryNodes: {e}")

if __name__ == "__main__":
    probe()
