import sys
import os

# Ensure backend root is in path
sys.path.append(os.getcwd())

from core.db import db
import time

def probe():
    tid = "probe-1"
    vec = [0.1] * 768 # Dummy vector

    print("Attempt 1: Raw list assignment: t.embedding = $vec")
    params = {"tid": tid, "vec": vec}
    cypher = "MERGE (t:Probe {id: $tid}) SET t.embedding = $vec RETURN t"
    try:
        db.query(cypher, params)
        print("Success! Raw list works.")
        return
    except Exception as e:
        print(f"Failed Attempt 1: {e}")

    print("Attempt 2: vecf32 assignment: t.embedding = vecf32($vec)")
    cypher = "MERGE (t:Probe {id: $tid}) SET t.embedding = vecf32($vec) RETURN t"
    try:
        db.query(cypher, params)
        print("Success! vecf32 works.")
        return
    except Exception as e:
        print(f"Failed Attempt 2: {e}")

if __name__ == "__main__":
    probe()
