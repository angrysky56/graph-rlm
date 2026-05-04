import asyncio

from graph_rlm.backend.src.core.db import db


async def test_db_indexes():
    print("Testing db.indexes() against FalkorDB...")

    try:
        # Try raw call without YIELD first to see columns
        print("\n1. Calling db.indexes() raw...")
        res = db.raw_graph.query("CALL db.indexes()")
        print(f"Columns: {res.header}")
        for row in res.result_set:
            print(f"Row: {row}")
    except Exception as e:
        print(f"Failed raw CALL: {e}")

    try:
        # Try the query currently in db.py:170
        print("\n2. Testing SET query in db.py (YIELD style)...")
        cypher = "CALL db.indexes() YIELD label, status RETURN label, status"
        res = db.raw_graph.query(cypher)
        print("Success!")
    except Exception as e:
        print(f"FAILED (as suspected): {e}")


if __name__ == "__main__":
    asyncio.run(test_db_indexes())
