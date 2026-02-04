import asyncio

from graph_rlm.backend.src.core.db import db


async def test_malformed_cypher():
    print("Testing Malformed Cypher diagnosis...")

    # 1. Drop existing index
    try:
        print("\n1. Dropping vector index...")
        db.raw_graph.query("DROP INDEX :Thought(embedding)")
        print("Success (or already gone)!")
    except Exception as e:
        print(f"Index drop skipped/failed: {e}")

    # 2. Try the "Neo4j Style" CREATE INDEX that is currently in db.py:141
    try:
        print("\n2. Attempting Neo4j-style CREATE INDEX (Suspect malformed)...")
        dim = 3072
        malformed_cypher = f"CREATE VECTOR INDEX FOR (t:Thought) ON (t.embedding) OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
        db.raw_graph.query(malformed_cypher)
        print(
            "Success! (If this worked, then this syntax is actually supported by your FalkorDB version)"
        )
    except Exception as e:
        print(f"FAILED (as suspected): {e}")

    # 3. Try the "FalkorDB Style" CREATE INDEX
    try:
        print("\n3. Attempting FalkorDB-style CREATE INDEX...")
        correct_cypher = f"CREATE VECTOR INDEX ON :Thought(embedding) OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
        db.raw_graph.query(correct_cypher)
        print("Success!")
    except Exception as e:
        print(f"FAILED Correct Syntax: {e}")


if __name__ == "__main__":
    asyncio.run(test_malformed_cypher())
