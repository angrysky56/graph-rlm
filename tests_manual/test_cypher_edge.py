import asyncio
import uuid

from graph_rlm.backend.src.core.config import settings
from graph_rlm.backend.src.core.db import db


async def test_cypher_edge_cases():
    print(f"Testing Cypher Edge Cases against FalkorDB at {settings.FALKOR_HOST}:{settings.FALKOR_PORT}...")

    # Session Details
    sid = str(uuid.uuid4())
    rid = str(uuid.uuid4())

    # 1. Test create_thought_node with ONLY REPL_ID (No embedding)
    # This tests the branch: cypher += ", t.repl_id = $repl_id"
    tid1 = "id-repl-only"
    try:
        print("\n1. Testing create_thought_node (REPL_ID only)...")
        db.create_thought_node(
            thought_id=tid1,
            prompt="Test REPL ID Only",
            session_id=sid,
            repl_id=rid,
            prompt_embedding=None
        )
        print("Success!")
    except Exception as e:
        print(f"FAILED 1 (REPL ID Only): {e}")

    # 2. Test create_thought_node with BOTH REPL_ID and Embedding
    tid2 = "id-both"
    try:
        print("\n2. Testing create_thought_node (Both)...")
        db.create_thought_node(
            thought_id=tid2,
            prompt="Test Both",
            session_id=sid,
            repl_id=rid,
            prompt_embedding=[0.2]*3072
        )
        print("Success!")
    except Exception as e:
        print(f"FAILED 2 (Both): {e}")

    # 3. Test update_thought_result with ONLY REPL_ID
    try:
        print("\n3. Testing update_thought_result (REPL_ID only)...")
        db.update_thought_result(
            thought_id=tid1,
            result="Success result",
            repl_id=rid,
            embedding=None
        )
        print("Success!")
    except Exception as e:
        print(f"FAILED 3 (Update REPL ID): {e}")

    # 4. Test find_similar_thoughts with exact dimensionality
    try:
        print("\n4. Testing find_similar_thoughts...")
        results = db.find_similar_thoughts([0.2]*3072, limit=2)
        print(f"Success! Found {len(results)} results.")
    except Exception as e:
        print(f"FAILED 4 (Vector Search): {e}")

    # 5. Test get_context_frontier
    try:
        print("\n5. Testing get_context_frontier...")
        results = db.get_context_frontier(sid, limit=5)
        print(f"Success! Found {len(results)} thoughts.")
    except Exception as e:
        print(f"FAILED 5 (Frontier): {e}")

if __name__ == "__main__":
    asyncio.run(test_cypher_edge_cases())
