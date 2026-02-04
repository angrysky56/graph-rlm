import asyncio

from graph_rlm.backend.src.core.config import settings
from graph_rlm.backend.src.core.db import db


async def test_cypher():
    print(f"Testing Cypher against FalkorDB at {settings.FALKOR_HOST}:{settings.FALKOR_PORT}...")

    # Create a dummy thought
    tid = "test-thought-id"
    try:
        print("\n1. Testing create_thought_node...")
        db.create_thought_node(
            thought_id=tid,
            prompt="Test Prompt",
            session_id="test-session",
            prompt_embedding=[0.1]*3072
        )
        print("Success!")
    except Exception as e:
        print(f"FAILED create_thought_node: {e}")

    try:
        print("\n2. Testing find_similar_thoughts...")
        # This will likely fail if index is bad or query is malformed
        results = db.find_similar_thoughts([0.1]*3072, limit=1)
        print(f"Success! Found {len(results)} results.")
    except Exception as e:
        print(f"FAILED find_similar_thoughts: {e}")

    try:
        print("\n3. Testing create_vector_index (manual check)...")
        # Try raw query with suspect syntax
        dim = 3072
        bad_cypher = f"CREATE VECTOR INDEX FOR (t:Thought) ON (t.embedding) OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
        db.raw_graph.query(bad_cypher)
        print("Success! (Wait, if this succeeded, then my diagnosis was wrong)")
    except Exception as e:
        print(f"FAILED manual index creation (as expected): {e}")

if __name__ == "__main__":
    asyncio.run(test_cypher())
