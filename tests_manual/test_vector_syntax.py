import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))


from graph_rlm.backend.src.core.db import db


def test_vector_search_syntax():
    print("Testing Vector Search Syntax in FalkorDB...")

    # 1. Ensure index exists (using correct syntax)
    dim = 3072
    label = "Thought"
    attr = "embedding"

    print(f"Creating index for {label}.{attr} (dim={dim})...")
    try:
        # FalkorDB Syntax
        db.query(f"CALL db.idx.vector.create('{label}', '{attr}', {dim}, 'cosine')")
        print("Index creation SUCCESS (or already exists).")
    except Exception as e:
        print(f"Index creation failed (might already exist): {e}")

    # 2. Add a dummy node with embedding
    print("Adding dummy node...")
    dummy_vec = [0.1] * dim
    try:
        db.query(
            "CREATE (t:Thought {id: 'test_vector', prompt: 'test', embedding: vecf32($vec)})",
            {"vec": dummy_vec},
        )
        print("Node created.")
    except Exception as e:
        print(f"Node creation failed: {e}")

    # 3. Test queryNodes
    print("\nTesting CALL db.idx.vector.queryNodes...")
    limit = 1
    # Note: queryNodes in FalkorDB often expects quoted strings for label/attr
    queries = [
        f"CALL db.idx.vector.queryNodes('{label}', '{attr}', {limit}, vecf32($vec)) YIELD node, score RETURN node.id, score",
        f"CALL db.idx.vector.queryNodes({label}, {attr}, {limit}, vecf32($vec)) YIELD node, score RETURN node.id, score",
    ]

    for q in queries:
        print(f"Trying query: {q}")
        try:
            res = db.query(q, {"vec": dummy_vec})
            print(f"SUCCESS: {res}")
        except Exception as e:
            print(f"FAILED: {e}")

    # cleanup
    db.query("MATCH (t:Thought {id: 'test_vector'}) DETACH DELETE t")


if __name__ == "__main__":
    test_vector_search_syntax()
