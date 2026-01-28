from graph_rlm.backend.src.core.db import db


def inspect():
    print("--- All Nodes ---")
    nodes = db.query(
        "MATCH (n:Thought) RETURN n.id, n.prompt, n.root_session_id, n.session_id LIMIT 10"
    )
    for n in nodes:
        print(n)

    print("\n--- Root Candidates (No Incoming DECOMPOSES_INTO) ---")
    roots = db.query("""
        MATCH (t:Thought)
        WHERE NOT ()-[:DECOMPOSES_INTO]->(t)
        RETURN t.id, t.prompt
    """)
    print(f"Found {len(roots)} roots.")
    for r in roots:
        print(r)

    print("\n--- Testing list_sessions query ---")
    q = """
    MATCH (t:Thought)
    WHERE NOT ()-[:DECOMPOSES_INTO]->(t)
    RETURN t.id AS id, t.prompt AS prompt, t.created_at AS created_at
    ORDER BY t.created_at DESC
    LIMIT 20
    """
    res = db.query(q)
    print(f"Result: {res}")


if __name__ == "__main__":
    inspect()
