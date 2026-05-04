import asyncio

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.scratchpad_builder import scratchpad_builder


async def get_test():
    q = """
    MATCH (n:Thought)
    WHERE n.repe_shakiness IS NOT NULL
    RETURN n.id, n.repe_shakiness, n.sheaf_score
    LIMIT 10
    """
    rows = db.query(q)
    print(f"Nodes with repe_shakiness: {len(rows)}")
    if rows:
        print(rows[0])

    q2 = """
    MATCH (n:Thought)
    RETURN n.id, n.repe_shakiness, n.sheaf_score
    ORDER BY n.created_at DESC
    LIMIT 3
    """
    rows2 = db.query(q2)
    print("Recent nodes:")
    for r in rows2:
        print(r)

if __name__ == "__main__":
    asyncio.run(get_test())
