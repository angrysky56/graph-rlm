import asyncio

from graph_rlm.backend.src.core.db import db


async def check_db():
    q = """
    MATCH (n:Thought)
    WHERE NOT n.status IN ["system", "reflexion", "sheaf", "omcd"]
      AND n.id CONTAINS ':T'
    RETURN n.id, n.prompt, n.repe_shakiness, n.sheaf_score, n.status
    ORDER BY n.created_at DESC
    LIMIT 20
    """
    rows = db.query(q)
    for r in rows:
        print(f"ID: {r['n.id']}")
        print(f"Status: {r['n.status']}")
        print(f"sheaf_score: {r['n.sheaf_score']}")
        print(f"repe_shakiness: {r['n.repe_shakiness']}")
        print("-------")

if __name__ == "__main__":
    asyncio.run(check_db())
