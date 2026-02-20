import asyncio
import json

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.repe import repe


async def check():
    print(f"Is repe calibrated directly? {repe.is_calibrated}")
    a = Agent()
    print(
        "Does Agent have same repe?",
        repe is a.repe if hasattr(a, "repe") else "Agent doesn't store repe instance",
    )

    print("\nReading recent db nodes:")
    q2 = """
    MATCH (n:Thought)
    RETURN n.id, n.prompt, n.repe_shakiness, n.status
    ORDER BY n.created_at DESC
    LIMIT 3
    """
    rows2 = db.query(q2)
    print("Recent nodes:")
    for r in rows2:
        print(f"ID: {r['n.id']}")
        print(f"Status: {r['n.status']}")
        print(f"repe_shakiness: {r['n.repe_shakiness']}")
        print(f"Prompt (first 50 chars): {str(r.get('n.prompt', ''))[:50]}")
        print("-------")


if __name__ == "__main__":
    asyncio.run(check())
