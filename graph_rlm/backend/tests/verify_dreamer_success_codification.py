"""
Verification script for Dreamer Codification on SUCCESS.
Triggers a successful agent run and checks if the Dreamer generates an axiom/insight.
"""

import asyncio
import sys
import uuid
from pathlib import Path

# Add the 'src' directory to sys.path
repo_root = Path(__file__).parent.parent.parent.parent.absolute()
backend_root = repo_root / "graph_rlm" / "backend"
backend_src = backend_root / "src"

if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))
if str(backend_src) not in sys.path:
    sys.path.insert(0, str(backend_src))

try:
    from src.core.agent import Agent
    from src.core.db import db
except ImportError:
    from core.agent import Agent
    from core.db import db


async def verify():
    print("--- Dreamer Success Codification Verification ---")
    agent = Agent()
    session_id = f"VERIFY_SUCCESS_{uuid.uuid4().hex[:8]}"

    # A simple task that should succeed and result in a learnable pattern or at least a peaceful insight
    prompt = "List the files in the current root directory using Python and summarize them. I want to see a Skill discovered for directory summarization."

    print(f"1. Starting Agent Query (Session: {session_id})...")

    await agent.query_sync(prompt=prompt, session_id=session_id)

    print("2. Checking for new Insights in DB...")
    # The agent.py awaits dream_cycle at the end of query_sync, so we don't need a long sleep
    # but let's give it a second for DB persistence
    await asyncio.sleep(2)

    q = """
    MATCH (i:Insight)
    WHERE i.session_id = $sid
    RETURN i.id as id, i.content as content
    """
    res = db.query(q, {"sid": session_id})

    if res:
        print(f"✅ SUCCESS: Found {len(res)} new Insight nodes!")
        for row in res:
            content = row['content']
            print(f"   - Insight ID: {row['id']}")
            print(f"     Preview: {content[:200]}...")

            # Check if it proceeded to codification
            if "Skill:" in content or "Rule:" in content:
                print("     🔥 [DREAMER] Found potential Axiom headers in insight!")
    else:
        print("❌ FAILURE: No new Insights found.")
        print("   This means the Dreamer didn't trigger or returned early.")

    # Check for actual axioms
    q_axiom = """
    MATCH (a:Axiom)
    WHERE a.description CONTAINS $sid OR a.name CONTAINS $sid
    RETURN a.id as id, a.name as name
    """
    # Note: Dreamer might not tag axioms with session_id directly in properties,
    # but it classifies them. We'll check by name/description if the dreamer generated any.
    # Actually, let's just look at the most recent axioms.
    q_all_axioms = "MATCH (a:Axiom) RETURN a.name as name, a.created_at as ts ORDER BY a.created_at DESC LIMIT 1"
    res_all = db.query(q_all_axioms)
    if res_all:
        print(f"   Latest Axiom in DB: {res_all[0]['name']}")


if __name__ == "__main__":
    asyncio.run(verify())
