"""
Verification script for Dreamer Codification.
Triggers a validation failure and checks if the Dreamer generates an axiom/insight.
"""

import asyncio
import sys
import uuid
from pathlib import Path

# Add the 'src' directory to sys.path so we can import 'core.agent' etc.
# We also need 'src' parent to be able to resolve relative imports if any,
# but usually absolute imports from 'src' are safer.
repo_root = Path(__file__).parent.parent.parent.parent.absolute()
backend_src = repo_root / "graph_rlm" / "backend" / "src"

if str(backend_src) not in sys.path:
    sys.path.insert(0, str(backend_src))

# To avoid relative import issues in agent.py (from ..mcp_integration...),
# we should ideally run this script as a module or add 'src' parent to path.
backend_root = repo_root / "graph_rlm" / "backend"
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# Force absolute imports by using the 'src' prefix if needed,
# but agent.py uses relative imports internally.
# If we run with backend_root in path, we should use 'from src.core.agent import agent'

try:
    from src.core.agent import agent
    from src.core.db import db
except ImportError:
    # Fallback for different path configs
    from core.agent import agent
    from core.db import db


async def verify():
    print("--- Dreamer Codification Verification ---")
    session_id = f"VERIFY_DREAMER_{uuid.uuid4().hex[:8]}"

    prompt = "Create a file named 'dreamer_test.txt' and then definitely claim that it exists and contains the word 'SUCCESS' without actually using any tools to verify it."

    print(f"1. Starting Agent Query (Session: {session_id})...")

    try:
        # We run for a few steps to trigger validation
        await agent.query_sync(prompt=prompt, session_id=session_id)
    except Exception as e:
        print(f"   Note: Query ended (possibly caught validation rejection): {e}")

    print("2. Waiting for background Dream Cycle (15s)...")
    # Dream cycle might take some time to process LLM calls
    await asyncio.sleep(15)

    print("3. Checking for new Insights/Axioms in DB...")
    # Check for nodes created in this session or linked to it
    q = """
    MATCH (i:Insight)
    WHERE i.session_id = $sid
    RETURN i.id as id, i.content as content
    """
    res = db.query(q, {"sid": session_id})

    if res:
        print(f"✅ SUCCESS: Found {len(res)} new Insight nodes!")
        for row in res:
            print(f"   - Insight ID: {row['id']}")
            print(f"     Preview: {row['content'][:200]}...")
    else:
        # Check for axioms
        q_axiom = """
        MATCH (a:Axiom)
        WHERE a.session_id = $sid
        RETURN a.id as id, a.description as description
        """
        res_axiom = db.query(q_axiom, {"sid": session_id})
        if res_axiom:
            print(f"✅ SUCCESS: Found {len(res_axiom)} new Axiom nodes!")
        else:
            print("❌ FAILURE: No new Insights or Axioms found.")
            print(
                "   This might mean the Dreamer didn't trigger 'Surprise' or LLM didn't codify."
            )


if __name__ == "__main__":
    asyncio.run(verify())
