import json
import os
import sys
from pathlib import Path

# Ensure repo root is in path
repo_root = Path(os.getcwd())
sys.path.append(str(repo_root))

from graph_rlm.backend.src.core.config import settings
from graph_rlm.backend.src.core.db import db


def debug_db():
    print(f"DB Host: {settings.FALKOR_HOST}:{settings.FALKOR_PORT}")
    print(f"Graph Name: {settings.GRAPH_NAME}")

    # 1. Get recent thoughts
    query = """
    MATCH (n:Thought)
    RETURN n.id, n.session_id, n.status, n.created_at, n.step_id, n.repl_id
    ORDER BY n.created_at DESC
    LIMIT 10
    """

    try:
        results = db.query(query)
        print(f"Found {len(results)} recent thoughts:")
        for r in results:
            print(f" - ID: {r.get('n.id')}")
            print(f"   Session: {r.get('n.session_id')}")
            print(f"   Status: {r.get('n.status')}")
            print(f"   Step: {r.get('n.step_id')}")
            print(f"   REPL: {r.get('n.repl_id')}")
            print("   ---")

        if not results:
            print("No Thought nodes found in DB.")

    except Exception as e:
        print(f"DB Query failed: {e}")


if __name__ == "__main__":
    debug_db()
