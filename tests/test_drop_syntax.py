import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from graph_rlm.backend.src.core.db import db


def test_drop_index_syntax():
    print("Testing DROP INDEX syntax...")

    # Try different variations
    queries = [
        "DROP INDEX FOR (t:Thought) ON (t.embedding)",
        "DROP INDEX ON :Thought(embedding)",
    ]

    for q in queries:
        print(f"Trying: {q}")
        try:
            res = db.query(q)
            print(f"SUCCESS: {res}")
        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    test_drop_index_syntax()
