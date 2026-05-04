import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from graph_rlm.backend.src.core.db import db


def test_create_index_syntax():
    print("Testing CREATE VECTOR INDEX syntax...")
    dim = 3072

    # Try different variations
    queries = [
        f"CREATE VECTOR INDEX FOR (t:Thought) ON (t.embedding) OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}",
        f"CREATE INDEX FOR (t:Thought) ON (t.embedding) OPTIONS {{indexConfig: {{dimension:{dim}, similarityFunction:'cosine'}}}}"
    ]

    for q in queries:
        print(f"Trying: {q}")
        try:
            res = db.query(q)
            print(f"SUCCESS: {res}")
            break
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    test_create_index_syntax()
