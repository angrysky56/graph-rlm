import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))


from graph_rlm.backend.src.core.db import db


def test_cosine_distance():
    print("Testing vec.cosineDistance in FalkorDB...")
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]

    # Simple query to test the function
    query = "RETURN vec.cosineDistance(vecf32($v1), vecf32($v2))"
    params = {"v1": v1, "v2": v2}

    try:
        res = db.query(query, params)
        print(f"Success! Result: {res}")
    except Exception as e:
        print(f"FAILED vec.cosineDistance: {e}")

    print("\nTesting vecdist as alternative...")
    query = "RETURN vecdist(vecf32($v1), vecf32($v2))"
    try:
        res = db.query(query, params)
        print(f"Success! Result: {res}")
    except Exception as e:
        print(f"FAILED vecdist: {e}")


if __name__ == "__main__":
    test_cosine_distance()
