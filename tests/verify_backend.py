import sys

try:
    from graph_rlm.backend.src.core import agent, db, endpoints
    print("SUCCESS: Modules imported correctly.")
except Exception as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
