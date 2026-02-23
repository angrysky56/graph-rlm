import os
import sys
import uuid
from pathlib import Path

# Add backend/src to path
# The file is in graph_rlm/backend/tests/verify_pruning_fix.py
# src is in graph_rlm/backend/src
backend_root = Path(__file__).parent.parent
sys.path.append(str(backend_root))

from src.core.config import settings
from src.core.db import GraphClient


def verify_pruning():
    db = GraphClient()
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    # Create nodes
    # 1. A failed node (should be pruned)
    failed_id = f"thought_{uuid.uuid4().hex[:8]}"
    db.create_thought_node(
        thought_id=failed_id,
        prompt="Test failed prompt",
        status="failed",
        session_id=session_id,
        root_session_id=session_id
    )

    # 2. A success node with high sheaf score (normally would be pruned, but should be protected now)
    success_id = f"thought_{uuid.uuid4().hex[:8]}"
    db.create_thought_node(
        thought_id=success_id,
        prompt="Test success prompt",
        status="success",
        result="Some successful retrieval",
        sheaf_score=0.9, # High score, triggers pruning normally
        session_id=session_id,
        root_session_id=session_id
    )

    print(f"Created nodes in session {session_id}:")
    print(f"  - Failed (prunable): {failed_id}")
    print(f"  - Success (protected): {success_id}")

    # Run pruning
    count, pruned_ids = db.force_consolidate_noisy_branches(session_id)
    print(f"Pruned {count} nodes: {pruned_ids}")

    # Verify
    failed_correct = failed_id in pruned_ids
    success_correct = success_id not in pruned_ids

    if failed_correct:
        print("✅ SUCCESS: Failed node was pruned.")
    else:
        print("❌ FAILURE: Failed node was NOT pruned.")

    if success_correct:
        print("✅ SUCCESS: Success node was protected from pruning.")
    else:
        print("❌ FAILURE: Success node was pruned despite protection logic.")

    # Check status in DB
    check_query = "MATCH (n:Thought) WHERE n.id IN [$f, $s] RETURN n.id as id, n.status as status"
    results = db.query(check_query, {"f": failed_id, "s": success_id})
    print("\nDB Status after pruning:")
    for r in results:
        print(f"  - {r['id']}: {r['status']}")

    if failed_correct and success_correct:
        print("\nOVERALL VERIFICATION: PASSED")
        sys.exit(0)
    else:
        print("\nOVERALL VERIFICATION: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    verify_pruning()
