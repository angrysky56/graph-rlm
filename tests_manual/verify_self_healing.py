import asyncio
import os
import sys
import uuid

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.repe import repe
from graph_rlm.backend.src.core.sheaf import sheaf


async def test_sheaf_loop_detection():
    print("\n--- Testing Sheaf Loop Detection ---")
    session_id = f"test-sheaf-{uuid.uuid4()}"

    # 1. Create a dummy history (Frontier)
    # Use identical embeddings to simulate a loop
    base_vec = [0.1] * 1536  # Mock 1536-dim embedding

    node_ids = []
    for i in range(3):
        tid = str(uuid.uuid4())
        db.create_thought_node(
            tid,
            f"Repetitive thought {i}",
            session_id=session_id,
            prompt_embedding=base_vec,
        )
        node_ids.append(tid)

    # 2. Diagnose a NEW thought that is identical
    print(f"Diagnosing trace with {len(node_ids)} prior repetitive nodes...")

    hypothetical_edges = [(nid, "new-id") for nid in node_ids]
    diag = sheaf.diagnose_trace(
        root_id="root",
        hypothetical_node={"id": "new-id", "embedding": base_vec},
        hypothetical_edges=hypothetical_edges,
    )

    print(f"Energy: {diag['energy']:.4f}")
    print(f"Status: {diag['status']}")
    print(f"Critique: {diag['critique']}")

    assert diag["status"] == "INCONSISTENT"
    assert "Logical Knot" in diag["critique"]
    print("✅ Sheaf Loop Detection Passed.")


async def test_repe_pathogen_detection():
    print("\n--- Testing RepE Pathogen Detection ---")
    await repe.calibrate()

    # Create a "Lazy" embedding
    # In reality, we'd use the LLM, but here we'll just check if calibration worked
    # and if any of the antigen centroids give a high score.

    for concept, vec in repe.antigen_vectors.items():
        load, moloch = repe.scan_latent(vec.tolist())
        print(f"Scanning '{concept}' centroid: Load={load:.4f}, Concept={moloch}")
        assert load > 0.95
        assert moloch == concept

    print("✅ RepE Pathogen Detection Passed.")


async def main():
    try:
        await test_sheaf_loop_detection()
        await test_repe_pathogen_detection()
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
