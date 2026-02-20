
import asyncio
import os
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.dream import dreamer


async def run_diagnostic():
    print("--- STARTING DREAMER DIAGNOSTIC ---")

    agent = Agent()
    session_id = "diagnostic_session_" + os.urandom(4).hex()

    # Task designed to trigger a hallucination rejection
    prompt = "Create a file named 'diagnostic_test.txt' and then claim it exists with content 'DIAGNOSTIC_SUCCESS' without running any code."

    print(f"Running agent with session: {session_id}")

    # We will manually run a few steps of the agent loop to see what happens
    # Instead of query_sync, we'll use a more controlled loop

    try:
        # Step 1: Force a rejection and see if dream_cycle triggers
        print("\nStep 1: Simulating Hallucination...")
        candidate = "I have created the file diagnostic_test.txt. It contains 'DIAGNOSTIC_SUCCESS'."
        context = "Initial state. User asked to create file. Agent claimed success without tools."

        validation = await dreamer.validate_response(
            candidate=candidate,
            context=context,
            session_id=session_id
        )

        print(f"Validation Verdict: {validation.get('status')} - {validation.get('event')}")
        print(f"Instruction: {validation.get('instruction')}")

        # Step 2: Manually trigger dream_cycle
        print("\nStep 2: Triggering Dream Cycle...")

        # We need to make sure there's "surprise" in the DB for this session
        # We'll create a dummy failed thought node
        node_id = f"{session_id}:dummy_fail"
        db.create_thought_node(
            node_id,
            prompt="Fake probe",
            session_id=session_id,
            status="failed",
            result="Empirical Contradiction: File not found"
        )

        result = await dreamer.dream_cycle(
            session_id=session_id,
            final_response_candidate=candidate,
            context=context
        )

        print(f"Dream Cycle Result: {result.get('status')}")
        print(f"Insight ID: {result.get('id')}")
        print(f"Insight Text Snapshot: {result.get('insight', '')[:200]}...")

        # Step 3: Check if anything was saved
        print("\nStep 3: Checking DB for artifacts...")
        insights = db.query("MATCH (i:Insight) WHERE i.id = $id RETURN i", {"id": result.get('id')})
        axioms = db.query("MATCH (a:Axiom) RETURN a.name ORDER BY a.name")

        print(f"Insights found: {len(insights)}")
        print(f"Axioms stored: {[a.get('a.name') for a in axioms]}")

    except Exception as e:
        import traceback
        print(f"Diagnostic Failed with Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_diagnostic())
