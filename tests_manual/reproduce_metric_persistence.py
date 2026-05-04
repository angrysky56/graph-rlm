import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock falkordb and redis
sys.modules["falkordb"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()
sys.modules["langchain_community"] = MagicMock()
sys.modules["langchain_community.graphs"] = MagicMock()

sys.path.append(os.getcwd())

from graph_rlm.backend.src.core.db import GraphClient
from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder


async def reproduce():
    print("=== Starting Cognitive Metric Persistence Verification ===\n")

    # --- 1. Test DB Persistence Logic ---
    print("\n--- Testing DB Persistence ---")
    mock_falkor = MagicMock()

    # We need to instantiate GraphClient and mock its internal components
    # But GraphClient __init__ calls self.create_vector_indexes() which calls query()
    # So we mock GraphClient.query directly for simplicity in testing thought creation

    with patch("graph_rlm.backend.src.core.db.GraphClient.query") as mock_query:
        # Instantiate
        # We also need to patch __init__ to avoid connection attempts
        with patch("graph_rlm.backend.src.core.db.GraphClient.__init__", return_value=None):
            client = GraphClient()
            client.raw_graph = MagicMock()
            client.query = mock_query

            # Test Data
            repe_profile = {"Shakiness": -0.5, "Evasion": -0.2}
            omcd_score = 0.15

            # Call create_thought_node
            client.create_thought_node(
                thought_id="t1",
                prompt="test",
                repe_profile=repe_profile,
                omcd_score=omcd_score,
                validate=False
            )

            # Verify Cypher contains new fields
            call_args = mock_query.call_args
            cypher, params = call_args[0]

            print(f"Cypher Query: {cypher}")
            print(f"Params: {params}")

            if "t.repe_shakiness = $repe_shakiness" in cypher:
                print("[PASS] repe_shakiness found in Cypher.")
            else:
                print("[FAIL] repe_shakiness missing from Cypher.")

            if "t.omcd_score = $omcd_score" in cypher:
                print("[PASS] omcd_score found in Cypher.")
            else:
                print("[FAIL] omcd_score missing from Cypher.")

            if params.get("repe_shakiness") == -0.5:
                print("[PASS] repe_shakiness param correct.")
            else:
                print(f"[FAIL] repe_shakiness param incorrect: {params.get('repe_shakiness')}")


    # --- 2. Test Scratchpad Rendering ---
    print("\n--- Testing Scratchpad Rendering ---")
    mock_db = MagicMock()

    # Mock Current Round Progress with Cognitive Metrics
    # The scratchpad builder expects list-of-lists (raw driver format) or dicts
    # We'll use dicts as our updated code handles both
    mock_db.query.return_value = [
        {
            "id": "thought_1",
            "prompt": "Thinking...",
            "status": "success",
            "result": "Done.",
            "created_at": 1700000200000,
            "repl_id": "repl_1",
            "turn_id": 1,
            "step_id": 1,
            "sheaf_score": 0.1,
            "spectral_energy": 0.1,
            "dreamer_analysis": None,
            "execution_summary": "Normal step.",
            "repe_shakiness": 0.5, # Healthy
            "repe_evasion": 0.5,
            "omcd_score": 0.9
        },
        {
            "id": "thought_2",
            "prompt": "Uncertain...",
            "status": "running",
            "result": "...",
            "created_at": 1700000205000,
            "repl_id": "repl_1",
            "turn_id": 1,
            "step_id": 2,
            "sheaf_score": 0.1,
            "spectral_energy": 0.1,
            "dreamer_analysis": None,
            "execution_summary": None,
            "repe_shakiness": -0.8, # ALERT: VERY SHAKY
            "repe_evasion": -0.6,   # ALERT: EVASIVE
            "omcd_score": 0.1       # ALERT: LOW CONFIDENCE
        },
    ]
    mock_db.get_completed_rounds.return_value = []

    # Patch LLM generation
    with patch(
        "graph_rlm.backend.src.core.scratchpad_builder.protected_llm_generate",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.return_value = "SUMMARY"

        # Patch db directly
        builder = ScratchpadBuilder()
        builder.db = mock_db

        # Build
        scratchpad = await builder.build_scratchpad(
            session_id="sess_test", root_session_id="root_test", task="Test Task"
        )

        print("\n=== Scratchpad Output Snippet ===")
        # Print the table part
        table_lines = [l for l in scratchpad.split('\n') if '|' in l]
        for l in table_lines:
            print(l)

        print("\n=== Verification Report ===")

        if "> [Ψ] SHAKINESS: -0.80" in scratchpad:
            print("[PASS] Shakiness Alert found.")
        else:
            print("[FAIL] Shakiness Alert missing.")

        if "> [Ψ] EVASION: -0.60" in scratchpad:
            print("[PASS] Evasion Alert found.")
        else:
            print("[FAIL] Evasion Alert missing.")

        if "> [Ω] LOW STOP CONFIDENCE: 0.10" in scratchpad:
             print("[PASS] oMCD Alert found.")
        else:
             print("[FAIL] oMCD Alert missing.")

if __name__ == "__main__":
    asyncio.run(reproduce())
