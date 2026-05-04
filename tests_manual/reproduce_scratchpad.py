import asyncio
import os

# Adjust import path as needed for reproduction context
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Mock falkordb and redis before importing backend modules that depend on them
sys.modules["falkordb"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()
sys.modules["langchain_community"] = MagicMock()
sys.modules["langchain_community.graphs"] = MagicMock()

sys.path.append(os.getcwd())

from graph_rlm.backend.src.core.config import settings
from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder


async def reproduce():
    print("=== Starting Scratchpad Reproduction ===\n")

    # Mock DB
    mock_db = MagicMock()

    # 1. Mock Completed Rounds (Previous Rounds)
    # Testing variable length summary logic
    mock_db.get_completed_rounds.return_value = [
        {
            "round_id": "rnd_long_1",
            "user_prompt": "A" * 600,  # Should trigger summarizer
            "final_response": "B" * 600,  # Should trigger summarizer
            "repl_ids": ["repl_1"],
            "ended_at": 1700000000000,
        },
        {
            "round_id": "rnd_short_2",
            "user_prompt": "Short prompt",  # Should be raw
            "final_response": "Short result",  # Should be raw
            "repl_ids": ["repl_1"],
            "ended_at": 1700000100000,
        },
    ]

    # 2. Mock Current Round Progress (Execution Trace)
    # Testing Observability Alerts
    mock_db.query.return_value = [
        {
            "id": "thought_1",
            "prompt": "Analyzing...",
            "status": "success",
            "result": "Done.",
            "created_at": 1700000200000,
            "repl_id": "repl_1",
            "turn_id": 1,
            "step_id": 1,
            "sheaf_score": 0.1,  # Normal
            "spectral_energy": 0.1,
            "dreamer_analysis": None,
            "execution_summary": "Analyzed inputs.",
        },
        {
            "id": "thought_2",
            "prompt": "Looping...",
            "status": "running",
            "result": "...",
            "created_at": 1700000205000,
            "repl_id": "repl_1",
            "turn_id": 1,
            "step_id": 2,
            "sheaf_score": 0.85,  # ALERT: Loop
            "spectral_energy": 0.6,  # ALERT: Drift
            "dreamer_analysis": "Critique: Stop this.",  # ALERT: Dreamer
            "execution_summary": None,
        },
    ]

    # Patch LLM generation
    with patch(
        "graph_rlm.backend.src.core.scratchpad_builder.protected_llm_generate",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.return_value = "LLM_GENERATED_SUMMARY"

        # Patch db on the singleton or instance
        builder = ScratchpadBuilder()
        builder.db = mock_db

        # Build
        scratchpad = await builder.build_scratchpad(
            session_id="sess_test", root_session_id="root_test", task="Test Task"
        )

        print(scratchpad)

        # Verifications
        print("\n=== Verification Report ===")

        # Check Block Format
        if "### Round 1 (ID: `rnd_long_1`)" in scratchpad:
            print("[PASS] Block format header found.")
        else:
            print("[FAIL] Block format header missing.")

        if "**User Prompt**:" in scratchpad and "**Agent Result**:" in scratchpad:
            print("[PASS] Block sections found.")
        else:
            print("[FAIL] Block sections missing.")

        # Check Summarization Logic
        # Round 1 (Long) should call LLM
        # Round 2 (Short) should NOT call LLM
        # We expect 2 calls for Round 1 (Prompt + Result) + 0 for Round 2 + 1 for Thought 2 (no execution summary) = 3 calls total
        # (Thought 1 has execution_summary so it skips LLM)
        print(f"LLM Call Count: {mock_llm.call_count}")
        if "LLM_GENERATED_SUMMARY" in scratchpad:
            print("[PASS] LLM Summary injected.")

        # Check Observability Alerts
        if "> [!] SHEAF: Loop Detected (0.85)" in scratchpad:
            print("[PASS] Sheaf Loop Alert found.")
        else:
            print("[FAIL] Sheaf Loop Alert missing.")

        if "> [!] DRIFT: High Deviation (0.60)" in scratchpad:
            print("[PASS] Drift Alert found.")
        else:
            print("[FAIL] Drift Alert missing.")

        if "> [!] DREAMER: Critique: Stop this." in scratchpad:
            print("[PASS] Dreamer Critique Alert found.")
        else:
            print("[FAIL] Dreamer Critique Alert missing.")

        if "**Thimac Gestalt (Memory Anchor)**" not in scratchpad:
            print("[WARN] Thimac Gestalt header missing (expected if None passed).")


if __name__ == "__main__":
    asyncio.run(reproduce())
