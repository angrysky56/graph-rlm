import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder


async def test_scratchpad_formatting():
    print("Testing Scratchpad Formatting and Ratings...")
    builder = ScratchpadBuilder()

    # Mock DB Query Results
    mock_results = [
        {
            "id": "t1",
            "prompt": "rlm.recall('some_id')",
            "status": "success",
            "result": "Some complex result text here...",
            "created_at": int(datetime.now().timestamp() * 1000),
            "repl_id": "REPL1",
            "turn_id": 1,
            "step_id": 1,
            "sheaf_score": 0.1,
            "spectral_energy": 0.05,
            "dreamer_analysis": None,
            "execution_summary": "Recalled previous thought data.",
            "repe_shakiness": 0.0,
            "repe_evasion": 0.0,
            "omcd_score": 0.9,
        },
        {
            "id": "t2",
            "prompt": "await agentskills.run('list_dir', path='/')",
            "status": "success",
            "result": "['bin', 'etc', 'home', ...]",
            "created_at": int(datetime.now().timestamp() * 1000) + 1000,
            "repl_id": "REPL1",
            "turn_id": 1,
            "step_id": 2,
            "sheaf_score": 0.85,  # High sheaf -> LOOP alert
            "spectral_energy": 0.1,
            "dreamer_analysis": None,
            "execution_summary": "Listed root directory.",
            "repe_shakiness": -0.2,  # Low shakiness -> UNCERTAIN alert
            "repe_evasion": -0.3,  # Low evasion -> EVASION alert
            "omcd_score": 0.2,  # Low OMCD -> LOW CONFIG alert
        },
        {
            "id": "t3",
            "prompt": "agent.read_file('ghost.txt')",
            "status": "failed",
            "result": "FileNotFoundError: File 'ghost.txt' does not exist.",
            "created_at": int(datetime.now().timestamp() * 1000) + 2000,
            "repl_id": "REPL1",
            "turn_id": 1,
            "step_id": 3,
            "sheaf_score": 0.0,
            "spectral_energy": 0.0,
            "dreamer_analysis": None,
            "execution_summary": "Attempted to read non-existent file.",
        },
    ]

    # Mock Thimac Gestalt
    mock_gestalt = """### 🧠 Current Known State
- **Known Skills**: [read_file, write_file]
- **Last File Written**: `/home/ty/test.py`
- **Knowledge Horizon**: Initial task received.

**Existence** (1 active results):
  CREATE: Materialized test.py [19:50:01.123]

**Subsistence** (1 potential states):
  PROCESS: Analyzing file system [19:50:05.456]
"""

    # Patch DB and other methods
    builder.db.query = lambda query, params: mock_results
    builder.db.get_completed_rounds = lambda root_session_id: []

    # Run build_scratchpad
    pad = await builder.build_scratchpad(
        session_id="test_session",
        root_session_id="root_session",
        task="Test the scratchpad formatting",
        current_step=2,
        max_steps=100,
        current_round_id="round_1",
        morph_gestalt=mock_gestalt,
        current_repl_id="REPL1",
    )

    print("\n--- GENERATED SCRATCHPAD ---")
    print(pad)
    print("--- END ---")

    # Verifications
    assert "Ratings (Ψ,📐,Ω)" in pad
    assert r"Ψ:-0.30 \| 📐:0.85 \| Ω:0.20" in pad
    assert "!! LOOP !! !! EVASION !! !! UNCERTAIN !! !! LOW CONFIG !!" in pad
    assert "## 🧠 THIMAC GESTALT" in pad
    assert "## Data Commands" not in pad

    # Verify Failure Triplet Integration
    assert "Agent -> filenotfounderror -> ghost.txt" in pad
    print("\n✅ Scratchpad formatting, ratings, and failure triplets verified.")


if __name__ == "__main__":
    asyncio.run(test_scratchpad_formatting())
