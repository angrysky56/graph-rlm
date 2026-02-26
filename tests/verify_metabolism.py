import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "graph_rlm/backend/src"))

from core.thimac_memory import ThimacIntention, ThimacMemory


def test_metabolism():
    memory = ThimacMemory()
    print(f"Initial State: Pi={memory.Pi}, Rg={memory.Rg}, Ee={memory.Ee}")

    # Simulate a grounded thought (Success)
    thought = {
        "id": "test_1",
        "prompt": "Testing grounding",
        "status": "success",
        "step_id": 1,
    }
    event = memory.ingest_thought(thought, intent_type=ThimacIntention.PROXIMAL)
    print(
        f"Grounded Thought: Pi={memory.Pi:.3f}, Rg={memory.Rg:.3f}, Ee={memory.Ee:.3f}, State={event.metabolic_state}"
    )

    # Simulate a loop (High pressure, failing to ground)
    for i in range(5):
        thought = {
            "id": f"loop_{i}",
            "prompt": "Looping",
            "status": "failed",
            "step_id": i + 2,
        }
        event = memory.ingest_thought(thought, intent_type=ThimacIntention.PROXIMAL)
        print(
            f"Loop {i}: Pi={memory.Pi:.3f}, Rg={memory.Rg:.3f}, Ee={memory.Ee:.3f}, State={event.metabolic_state}"
        )


if __name__ == "__main__":
    test_metabolism()
