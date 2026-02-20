import asyncio
import time

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.repe import repe


async def run_agent_test():
    await repe.calibrate()

    agent = Agent()
    agent.session_cache["task_embedding"] = [0.1] * 384
    tid = f"test_thought_{time.time()}"

    vec = [0.1] * 384

    # Pre-commit
    print("Pre-committing...")
    db.create_thought_node(
        thought_id=tid, prompt="Running test", status="running", validate=False
    )

    # Compute psych
    print("Is repe calibrated:", repe.is_calibrated)
    # mock vec up to 3072 because text-embedding-3-small
    real_vec = [0.1] * 3072
    psych = repe.scan_thought(real_vec)
    print("Psych Profile:", psych)

    shaky = psych.get("Shakiness") if psych else None
    print("Passing Shaky type:", type(shaky))

    # Final commit directly simulating agent.py line 1902
    db.create_thought_node(
        thought_id=tid,
        prompt="Running test \n\n[Output]:\nSuccess",
        status="success",
        validate=False,
        repe_shakiness=shaky,
    )

    q = "MATCH (n:Thought {id: $id}) RETURN n.id, n.repe_shakiness, n.status"
    rows = db.query(q, {"id": tid})
    print("Final Query Result:", rows)


if __name__ == "__main__":
    asyncio.run(run_agent_test())
