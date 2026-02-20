import asyncio

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.dream import dreamer
from graph_rlm.backend.src.core.repe import repe


async def test_discrepancy():
    print("Initial repe.is_calibrated:", repe.is_calibrated)

    agent = Agent()
    print("Agent init repe.is_calibrated:", repe.is_calibrated)

    # Try calibrating explicitly
    await repe.calibrate()
    print("After calibrate repe.is_calibrated:", repe.is_calibrated)

    # Agent scan thought
    vec = [0.1] * 384
    psych1 = repe.scan_thought(vec)
    print("Agent-side psych profile length:", len(psych1))
    print(psych1)

    # Dreamer validation
    psych2 = repe.scan_thought(vec)
    print("Dreamer-side psych profile length:", len(psych2))
    print(psych2)

if __name__ == "__main__":
    asyncio.run(test_discrepancy())
