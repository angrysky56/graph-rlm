import asyncio
import sys
import uuid
from unittest.mock import MagicMock

# Mock out the database to avoid ConnectionError during import if not needed
# But since user said falkor is available, we can try to use it.
# However, for a unit-like test of recall_node RAM logic, mocks are safer.

try:
    import graph_rlm.backend.src.core.db as db_module
except ImportError:
    # Fallback/Mock
    db_module = MagicMock()

from graph_rlm.backend.src.core.rlm_interface import RLMInterface
from graph_rlm.backend.src.core.thimac_memory import ThimacMemory


class MockAgent:
    def __init__(self):
        self.current_thought_id = "test_root"
        self.db = db_module.db if hasattr(db_module, "db") else MagicMock()
        self.llm = MagicMock()
        self.morph_memory = ThimacMemory()
        self.execution_logs = {}
        self.current_turn = 1
        self.active_repls = {}
        self.stop_requested = False

    def emit_event(self, event_type, content):
        print(f"EVENT [{event_type}]: {content}")

    def record_tool_use(self, tool_name):
        print(f"TOOL USE: {tool_name}")
        if "test_session" not in self.execution_logs:
            self.execution_logs["test_session"] = []
        self.execution_logs["test_session"].append(tool_name)


async def test_recall_node():
    print("--- Testing rlm.recall_node (In-Memory) ---")
    agent = MockAgent()
    rlm = RLMInterface(
        agent_instance=agent, session_id="test_session", root_session_id="test_root"
    )
    agent.rlm = rlm

    # 1. Ingest a large thought into Thimac
    large_result = "DATA_" * 1000  # 5000 chars
    thought_id = str(uuid.uuid4())

    agent.morph_memory.ingest_thought(
        thought={
            "id": thought_id,
            "prompt": "Generating large data",
            "status": "success",
            "result": large_result,
            "repl_id": "test_repl",
        }
    )

    # 2. Test recall_node with different offsets
    print("\n[Test 1] Recalling first 1000 chars:")
    res1 = await rlm.recall_node(thought_id, offset=0, limit=1000)
    print(f"Result Snippet Length: {len(res1)}")
    assert "DATA_" in res1
    assert "truncated 4000 remaining chars" in res1

    print("\n[Test 2] Recalling middle 1000 chars:")
    res2 = await rlm.recall_node(thought_id, offset=2000, limit=1000)
    print(f"Result Snippet Length: {len(res2)}")
    assert "DATA_" in res2
    assert "chars 2000-3000/5000" in res2

    print("\n[Test 3] Recalling end of data:")
    res3 = await rlm.recall_node(thought_id, offset=4500, limit=1000)
    print(f"Result Snippet Length: {len(res3)}")
    assert "DATA_" in res3
    assert "remaining chars" not in res3

    print("\n--- Recall Node Verification SUCCESS ---")


if __name__ == "__main__":
    asyncio.run(test_recall_node())
