import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Mock dependencies before further imports
sys.modules["falkordb"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.db"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.llm"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.sheaf"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.context_index"] = MagicMock()

import queue

from graph_rlm.backend.src.core.agent import Agent, RLMInterface, execution_events


async def test_namespacing():
    agent = Agent()
    agent.repl_manager.create_repl()

    # Simulate execution_events context var for emission
    execution_events.set(queue.Queue())

    print("Checking REPL namespace after injection simulation...")
    # Directly call _execute_code to verify injection logic
    try:
        await agent._execute_code("print('test')", "thought_1", "test_session")
    except Exception as e:
        print(f"Ignored execution error: {e}")

    repl_id = agent.active_repls.get("test_session")
    repl = agent.repl_manager.get_repl(repl_id)
    ns = repl.namespace

    print(f"Namespace keys: {list(ns.keys())}")
    assert "rlm" in ns, "rlm namespace missing"
    assert "mcp" in ns, "mcp namespace missing"
    assert "done" in ns, "done helper missing"

    # Check if a known MCP server is present
    mcp = ns["mcp"]
    servers = [s for s in dir(mcp) if not s.startswith("_")]
    print(f"Discovered MCP servers: {servers}")
    assert len(servers) > 0, "No MCP servers discovered"

    # Check RLMInterface
    assert isinstance(ns["rlm"], RLMInterface)

    print("Verification Successful: Namespaces present and populated.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_namespacing())
