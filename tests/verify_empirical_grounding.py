import asyncio
import uuid

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.scratchpad_builder import scratchpad_builder


async def test_empirical_grounding():
    print("Starting Empirical Grounding Test...")

    session_id = f"test-{uuid.uuid4()}"
    agent = Agent()

    # 1. Simulate a PROCESS thought with tool calls (Subsistence)
    thought_id_1 = str(uuid.uuid4())
    agent.execution_logs[session_id] = ["rlm.recall", "mcp.search"]

    print(
        f"Ingesting PROCESS thought {thought_id_1} with tools: {agent.execution_logs[session_id]}"
    )

    res_1 = agent._sync_thimac(
        thought_id=thought_id_1,
        prompt="Searching for something...",
        status="pending",
        result=None,
        step=1,
        tool_calls=agent.execution_logs[session_id],
    )

    if not res_1:
        print("Error: _sync_thimac returned None")
        return

    print(f"PROCESS result op: {res_1['operation']}")
    # Should be ARRIVE because tool calls contain search/recall
    assert res_1["operation"] == "ARRIVE"

    # Check if summary contains the tool
    event_1 = agent.morph_memory._all_events[-1]
    print(f"PROCESS summary: {event_1.summary}")
    assert "T: rlm.recall" in event_1.summary

    # 2. Simulate a thought with a NAV label
    thought_id_2 = str(uuid.uuid4())
    res_2 = agent._sync_thimac(
        thought_id=thought_id_2,
        prompt="Grounding...",
        status="success",
        result="Success",
        step=2,
        logical_id="session:T1:S2:NAV",
    )
    event_2 = agent.morph_memory._all_events[-1]
    print(f"NAV summary: {event_2.summary}")
    assert event_2.summary == "NAV"  # Should not be "N A V"

    # 3. Verify DB storage
    print("Verifying DB property storage...")
    db.create_thought_node(
        thought_id=thought_id_1,
        prompt="Initial prompt",
        session_id=session_id,
        thimac_op=res_1["operation"],
        thimac_level=res_1["level"],
        navigator_insight="CLASS 4 | Progress: 0.1234",
    )

    # Query back
    q = "MATCH (n:Thought {id: $id}) RETURN n.thimac_op as op, n.thimac_level as level, n.navigator_insight as insight"
    rows = db.query(q, {"id": thought_id_1})
    print(f"DB Row: {rows[0]}")
    assert rows[0]["op"] == "ARRIVE"
    assert rows[0]["insight"] == "CLASS 4 | Progress: 0.1234"

    # 4. Verify Thimac Gestalt Output
    print("Verifying Thimac Gestalt Truncation Limits...")
    # Long result
    long_res = "Materializing evidence for " + "X " * 100
    agent._sync_thimac(
        thought_id=str(uuid.uuid4()),
        prompt="Implementing...",
        status="success",
        result=long_res,
        step=3,
    )
    gestalt = agent.morph_memory.get_gestalt_string()
    print("\nGestalt Snippet (First 1000 chars):")
    print(gestalt[:1000])

    # Instead of looking for "ACCEPT:", find the line containing the long string fragment
    found_long_summary = False
    for line in gestalt.split("\n"):
        if "Materializing evidence for X X" in line:
            print(f"Found long summary line: {line}")
            if len(line) > 100:
                found_long_summary = True
                break

    assert (
        found_long_summary
    ), "Could not find the long materialized summary in gestalt output"

    print("\nSUCCESS: Empirical grounding and Thimac refinements verified.")


if __name__ == "__main__":
    asyncio.run(test_empirical_grounding())
