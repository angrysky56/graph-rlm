import asyncio
from unittest.mock import AsyncMock, patch

from graph_rlm.backend.src.core.agent import Agent


async def verify_termination_unit():
    print("🚀 Starting Termination Flow Unit Verification...")

    agent = Agent()
    events = []

    def on_event(event_type, **kwargs):
        events.append((event_type, kwargs.get("content")))
        if event_type == "RLM_FINAL_OUTPUT":
            print(f"✅ RLM_FINAL_OUTPUT captured: {kwargs.get('content')[:100]}...")

    agent.emit_event = on_event
    agent.final_result = "Some draft result"

    # Mock dreamer.validate_response to return "exhausted"
    with patch(
        "graph_rlm.backend.src.core.agent.dreamer.validate_response",
        new_callable=AsyncMock,
    ) as mock_validate, patch(
        "graph_rlm.backend.src.core.agent.sheaf.check_axiomatic_consistency",
        new_callable=AsyncMock,
    ) as mock_sheaf:
        mock_validate.return_value = {
            "status": "exhausted",
            "instruction": "SYSTEM CRITICAL: Economic budget exhausted.",
            "reasons": ["oMCD high Q_stop"],
        }
        mock_sheaf.return_value = {"status": "HEALTHY"}

        # Mock create_system_node
        agent.create_system_node = AsyncMock(return_value="thought-123")

        print("Calling _validate_and_finalize with exhausted status...")
        # (response_text, context_scratchpad, prompt, session_id, root_id, step, round_id, repl_id, bool(code), code_hash)
        result = await agent._validate_and_finalize(
            "Final text RLM_FINAL_OUTPUT",
            "context",
            "prompt",
            "session-1",
            "root-1",
            1,
            "round-1",
            "repl-1",
            False,
            code_hash="c123",
        )

    print(f"Method returned: {result}")

    # Check if RLM_FINAL_OUTPUT was emitted
    final_outputs = [
        content for etype, content in events if etype == "RLM_FINAL_OUTPUT"
    ]

    if result is True and any(
        "BUDGET_EXHAUSTED" in content for content in final_outputs
    ):
        print(
            "✅ SUCCESS: _validate_and_finalize handled 'exhausted' as terminal and emitted report."
        )
    else:
        print("❌ FAILED: Terminal report not sent or loop not broken.")

    print("\n🎉 Termination Flow Unit Verification Complete.")


if __name__ == "__main__":
    asyncio.run(verify_termination_unit())
