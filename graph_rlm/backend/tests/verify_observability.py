import asyncio
import uuid

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.repe import repe
from graph_rlm.backend.src.core.scratchpad_builder import scratchpad_builder
from graph_rlm.backend.src.core.state import ExecutionState, agent_state


async def verify_observability():
    print("🚀 Starting Targeted Observability Verification...")

    agent = Agent()
    session_id = f"test-obs-{uuid.uuid4().hex[:8]}"

    # Mock state
    state = ExecutionState(round_id=f"{session_id}:R1", turn_id=1)
    agent_state.set(state)

    # 1. Create a "shaky" thinking node via _sync_thimac directly
    # This simulates what query_sync now does
    shaky_text = "I assume that the file exists. acting as if I know."
    thought_id = str(uuid.uuid4())

    print(f"Simulating sync for shaky thought: '{shaky_text}'")

    # Get embedding for scanning
    embedding = await agent.llm.get_embedding(shaky_text)
    await repe.calibrate()
    repe_res = repe.scan_thought(embedding, shaky_text)
    repe_scores = repe_res.get("scores", {})
    repe_rationale = repe_res.get("rationale")

    await agent._sync_thimac(
        thought_id=thought_id,
        prompt=shaky_text,
        status="thinking",
        result=None,
        step=1,
        session_id=session_id,
        round_id=f"{session_id}:R1",
        turn_id=1,
        embedding=embedding,
        repe_shakiness=repe_scores.get("Shakiness"),
        repe_confluence=repe_scores.get("Confluence"),
        repe_evasion=repe_scores.get("Evasion"),
        repe_freedom=repe_scores.get("Freedom"),
        repe_rationale=repe_rationale,
    )

    print("Node synced. Building scratchpad...")

    # 2. Build scratchpad
    pad = await scratchpad_builder.build_scratchpad(
        session_id=session_id,
        root_session_id=session_id,
        task="Test observability mission",
        current_step=1,
        execution_state=state,
        current_round_id=f"{session_id}:R1",
    )
    # 3. Test Dreamer Validation (Stability Fix Check)
    print("Testing Dreamer validation stability...")
    from graph_rlm.backend.src.core.dream import dreamer

    val_res = await dreamer.validate_response(
        candidate="The file exists and I have verified its contents.",
        context=pad,
        session_id=session_id,
        current_step=2,
        root_session_id=session_id,
    )
    print(f"Validation Verdict: {val_res.get('status')}")
    assert val_res.get("status") in [
        "valid",
        "invalid",
        "exhausted",
    ], "Dreamer crashed or returned unexpected status"

    print("\n--- SCRATCHPAD OUTPUT ---")
    print(pad)
    print("--- END SCRATCHPAD ---\n")

    # Verification assertions
    assert "Ψ(S:" in pad, "RepE metrics missing from scratchpad"
    assert "!! UNCERTAIN !!" in pad, "Uncertainty alert missing"
    assert (
        "Insight: 🧠 *Psychological triggers detected" in pad
    ), "Rationale/Insight missing"
    assert (
        "Shakiness: 'I assume that'" in pad
    ), "Specific trigger missing from rationale"

    print("✅ Targeted Observability Verification PASSED!")


if __name__ == "__main__":
    asyncio.run(verify_observability())
