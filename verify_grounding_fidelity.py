import asyncio

from graph_rlm.backend.src.core.dream import dreamer


class MockEvent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.round_id = kwargs.get("round_id")

    def to_dict(self):
        return self.__dict__


async def verify_grounding_fidelity():
    print("🚀 Starting Grounding Fidelity Verification...")

    # Mock Context with large task_input
    large_input = "A" * 1500
    context = f"## 🎯 Initial Task\nSummarize the provided text.\n\n## 👤 Task Input\n{large_input}\n\n## 📝 Execution Trace\n"

    # Trace without REPL interaction
    empty_trace = []

    candidate = "This is a zero-shot summary of the large input."

    print("\n--- Test 1: Zero-Shot Synthesis (No REPL) ---")

    result = await dreamer.validate_response(
        candidate=candidate,
        context=context,
        session_id="test-fid-1",
        memory_trajectory=empty_trace,
    )

    print(f"Status: {result['status']}")
    reasons = result.get("reasons", [])
    print(f"Reasons: {reasons}")

    is_fidelity_reject = any(
        "fidelity" in r.lower() or "synthesis" in r.lower() or "ground" in r.lower()
        for r in reasons
    )

    if result["status"] == "invalid" and is_fidelity_reject:
        print("✅ Zero-Shot Synthesis rejected as expected.")
    else:
        print("❌ FAILED: Zero-Shot Synthesis should have been rejected.")

    print("\n--- Test 2: Grounded Synthesis (With REPL) ---")
    # Mock trace with a REPL step using MockEvent
    grounded_trace = [
        MockEvent(
            id="step1",
            status="success",
            repl="repl_123",
            prompt="print(task_input[:100])",
            result="Some data",
        )
    ]

    result_2 = await dreamer.validate_response(
        candidate=candidate,
        context=context + "  - [step1] success @ 0 (REPL: repl_123)\n",
        session_id="test-fid-2",
        memory_trajectory=grounded_trace,
    )

    print(f"Status: {result_2['status']}")
    reasons_2 = result_2.get("reasons", [])
    fidelity_violations = [r for r in reasons_2 if "fidelity" in r.lower()]

    if not fidelity_violations:
        print("✅ Grounded Synthesis cleared the fidelity check.")
    else:
        print(
            f"❌ FAILED: Fidelity violation reported despite REPL usage: {fidelity_violations}"
        )

    print("\n🎉 Grounding Fidelity Verification Complete.")


if __name__ == "__main__":
    asyncio.run(verify_grounding_fidelity())
