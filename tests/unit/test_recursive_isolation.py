import asyncio
import uuid

from graph_rlm.backend.src.core.agent import agent


async def test_recursive_isolation():
    print("🚀 Starting Recursive Isolation Test...")

    # Parent Query
    # The parent will call a sub-agent.
    # We want to see if the parent continues AFTER the sub-agent finishes.

    prompt = (
        "Run a sub-query using 'rlm.query(\"Calculate 5+5\", context=\"Test\")'. "
        "After it returns, state the result and then say 'Final Answer: [Parent confirmed result is X]'."
    )

    session_id = f"test-{uuid.uuid4().hex[:8]}"
    print(f"Session: {session_id}")

    events = []
    async for event in agent.stream_query(prompt, session_id=session_id):
        events.append(event)
        if event["type"] == "thinking":
            print(f"  [Thinking] {event['content'][:100]}...")
        elif event["type"] == "final_answer":
            print(f"  [Final Answer] {event['content']}")

    # Verify that the parent produced its own final answer
    final_answers = [e["content"] for e in events if e["type"] == "final_answer"]

    print("\n--- RESULTS ---")
    if any("Parent confirmed result is 10" in fa for fa in final_answers):
        print("✅ SUCCESS: Parent successfully synthesized child result.")
    else:
        print("❌ FAILURE: Parent failed to produce correct final answer or aborted early.")
        for fa in final_answers:
            print(f"  Actual Final Answer: {fa}")

if __name__ == "__main__":
    asyncio.run(test_recursive_isolation())
