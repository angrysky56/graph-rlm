
import asyncio
import os
import sys

from core.agent import Agent
from core.db import db
from core.sheaf import sheaf


async def verify():
    print("--- 1. Verifying Axiomatic Bypass ---")
    # This should return HEALTHY despite any axioms because of 'diagnostic' tag
    res = await sheaf.check_axiomatic_consistency(
        proposed_code="print('hello')",
        task_tags=["diagnostic"]
    )
    print(f"Axiomic Bypass Result: {res}")
    assert res["status"] == "HEALTHY"
    assert res.get("mode") == "diagnostic"

    print("\n--- 2. Verifying Metadata Injection ---")
    agent = Agent()
    session_id = "test_verification"
    thought_id = "test_thought_1"
    agent.current_turn = 5

    # We need to mock some things to run _execute_code standalone or just trust the code
    # But let's try a small execution
    output, failed = await agent._execute_code(
        code="print(f'Turn: {turn_id}, Step: {step_id}, REPL: {repl_id}')",
        thought_id=thought_id,
        session_id=session_id,
        turn_id=5,
        step_id=3
    )
    print(f"Execution Output: {output}")
    assert "Turn: 5, Step: 3" in output

    print("\n--- 3. Verifying Code Hash in DB ---")
    # Clean up test node if exists
    db.query("MATCH (n:Thought {id: 'test_thought_trace'}) DELETE n")

    db.create_thought_node(
        thought_id="test_thought_trace",
        prompt="test prompt",
        session_id="test_session",
        code_hash="fake_hash_12345"
    )

    res = db.query("MATCH (n:Thought {id: 'test_thought_trace'}) RETURN n.code_hash as hash")
    print(f"Stored Hash: {res[0]['hash'] if res else 'Not found'}")
    assert res[0]['hash'] == "fake_hash_12345"

    print("\nVerification Complete!")

if __name__ == "__main__":
    asyncio.run(verify())
