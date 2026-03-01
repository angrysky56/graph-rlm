import asyncio
import json
import os
import sys
import uuid

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.db import db


async def verify_recursion_and_gea():
    print("🚀 Starting GEA and Recursion Verification...")

    agent = Agent()
    session_id = f"test-gea-{uuid.uuid4().hex[:8]}"
    root_id = session_id
    round_id = f"round-{uuid.uuid4().hex[:4]}"

    print(f"Session: {session_id}")
    print(f"Round: {round_id}")

    # --- 1. Test Recursion (Amnesia & Silent UI Bug) ---
    prompt = "Use rlm.query to check the current system time and then report it back to me. This serves as a recursion test."

    print("\n--- Testing Recursion and UI Events ---")
    sub_agent_found = False
    done_event_found = False

    async for event in agent.stream_query(
        prompt, session_id=session_id, round_id=round_id, root_session_id=root_id
    ):
        event_type = event.get("type")
        content = event.get("content", "")

        if event_type == "thinking" and "rlm.query" in str(content):
            print("  ✓ Found rlm.query action in thinking.")

        if event_type == "thinking" and "[Sub-Agent]" in str(content):
            print("  ✓ Found Sub-Agent activity in thinking.")
            sub_agent_found = True

        if event_type == "done" or event_type == "RLM_FINAL_OUTPUT":
            print(
                f"  ✓ Found '{event_type}' event with content: {str(content)[:50]}..."
            )
            done_event_found = True

    if not done_event_found:
        print("  ✗ FAILED: 'done' or 'RLM_FINAL_OUTPUT' event not found in stream.")

    # --- 2. Test Shared Experience Pool (Insight Registration) ---
    print("\n--- Testing Insight Registration ---")
    cypher = "MATCH (i:Insight) WHERE i.root_session_id = $rsid RETURN i"
    insights = db.query(cypher, {"rsid": root_id})

    if insights:
        print(f"  ✓ Found {len(insights)} Insight nodes in the experience pool.")
        for idx, ins in enumerate(insights):
            content = ins.get("content") or "(no content)"
            itype = ins.get("type") or "unknown"
            print(
                f"    - Insight {idx+1}: Type={itype}, Content sample='{str(content)[:30]}...'"
            )
    else:
        print("  ✗ FAILED: No Insight nodes found for this session.")

    # --- 3. Test Group Cohesion (H0 Rank Monitoring) ---
    print("\n--- Testing Group Cohesion (H0 Rank) ---")
    cypher = "MATCH (n:Thought) WHERE n.root_session_id = $rsid RETURN n.h0_rank as h0 ORDER BY n.created_at DESC LIMIT 1"
    h0_res = db.query(cypher, {"rsid": root_id})

    if h0_res and h0_res[0].get("h0") is not None:
        h0 = h0_res[0]["h0"]
        print(f"  ✓ Found H0 Rank in database: {h0}")
    else:
        print("  ✗ FAILED: H0 Rank not found in latest thought node.")

    print("\n--- Summary ---")
    success = done_event_found and len(insights) > 0 and h0_res
    if success:
        print("✅ GEA Integration and Recursion Fixes VERIFIED.")
    else:
        print("❌ Verification FAILED. See logs above.")


if __name__ == "__main__":
    asyncio.run(verify_recursion_and_gea())
