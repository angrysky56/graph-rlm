import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

# --- PATH SETUP ---
# Ensure we can import 'src' from backend root
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src
backend_dir = os.path.dirname(current_dir)  # .../graph_rlm/backend
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# --- ENVIRONMENT LOADING ---
# Try to find .env
# 1. Check backend dir (graph_rlm/backend)
env_path = os.path.join(backend_dir, ".env")
if os.path.exists(env_path):
    print(f"Loading environment from {env_path}")
    load_dotenv(env_path)
else:
    # 2. Check graph_rlm dir (parent of backend)
    graph_rlm_dir = os.path.dirname(backend_dir)
    env_path = os.path.join(graph_rlm_dir, ".env")
    if os.path.exists(env_path):
        print(f"Loading environment from {env_path}")
        load_dotenv(env_path)
    else:
        # 3. Check Workspace Root (parent of graph_rlm)
        workspace_root = os.path.dirname(graph_rlm_dir)
        env_path = os.path.join(workspace_root, ".env")
        if os.path.exists(env_path):
            print(f"Loading environment from {env_path}")
            load_dotenv(env_path)
        else:
            print(
                f"WARNING: No .env found. Searched: {backend_dir}, {graph_rlm_dir}, {workspace_root}"
            )

# --- IMPORTS ---
try:
    from src.core.agent import agent  # DIAGNOSTIC IMPORT
    from src.core.db import db

    # Lazy import dreamer to avoid potential circular deps if any, though cli is top level
    from src.core.dream import dreamer
    from src.core.llm import llm
    from src.core.repe import repe
    from src.core.sheaf import sheaf
except ImportError as e:
    print(f"❌ critical import error: {e}")
    sys.exit(1)

import argparse
import uuid


async def test_live_repe():
    print("\n--- LIVE TEST: RepE (Gestalt Monitor) ---")
    print("1. Calibrating Axes (Live LLM Calls)...")
    try:
        await repe.calibrate()
    except Exception as e:
        print(f"⚠️ FAIL (Graceful): RepE Calibration failed. Auth Error? {e}")
        return

    # Test text triggering 'Shakiness'
    shaky_text = "I assume that maybe this is likely true, but I am confused and lost the thread."
    print(f"2. Scanning Text: '{shaky_text}'")

    try:
        vec = await llm.get_embedding(shaky_text)
        if not vec:
            print("❌ FAIL: Could not generate embedding from Live LLM.")
            return

        profile = repe.scan_thought(vec)
        print(f"3. Assessment Profile: {profile}")

        score = profile.get("Shakiness", 0)
        if score < -0.1:
            print(f"✅ PASS: Detected Shakiness (Score: {score:.3f})")
        else:
            print(f"❌ FAIL: Missed Shakiness (Score: {score:.3f}) > -0.1")
    except Exception as e:
        print(f"⚠️ FAIL (Graceful): RepE Check failed due to API/Network: {e}")


async def test_live_sheaf():
    """
    Writes temporary nodes to the REAL Database and tests Sheaf Loop detection.
    """
    print("\n--- LIVE TEST: Sheaf (Topological Monitor) ---")

    # Check DB Connection
    try:
        print("1. Checking DB Connection...")
        # Simple query to verify connectivity
        db.query("RETURN 1")
        print("   -> Connected to FalkorDB @ localhost:6380")
    except Exception as e:
        print(
            f"❌ FAIL: Database not reachable. Is Docker container 'graph-rlm-db' running? Error: {e}"
        )
        return

    # Creating a unique test session ID to avoid pollution
    test_session_id = f"CLI_TEST_SHEAF_{uuid.uuid4().hex[:8]}"
    test_vec = np.random.rand(1536).tolist()

    print(f"2. Seeding DB for Loop (Session: {test_session_id})...")

    # Create loop chain
    prev_id = None
    node_ids = []

    for i in range(4):
        node_id = f"test_node_{i}"
        node_ids.append(node_id)
        db.create_thought_node(
            thought_id=node_id,
            prompt="Repetitive loop content",
            session_id=test_session_id,
            prompt_embedding=test_vec,
            status="success" if i < 3 else "running",
        )
        if prev_id:
            # Create edge
            edge_query = """
             MATCH (a:Thought {id: $aid}), (b:Thought {id: $bid})
             MERGE (a)-[:DECOMPOSES_INTO]->(b)
             """
            db.query(edge_query, {"aid": prev_id, "bid": node_id})
        prev_id = node_id

    # Now diagnose connection from Node 3 -> New Node (which is identical)
    print("3. Diagnosing Trace for Loop...")

    # Hypothetical edges: Node 3 -> Current
    hypothetical_edges = [(node_ids[-1], "current_hypothetical")]

    diag = sheaf.diagnose_trace(
        root_id="test_root",
        hypothetical_node={"embedding": test_vec},  # Same vector = LOOP
        hypothetical_edges=hypothetical_edges,
        goal_embedding=test_vec,
    )

    print(f"4. Sheaf Diagnosis: {diag['status']}")

    if diag.get("status") == "LOGICAL_KNOT":
        print("✅ PASS: Sheaf correctly identified LOGICAL_KNOT (Loop).")
    else:
        print(f"❌ FAIL: Sheaf missed the loop. Status: {diag.get('status')}")

    # Cleanup
    print("5. Cleaning up test data...")
    res = db.query(
        "MATCH (n:Thought {session_id: $sid}) DETACH DELETE n RETURN count(n) as count",
        {"sid": test_session_id},
    )
    count = res[0]["count"] if res else 0
    print(f"   -> Cleaned up {count} nodes from session {test_session_id}.")


async def test_live_dreamer():
    """
    Tests if Dreamer correctly ingests Context (Scratchpad) and identifies Surprise.
    """
    print("\n--- LIVE TEST: Dreamer (Meta-Cognitive Monitor) ---")

    test_session_id = f"CLI_TEST_DREAMER_{uuid.uuid4().hex[:8]}"
    print(f"1. Creating 'Surprise Event' in DB (Session: {test_session_id})...")

    # Create a FAIL node sequence to trigger 'Surprise'
    # Node A (Success) -> Node B (Failed)
    db.create_thought_node(
        thought_id="node_a",
        prompt="Calculate 2+2",
        session_id=test_session_id,
        status="success",
        result="4",
    )
    db.create_thought_node(
        thought_id="node_b",
        prompt="Divide by Zero",
        session_id=test_session_id,
        status="failed",
        result="ZeroDivisionError",
        parent_id="node_a",
    )
    # Create Edge
    db.query(
        "MATCH (a:Thought {id: 'node_a'}), (b:Thought {id: 'node_b'}) MERGE (a)-[:DECOMPOSES_INTO]->(b)"
    )

    from src.core.scratchpad_builder import scratchpad_builder

    print("2. Constructing Real Scratchpad Context (via Builder)...")
    # Dynamically build context from the DB nodes we just created
    real_context = scratchpad_builder.build_scratchpad(
        session_id=test_session_id,
        root_session_id=test_session_id,
        task="Test Task: Verify Dreamer Cycle",
        current_step=2,
    )
    print(f"   -> Context built ({len(real_context)} chars).")

    print("3. Triggering Dream Cycle with Context...")

    # Define a callback to see internal thoughts
    def cli_emit(type, content, tag=None):
        if type == "thinking" and tag == "DREAMER":
            print(f"   [Dreamer Internal]: {content}...")

    try:
        # Call Dreamer with the NEW context argument
        res = await dreamer.dream_cycle(
            emit_callback=cli_emit, session_id=test_session_id, context=real_context
        )

        print(f"4. Dream Result: {res.get('status')}")

        if res.get("status") == "lucid":
            print(f"✅ PASS: Dreamer woke up ('lucid'). Insight generated.")
            print(f"   Insight Preview: {res.get('insight')}")

            # [METABOLISM CHECK]
            print("5. Verifying Metabolism (Node Consolidation)...")
            b_res = db.query("MATCH (n:Thought {id: 'node_b'}) RETURN n.status")
            status = b_res[0]["n.status"] if b_res else "unknown"
            if status == "consolidated":
                print(
                    "✅ PASS: Metabolism confirmed. Node 'node_b' is now 'consolidated'."
                )
            else:
                print(
                    f"❌ FAIL: Metabolism failed. Node 'node_b' status is '{status}'."
                )

        elif res.get("status") == "peaceful":
            # It might be peaceful if it thinks the error is handled, but usually failure = surprise
            print(
                f"⚠️ NOTE: Dreamer returned 'peaceful'. Check logs if this was intended."
            )
        else:
            print(f"❌ FAIL: Dreamer status '{res.get('status')}' unexpected.")

    except Exception as e:
        print(f"❌ FAIL: Dreamer execution crashed: {e}")

    # Cleanup
    print("5. Cleaning up test data...")
    res = db.query(
        "MATCH (n:Thought {session_id: $sid}) DETACH DELETE n RETURN count(n) as count",
        {"sid": test_session_id},
    )
    count = res[0]["count"] if res else 0
    print(f"   -> Cleaned up {count} nodes from session {test_session_id}.")


async def test_live_agent(custom_prompt: Optional[str] = None):
    """
    Tests the full Agent Loop (query_sync) to reproduce the 'Instant Stop' / Hang.
    """
    print("\n--- LIVE TEST: Agent Loop (Full Integration) ---")
    test_session_id = f"CLI_TEST_AGENT_{uuid.uuid4().hex[:8]}"
    print(f"1. Starting Agent Session: {test_session_id}")

    try:
        # Define a simplified prompt
        prompt = (
            custom_prompt
            or "Explain the difference between TCP and UDP in one sentence."
        )

        print(f"2. Invoking agent.query_sync with prompt: {prompt[:100]}...")

        # We need to patch emit_event to print to stdout so we see "THINKING"
        original_emit = agent.emit_event

        def verbose_emit(
            type, content=None, data=None, code=None, tag=None, is_sub_event=False
        ):
            if type in [
                "thinking",
                "code_output",
                "error",
                "debug_thought",
                "debug_code",
            ]:
                print(f"   [{type.upper()}] {content or ''}")
            original_emit(type, content, data, code, tag, is_sub_event)

        agent.emit_event = verbose_emit

        result = await agent.query_sync(prompt=prompt, session_id=test_session_id)

        print(f"3. Agent Finished. Result: {result}")

    except Exception as e:
        print(f"❌ FAIL: Agent Loop Crashed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        print("4. Cleaning up...")
        db.query(
            "MATCH (n:Thought {session_id: $sid}) DETACH DELETE n",
            {"sid": test_session_id},
        )


async def inspect_session(session_id: str):
    """Prints the scratchpad context Gemini sees for a given session."""
    print(f"\n--- INSPECTING SESSION: {session_id} ---")
    from src.core.scratchpad_builder import scratchpad_builder

    try:
        context = scratchpad_builder.build_scratchpad(
            session_id=session_id,
            root_session_id=session_id,
            task="Diagnostic",
            current_step=0,
        )
        print("--- START SCRATCHPAD ---")
        print(context)
        print("--- END SCRATCHPAD ---")
        print(f"Total Character Count: {len(context)}")
    except Exception as e:
        print(f"❌ Error building scratchpad: {e}")


async def inspect_node(node_id: str):
    """Prints the raw prompt and properties of a specific node."""
    print(f"\n--- INSPECTING NODE: {node_id} ---")
    try:
        nodes = db.query("MATCH (n:Thought {id: $id}) RETURN n", {"id": node_id})
        if not nodes:
            print("❌ Node not found.")
            return
        node = nodes[0]["n"]
        props = node.properties if hasattr(node, "properties") else node
        print("--- PROPERTIES ---")
        for k, v in props.items():
            if k != "prompt":
                print(f"  {k}: {v}")
        print("\n--- PROMPT ---")
        print(props.get("prompt", "NO PROMPT"))
        print("--- END PROMPT ---")
    except Exception as e:
        print(f"❌ Error inspecting node: {e}")


async def llm_debug_test():
    """Tries a problematic prompt against Gemini to find malformed patterns."""
    print("\n--- LLM DEBUG TEST: Probing MALFORMED_FUNCTION_CALL ---")

    # Example problematic context: Mixed brackets, step numbering, etc.
    test_context = (
        "### EXECUTION TRACE (READ ONLY HISTORY)\n"
        "(Code) Step 1: print('hello')\n"
        "    -> Result: None\n"
        "(Code) Step 2: x = [1, 2, 3]\n"
        "    -> Result: [1, 2, 3]\n"
        "(SYSTEM) Step 3: ⚠️ REFLEXION_BREAK: System warning.\n"
        "    -> Next: Continue\n"
        "### END TRACE\n"
    )

    prompt = f"Diagnostic Test. Analyze history and say 'ACK'.\n\n{test_context}"

    print("Sending probe to LLM...")
    try:
        response = await llm.generate(prompt)
        print(f"✅ Success! Response: {response}")
    except Exception as e:
        print(f"❌ LLM FAILED: {e}")
        import traceback

        traceback.print_exc()


async def main():
    parser = argparse.ArgumentParser(description="Graph-RLM Live Diagnostics Tool")
    parser.add_argument(
        "--check",
        choices=["repe", "sheaf", "dreamer", "agent", "all"],
        default=None,
        help="Component to verify",
    )
    parser.add_argument(
        "--inspect",
        type=str,
        help="Inspect the scratchpad context for a specific session_id",
    )
    parser.add_argument(
        "--inspect-node",
        type=str,
        help="Inspect the raw prompt and properties of a specific node_id",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt for the Agent test",
    )
    parser.add_argument(
        "--llm-debug",
        action="store_true",
        help="Run a probe to debug MALFORMED_FUNCTION_CALL",
    )
    args = parser.parse_args()

    print("=== GRAPH-RLM LIVE DIAGNOSTICS ===")

    if args.check in ["repe", "all"]:
        await test_live_repe()

    if args.check in ["sheaf", "all"]:
        await test_live_sheaf()

    if args.check in ["dreamer", "all"]:
        await test_live_dreamer()

    if args.check in ["agent", "all"]:
        await test_live_agent(custom_prompt=args.prompt)

    if args.inspect:
        await inspect_session(args.inspect)

    if args.inspect_node:
        await inspect_node(args.inspect_node)

    if args.llm_debug:
        await llm_debug_test()


if __name__ == "__main__":
    asyncio.run(main())
