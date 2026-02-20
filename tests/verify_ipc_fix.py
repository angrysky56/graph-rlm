import asyncio
import logging

from graph_rlm.backend.src.core.agent import agent
from graph_rlm.backend.src.core.rlm_interface import RLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graph_rlm.test")


async def test_dynamic_dispatch():
    print("--- Testing Dynamic Skill Dispatch ---")

    # Initialize Interface
    rlm = RLMInterface(
        agent_instance=agent,
        session_id="test_session_verify_ipc",
        root_session_id="test_root_verify_ipc",
    )

    # 1. Test existing methods still work
    try:
        print("\n[TEST 1] Calling existing method 'describe_tools'...")
        desc = await rlm.describe_tools("mcp.brave_search")
        print(f"✅ Success: describe_tools returned {len(desc)} chars.")
    except Exception as e:
        print(f"❌ Failed: describe_tools raised {e}")

    # 2. Test Dynamic Dispatch (simulating run_code_agency)
    # We use a known existing skill or a dummy name to check dispatch logic
    # Since we can't easily install a new skill here, we'll try to call 'run_code_agency'
    # and expect it to fail within the semantic search/skill loading phase,
    # BUT NOT fail with "AttributeError: 'RLMInterface' object has no attribute 'run_code_agency'"

    try:
        print("\n[TEST 2] Calling dynamic method 'run_code_agency'...")
        # We expect this to execute the dynamic_skill_wrapper -> run_skill
        # It might return a string about the skill not being found or an error from run_skill
        # but it should NOT be an AttributeError.
        result = await rlm.run_code_agency(
            repo_path="/tmp/test_repo", objective="Verify IPC", phase="evaluate"
        )
        print(
            f"✅ Success: run_code_agency dispatch worked. Result type: {type(result)}"
        )
        print(f"Result preview: {str(result)[:100]}")

    except AttributeError as e:
        print(
            f"❌ Failed: AttributeError raised. Dynamic dispatch NOT working. Error: {e}"
        )
    except Exception as e:
        # If it fails inside run_skill, that's fine, it means dispatch worked
        print(
            f"✅ Success (Partial): Dispatch worked, but skill execution failed (expected in test env): {e}"
        )

    # 3. Test Positional Args Rejection
    try:
        print("\n[TEST 3] Calling dynamic method with positional args (should fail)...")
        res = await rlm.run_code_agency("arg1", "arg2")
        if "Error: Dynamic skill calls only support keyword arguments" in res:
            print("✅ Success: Positional args correctly rejected.")
        else:
            print(f"❌ Failed: Did not reject positional args. Result: {res}")
    except Exception as e:
        print(f"❌ Failed: Exception raised during positional arg test: {e}")


async def test_axiom_query_generation():
    print("\n--- Testing Axiom Query Generation ---")

    prompt = "Create a python script that deletes files from the system."
    try:
        query = await agent._generate_axiom_search_query(prompt)
        print(f"Generated Query: {query}")

        # Validation
        if (
            "python" in query.lower()
            or "persistence" in query.lower()
            or "file" in query.lower()
        ):
            print("✅ Success: Query contains relevant keywords.")
        elif query == prompt[:300]:
            print(
                "⚠️  Warning: Fallback used (LLM might be offline or returned error)."
            )
        else:
            print(f"❓ result: {query}")

    except Exception as e:
        print(f"❌ Failed: _generate_axiom_search_query raised {e}")


if __name__ == "__main__":
    asyncio.run(test_dynamic_dispatch())
    asyncio.run(test_axiom_query_generation())
