import asyncio

from graph_rlm.backend.src.core.sheaf import sheaf
from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager


async def test_axiom_transparency():
    print("\n--- Testing Axiom Transparency (Sandbox Isolation) ---")

    # 1. Register a dummy general axiom
    mgr = get_skills_manager()
    axiom_code = """
def validate_general_test(result):
    if result == "TRIGGER_VIOLATION":
        return False
    return True
"""
    mgr.save_skill(
        name="axiom_test_transparency",
        code=axiom_code,
        description="Axiom transparency test.",
        tags=["general"],
    )

    # 2. Proposed code using 'rlm' and 'mcp' (should NOT crash)
    proposed_code = """
async def run():
    await rlm.recall("test")
    return "TRIGGER_VIOLATION"
result = asyncio.run(run())
"""

    print("Running axiomatic check on code using rlm/mcp/asyncio...")
    # PASS THE TAG to filter for ONLY our test axiom and avoid noise from other crashing axioms
    diag = sheaf.check_axiomatic_consistency(
        proposed_code, task_tags=["transparency_test"]
    )

    print(f"Status: {diag['status']}")
    if diag["status"] == "AXIOMATIC_VIOLATION":
        print(f"Critique: {diag['critique']}")
        print("✅ Axiom Violation explicitly detected despite rlm/mcp usage.")
    else:
        print(f"❌ Failed to detect violation or crashed silently. Diag: {diag}")


if __name__ == "__main__":
    asyncio.run(test_axiom_transparency())
