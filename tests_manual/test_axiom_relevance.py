import asyncio

from graph_rlm.backend.src.core.sheaf import sheaf


async def test_axiom_filtering():
    print("🧪 Testing Axiomatic Filtering...")

    # Mock some skills in a temporary way or assume they exist
    # For a real test, we'd use the SkillsManager to inject them

    proposed_code = "print('Hello World'); result = 42"

    # Case 1: Physics task
    print("  Testing physics task tags...")
    diag = sheaf.check_axiomatic_consistency(proposed_code, task_tags=["physics"])
    print(f"    -> Status: {diag['status']}")

    # Case 2: Coding task
    print("  Testing coding task tags...")
    diag = sheaf.check_axiomatic_consistency(proposed_code, task_tags=["coding"])
    print(f"    -> Status: {diag['status']}")

    # Case 3: Untagged task (should still run untagged axioms)
    print("  Testing untagged task...")
    diag = sheaf.check_axiomatic_consistency(proposed_code, task_tags=[])
    print(f"    -> Status: {diag['status']}")

    print("✅ Axiomatic filtering verification logic completed!")


if __name__ == "__main__":
    asyncio.run(test_axiom_filtering())
