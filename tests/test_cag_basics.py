import asyncio
import os
import sys
import uuid
from pathlib import Path

# Setup PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from graph_rlm.backend.src.core.sheaf import sheaf
from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager


async def test_axiom_blocking():
    print("🚀 Starting Axiomatic Consistency Verification...")

    # 1. Manually create an axiom (guardrail)
    mgr = get_skills_manager()
    # We use a unique name to avoid collision during parallel runs/reruns
    axiom_name = f"axiom_test_max_val_{uuid.uuid4().hex[:4]}"
    axiom_code = f"""
def validate_max_value(val):
    \"\"\"Constraint: Value must be less than 50.\"\"\"
    if isinstance(val, (int, float)) and val >= 50:
        return False
    return True
"""
    mgr.save_skill(
        name=axiom_name,
        code=axiom_code,
        description="Value must be less than 50.",
        tags=["axiom", "test"],
    )

    try:
        # 2. Test code that violates the axiom
        # Note: SheafMonitor calls repl.execute which might need 'result' variable set
        # Our check_axiomatic_consistency executes code and then calls the validator on 'result'
        violating_code = "result = 100\nresult"
        diag = sheaf.check_axiomatic_consistency(violating_code)
        print(f"  -> Violating Code Status: {diag['status']}")
        if diag["status"] != "AXIOMATIC_VIOLATION":
            print(f"  ❌ ERROR: Expected AXIOMATIC_VIOLATION, got {diag['status']}")
            return False

        # 3. Test code that satisfies the axiom
        valid_code = "result = 10\nresult"
        diag = sheaf.check_axiomatic_consistency(valid_code)
        print(f"  -> Valid Code Status: {diag['status']}")
        if diag["status"] != "HEALTHY":
            print(f"  ❌ ERROR: Expected HEALTHY, got {diag['status']}")
            return False

        print("✅ Axiomatic consistency basics passed!")
        return True
    finally:
        # Cleanup
        skill_file = mgr.skills_dir / f"{axiom_name}.py"
        if skill_file.exists():
            skill_file.unlink()


if __name__ == "__main__":
    success = asyncio.run(test_axiom_blocking())
    sys.exit(0 if success else 1)
