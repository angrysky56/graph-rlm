import asyncio
import sys
from pathlib import Path

# Setup PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from graph_rlm.backend.src.core.dream import dreamer
from graph_rlm.backend.src.mcp_integration.skill_storage import get_skills_manager


async def test_ingestion():
    print("🚀 Starting Document Ingestion Verification...")

    # 1. Create a dummy document
    doc_path = project_root / "tests" / "test_knowledge.txt"
    doc_content = """
    Fluid Dynamics Rules:
    1. The pressure P must always be positive.
    2. The velocity V must never exceed 300 units.
    """
    doc_path.write_text(doc_content)

    try:
        # 2. Perform ingestion
        print(f"  -> Ingesting {doc_path}...")
        res = await dreamer.ingest_document(str(doc_path), "fluid_dynamics")

        if res.get("status") != "success":
            print(f"  ❌ ERROR: Ingestion failed: {res.get('message')}")
            return False

        codified = res.get("codified_axioms", [])
        print(f"  -> Codified Axioms: {codified}")

        if not codified:
            print("  ❌ ERROR: No axioms codified.")
            return False

        # 3. Verify skills exist
        mgr = get_skills_manager()
        for axiom in codified:
            skill = mgr.get_skill(axiom)
            if not skill:
                print(f"  ❌ ERROR: Skill {axiom} not found in library.")
                return False
            print(f"  ✅ Verified Skill: {axiom}")

        print("✅ Document Ingestion Verification Passed!")
        return True

    finally:
        # Cleanup
        if doc_path.exists():
            doc_path.unlink()


if __name__ == "__main__":
    success = asyncio.run(test_ingestion())
    sys.exit(0 if success else 1)
