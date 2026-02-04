import asyncio
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.llm import llm
from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager


async def verify_discovery():
    print("🚀 Starting Agentic Axiom Discovery Verification...")

    agent = Agent()
    mgr = get_skills_manager()

    # 1. Prepare a Mock Axiom in the Skills DB
    print("📝 Preparing Mock Physics Axiom...")
    axiom_code = """
def verify_mass_conservation(initial_mass, final_mass):
    \"\"\"AXIOM: Mass must be conserved in closed systems.\"\"\"
    return abs(initial_mass - final_mass) < 1e-6
"""
    # Use save_skill (which now embeds)
    mgr.save_skill(
        name="physics_mass_axiom",
        code=axiom_code,
        description="Axiomatic verification of mass conservation in physics simulations.",
        tags=["physics", "axiom"],
    )

    # 2. Test the discovery loop
    prompt = "Simulate a laminar flow in a pipe and ensure mass is conserved."
    code = "import numpy as np; # simulation logic here"

    print(f"🔍 Testing discovery for prompt: '{prompt}'")

    # We expect _detect_required_axioms_agentic to:
    # 1. Ask LLM for invariants (e.g. 'mass conservation')
    # 2. Find 'physics_mass_axiom' via semantic search
    # 3. Return ['physics'] (or similar)

    tags = await agent._detect_required_axioms_agentic(prompt, code)

    print(f"🛡️  Discovered Tags: {tags}")

    if "physics" in tags:
        print("✅ SUCCESS: Physics domain discovered agentically.")
    else:
        print("❌ FAILURE: Physics domain NOT discovered.")

    # 3. Verify Async LLM Stability
    print("⚡ Verifying Async LLM Stability...")
    try:
        resp = await llm.generate("Hello, are you operational?")
        print(f"LLM Response: {resp[:50]}...")
        print("✅ SUCCESS: Async LLM is stable.")
    except Exception as e:
        print(f"❌ FAILURE: Async LLM failed: {e}")


if __name__ == "__main__":
    asyncio.run(verify_discovery())
