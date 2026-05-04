import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add repo root to sys.path
repo_root = Path("/home/ty/Repositories/ai_workspace/graph-rlm")
sys.path.insert(0, str(repo_root))

# Mock DB before importing meta_agents
from unittest.mock import MagicMock

sys.modules["graph_rlm.backend.src.core.db"] = MagicMock()
sys.modules["graph_rlm.backend.src.core.db"].db = MagicMock()

from graph_rlm.backend.src.core.meta_agents import SubAgentProfile, meta_agents


async def test_meta_agent_discovery():
    print("Testing async meta-agent profiling...")

    task = "Research latest gold prices and save to a spreadsheet"

    # Mock Skills Manager
    mock_skills_mgr = AsyncMock()
    mock_skills_mgr.find_similar_skills.return_value = [
        {
            "name": "spreadsheet_tool",
            "function_name": "save_to_csv",
            "description": "Saves data to a CSV file.",
        }
    ]

    # Mock LLM generate_structured
    from graph_rlm.backend.src.core.llm import llm

    # We need to mock generate_structured because it uses Pydantic AI/LLM
    original_generate_structured = llm.generate_structured
    llm.generate_structured = AsyncMock()

    # Simulate LLM response
    mock_profile = SubAgentProfile(
        persona="Data Researcher",
        tools=["mcp.brave_search", "skills.spreadsheet_tool", "rlm.recall"],
        reasoning="Task requires web search for prices and spreadsheet tool for saving.",
    )
    llm.generate_structured.return_value = mock_profile

    try:
        profile = await meta_agents.generate_sub_agent_profile(
            task, skills_manager=mock_skills_mgr, mcp_names=["brave_search", "files"]
        )

        print(f"Synthesized Persona: {profile['persona']}")
        print(f"Assigned Tools: {profile['tools']}")
        print(f"Reasoning: {profile.get('reasoning')}")

        assert profile["persona"] == "Data Researcher"
        assert "mcp.brave_search" in profile["tools"]
        assert "skills.spreadsheet_tool" in profile["tools"]
        print("SUCCESS: Meta-agent discovery verified.")

    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore mock
        llm.generate_structured = original_generate_structured


if __name__ == "__main__":
    asyncio.run(test_meta_agent_discovery())
