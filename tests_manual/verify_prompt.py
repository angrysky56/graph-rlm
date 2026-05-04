
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from graph_rlm.backend.src.core.agent import Agent


def test_prompt():
    agent = Agent()
    try:
        prompt = agent._build_system_prompt()
        print(f"Prompt Length: {len(prompt)}")
        print("--- PROMPT START ---")
        print(prompt[:500])
        print("...")
        print(prompt[-500:])
        print("--- PROMPT END ---")

        # Check for placeholders or missing vars
        if "{tool_list_str}" in prompt:
            print("!!! ERROR: tool_list_str not interpolated !!!")
        if "{skills_list_str}" in prompt:
            print("!!! ERROR: skills_list_str not interpolated !!!")

    except Exception as e:
        print(f"!!! PROMPT GEN FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prompt()
