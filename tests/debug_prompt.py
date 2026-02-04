import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path("/home/ty/Repositories/ai_workspace/graph-rlm")
sys.path.append(str(project_root))

from graph_rlm.backend.src.core.agent import Agent

agent = Agent()
prompt = agent._build_system_prompt()
print("--- SYSTEM PROMPT ---")
print(prompt)
print("--- END ---")
