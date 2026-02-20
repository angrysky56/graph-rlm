
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
repo_root = Path("/home/ty/Repositories/ai_workspace/graph-rlm").resolve()
sys.path.insert(0, str(repo_root))

from graph_rlm.backend.src.mcp_integration.discovery import discover_all_servers


async def main():
    config_path = repo_root / "mcp_servers.json"
    print(f"Testing discovery with {config_path}")
    results = await discover_all_servers(config_path)

    for name, info in results.items():
        if "error" in info:
            print(f"❌ {name}: {info['error']}")
        else:
            print(f"✅ {name}: {len(info.get('tools', {}))} tools found")

if __name__ == "__main__":
    asyncio.run(main())
