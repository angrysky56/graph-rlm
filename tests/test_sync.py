
import asyncio
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "graph_rlm", "backend", "src"))

async def manual_sync():
    print("Starting manual skill sync...")
    try:
        from mcp_integration.skill_storage import get_skills_manager
        mgr = get_skills_manager()
        print(f"Skills directory: {mgr.skills_dir}")
        await mgr.sync_from_disk()

        from core.db import db
        res = db.query('MATCH (s:Skill) RETURN count(s)')
        print(f"Skills count after sync: {res}")
    except Exception as e:
        print(f"Sync failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(manual_sync())
