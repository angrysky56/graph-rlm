
import asyncio
import os
from pathlib import Path

from graph_rlm.backend.src.mcp_integration.runtime import AgentRuntime


async def verify_skill_import():
    project_root = Path(os.getcwd())
    runtime = AgentRuntime(project_root)
    skills_dir = runtime.backend_root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy skill
    skill_file = skills_dir / "test_skill_import.py"
    skill_file.write_text("def hello(): return 'world'")

    try:
        # Try to import it in the kernel
        code = "import test_skill_import; print(test_skill_import.hello())"
        print(f"Executing: {code}")
        stdout, stderr, result, code = await runtime.execute(code, {"session_id": "test_import"})
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")

        if "world" in stdout:
            print("SUCCESS: Skill imported and executed!")
        else:
            print("FAILURE: Skill not imported correctly.")

    finally:
        # Cleanup
        if skill_file.exists():
            skill_file.unlink()
        # Kill session
        if "test_import" in runtime.sessions:
             runtime.sessions["test_import"]["process"].kill()

if __name__ == "__main__":
    asyncio.run(verify_skill_import())
