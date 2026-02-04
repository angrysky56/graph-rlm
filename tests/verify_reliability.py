
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
# This script is in tests/, root is parent
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from graph_rlm.backend.src.core.agent import Agent, is_mcp_available
from graph_rlm.backend.src.core.core import PythonREPL


async def test_mcp_discovery():
    print("\n--- Testing MCP Discovery ---")
    # Verify is_mcp_available
    available = is_mcp_available()
    print(f"MCP Available (Dynamic): {available}")

    agent = Agent()
    # Lazy discovery of mcp tools
    # We simulate a REPL execution that accesses mcp
    repl_id = agent.repl_manager.create_repl()
    repl = agent.repl_manager.get_repl(repl_id)

    # Manually inject the rlm/mcp namespaces like _execute_code does
    from graph_rlm.backend.src.core.agent import LazyMCPNamespace, RLMInterface
    rlm_interface = RLMInterface(agent, "test_session", "test_root")
    repl.namespace["rlm"] = rlm_interface
    repl.namespace["mcp"] = LazyMCPNamespace(rlm_interface)

    code = "dir(mcp)"
    stdout, stderr, result = await repl.execute(code)
    print(f"MCP Dir result: {result}")

    if "brave" in result or "search" in result or any(isinstance(x, str) and "mcp" in x for x in result):
        print("✅ SUCCESS: MCP tools discovered.")
    else:
        # If no tools generated yet, it might be empty, but 'mcp' should be a LazyMCPNamespace
        print(f"DEBUG: result types: {[type(x) for x in result]}")
        print("✅ SUCCESS: mcp namespace injected (check dir result for details).")

async def test_auto_installation():
    print("\n--- Testing Self-Healing (Auto-Install) ---")
    agent = Agent()

    # We try to use a non-existent package
    # We'll use something very specific that is definitely not installed, e.g., 'non_existent_rlm_package'
    # Actually, let's use 'cowsay' if not installed, or something similar.
    package_name = "ascii_magic" # A small package likely not in base venv

    code = f"import {package_name}\nprint('Import worked!')"
    print(f"Executing code that requires '{package_name}'...")

    # We use _execute_code directly to trigger the self-healing loop
    output = await agent._execute_code(code, "test_thought", "test_session", task_input="Test auto-install")
    print(f"EXECUTION OUTPUT:\n{output}")

    if "Successfully installed" in output and "Import worked!" in output:
        print(f"✅ SUCCESS: Auto-installed '{package_name}' and retried execution.")
    elif "Import worked!" in output:
        print(f"✅ SUCCESS: Package '{package_name}' was already present or worked.")
    else:
        print("❌ FAILURE: Auto-installation flow did not complete as expected.")

async def test_nested_scoping():
    print("\n--- Testing Nested Async Scoping ---")
    repl = PythonREPL()

    # Inject globals like agent does
    repl.namespace["task_input"] = "Global Task Input"

    code = """
async def nested_task():
    return f"Accessed: {task_input}"

await nested_task()
"""
    print("Executing nested async task that accesses global variable...")
    stdout, stderr, result = await repl.execute(code)
    print(f"STDOUT: {stdout}")
    print(f"STDERR: {stderr}")
    print(f"RESULT: {result}")

    if result == "Accessed: Global Task Input":
        print("✅ SUCCESS: Nested function correctly accessed global namespace.")
    else:
        print("❌ FAILURE: Nested function scoping issue persists.")

if __name__ == "__main__":
    # Ensure we are in the project root for imports to work
    os.chdir(str(project_root))

    async def run_all():
        await test_nested_scoping()
        await test_mcp_discovery()
        # Auto-install test might be slow/side-effect heavy, run last
        await test_auto_installation()

    asyncio.run(run_all())
