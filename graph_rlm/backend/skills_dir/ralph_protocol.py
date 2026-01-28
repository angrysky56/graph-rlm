import os

# trunk-ignore(bandit/B404)
import subprocess
from typing import Callable


class RalphProtocol:
    """
    Ralph: Recursive Adaptive Logic Processing Hub (Stateless)
    Implementation of the 'Industrial' over 'Agentic' philosophy.
    """

    def __init__(self, workspace_root: str):
        self.workspace = workspace_root
        self.spec_path = os.path.join(workspace_root, "PROMPT.md")
        self.test_dir = os.path.join(workspace_root, "tests")

    def die(self):
        """Wipes the transient memory/context (The 'Die' step)."""
        # In a Graph-RLM context, this triggers a new REPL ID and
        # flushes the local scratchpad to the Global Workspace.
        print("[Ralph] Segment complete. Terminating current state context...")
        return None

    def execute_and_verify(self, logic_fn: Callable, *args) -> bool:
        """
        The Backpressure loop.
        Instead of asking the LLM if it's correct, we run the artifacts.
        """
        print("[Ralph] Executing logic chunk...")
        logic_fn(*args)

        # External Verification (Compiler/Test Runner)
        print("[Ralph] Applying Backpressure (External Verification)...")
        # trunk-ignore(bandit/B603)
        # trunk-ignore(bandit/B607)
        result = subprocess.run(["pytest", self.test_dir], capture_output=True)

        if result.returncode == 0:
            print("[Ralph] Verification Passed. Committing to Disk (Global State).")
            return True
        else:
            print(f"[Ralph] Verification Failed. Error: {result.stderr.decode()}")
            return False

    def ralph_loop(self, subtasks: list):
        """The 'Repeat' cycle: Polynomial execution of N tasks."""
        for i, task in enumerate(subtasks):
            print(f"\n--- Starting Task {i+1}: {task['name']} ---")
            success = False
            attempts = 0
            while not success and attempts < 3:
                # 1. Zero-Shot/Few-Shot Execution based on current Disk state
                success = self.execute_and_verify(task["func"], task["params"])
                attempts += 1

            # 2. Mandatory Memory Flush
            self.die()

            if not success:
                print(
                    f"[Ralph] Task {task['name']} reached exhaustion. Manual intervention required."
                )
                break
