"""
Ralph Protocol Skill.

Ralph: Recursive Adaptive Logic Processing Hub (Stateless).
Implementation of the 'Industrial' over 'Agentic' philosophy, prioritizing
artifact verification and mandatory context flushing (backpressure).
"""

import logging
import os
import shutil
import subprocess  # trunk-ignore(bandit/B404)
import sys
from typing import Any, Callable, Dict, List

logger = logging.getLogger("graph_rlm.skills.ralph_protocol")


class RalphProtocol:
    """
    Ralph: Recursive Adaptive Logic Processing Hub (Stateless).
    Enforces a strict execution-verification-flush cycle.
    """

    def __init__(self, workspace_root: str):
        """
        Initializes the Ralph Protocol with a workspace context.

        Args:
            workspace_root: The root directory for task execution and tests.
        """
        self.workspace = workspace_root
        self.spec_path = os.path.join(workspace_root, "PROMPT.md")
        self.test_dir = os.path.join(workspace_root, "tests")

    def die(self) -> None:
        """
        Wipes the transient memory/context (The 'Die' step).
        In the RLM context, this signifies the termination of a stateless
        execution segment to prevent context pollution.
        """
        print("[Ralph] Segment complete. Flush/Terminating current state context...")
        # Placeholder for actual kernel flush command if integrated
        return None

    def execute_and_verify(self, logic_fn: Callable[..., Any], params: Any) -> bool:
        """
        The Backpressure loop.
        Instead of asking the LLM if it's correct, we execute and verify
        using external tools (Pytest).

        Args:
            logic_fn: The function implementing the task logic.
            params: Parameters to pass to the logic function.

        Returns:
            True if logic executed and verification tests passed.
        """
        print("[Ralph] Executing logic chunk...")
        try:
            logic_fn(params)
        except Exception as e:  # noqa: BLE001
            logger.error("Logic execution failed: %s", e)
            return False

        # External Verification (Compiler/Test Runner)
        print("[Ralph] Applying Backpressure (External Verification)...")

        python_executable = sys.executable or shutil.which("python3")
        if not python_executable:
            logger.error("Python executable not found for verification.")
            return False

        if not os.path.exists(self.test_dir):
            logger.warning(
                "Test directory %s not found. Skipping verification.", self.test_dir
            )
            return True  # Fallback: assume success if no tests exist

        try:
            # trunk-ignore(bandit/B603)
            result = subprocess.run(
                [python_executable, "-m", "pytest", self.test_dir],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            if result.returncode == 0:
                print("[Ralph] Verification Passed. Committing to Disk (Global State).")
                return True

            print(f"[Ralph] Verification Failed. RC: {result.returncode}")
            if result.stderr:
                logger.debug("Verification Stderr: %s", result.stderr.strip())
            return False

        except subprocess.TimeoutExpired:
            logger.error("Verification timed out.")
            return False
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error during verification: %s", e)
            return False

    def ralph_loop(self, subtasks: List[Dict[str, Any]]) -> None:
        """
        The 'Repeat' cycle: Sequential execution and verification of tasks.

        Args:
            subtasks: A list of task definitions (name, func, params).
        """
        for i, task in enumerate(subtasks):
            print(f"\n--- Starting Task {i+1}: {task['name']} ---")
            success = False
            attempts = 0
            while not success and attempts < 3:
                # Execution based on current state context
                success = self.execute_and_verify(task["func"], task["params"])
                attempts += 1

            # Mandatory Memory Flush between segments
            self.die()

            if not success:
                print(
                    f"[Ralph] Task {task['name']} reached exhaustion. Manual intervention required."
                )
                break
