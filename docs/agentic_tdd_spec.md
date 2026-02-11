# Agentic TDD System Specification

Here is a formal pseudo-code definition for an **Agentic TDD System** utilizing a mandatory `uv` execution environment.

This specification integrates the **Role-Goal-Backstory** framework with the **Iterative Feedback Loops** characteristic of agentic coding.

## System Specification: Agentic TDD Orchestrator

```python
"""
AGENTIC TDD ORCHESTRATOR
Version: 1.0.0
Architecture: Chicago School TDD (State Verification)
Environment: uv (Mandatory Activation)
"""

class SystemEnvironment:
    def __init__(self):
        # INVARIANT: The environment is deterministic and enforced by the system, not the agent.
        self.project_root = "./workspace"
        self.venv_status = "LOCKED" 
    
    def initialize(self):
        """
        Phase 0: The Lock-In.
        Agents cannot bypass this. If this fails, the system halts.
        """
        execute_shell("uv init")
        execute_shell("uv venv")
        if not check_exit_code(0):
            raise SystemError("CRITICAL: uv environment failed to initialize.")
        self.venv_status = "ACTIVE"

    def run_command_securely(self, command: str) -> ExecutionResult:
        """
        The Gatekeeper.
        Wraps ALL agent execution requests in 'uv run' to enforce dependencies.
        """
        # Security: Prevent agents from breaking out of the environment
        if "pip install" in command:
             return ExecutionResult(status="DENIED", log="Direct pip usage forbidden. Edit pyproject.toml.")
        
        # Enforce uv execution
        secure_cmd = f"uv run {command}"
        result = execute_shell(secure_cmd)
        
        return ExecutionResult(
            status="PASS" if result.exit_code == 0 else "FAIL",
            stdout=result.stdout,
            [cite_start]stderr=result.stderr  # Critical for the feedback loop [cite: 320]
        )

class Agent:
    """
    [cite_start]Base class implementing the CrewAI 'Role-Goal-Backstory' framework[cite: 957].
    """
    def __init__(self, role, goal, backstory, allowed_tools):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = allowed_tools

# --- AGENT DEFINITIONS ---

TestArchitect = Agent(
    role="Senior Test Architect",
    goal="Create failing tests that define behavior. Do NOT implement logic.",
    backstory="You are a QA veteran who strictly follows Chicago School TDD.",
    allowed_tools=["write_file(tests/*)", "read_file(src/*)"]
)

ImplementationEngineer = Agent(
    role="Pragmatic Systems Engineer",
    goal="Write minimal code to pass the current error log.",
    backstory="You rely strictly on error logs. You do not over-engineer.",
    allowed_tools=["write_file(src/*)", "read_file(tests/*)", "read_file(logs/*)"]
)

# --- THE MAIN EXECUTION LOOP ---

def main(user_feature_request: str):
    # 1. System Bootstrap
    Environment = SystemEnvironment()
    Environment.initialize()

    # 2. Phase 1: The "Red" State (Specification)
    print("--- PHASE 1: RED (TEST CREATION) ---")
    
    test_file_path = f"tests/test_{generate_id()}.py"
    
    # Architect drafts the test based on requirements
    TestArchitect.act(
        task=f"Create a test for: {user_feature_request}",
        constraint="Test must be self-contained and strictly assert inputs/outputs.",
        output_file=test_file_path
    )

    # VERIFICATION: The test MUST fail.
    # If it passes now, the test is invalid (False Positive).
    initial_run = Environment.run_command_securely(f"pytest {test_file_path}")
    
    if initial_run.status == "PASS":
        raise ProcessError("VIOLATION: Test passed before implementation. Architect must rewrite.")
    else:
        current_error_log = initial_run.stderr
        print(f"CONFIRMED RED: Test failed as expected. Error: {current_error_log}")

    # 3. Phase 2: The "Green" State (Implementation Loop)
    print("--- PHASE 2: GREEN (IMPLEMENTATION) ---")
    
    max_retries = 5
    attempts = 0
    
    while attempts < max_retries:
        # [cite_start]Agentic Feedback Loop: The log is the prompt [cite: 275]
        ImplementationEngineer.act(
            task="Fix the codebase to resolve the error log.",
            context={
                "test_content": read_file(test_file_path),
                "error_log": current_error_log # The critical feedback mechanism
            },
            constraint="Do NOT edit the test file. Only edit src/ files."
        )

        # Execute immediately via uv
        verification_run = Environment.run_command_securely(f"pytest {test_file_path}")

        if verification_run.status == "PASS":
            print("SUCCESS: Tests passed.")
            break
        else:
            # [cite_start]Self-Correction: Update the log and recurse [cite: 423]
            current_error_log = verification_run.stderr
            attempts += 1
            print(f"RETRY {attempts}/{max_retries}: New error detected -> {current_error_log}")

    if attempts == max_retries:
        print("FAILURE: Max retries exceeded. Human intervention required.")
        exit(1)

    # 4. Phase 3: The "Refactor" State (Optimization)
    print("--- PHASE 3: REFACTOR ---")
    
    # Optional: Engineer cleans up code while ensuring tests still pass
    ImplementationEngineer.act(
        task="Refactor the code for readability without changing behavior.",
        constraint="Run tests after every change."
    )
    
    final_check = Environment.run_command_securely(f"pytest {test_file_path}")
    if final_check.status == "PASS":
        print("TDD CYCLE COMPLETE: Feature deployed.")
    else:
        print("REFACTOR FAILED: Reverting to last known green state.")

if __name__ == "__main__":
    main("Create a Fibonacci calculator")
```
