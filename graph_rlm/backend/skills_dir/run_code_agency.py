import logging
import asyncio
from pathlib import Path
from typing import Literal

from graph_rlm.backend.src.core.agent import agent
from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

logger = logging.getLogger("graph_rlm.skills.code_agency")

# --- Model Roster (Free/Cheap Tier) ---
# Kept for reference. Graph-RLM Agent currently uses the globally configured model.
MODEL_ROSTER = {
    "architect": "google/gemini-3-pro-preview",
    "planner": "deepseek/deepseek-v3.2",
    "engineer": "moonshotai/kimi-k2-thinking",
    "qa": "x-ai/grok-4.1-fast",
}


async def run_code_agency(
    repo_path: str,
    objective: str,
    phase: Literal["evaluate", "plan", "execute", "verify"] = "evaluate",
) -> str:
    """
    Run the Autonomous Code Agency on a target repository.

    The Agency simulates a professional team with specialized roles:
    - Architect (Evaluate): Analyzes system design and risks.
    - Tech Lead (Plan): Creates atomic implementation steps.
    - Engineer (Execute): Writes code.
    - QA (Verify): Runs tests and debugs.

    Args:
        repo_path: Absolute path to the target repository.
        objective: The high-level goal (e.g., "Refactor async cleanup").
        phase: The current phase of the workflow to execute.

    Returns:
        A report string regarding phase completion.
    """
    repo = Path(repo_path)

    if not repo.exists():
        return f"Error: Repository path {repo} does not exist."

    # Ensure agency directory exists
    agency_dir = repo / ".CODEAGENCY"
    agency_dir.mkdir(exist_ok=True)

    # --- Helper: Load Prompt ---
    def load_prompt(role: str) -> str:
        # Fallback for knowledge base path
        kb_path = Path("/home/ty/Repositories/ai_workspace/graph-rlm/knowledge_base")
        prompt_path = kb_path / "prompts" / "agency" / f"{role}.md"

        if not prompt_path.exists():
            logger.warning(
                f"Prompt for {role} not found at {prompt_path}, using fallback."
            )
            return f"You are the {role}. Please perform your duties responsibly."
        return prompt_path.read_text()

    # --- Phase 1: Evaluate (Architect) ---
    if phase == "evaluate":
        logger.info(f"Starting Agency Evaluation on {repo.name}...")

        # 1. Recall Institutional Knowledge
        memories_result = await call_mcp_tool("memory", "search_memory", {"query": objective, "n_results": 3})
        memories = memories_result if isinstance(memories_result, list) else []

        memory_text = (
            "\n".join([f"- {m}" for m in memories])
            if memories
            else "No relevant past memories."
        )

        system_prompt = load_prompt("architect")
        eval_prompt = f"""
        {system_prompt}

        # Context
        PROJECT: {repo.name}
        TARGET_REPOSITORY_PATH: {repo}
        OBJECTIVE: {objective}

        # CRITICAL: The target repository to analyze is at: {repo}
        # Do NOT analyze /app - that is the mcp_coordinator source, NOT the target.
        # Use tools like read_file, list_directory to explore the TARGET repository.

        # Institutional Memory (Lessons Learned)
        {memory_text}

        Please proceed with the analysis of the repository at {repo}.
        """

        report = await asyncio.to_thread(
            agent.query_sync,
            prompt=f"IMPORTANT: Analyze the repository at {repo} (NOT /app). Use file reading tools to explore {repo}. Objective: {objective}",
            session_id=f"agency_architect_{repo.name}",
        )

        output_file = agency_dir / "evaluation.md"
        output_file.write_text(report)

        return f"Evaluation Phase Complete.\nReport saved to: {output_file}\n\nSummary:\n{report[:500]}..."

    # --- Phase 2: Plan (Tech Lead) ---
    elif phase == "plan":
        logger.info(f"Starting Agency Planning on {repo.name}...")

        eval_file = agency_dir / "evaluation.md"
        if not eval_file.exists():
            return (
                "Error: specific evaluation.md not found. Run 'evaluate' phase first."
            )

        eval_content = eval_file.read_text()
        system_prompt = load_prompt("planner")

        plan_prompt = f"""
        {system_prompt}

        # Context
        PROJECT: {repo.name}
        TARGET_REPOSITORY_PATH: {repo}
        OBJECTIVE: {objective}

        # CRITICAL: The target repository is at: {repo}
        # Do NOT reference /app - that is NOT the target repository.

        # Architect's Evaluation
        {eval_content}

        Create the implementation plan for files in {repo}.
        """

        plan = await asyncio.to_thread(
            agent.query_sync,
            prompt=plan_prompt + "\n\nTask: Create the implementation plan based on the evaluation.",
            session_id=f"agency_planner_{repo.name}",
        )

        output_file = agency_dir / "plan.md"
        output_file.write_text(plan)

        return f"Planning Phase Complete.\nPlan saved to: {output_file}\n\nSummary:\n{plan[:500]}..."

    # --- Phase 3: Execute (Engineer) ---
    elif phase == "execute":
        logger.info(f"Starting Agency Execution on {repo.name}...")

        plan_file = agency_dir / "plan.md"
        if not plan_file.exists():
            return "Error: plan.md not found. Run 'plan' phase first."

        plan_content = plan_file.read_text()
        system_prompt = load_prompt("engineer")

        exec_prompt = f"""
        {system_prompt}

        # Context
        PROJECT: {repo.name}
        TARGET_REPOSITORY_PATH: {repo}

        # CRITICAL: All file edits must be in {repo}, NOT /app.
        # /app is the coordinator source - DO NOT modify it.

        # Implementation Plan
        {plan_content}

        Start executing the plan in {repo}. Update the work log as you go.
        """

        # Initialize work log
        log_file = agency_dir / "work_log.md"
        if not log_file.exists():
            log_file.write_text(
                f"# Work Log ({repo.name})\nStarted execution for: {objective}\n"
            )

        result = await asyncio.to_thread(
            agent.query_sync,
            prompt=exec_prompt + "\n\nTask: Execute Phase 1 and 2 of the plan. Stop if you encounter critical blocking errors.",
            session_id=f"agency_engineer_{repo.name}",
        )

        return f"Execution Phase Complete.\nResult: {result}"

    # --- Phase 4: Verify (QA) ---
    elif phase == "verify":
        logger.info(f"Starting Agency Verification on {repo.name}...")

        system_prompt = load_prompt("qa")

        qa_prompt = f"""
        {system_prompt}

        # Context
        PROJECT: {repo.name}
        TARGET_REPOSITORY_PATH: {repo}
        OBJECTIVE: {objective}

        # CRITICAL: Verify files in {repo}, NOT /app.

        Please verify the current state of the repository at {repo}.
        """

        report = await asyncio.to_thread(
            agent.query_sync,
            prompt=qa_prompt + "\n\nTask: Run necessary verification steps (tests, linters) and report status.",
            session_id=f"agency_qa_{repo.name}",
        )

        output_file = agency_dir / "test_report.md"
        output_file.write_text(report)

        return f"Verification Phase Complete.\nReport saved to: {output_file}\n\nSummary:\n{report[:5000]}..."

    else:
        return f"Unknown phase: {phase}. Valid phases: evaluate, plan, execute, verify."
