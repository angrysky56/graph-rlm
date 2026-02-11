"""
Debug Team Skill.

Orchestrates a multi-agent debugging workflow using specialized roles:
- Architect: Evaluates codebase structure using AST.
- Planner: Creates step-by-step fix strategies.
- Engineer: Implements code changes.
- QA: Verifies fixes and reports outcome.
"""

import logging
from pathlib import Path
from typing import Callable

from graph_rlm.backend.mcp_tools import call_tool
from graph_rlm.backend.src.core.agent import agent

logger = logging.getLogger("graph_rlm.skills.debug_team")


async def debug_codebase(target_path: str, issue_description: str = "") -> str:
    """
    Main entry point: Orchestrate full debugging workflow across multiple roles.

    Args:
        target_path: Absolute path to the repository to debug.
        issue_description: Description of the bug or optimization goal.

    Returns:
        A summary report of the debugging session.
    """
    repo = Path(target_path)
    if not repo.exists():
        return f"Error: {repo} does not exist."

    debug_dir = repo / "DEBUGTEAM"
    try:
        debug_dir.mkdir(exist_ok=True)
    except OSError as e:
        return f"Error creating debug directory: {e}"

    objective = (
        f"Debug and fix: {issue_description}"
        if issue_description
        else "Full debug/optimize."
    )

    def load_prompt(role: str) -> str:
        """Loads specialized system prompts from the knowledge base."""
        # Workspace-specific Knowledge Base path
        kb_path = Path("/home/ty/Repositories/ai_workspace/graph-rlm/knowledge_base")
        prompt_path = kb_path / "prompts" / "agency" / f"{role}.md"

        if prompt_path.exists():
            try:
                return prompt_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("Failed to read prompt for %s: %s", role, e)

        return f"You are {role.title()}. Focus on debugging and high-quality results."

    # 1. analyze_codebase
    print(f"🔬 Phase 1: Analyzing {repo.name}...")
    analysis = await _run_analyze_phase(repo, debug_dir, objective, load_prompt)
    try:
        await call_tool(
            "memory",
            "save_memory",
            {
                "text": analysis,
                "metadata": {"type": "debug_analysis", "repo": repo.name},
            },
        )
    except RuntimeError:
        pass

    # 2. create_debug_plan
    print(f"📋 Phase 2: Planning fixes for {repo.name}...")
    plan = await _run_plan_phase(repo, debug_dir, objective, load_prompt, analysis)

    # 3. execute_fixes
    print(f"🛠️ Phase 3: Executing fixes in {repo.name}...")
    fixes = await _run_execute_phase(repo, debug_dir, objective, load_prompt, plan)

    # 4. verify_fixes
    print(f"✅ Phase 4: Verifying {repo.name}...")
    verification = await _run_verify_phase(repo, debug_dir, objective, load_prompt)

    final_report = f"""
=== DEBUG TEAM COMPLETE ===
Repo: {repo.name}
Objective: {objective}

Analysis: {debug_dir / 'analysis.md'}
Plan: {debug_dir / 'debug_plan.md'}
Fixes: {debug_dir / 'fixes_log.md'} ({len(fixes)} chars)
Verification: {debug_dir / 'verification.md'}

{verification[:1500]}...
    """

    try:
        await call_tool(
            "memory",
            "save_memory",
            {
                "text": final_report,
                "metadata": {"type": "debug_complete", "repo": repo.name},
            },
        )
    except RuntimeError:
        pass

    return final_report


# Placeholder functions for tool discovery if needed, though workflow is inside debug_codebase
def analyze_codebase(path: str, issue: str = "") -> str:
    """Architect evaluates using AST. (Part of debug_codebase workflow)"""
    _ = path, issue  # Mark as used for linting
    return "Use debug_codebase for full run."


def create_debug_plan(path: str, evaluation: str) -> str:
    """Planner designs the strategy. (Part of debug_codebase workflow)"""
    _ = path, evaluation  # Mark as used for linting
    return "Use debug_codebase for full run."


async def _run_analyze_phase(
    repo: Path, debug_dir: Path, objective: str, load_prompt: Callable[[str], str]
) -> str:
    logger.info("Analyze: %s", repo.name)

    # AST Project Analysis integration
    try:
        ast_result = await call_tool(
            "ast-asg",
            "analyze_project",
            {"project_path": str(repo), "project_name": repo.name},
        )
        ast_str = str(ast_result)[:30000]
        try:
            await call_tool(
                "memory",
                "save_memory",
                {
                    "text": ast_str,
                    "metadata": {"type": "ast_analysis", "repo": repo.name},
                },
            )
        except RuntimeError:
            pass
        analysis_context = f"AST Project Analysis:\n{ast_str[:16000]}...\n"
    except RuntimeError as e:
        logger.warning("AST Analysis failed: %s", e)
        analysis_context = ""

    # Check memory for relevant prior experience
    try:
        memories_result = await call_tool(
            "memory", "search_memory", {"query": objective, "n_results": 10}
        )
        memories = memories_result if isinstance(memories_result, list) else []
        memory_text = (
            "\n".join(f"- {m}" for m in memories)
            if memories
            else "No relevant memories found."
        )
    except RuntimeError:
        memory_text = "Memory search unavailable."

    system_prompt = load_prompt("architect")
    full_prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT:\n{analysis_context}\n"
        f"MEMORIES:\n{memory_text}\n"
        f"OBJECTIVE: {objective}\n\n"
        f"Analyze this codebase for bugs and provide a comprehensive report."
    )

    # Use agent.query_sync directly (it is an async function in this codebase)
    report = await agent.query_sync(
        prompt=full_prompt,
        session_id=f"debug_architect_{repo.name}",
    )

    try:
        (debug_dir / "analysis.md").write_text(report, encoding="utf-8")
    except OSError as e:
        logger.error("Failed to write analysis.md: %s", e)

    return report


async def _run_plan_phase(
    repo: Path,
    debug_dir: Path,
    objective: str,
    load_prompt: Callable[[str], str],
    analysis: str,
) -> str:
    logger.info("Plan: %s", repo.name)
    system_prompt = load_prompt("planner")
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Analysis: {analysis}\n"
        f"OBJECTIVE: {objective}\n"
        f"Plan the debugging and fix steps sequentially."
    )

    plan = await agent.query_sync(
        prompt=full_prompt,
        session_id=f"debug_planner_{repo.name}",
    )

    try:
        (debug_dir / "debug_plan.md").write_text(plan, encoding="utf-8")
    except OSError as e:
        logger.error("Failed to write debug_plan.md: %s", e)

    return plan


async def _run_execute_phase(
    repo: Path,
    debug_dir: Path,
    objective: str,
    load_prompt: Callable[[str], str],
    plan: str,
) -> str:
    _ = objective  # Mark as used for linting
    logger.info("Execute: %s", repo.name)
    system_prompt = load_prompt("engineer")
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Plan:\n{plan}\n\n"
        f"Implement these fixes carefully. Use AST transformations if applicable."
    )

    result = await agent.query_sync(
        prompt=full_prompt,
        session_id=f"debug_engineer_{repo.name}",
    )

    try:
        (debug_dir / "fixes_log.md").write_text(result, encoding="utf-8")
    except OSError as e:
        logger.error("Failed to write fixes_log.md: %s", e)

    return result


async def _run_verify_phase(
    repo: Path, debug_dir: Path, objective: str, load_prompt: Callable[[str], str]
) -> str:
    _ = objective  # Mark as used for linting
    logger.info("Verify: %s", repo.name)
    system_prompt = load_prompt("qa")
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Verify the implemented fixes. Re-inspect the codebase to ensure no new bugs were introduced."
    )

    report = await agent.query_sync(
        prompt=full_prompt,
        session_id=f"debug_qa_{repo.name}",
    )

    try:
        (debug_dir / "verification.md").write_text(report, encoding="utf-8")
    except OSError as e:
        logger.error("Failed to write verification.md: %s", e)

    return report
