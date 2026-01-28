import logging
import asyncio
from pathlib import Path
from graph_rlm.backend.src.core.agent import agent
from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

logger = logging.getLogger("graph_rlm.skills.debug_team")

# Model Roster (optimized for free tiers) - Kept for reference, but currently Agent uses global config
# future-todo: allow passing model override to agent query
MODEL_ROSTER = {
    "architect": "google/gemini-3-flash-preview",
    "planner": "x-ai/grok-code-fast-1",
    "engineer": "z-ai/glm-4.7",
    "qa": "x-ai/grok-4.1-fast",
}


async def debug_codebase(target_path: str, issue_description: str = "") -> str:
    """
    Main entry point: Orchestrate full debugging workflow.
    """
    repo = Path(target_path)
    if not repo.exists():
        return f"Error: {repo} does not exist."

    debug_dir = repo / "DEBUGTEAM"
    debug_dir.mkdir(exist_ok=True)

    objective = (
        f"Debug and fix: {issue_description}"
        if issue_description
        else "Full debug/optimize."
    )

    def load_prompt(role: str) -> str:
        # Assuming knowledge_base is at graph-rlm root for now, or relative to cwd
        # Ideally we use a fixed project root.
        # graph-rlm structure: /home/ty/Repositories/ai_workspace/graph-rlm/knowledge_base
        # valid heuristic: look for knowledge_base in current working dir or parents

        # Hardcoded fallback for typical workspace
        kb_path = Path("/home/ty/Repositories/ai_workspace/graph-rlm/knowledge_base")
        prompt_path = kb_path / "prompts" / "agency" / f"{role}.md"

        if prompt_path.exists():
            return prompt_path.read_text()

        return f"You are {role.title()}. Focus on debugging."

    # 1. analyze_codebase
    analysis = await _run_analyze_phase(
        repo, debug_dir, objective, load_prompt
    )
    await call_mcp_tool("memory", "save_memory", {
        "text": analysis,
        "metadata": {"type": "debug_analysis", "repo": repo.name}
    })

    # 2. create_debug_plan
    plan = await _run_plan_phase(
        repo, debug_dir, objective, load_prompt, analysis
    )

    # 3. execute_fixes
    fixes = await _run_execute_phase(
        repo, debug_dir, objective, load_prompt, plan
    )

    # 4. verify_fixes
    verification = await _run_verify_phase(
        repo, debug_dir, objective, load_prompt
    )

    final_report = f"""
=== DEBUG TEAM COMPLETE ===
Repo: {repo.name}
Objective: {objective}

Analysis: {debug_dir / 'analysis.md'}
Plan: {debug_dir / 'debug_plan.md'}
Fixes: {debug_dir / 'fixes_log.md'} ({len(fixes)} chars)
Verification: {debug_dir / 'verification.md'}

{verification[:1500]}
    """
    await call_mcp_tool("memory", "save_memory", {
        "text": final_report,
        "metadata": {"type": "debug_complete", "repo": repo.name}
    })
    return final_report


def analyze_codebase(path: str, issue: str = "") -> str:
    """Architect evaluates using AST."""
    return "Implemented as part of full workflow. Use debug_codebase for full run."


def create_debug_plan(path: str, evaluation: str) -> str:
    return "Implemented in workflow."


def execute_fixes(path: str, plan: str) -> str:
    return "Implemented in workflow."


def verify_fixes(path: str) -> str:
    return "Implemented in workflow."


async def _run_analyze_phase(repo, debug_dir, objective, load_prompt):
    logger.info(f"Analyze: {repo.name}")

    # ASTmcp integration
    try:
        ast_result = await call_mcp_tool(
            "ast-asg", "analyze_project", {"project_path": str(repo), "project_name": repo.name}
        )
        ast_str = str(ast_result)[:30000]
        await call_mcp_tool("memory", "save_memory", {
            "text": ast_str,
            "metadata": {"type": "ast_analysis", "repo": repo.name}
        })
        analysis_context = f"AST Project Analysis:\n{ast_str[:16000]}...\n"
    except Exception as e:
        logger.warning(f"AST Analysis failed: {e}")
        analysis_context = ""

    memories_result = await call_mcp_tool("memory", "search_memory", {"query": objective, "n_results": 10})
    # Handle direct list return or dict with 'results'
    memories = memories_result if isinstance(memories_result, list) else []
    memory_text = "\n".join(f"- {m}" for m in memories) if memories else "No memories."

    system_prompt = load_prompt("architect")
    full_prompt = f"{system_prompt}\n\nCONTEXT:\n{analysis_context}MEMORIES:\n{memory_text}\nOBJECTIVE: {objective}\n\nAnalyze for bugs."

    # Use agent.query_sync wrapped in thread
    report = await asyncio.to_thread(
        agent.query_sync,
        prompt=full_prompt,
        session_id=f"debug_architect_{repo.name}",
    )

    file = debug_dir / "analysis.md"
    file.write_text(report)
    return report


async def _run_plan_phase(repo, debug_dir, objective, load_prompt, analysis):
    logger.info(f"Plan: {repo.name}")
    system_prompt = load_prompt("planner")
    full_prompt = f"{system_prompt}\n\nAnalysis: {analysis}\nOBJECTIVE: {objective}\nPlan fixes step-by-step."

    plan = await asyncio.to_thread(
        agent.query_sync,
        prompt=full_prompt,
        session_id=f"debug_planner_{repo.name}",
    )

    file = debug_dir / "debug_plan.md"
    file.write_text(plan)
    return plan


async def _run_execute_phase(repo, debug_dir, objective, load_prompt, plan):
    logger.info(f"Execute: {repo.name}")
    system_prompt = load_prompt("engineer")
    full_prompt = f"{system_prompt}\n\nPlan: {plan}\nImplement fixes carefully, backup changes. Use AST transform if needed."

    result = await asyncio.to_thread(
        agent.query_sync,
        prompt=full_prompt,
        session_id=f"debug_engineer_{repo.name}",
    )

    file = debug_dir / "fixes_log.md"
    file.write_text(result)
    return result


async def _run_verify_phase(repo, debug_dir, objective, load_prompt):
    logger.info(f"Verify: {repo.name}")
    system_prompt = load_prompt("qa")
    full_prompt = f"{system_prompt}\n\nVerify all changes, re-analyze with AST, test. Report issues."

    report = await asyncio.to_thread(
        agent.query_sync,
        prompt=full_prompt,
        session_id=f"debug_qa_{repo.name}",
    )

    file = debug_dir / "verification.md"
    file.write_text(report)
    return report
