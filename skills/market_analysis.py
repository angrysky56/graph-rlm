"""
Market Analysis Skill.

Orchestrates a market analysis for a specific sector by combining
advanced reasoning with academic paper searches.
"""

import logging
from typing import Any, Dict

from graph_rlm.backend.mcp_tools import call_tool

logger = logging.getLogger("graph_rlm.skills.market_analysis")


async def market_analysis(sector: str) -> Dict[str, Any]:
    """
    Performs a multi-step market analysis for a given sector.

    Args:
        sector: The market sector to analyze (e.g., 'Semiconductors', 'SaaS').

    Returns:
        A dictionary containing the analysis plan, relevant papers, and insights.
    """
    # 1. Quick Plan
    print(f"Planning analysis for {sector}...")
    try:
        plan = await call_tool(
            "advanced_reasoning_server",
            "advanced_reasoning",
            {
                "thought": f"Analyze market trends for {sector}.",
                "nextThoughtNeeded": True,
                "thoughtNumber": 1,
                "totalThoughts": 2,
                "confidence": 0.8,
            },
        )
    except RuntimeError as e:
        logger.error("Planning phase failed: %s", e)
        plan = f"Planning error: {e}"

    # 2. Search
    print("Searching for recent papers...")
    try:
        data = await call_tool(
            "arxiv_mcp_server", "search_papers", {"query": sector, "max_results": 3}
        )
    except RuntimeError as e:
        logger.error("Search phase failed: %s", e)
        data = []

    # 3. Insight
    print("Generating insights...")
    try:
        evidence_list = [str(p) for p in data] if isinstance(data, list) else []
        insight = await call_tool(
            "advanced_reasoning_server",
            "advanced_reasoning",
            {
                "thought": f"Synthesize trends for {sector} based on recent findings.",
                "evidence": evidence_list,
                "nextThoughtNeeded": False,
                "thoughtNumber": 2,
                "totalThoughts": 2,
                "confidence": 0.9,
            },
        )
    except RuntimeError as e:
        logger.error("Insight generation failed: %s", e)
        insight = f"Insight error: {e}"

    return {
        "sector": sector,
        "plan": plan,
        "paper_count": len(data) if isinstance(data, list) else 0,
        "insight": str(insight)[:500],
    }
