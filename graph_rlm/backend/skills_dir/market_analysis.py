async def market_analysis(sector: str) -> dict:
    from graph_rlm.backend.mcp_tools.advanced_reasoning import advanced_reasoning
    from graph_rlm.backend.mcp_tools.arxiv_mcp_server import search_papers

    # 1. Quick Plan
    print(f"Planning analysis for {sector}...")
    plan = await advanced_reasoning(
        thought=f"Analyze market trends for {sector}.",
        nextThoughtNeeded=True,
        thoughtNumber=1,
        totalThoughts=2,
        confidence=0.8,
    )

    # 2. Search
    print("Searching for recent papers...")
    data = await search_papers(query=sector, max_results=3)

    # 3. Insight
    print("Generating insights...")
    insight = await advanced_reasoning(
        thought=f"Synthesize trends for {sector} based on recent papers.",
        evidence=[str(p) for p in data] if isinstance(data, list) else [],
        nextThoughtNeeded=False,
        thoughtNumber=2,
        totalThoughts=2,
        confidence=0.9,
    )

    return {
        "sector": sector,
        "plan": plan,
        "paper_count": len(data) if isinstance(data, list) else 0,
        "insight": str(insight)[:200],
    }
