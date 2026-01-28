async def wolfram_query(query: str, format: str = "text") -> str:
    """
    Executes a query using the Wolfram Language MCP server.

    Args:
        query: The Wolfram Language code or natural language query (if the server supports it).
        format: The desired output format (default: "text").

    Returns:
        The result of the Wolfram computation.
    """
    from graph_rlm.backend.mcp_tools.wolfram import WolframLanguageEvaluate

    try:
        # We use WolframLanguageEvaluate as the primary interface for the rhennigan server
        result = await WolframLanguageEvaluate(code=query)
        return result
    except Exception as e:
        return f"Error executing Wolfram query: {str(e)}"
