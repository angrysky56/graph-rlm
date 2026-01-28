"""
Auto-generated wrapper for coinbase-trade-assistant MCP server.

This module provides Python function wrappers for all tools
exposed by the coinbase-trade-assistant server.

Do not edit manually.
"""

from typing import Any


def screen_all_coins(limit: int | None = None, min_volume_usd: float | None = None) -> Any:
    """
Run comprehensive technical screening across all available coins

Args:
    limit: Maximum number of coins to screen (default: 50)
    min_volume_usd: Minimum 24h volume in USD to consider (default: 100000)

Returns:
    JSON string with screening results


    Args:
        limit: 
        min_volume_usd: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if limit is not None:
        params["limit"] = limit
    if min_volume_usd is not None:
        params["min_volume_usd"] = min_volume_usd


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="coinbase-trade-assistant",
            tool_name="screen_all_coins",
            arguments=params,
        )
    return asyncio.run(_async_call())


def analyze_coin(product_id: str, timeframe: str | None = None) -> Any:
    """
Perform deep technical analysis on a specific cryptocurrency

Args:
    product_id: Trading pair to analyze (e.g., 'BTC-USD')
    timeframe: Analysis timeframe ('1h', '4h', '1d')

Returns:
    JSON string with detailed analysis


    Args:
        product_id: 
        timeframe: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if product_id is not None:
        params["product_id"] = product_id
    if timeframe is not None:
        params["timeframe"] = timeframe


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="coinbase-trade-assistant",
            tool_name="analyze_coin",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_market_data(product_ids: str | None = None) -> Any:
    """
Get current market data for specified cryptocurrencies

Args:
    product_ids: Comma-separated list of trading pairs

Returns:
    JSON string with current market data


    Args:
        product_ids: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if product_ids is not None:
        params["product_ids"] = product_ids


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="coinbase-trade-assistant",
            tool_name="get_market_data",
            arguments=params,
        )
    return asyncio.run(_async_call())


def check_signals() -> Any:
    """
Review recent trading signals from the last screening run

Returns:
    JSON string with recent signals and their status


    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="coinbase-trade-assistant",
            tool_name="check_signals",
            arguments=params,
        )
    return asyncio.run(_async_call())


def monitor_portfolio(watchlist: str | None = None) -> Any:
    """
Monitor a custom watchlist of cryptocurrencies for trading signals

Args:
    watchlist: Comma-separated list of trading pairs to monitor

Returns:
    JSON string with monitoring results


    Args:
        watchlist: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if watchlist is not None:
        params["watchlist"] = watchlist


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="coinbase-trade-assistant",
            tool_name="monitor_portfolio",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['screen_all_coins', 'analyze_coin', 'get_market_data', 'check_signals', 'monitor_portfolio']
