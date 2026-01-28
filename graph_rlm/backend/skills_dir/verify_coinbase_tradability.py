async def verify_coinbase_tradability(token_symbol: str):
    """
    Verifies if a token is tradable on Coinbase by checking for a USD or USDC product pair.
    
    Args:
        token_symbol: The ticker symbol (e.g., 'BTC', 'ETH', 'CBETH')
    
    Returns:
        dict: {'tradable': bool, 'product_id': str or None, 'error': str or None}
    """
    from graph_rlm.backend.mcp_tools import call_tool
    
    symbol = token_symbol.upper().strip()
    # Common pairs on Coinbase
    pairs_to_check = [f"{symbol}-USD", f"{symbol}-USDC"]
    
    for product_id in pairs_to_check:
        try:
            # Using get_market_data as a proxy for existence/tradability
            result = await call_tool("coinbase-trade-assistant", "get_market_data", {"product_id": product_id})
            if result and "error" not in str(result).lower():
                return {"tradable": True, "product_id": product_id, "error": None}
        except Exception as e:
            continue
            
    return {"tradable": False, "product_id": None, "error": f"No USD/USDC pair found for {symbol}"}
