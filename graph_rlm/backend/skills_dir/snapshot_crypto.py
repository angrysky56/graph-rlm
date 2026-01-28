def snapshot_crypto(symbols: list[str]) -> dict:
    """
    Fetch current market snapshot for a list of crypto symbols (e.g., ['BTC-USD', 'ETH-USD'])
    using the coinbase-trade-assistant MCP server wrapper.

    Args:
        symbols: List of trading pairs like 'BTC-USD'

    Returns:
        Dict of market data for each symbol.
    """
    try:
        from coinbase_trade_assistant import get_market_data

        product_ids = ",".join(symbols)
        raw_result = get_market_data(product_ids=product_ids)
        import json

        parsed = json.loads(raw_result)
        return parsed.get("market_data", [])
    except Exception as e:
        return {"error": str(e)}
