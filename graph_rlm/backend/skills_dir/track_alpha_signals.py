async def track_alpha_signals(top_n: int = 5) -> dict:
    """
    REFINED ALPHA TRACKER (Titans/Miras Logic):
    Identifies high-potential 'alpha' signals by correlating social media 'Attentional Spikes'
    with market data status.

    Orchestration Pipeline:
    1. Multi-Vector Attentional Search: Targets high-noise social hubs (X, Reddit).
    2. Heuristic Ticker Extraction: Identifies $TICKER and uppercase clusters via regex.
    3. Attentional Density Scoring: Measures mention frequency as a proxy for social momentum.
    4. Market Gating: Validates against Coinbase to classify as 'Early Stage' (DEX) vs 'Mainstream'.
    5. Alpha Scoring: Multiplier applied to Early Stage assets to highlight high-potential signals.

    Args:
        top_n: Number of top signals to return.

    Returns:
        Dict containing top_signals, metadata, and status.
    """
    import re
    import json
    from graph_rlm.backend.mcp_tools import call_tool

    # 1. Attentional Search
    queries = [
        "site:x.com 'pump' 'moon' '100x' crypto",
        "site:reddit.com/r/CryptoMoonShots 'gem' 'launch'",
        "top trending crypto tokens social volume spike"
    ]
    
    aggregated_text = ""
    for q in queries:
        try:
            res = await call_tool("brave-search", "brave_web_search", {"query": q})
            aggregated_text += str(res) + " "
        except:
            pass

    # 2. Heuristic Extraction
    # Pattern: $TICKER or uppercase words 3-6 chars long
    raw_tickers = re.findall(r"\$([A-Z0-9]{3,8})|(?<=\s)([A-Z]{3,6})(?=\s|\.|\,)", aggregated_text)
    
    extracted = []
    for group in raw_tickers:
        for item in group:
            if item: extracted.append(item.upper())
            
    # Blacklist common noise words
    blacklist = {
        'BTC', 'ETH', 'SOL', 'USD', 'USDT', 'USDC', 'THE', 'AND', 'FOR', 'ARE', 
        'THIS', 'THAT', 'NEW', 'TOP', 'X', 'MEME', 'CRYPTO', 'MOON', 'PUMP', 'BUY',
        'CALL', 'TOKEN', 'OUR', 'LOCKED', 'LP', 'CA', 'GET', 'BIG'
    }
    candidates = [t for t in extracted if t not in blacklist]
    
    # 3. Attentional Density Scoring
    counts = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
        
    sorted_signals = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Market Gating and Alpha Scoring
    final_results = []
    for ticker, mentions in sorted_signals[:10]:
        market_status = "EARLY_STAGE (DEX/Pre-Listing)"
        price = 0.0
        volume = 0.0
        
        try:
            market_res = await call_tool("coinbase-trade-assistant", "get_market_data", {"product_ids": f"{ticker}-USD"})
            if isinstance(market_res, str):
                m_data = json.loads(market_res)
                products = m_data.get("market_data", [])
                if products and "error" not in products[0]:
                    market_status = "MAINSTREAM (Coinbase Listed)"
                    price = float(products[0].get("price", 0))
                    volume = float(products[0].get("volume_24h", 0))
        except:
            pass

        # Alpha Score: Mentions * (2.5 multiplier for Early Stage assets)
        multiplier = 2.5 if market_status.startswith("EARLY") else 1.0
        alpha_score = round(mentions * multiplier, 2)
        
        final_results.append({
            "ticker": ticker,
            "mentions": mentions,
            "market_status": market_status,
            "price": price,
            "volume_24h": volume,
            "alpha_score": alpha_score
        })

    final_results.sort(key=lambda x: x["alpha_score"], reverse=True)

    return {
        "status": "success",
        "top_signals": final_results[:top_n],
        "metadata": {
            "total_raw_mentions": len(extracted),
            "unique_candidates": len(counts)
        }
    }
