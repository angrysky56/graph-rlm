import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rlm.backend.src.core.db import db


async def manual_db_check():
    print("--- Manual DB Check for GEA Proof ---")

    # Check Insights
    cypher_insights = "MATCH (i:Insight) RETURN i.type as type, i.content as content, i.root_session_id as rsid LIMIT 10"
    insights = db.query(cypher_insights)
    print(f"\nRecent Insights ({len(insights)}):")
    for ins in insights:
        itype = ins.get("type", "unknown")
        content = ins.get("content", "") or ""
        rsid = ins.get("rsid", "none") or "none"
        print(f"  - [{itype}] {str(content)[:50]}... (Session: {str(rsid)[:8]})")

    # Check H0 Rank
    cypher_h0 = "MATCH (n:Thought) WHERE n.h0_rank IS NOT NULL RETURN n.h0_rank as h0, n.id as id ORDER BY n.created_at DESC LIMIT 10"
    h0_nodes = db.query(cypher_h0)
    print(f"\nRecent H0 Ranks ({len(h0_nodes)}):")
    for node in h0_nodes:
        h0 = node.get("h0", "N/A")
        nid = node.get("id", "unknown") or "unknown"
        print(f"  - H0: {h0} (Node: {str(nid)[:8]})")


if __name__ == "__main__":
    asyncio.run(manual_db_check())
