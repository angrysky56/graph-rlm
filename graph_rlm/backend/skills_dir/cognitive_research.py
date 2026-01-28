"""
Cognitive Research skill that combines reasoning, external research, and knowledge storage.
"""

import datetime
import uuid
from typing import Any

# Import wrappers for our desired MCP capabilities
from graph_rlm.backend.mcp_tools.advanced_reasoning import advanced_reasoning
from graph_rlm.backend.mcp_tools.arxiv_mcp_server import search_papers
from graph_rlm.backend.mcp_tools.chroma import chroma_add_documents, chroma_create_collection
from graph_rlm.backend.mcp_tools.neo4j_mcp import write_cypher


async def perform_cognitive_research(
    topic: str, max_papers: int = 3, store_results: bool = True
) -> dict[str, Any]:
    """
    Perform a complete cognitive research cycle: Plan -> Search -> Synthesize -> Store.

    Args:
        topic: The research topic to investigate
        max_papers: Number of papers to include in analysis
        store_results: Whether to persist the session to the Neo4j graph and Chroma vector DB

    Returns:
        Dictionary containing the synthesis, found papers, and any storage results.
    """

    # 1. Plan
    print(f"🤔 Planning research on: {topic}")
    plan = await advanced_reasoning(
        thought=f"I need to research {topic}. I will find key papers and synthesize the current state of the art.",
        nextThoughtNeeded=True,
        thoughtNumber=1,
        totalThoughts=3,
        confidence=0.9,
        goal=f"Research {topic}",
    )

    # 2. Search
    print(f"📚 Searching Arxiv for: {topic}")
    raw_results = await search_papers(
        query=topic, max_results=max_papers, sort_by="relevance"
    )

    # Normalize results (handle potential text/dict variations)
    papers = []
    if isinstance(raw_results, list):
        papers = raw_results
    elif isinstance(raw_results, str):
        # Fallback if it returns a string representation
        papers = [{"title": "Raw Results", "summary": raw_results}]

    print(f"📄 Found {len(papers)} papers")

    # 3. Synthesize
    print("🧠 Synthesizing findings...")
    synthesis = await advanced_reasoning(
        thought=f"I found {len(papers)} papers on {topic}. I will now synthesize the key insights.",
        nextThoughtNeeded=False,
        thoughtNumber=2,
        totalThoughts=3,
        evidence=[str(p) for p in papers],
        confidence=0.95,
    )

    # Extract string content from synthesis for storage
    synthesis_text = str(synthesis)
    if hasattr(synthesis, "content") and isinstance(synthesis.content, list):
        synthesis_text = synthesis.content[0].text

    result = {
        "topic": topic,
        "plan": plan,
        "paper_count": len(papers),
        "papers": papers,
        "synthesis": synthesis,
    }

    # 4. Store (Optional)
    if store_results:
        # A. Store Structured Data in Neo4j
        print("💾 Storing Graph Data (Neo4j)...")
        try:
            cypher = """
            MERGE (r:ResearchSession {topic: $topic})
            SET r.timestamp = datetime(),
                r.paper_count = $count,
                r.synthesis_preview = $synthesis
            RETURN r
            """
            params = {
                "topic": topic,
                "count": len(papers),
                "synthesis": synthesis_text[:100],  # Truncate for property
            }

            storage_res = await write_cypher(query=cypher, params=params)
            result["neo4j_result"] = str(storage_res)
            print("   ✅ Neo4j storage successful")
        except Exception as e:
            print(f"   ⚠️ Neo4j storage failed: {e}")
            result["neo4j_error"] = str(e)

        # B. Store Vector Data in Chroma
        print("🧠 Storing Vector Data (Chroma)...")
        try:
            # Ensure collection exists
            collection_name = "cognitive_research"
            try:
                await chroma_create_collection(collection_name=collection_name)
            except Exception:
                pass  # Collection likely exists/error handled in server

            # Prepare documents
            documents = []
            ids = []
            metadatas = []

            # Add synthesis
            documents.append(f"Synthesis for {topic}: {synthesis_text}")
            ids.append(f"synthesis_{uuid.uuid4().hex[:8]}")
            metadatas.append(
                {
                    "type": "synthesis",
                    "topic": topic,
                    "timestamp": str(datetime.datetime.now()),
                }
            )

            # Add papers
            for i, p in enumerate(papers):
                if isinstance(p, dict):
                    content = p.get("summary", p.get("title", "Unknown"))
                    documents.append(f"Paper: {p.get('title', 'Unknown')}\n\n{content}")
                    ids.append(f"paper_{uuid.uuid4().hex[:8]}")
                    metadatas.append(
                        {"type": "paper", "topic": topic, "source": "arxiv"}
                    )

            # Batch add
            if documents:
                chroma_res = await chroma_add_documents(
                    collection_name=collection_name,
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas,
                )
                result["chroma_result"] = str(chroma_res)
                print(f"   ✅ Chroma storage successful ({len(documents)} docs)")
            else:
                print("   ℹ️ No documents to store in Chroma")

        except Exception as e:
            print(f"   ⚠️ Chroma storage failed: {e}")
            result["chroma_error"] = str(e)

    return result
