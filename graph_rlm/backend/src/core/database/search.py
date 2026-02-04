from typing import Any, Dict, List, Optional
import time

from ..logger import get_logger
from .client import client

logger = get_logger("graph_rlm.db.search")

def find_similar_thoughts(
    query_embedding: list[float], limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Finds thoughts with similar embeddings to the query and returns structured results.
    """
    if not client.use_falkor:
        # NetworkX backend does not support vector search yet.
        return []

    # Ensure embedding is the correct length
    if len(query_embedding) != 3072:
        logger.warning(
            f"Vector search failed: Embedding dimension mismatch (expected 3072, got {len(query_embedding)})"
        )
        return []

    params: Dict[str, Any] = {"vec": query_embedding}
    # FalkorDB syntax requires quoted strings for label and property name
    cypher = f"CALL db.idx.vector.queryNodes('Thought', 'embedding', {limit}, vecf32($vec)) YIELD node, score RETURN node.id, node.prompt, node.result, score"

    try:
        res = client.raw_graph.query(cypher, params)
        results = []
        for row in res.result_set:
            results.append(
                {"id": row[0], "prompt": row[1], "result": row[2], "score": row[3]}
            )
        return results
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []

def create_vector_indexes():
    """
    Creates vector indexes on Thought.embedding and Skill.embedding.
    """
    if not client.use_falkor:
        return

    dim = 3072  # Gemini default

    # 1. Thought Index
    try:
        cypher = f"CREATE VECTOR INDEX FOR (t:Thought) ON (t.embedding) OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
        client.raw_graph.query(cypher)
        logger.info(f"Sync: Vector Index on Thought(embedding) created (dim={dim})")
    except Exception as e:
        if "already indexed" not in str(e).lower():
            logger.warning(f"Thought vector index creation skipped: {e}")

    # 2. Skill Index
    try:
        cypher = f"CREATE VECTOR INDEX FOR (s:Skill) ON (s.embedding) OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
        client.raw_graph.query(cypher)
        logger.info(f"Sync: Vector Index on Skill(embedding) created (dim={dim})")
    except Exception as e:
        if "already indexed" not in str(e).lower():
            logger.warning(f"Skill vector index creation skipped: {e}")

    # 3. Axiom Index
    try:
        cypher = f"CREATE VECTOR INDEX FOR (a:Axiom) ON (a.embedding) OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
        client.raw_graph.query(cypher)
        logger.info(f"Sync: Vector Index on Axiom(embedding) created (dim={dim})")
    except Exception as e:
        if "already indexed" not in str(e).lower():
            logger.warning(f"Axiom vector index creation skipped: {e}")

def drop_vector_index():
    """
    Drops the vector index on Thought.embedding.
    """
    if not client.use_falkor:
        return

    try:
        client.query("DROP INDEX FOR (t:Thought) ON (t.embedding)")
        logger.info("Dropped Vector Index on Thought.embedding")
    except Exception as e:
        logger.info(f"Vector index drop skipped: {e}")

def wait_for_index(label: str):
    if not client.use_falkor:
        return

    # Poll db.indexes() until status is OPERATIONAL
    for _ in range(20):
        try:
            res = client.query(
                "CALL db.indexes() YIELD label, status RETURN label, status"
            )
            # res is List[Dict] e.g. [{'label': 'Thought', 'status': 'OPERATIONAL'}]
            for row in res:
                # Handle both list (driver) and dict (wrapper) formats
                r_label, r_status = None, None
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    r_label = row[0]
                    r_status = row[1]
                elif isinstance(row, dict):
                    r_label = row.get("label")
                    r_status = row.get("status")

                if r_label == label and r_status == "OPERATIONAL":
                    return
        except Exception as e:
            logger.debug(f"Index check polling error: {e}")
        time.sleep(0.5)

def reembed_all_thoughts(llm_service: Any):
    """
    Iterates through all Thought nodes and refreshes their embeddings.
    Useful when switching embedding models.
    """
    if not client.use_falkor:
        logger.warning("Re-embedding skipped: Not using FalkorDB")
        return 0

    from .operations import update_thought_result

    logger.info("Starting graph-wide re-embedding process...")
    # 1. Fetch all nodes with enough text to embed
    cypher = "MATCH (n:Thought) RETURN n.id as id, n.prompt as prompt, n.result as result"
    # Using raw client for consistent list-of-lists format
    res = client.raw_graph.query(cypher)
    nodes = res.result_set

    count = 0
    for row in nodes:
        # Result set is list of lists: [id, prompt, result]
        if not row or len(row) < 2:
            continue

        node_id = row[0]
        prompt = row[1] if row[1] is not None else ""
        result = row[2] if len(row) > 2 and row[2] is not None else ""

        if not isinstance(node_id, str):
            continue

        # Combine prompt and result for better context representation if both exist
        text_to_embed = prompt
        if result:
            text_to_embed += f"\nResult: {result}"

        if not text_to_embed:
            continue

        try:
            # Use provided LLM service to get NEW embedding
            new_vec = llm_service.get_embedding(text_to_embed)
            if new_vec:
                # Update node in FalkorDB
                update_thought_result(
                    thought_id=node_id,
                    result=result,  # Keep existing result
                    embedding=new_vec,
                    status="complete",
                )
                count += 1
        except Exception as e:
            logger.error(f"Failed to re-embed thought {node_id}: {e}")

    logger.info(f"Re-embedding complete. Updated {count} thoughts.")
    return count
