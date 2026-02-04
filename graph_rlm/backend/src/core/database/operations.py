from typing import Any, Dict, List, Optional
import time

from ..logger import get_logger
from ..guardrails import GuardrailError, validate_thought_node
from .client import client

logger = get_logger("graph_rlm.db.operations")

def create_thought_node(
    thought_id: str,
    prompt: str,
    parent_id: Optional[str] = None,
    prompt_embedding: Optional[List[float]] = None,
    session_id: str = "default",
    root_session_id: Optional[str] = None,
    repl_id: Optional[str] = None,
    status: str = "pending",
    execution_summary: Optional[str] = None,
    result: Optional[str] = None,
    next_action: Optional[str] = None,
    dreamer_analysis: Optional[str] = None,
    final_response: Optional[str] = None,
    round_id: Optional[str] = None,
):
    """
    Creates a 'Thought' node in the graph.
    If parent_id is provided, creates a DECOMPOSES_INTO edge from parent to child.
    """
    # If root_session_id is not provided, default to the session_id (implies this IS the root)
    final_root = root_session_id if root_session_id else session_id

    # --- GUARDRAILS ---
    try:
        # Check if parent exists and get its metadata for continuity check
        parent_meta = None
        if parent_id:
            p_res = client.query(
                "MATCH (p:Thought {id: $pid}) RETURN p.session_id as session_id, p.root_session_id as root_session_id",
                {"pid": parent_id},
            )
            if p_res:
                parent_meta = p_res[0]

        validate_thought_node(
            thought_id=thought_id,
            prompt=prompt,
            parent_id=parent_id,
            session_id=session_id,
            root_session_id=final_root,
            parent_metadata=parent_meta,
        )
    except GuardrailError as ge:
        logger.error(f"Guardrail Violation: {ge}")
        raise
    except Exception as e:
        logger.error(f"Guardrail internal error: {e}")

    params: Dict[str, Any] = {
        "tid": thought_id,
        "prompt": prompt,
        "sid": session_id,
        "rsid": final_root,
        "status": status,
    }

    # Create the node
    cypher = """
    MERGE (t:Thought {id: $tid})
    SET t.prompt = $prompt, t.status = $status, t.created_at = timestamp(), t.session_id = $sid, t.root_session_id = $rsid
    """
    if prompt_embedding:
        params["vec"] = prompt_embedding
        cypher += ", t.embedding = vecf32($vec)"

    if repl_id:
        params["repl_id"] = repl_id
        cypher += ", t.repl_id = $repl_id"

    if round_id:
        params["rid"] = round_id
        cypher += ", t.round_id = $rid"

    if execution_summary:
        params["exec_summary"] = execution_summary
        cypher += ", t.execution_summary = $exec_summary"

    if result:
        params["result"] = result
        cypher += ", t.result = $result"

    if next_action:
        params["next_action"] = next_action
        cypher += ", t.next_action = $next_action"

    if dreamer_analysis:
        params["dreamer_analysis"] = dreamer_analysis
        cypher += ", t.dreamer_analysis = $dreamer_analysis"

    if final_response:
        params["final_response"] = final_response
        cypher += ", t.final_response = $final_response"

    client.query(cypher, params)

    # Link to parent if exists
    if parent_id:
        edge_params = {"tid": thought_id, "pid": parent_id}
        edge_cypher = """
        MATCH (parent:Thought {id: $pid})
        MATCH (child:Thought {id: $tid})
        MERGE (parent)-[:DECOMPOSES_INTO]->(child)
        """
        client.query(edge_cypher, edge_params)

def get_parent_id(thought_id: str) -> Optional[str]:
    """
    Retrieves the parent ID of a thought node.
    """
    cypher = """
    MATCH (p:Thought)-[:DECOMPOSES_INTO]->(c:Thought {id: $tid})
    RETURN p.id as pid
    LIMIT 1
    """
    res = client.query(cypher, {"tid": thought_id})
    if res and "pid" in res[0]:
        return res[0]["pid"]
    return None

def delete_thought_node(thought_id: str):
    """
    Physically deletes a thought node and its interactions from the graph.
    """
    cypher = "MATCH (n:Thought {id: $tid}) DETACH DELETE n"
    client.query(cypher, {"tid": thought_id})
    logger.info(f"♻️ Graph Hygiene: Pruned thought node {thought_id}")

def update_thought_result(
    thought_id: str,
    result: str,
    embedding: Optional[List[float]] = None,
    repl_id: Optional[str] = None,
    status: str = "complete",
):
    params: Dict[str, Any] = {
        "tid": thought_id,
        "result": result,
        "status": status,
    }
    cypher = """
    MATCH (t:Thought {id: $tid})
    SET t.result = $result, t.status = $status, t.completed_at = timestamp()
    """
    if embedding:
        params["vec"] = embedding
        cypher += ", t.embedding = vecf32($vec)"

    if repl_id:
        params["repl_id"] = repl_id
        cypher += ", t.repl_id = $repl_id"

    client.query(cypher, params)

def get_graph_state():
    """
    Returns the entire graph structure for visualization.
    """
    cypher = """
    MATCH (n:Thought)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN n, r, m
    """
    return client.query(cypher)

def get_context_frontier(
    repl_id: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieves the 'Frontier' of the conversation for a given session.
    """
    params = {"sid": repl_id, "limit": limit}

    cypher = f"""
    MATCH (n:Thought)
    WHERE n.session_id = $sid
    RETURN n
    ORDER BY n.created_at DESC
    LIMIT {limit}
    """

    try:
        return client.query(cypher, params)
    except Exception as e:
        logger.error(f"Failed to get context frontier: {e}")
        return []

# ===== ROUND MANAGEMENT =====

def save_round(
    round_id: str,
    root_session_id: str,
    user_prompt: str,
    repl_ids: List[str],
    final_response: str,
    full_scratchpad: str,
    started_at: int,
    ended_at: int,
):
    """
    Archives a completed round to the graph.
    """
    params = {
        "rid": round_id,
        "rsid": root_session_id,
        "prompt": user_prompt,
        "repl_ids": repl_ids,
        "final": final_response,
        "scratchpad": full_scratchpad,
        "started": started_at,
        "ended": ended_at,
    }

    cypher = """
    CREATE (r:Round {
        round_id: $rid,
        root_session_id: $rsid,
        user_prompt: $prompt,
        repl_ids: $repl_ids,
        final_response: $final,
        full_scratchpad: $scratchpad,
        started_at: $started,
        ended_at: $ended
    })
    """
    client.query(cypher, params)
    logger.info(f"Archived Round {round_id} for session {root_session_id}")

def get_completed_rounds(root_session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all completed rounds for a session.
    """
    params = {"rsid": root_session_id}
    cypher = """
    MATCH (r:Round)
    WHERE r.root_session_id = $rsid
    RETURN r.round_id as round_id,
           r.user_prompt as user_prompt,
           r.repl_ids as repl_ids,
           r.final_response as final_response,
           r.ended_at as ended_at
    ORDER BY r.ended_at ASC
    """
    return client.query(cypher, params)

def get_session_trace(root_session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves the full trace of thoughts for a given root session.
    """
    q = """
    MATCH (n:Thought)
    WHERE n.root_session_id = $rsid
    RETURN n.id as id,
           n.prompt as prompt,
           n.status as status,
           n.result as result,
           n.created_at as created_at,
           n.repl_id as repl_id,
           n.execution_summary as execution_summary,
           n.next_action as next_action,
           n.dreamer_analysis as dreamer_analysis,
           n.final_response as final_response,
           n.round_id as round_id
    ORDER BY n.created_at ASC
    """
    return client.query(q, {"rsid": root_session_id})

def delete_session(root_session_id: str):
    """
    Deletes an entire session context.
    """
    # 1. Delete Thoughts
    cypher_thoughts = (
        "MATCH (n:Thought) WHERE n.root_session_id = $rsid DETACH DELETE n"
    )
    client.query(cypher_thoughts, {"rsid": root_session_id})

    # 2. Delete Rounds
    cypher_rounds = (
        "MATCH (r:Round) WHERE r.root_session_id = $rsid DETACH DELETE r"
    )
    client.query(cypher_rounds, {"rsid": root_session_id})

    logger.info(f"🗑️ Deleted session {root_session_id}")

def prune_orphans(older_than_hours: int = 1) -> int:
    """
    Deletes orphaned Thought nodes.
    """
    cypher = """
    MATCH (n:Thought)
    WHERE NOT (n)--()
    AND n.created_at < $cutoff
    DETACH DELETE n
    RETURN count(n) as count
    """
    current_millis = int(time.time() * 1000)
    cutoff_millis = current_millis - (older_than_hours * 3600 * 1000)

    res = client.query(cypher, {"cutoff": cutoff_millis})
    count = res[0]["count"] if res else 0
    logger.info(f"🧹 Pruned {count} orphan nodes (older than {older_than_hours}h)")
    return count

def reset_graph():
    """
    NUCLEAR OPTION: Wipes the entire database.
    """
    from .search import create_vector_indexes

    client.query("MATCH (n) DETACH DELETE n")
    logger.warning("☢️ GRAPH RESET PERFORMED ☢️")
    create_vector_indexes()

def mark_nodes_as_consolidated(node_ids: List[str], insight_id: str):
    """
    Closes the Gestalt on failed nodes.
    """
    cypher = """
    MATCH (t:Thought)
    WHERE t.id IN $ids
    SET t.status = 'consolidated', t.consolidated_at = timestamp()
    WITH t
    MATCH (i:Insight {id: $iid})
    MERGE (t)-[:CONSOLIDATED_INTO]->(i)
    """
    client.query(cypher, {"ids": node_ids, "iid": insight_id})

def perform_synaptic_homeostasis(retention_window: int = 24):
    """
    Implements the Synaptic Homeostasis Hypothesis (SHY).
    """
    current_ms = int(time.time() * 1000)
    cutoff = current_ms - (retention_window * 3600 * 1000)

    cypher = """
    MATCH (t:Thought)
    WHERE t.status = 'consolidated'
      AND t.consolidated_at < $cutoff
    DETACH DELETE t
    RETURN count(t) as count
    """
    res = client.query(cypher, {"cutoff": cutoff})
    count = res[0]["count"] if res else 0
    if count > 0:
        logger.info(
            f"🧠 Synaptic Homeostasis: Pruned {count} saturated memory traces."
        )
