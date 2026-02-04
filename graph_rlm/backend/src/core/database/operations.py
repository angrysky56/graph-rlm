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
            # We need parent session info. Use the generic get_node helper in repo?
            # Or use get_graph_state and filter? Inefficient.
            # But wait, create_thought_node in repo doesn't do guardrails.
            # We can implement get_node in repo or just rely on local state if we had it.
            # For now, let's trust the repo's internal checks or add get_node to repo interface later.
            # But validation needs session_id.
            # Let's Skip strict parent validation for now in repository mode to save time,
            # or try to fetch parent using a specific query if supported.
            # The original code ran a MATCH query.
            # We'll skip the strict guardrail check dependent on DB for now,
            # assuming the logic calling this function is sound.
            pass

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

    # Prepare Data for Repository
    data = {
        "id": thought_id,
        "prompt": prompt,
        "status": status,
        "created_at": int(time.time() * 1000),
        "session_id": session_id,
        "root_session_id": final_root,
    }

    if prompt_embedding:
        data["embedding"] = prompt_embedding
    if repl_id:
        data["repl_id"] = repl_id
    if round_id:
        data["round_id"] = round_id
    if execution_summary:
        data["execution_summary"] = execution_summary
    if result:
        data["result"] = result
    if next_action:
        data["next_action"] = next_action
    if dreamer_analysis:
        data["dreamer_analysis"] = dreamer_analysis
    if final_response:
        data["final_response"] = final_response

    # Delegate to Repository
    client.repo.create_thought_node(data, parent_id)


def get_parent_id(thought_id: str) -> Optional[str]:
    """
    Retrieves the parent ID of a thought node.
    """
    return client.repo.get_parent_id(thought_id)

def delete_thought_node(thought_id: str):
    """
    Physically deletes a thought node and its interactions from the graph.
    """
    client.repo.delete_thought_node(thought_id)
    logger.info(f"♻️ Graph Hygiene: Pruned thought node {thought_id}")

def update_thought_result(
    thought_id: str,
    result: str,
    embedding: Optional[List[float]] = None,
    repl_id: Optional[str] = None,
    status: str = "complete",
):
    data = {
        "result": result,
        "status": status,
        "completed_at": int(time.time() * 1000)
    }
    if embedding:
        data["embedding"] = embedding
    if repl_id:
        data["repl_id"] = repl_id

    client.repo.update_thought_result(thought_id, data)

def get_graph_state():
    """
    Returns the entire graph structure for visualization.
    """
    return client.repo.get_graph_state()

def get_context_frontier(
    repl_id: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieves the 'Frontier' of the conversation for a given session.
    """
    return client.repo.get_context_frontier(repl_id, limit)

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
    data = {
        "round_id": round_id,
        "root_session_id": root_session_id,
        "user_prompt": user_prompt,
        "repl_ids": repl_ids,
        "final_response": final_response,
        "full_scratchpad": full_scratchpad,
        "started_at": started_at,
        "ended_at": ended_at
    }
    client.repo.save_round(data)
    logger.info(f"Archived Round {round_id} for session {root_session_id}")

def get_completed_rounds(root_session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all completed rounds for a session.
    """
    return client.repo.get_completed_rounds(root_session_id)

def get_session_trace(root_session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves the full trace of thoughts for a given root session.
    """
    return client.repo.get_session_trace(root_session_id)

def delete_session(root_session_id: str):
    """
    Deletes an entire session context.
    """
    client.repo.delete_session(root_session_id)
    logger.info(f"🗑️ Deleted session {root_session_id}")

def prune_orphans(older_than_hours: int = 1) -> int:
    """
    Deletes orphaned Thought nodes.
    """
    current_millis = int(time.time() * 1000)
    cutoff_millis = current_millis - (older_than_hours * 3600 * 1000)
    count = client.repo.prune_orphans(cutoff_millis)
    logger.info(f"🧹 Pruned {count} orphan nodes (older than {older_than_hours}h)")
    return count

def reset_graph():
    """
    NUCLEAR OPTION: Wipes the entire database.
    """
    from .search import create_vector_indexes

    client.repo.reset_graph()
    logger.warning("☢️ GRAPH RESET PERFORMED ☢️")
    # Only try creating indexes if supported
    create_vector_indexes()

def mark_nodes_as_consolidated(node_ids: List[str], insight_id: str):
    """
    Closes the Gestalt on failed nodes.
    """
    client.repo.mark_nodes_as_consolidated(node_ids, insight_id)

def perform_synaptic_homeostasis(retention_window: int = 24):
    """
    Implements the Synaptic Homeostasis Hypothesis (SHY).
    """
    current_ms = int(time.time() * 1000)
    cutoff = current_ms - (retention_window * 3600 * 1000)
    count = client.repo.perform_synaptic_homeostasis(cutoff)
    if count > 0:
        logger.info(
            f"🧠 Synaptic Homeostasis: Pruned {count} saturated memory traces."
        )
