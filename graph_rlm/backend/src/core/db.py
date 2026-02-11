"""
Database core module for Graph-RLM.
Handles interactions with FalkorDB, including thought node creation,
vector indexing, and session management.
"""

import time
import traceback
from typing import Any, Dict, List, Optional

from falkordb import FalkorDB
from langchain_community.graphs import FalkorDBGraph

from .config import settings
from .guardrails import GuardrailError, validate_thought_node
from .logger import get_logger

logger = get_logger("graph_rlm.db")


class GraphClient:
    """
    Client for interacting with FalkorDB.
    Provides methods for querying, creating thoughts, and managing the graph state.
    """

    def __init__(self):
        self.graph = FalkorDBGraph(
            database=settings.GRAPH_NAME,
            host=settings.FALKOR_HOST,
            port=settings.FALKOR_PORT,
        )
        self.client = FalkorDB(
            host=settings.FALKOR_HOST,
            port=settings.FALKOR_PORT,
        )
        self.raw_graph = self.client.select_graph(settings.GRAPH_NAME)
        # Ensure indexes exist
        self.create_vector_indexes()

    def query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query on FalkorDB.
        Uses the raw falkordb-py client to ensure parameter support.
        """
        try:
            res = self.raw_graph.query(query, params if params else {})
            results = []

            # Defensive check: Write queries (SET, MERGE, etc.) may return an empty header list.
            if not getattr(res, "header", None) or not res.header:
                return []

            # FalkorDB headers are of the form: [[type, name], [type, name], ...]
            column_names = [h[1] for h in res.header]
            for row in res.result_set:
                results.append(dict(zip(column_names, row, strict=True)))
            return results
        except Exception as e:
            logger.error(
                "FalkorDB Query Error: %s\nQuery: %s\nParams: %s", e, query, params
            )
            logger.error("%s", traceback.format_exc())
            return []

    def create_thought_node(
        self,
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
        turn_id: Optional[int] = None,
        step_id: Optional[int] = None,
        code_hash: Optional[str] = None,
        sheaf_score: Optional[float] = None,
        spectral_energy: Optional[float] = None,
        h0_rank: Optional[int] = None,
        validate: bool = True,
    ):
        """
        Creates a 'Thought' node in the graph.
        If parent_id is provided, creates a DECOMPOSES_INTO edge from parent to child.

        Args:
            thought_id: Unique ID for the thought
            prompt: Reasoning/action prompt
            parent_id: Optional ID of parent thought
            prompt_embedding: Optional vector representation
            session_id: Active session ID
            root_session_id: Root of the reasoning tree
            repl_id: Associated REPL ID
            status: Execution status
            execution_summary: Brief summary of execution result
            result: Full execution output
            next_action: What the agent should do next
            dreamer_analysis: Analysis from dreamer cycle
            final_response: RLM_FINAL_RESPONSE if terminal step
            round_id: ID of the current round
            turn_id: High-level turn counter
            step_id: Atomic step counter
            code_hash: SHA256 of executed code block
            sheaf_score: Local Consistency (1.0 = perfect)
            spectral_energy: Topological Stress
            h0_rank: Number of connected components
            validate: Whether to run guardrail validation (skip for updates)
        """
        # If root_session_id is not provided, default to the session_id (implies this IS the root)
        final_root = root_session_id if root_session_id else session_id

        # --- GUARDRAILS ---
        if validate:
            try:
                # Check if parent exists and get its metadata for continuity check
                parent_meta = None
                if parent_id:
                    p_res = self.query(
                        "MATCH (p:Thought {id: $pid}) "
                        "RETURN p.session_id as session_id, p.root_session_id as root_session_id",
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
                    repl_id=repl_id,
                    turn_id=turn_id,
                    step_id=step_id,
                    parent_metadata=parent_meta,
                )
            except GuardrailError as ge:
                logger.error("Guardrail Violation: %s", ge)
                raise
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Guardrail internal error: %s", e)

        params: Dict[str, Any] = {
            "tid": thought_id,
            "prompt": prompt,
            "sid": session_id,
            "rsid": final_root,
            "status": status,
        }

        # Create the node
        cypher = (
            "MERGE (t:Thought {id: $tid}) "
            "SET t.prompt = $prompt, t.status = $status, t.created_at = timestamp(), "
            "t.session_id = $sid, t.root_session_id = $rsid"
        )

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

        if turn_id is not None:
            params["turn_id"] = turn_id
            cypher += ", t.turn_id = $turn_id"

        if step_id is not None:
            params["step_id"] = step_id
            cypher += ", t.step_id = $step_id"

        if code_hash:
            params["code_hash"] = code_hash
            cypher += ", t.code_hash = $code_hash"

        if sheaf_score is not None:
            params["sheaf_score"] = sheaf_score
            cypher += ", t.sheaf_score = $sheaf_score"

        if spectral_energy is not None:
            params["spectral_energy"] = spectral_energy
            cypher += ", t.spectral_energy = $spectral_energy"

        if h0_rank is not None:
            params["h0_rank"] = h0_rank
            cypher += ", t.h0_rank = $h0_rank"

        self.query(cypher, params)

        # Link to parent if exists
        if parent_id:
            edge_params = {"tid": thought_id, "pid": parent_id}
            edge_cypher = """
            MATCH (parent:Thought {id: $pid})
            MATCH (child:Thought {id: $tid})
            MERGE (parent)-[:DECOMPOSES_INTO]->(child)
            """
            self.query(edge_cypher, edge_params)

    def get_parent_id(self, thought_id: str) -> Optional[str]:
        """
        Retrieves the parent ID of a thought node.
        Used for rewiring graph topology after pruning.
        """
        cypher = """
        MATCH (p:Thought)-[:DECOMPOSES_INTO]->(c:Thought {id: $tid})
        RETURN p.id as pid
        LIMIT 1
        """
        res = self.query(cypher, {"tid": thought_id})
        if res and "pid" in res[0]:
            return res[0]["pid"]
        return None

    def delete_thought_node(self, thought_id: str):
        """
        Physically deletes a thought node and its interactions from the graph.
        Used for 'Active Pruning' of resolved error chains.
        """
        # Detach delete removes the node and all connected edges
        cypher = "MATCH (n:Thought {id: $tid}) DETACH DELETE n"
        self.query(cypher, {"tid": thought_id})
        logger.info("♻️ Graph Hygiene: Pruned thought node %s", thought_id)

    def update_thought_result(
        self,
        thought_id: str,
        result: str,
        embedding: Optional[List[float]] = None,
        repl_id: Optional[str] = None,
        status: Optional[str] = "complete",
        sheaf_score: Optional[float] = None,
        spectral_energy: Optional[float] = None,
        h0_rank: Optional[int] = None,
    ):
        """
        Updates the execution result and status of an existing thought node.
        """
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
            # Note: Storing vectors in FalkorDB enables vector search
            # We assume embedding is a list of floats
            params["vec"] = embedding
            cypher += ", t.embedding = vecf32($vec)"

        if repl_id:
            params["repl_id"] = repl_id
            cypher += ", t.repl_id = $repl_id"

        if sheaf_score is not None:
            params["sheaf_score"] = sheaf_score
            cypher += ", t.sheaf_score = $sheaf_score"

        if spectral_energy is not None:
            params["spectral_energy"] = spectral_energy
            cypher += ", t.spectral_energy = $spectral_energy"

        if h0_rank is not None:
            params["h0_rank"] = h0_rank
            cypher += ", t.h0_rank = $h0_rank"

        self.query(cypher, params)

    def find_similar_thoughts(
        self, query_embedding: list[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Finds thoughts with similar embeddings to the query and returns structured results.
        """
        # Ensure embedding is the correct length
        if len(query_embedding) != 3072:
            logger.warning(
                "Vector search failed: Embedding dimension mismatch (expected 3072, got %d)",
                len(query_embedding),
            )
            return []

        params: Dict[str, Any] = {"vec": query_embedding, "limit": limit}
        # FalkorDB syntax requires quoted strings for label and property name
        cypher = (
            "CALL db.idx.vector.queryNodes('Thought', 'embedding', $limit, vecf32($vec)) "
            "YIELD node, score RETURN node.id, node.prompt, node.result, score"
        )

        try:
            res = self.raw_graph.query(cypher, params)
            results = []
            for row in res.result_set:
                results.append(
                    {"id": row[0], "prompt": row[1], "result": row[2], "score": row[3]}
                )
            return results
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Vector search failed: %s", e)
            return []

    def create_vector_indexes(self):
        """
        Creates vector indexes on Thought.embedding and Skill.embedding.
        """
        dim = 3072  # Gemini default

        # 1. Thought Index
        try:
            cypher = (
                f"CREATE VECTOR INDEX FOR (t:Thought) ON (t.embedding) "
                f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
            )
            self.raw_graph.query(cypher)
            logger.info(
                "Sync: Vector Index on Thought(embedding) created (dim=%d)", dim
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            if "already indexed" not in str(e).lower():
                logger.warning("Thought vector index creation skipped: %s", e)

        # 2. Skill Index
        try:
            cypher = (
                f"CREATE VECTOR INDEX FOR (s:Skill) ON (s.embedding) "
                f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
            )
            self.raw_graph.query(cypher)
            logger.info("Sync: Vector Index on Skill(embedding) created (dim=%d)", dim)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if "already indexed" not in str(e).lower():
                logger.warning("Skill vector index creation skipped: %s", e)

        # 3. Axiom Index
        try:
            cypher = (
                f"CREATE VECTOR INDEX FOR (a:Axiom) ON (a.embedding) "
                f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
            )
            self.raw_graph.query(cypher)
            logger.info("Sync: Vector Index on Axiom(embedding) created (dim=%d)", dim)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if "already indexed" not in str(e).lower():
                logger.warning("Axiom vector index creation skipped: %s", e)

    def drop_vector_index(self):
        """
        Drops the vector index on Thought.embedding.
        """
        try:
            # FalkorDB standard index drop syntax
            self.query("DROP INDEX FOR (t:Thought) ON (t.embedding)")
            logger.info("Dropped Vector Index on Thought.embedding")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Vector index drop skipped: %s", e)

    def wait_for_index(self, label: str):
        """
        Polls the database until the specified vector index is OPERATIONAL.
        """
        # Poll db.indexes() until status is OPERATIONAL
        for _ in range(20):
            try:
                res = self.query(
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
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("Index check polling error: %s", e)
            time.sleep(0.5)

    def get_graph_state(self):
        """
        Returns the entire graph structure for visualization.
        """
        # Return all Thoughts and their relationships
        cypher = """
        MATCH (n:Thought)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        return self.query(cypher)

    def get_context_frontier(
        self, repl_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the 'Frontier' of the conversation for a given session.
        The frontier consists of:
        1. Leaf nodes (thoughts with no children in this session).
        2. Recent linear history (if single thread).

        Used by the Stateless Agent to 'Wake Up' and load context.
        """
        params = {"sid": repl_id, "limit": limit}

        cypher = """
        MATCH (n:Thought)
        WHERE n.session_id = $sid
        RETURN n
        ORDER BY n.created_at DESC
        LIMIT $limit
        """

        try:
            return self.query(cypher, params)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get context frontier: %s", e)
            return []

    def find_thought_by_repl_id(
        self, repl_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Finds thoughts associated with a specific REPL ID.
        """
        params = {"rid": repl_id, "limit": limit}
        cypher = """
        MATCH (n:Thought)
        WHERE n.repl_id = $rid
        RETURN n.id as id, n.prompt as prompt, n.result as result, n.created_at as created_at
        ORDER BY n.created_at DESC
        LIMIT $limit
        """
        return self.query(cypher, params)

    def reembed_all_thoughts(self, llm_service: Any):
        """
        Iterates through all Thought nodes and refreshes their embeddings.
        Useful when switching embedding models.
        """
        logger.info("Starting graph-wide re-embedding process...")
        # 1. Fetch all nodes with enough text to embed
        cypher = "MATCH (n:Thought) RETURN n.id as id, n.prompt as prompt, n.result as result"
        # Using raw client for consistent list-of-lists format
        res = self.raw_graph.query(cypher)
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
                    self.update_thought_result(
                        thought_id=node_id,
                        result=result,  # Keep existing result
                        embedding=new_vec,
                        status="complete",
                    )
                    count += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Failed to re-embed thought %s: %s", node_id, e)

        logger.info("Re-embedding complete. Updated %d thoughts.", count)
        return count

    # ===== ROUND MANAGEMENT (for stateless agent context compression) =====

    def save_round(
        self,
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

        A 'round' is one user prompt -> agent steps -> RLM_FINAL_RESPONSE cycle.
        Saved for compressed reference in future scratchpads.
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
        self.query(cypher, params)
        logger.info("Archived Round %s for session %s", round_id, root_session_id)

    def get_completed_rounds(self, root_session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all completed rounds for a session, ordered by completion time.
        Used to build compressed previous rounds summary in scratchpad.
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
        return self.query(cypher, params)

    def get_session_trace(
        self, root_session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the full trace of thoughts for a given root session.
        Used by scratchpad_builder and rlm.history.
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
               n.round_id as round_id,
               n.turn_id as turn_id,
               n.step_id as step_id,
               n.code_hash as code_hash,
               n.sheaf_score as sheaf_score,
               n.spectral_energy as spectral_energy,
               n.h0_rank as h0_rank,
               n.sheaf_score as sheaf_score,
               n.spectral_energy as spectral_energy,
               n.h0_rank as h0_rank
        """
        if limit:
            q += "\n        ORDER BY n.created_at DESC"
            q += f"\n        LIMIT {limit}"
        else:
            q += "\n        ORDER BY n.created_at ASC"

        res = self.query(q, {"rsid": root_session_id})
        if limit and res:
            res.reverse()  # Return in chronological order
        return res

    def delete_session(self, root_session_id: str):
        """
        Deletes an entire session context, including all thoughts and rounds.
        """
        # 1. Delete Thoughts
        cypher_thoughts = (
            "MATCH (n:Thought) WHERE n.root_session_id = $rsid DETACH DELETE n"
        )
        self.query(cypher_thoughts, {"rsid": root_session_id})

        # 2. Delete Rounds
        cypher_rounds = (
            "MATCH (r:Round) WHERE r.root_session_id = $rsid DETACH DELETE r"
        )
        self.query(cypher_rounds, {"rsid": root_session_id})

        logger.info("🗑️ Deleted session %s", root_session_id)

    def prune_orphans(self, older_than_hours: int = 1) -> int:
        """
        Deletes orphaned Thought nodes (no relationships) created before the cutoff.
        Returns count of deleted nodes.
        """
        # Timestamp in milliseconds
        cypher = """
        MATCH (n:Thought)
        WHERE NOT (n)--()
        AND n.created_at < $cutoff
        DETACH DELETE n
        RETURN count(n) as count
        """
        current_millis = int(time.time() * 1000)
        cutoff_millis = current_millis - (older_than_hours * 3600 * 1000)

        res = self.query(cypher, {"cutoff": cutoff_millis})
        count = res[0]["count"] if res else 0
        logger.info(
            "🧹 Pruned %d orphan nodes (older than %dh)", count, older_than_hours
        )
        return count

    def reset_graph(self):
        """
        NUCLEAR OPTION: Wipes the entire database.
        """
        self.query("MATCH (n) DETACH DELETE n")
        logger.warning("☢️ GRAPH RESET PERFORMED ☢️")

        # Re-create indexes immediately
        self.create_vector_indexes()

    def mark_nodes_as_consolidated(self, node_ids: List[str], insight_id: str):
        """
        Closes the Gestalt on failed nodes.
        1. Changes status from 'failed'/'error' to 'consolidated'.
        2. Links them to the Insight that resolved them (for traceability).
        """
        cypher = """
        MATCH (t:Thought)
        WHERE t.id IN $ids
        SET t.status = 'consolidated', t.consolidated_at = timestamp()
        WITH t
        MATCH (i:Insight {id: $iid})
        MERGE (t)-[:CONSOLIDATED_INTO]->(i)
        """
        self.query(cypher, {"ids": node_ids, "iid": insight_id})

    def perform_synaptic_homeostasis(self, retention_window: int = 24):
        """
        Implements the Synaptic Homeostasis Hypothesis (SHY).
        We 'downscale' (delete) detailed thought chains that have been
        consolidated into Insights, preserving global plasticity.
        """
        # Current time in ms
        current_ms = int(time.time() * 1000)
        cutoff = current_ms - (retention_window * 3600 * 1000)

        cypher = """
        MATCH (t:Thought)
        WHERE t.status = 'consolidated'
          AND t.consolidated_at < $cutoff
        DETACH DELETE t
        RETURN count(t) as count
        """
        res = self.query(cypher, {"cutoff": cutoff})
        count = res[0]["count"] if res else 0
        if count > 0:
            logger.info(
                "🧠 Synaptic Homeostasis: Pruned %d saturated memory traces.", count
            )


db = GraphClient()
