"""
Database core module for Graph-RLM.
Handles interactions with FalkorDB, including thought node creation,
vector indexing, and session management.
"""

import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import redis
from falkordb import FalkorDB
from langchain_community.graphs import FalkorDBGraph

from .config import settings
from .guardrails import ValidationError, validate_thought_node
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
        except (redis.exceptions.RedisError, redis.exceptions.ResponseError) as e:
            logger.error(
                "FalkorDB Query Error: %s\nQuery: %s\nParams: %s", e, query, params
            )
            logger.error("%s", traceback.format_exc())
            return []

    def create_thought_node(
        self,
        thought_id: str,
        prompt: str,
        logical_id: Optional[str] = None,
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
        thimac_state: Optional[str] = None,
        reflexion_analysis: Optional[str] = None,
        final_response: Optional[str] = None,
        round_id: Optional[str] = None,
        turn_id: Optional[int] = None,
        step_id: Optional[int] = None,
        code_hash: Optional[str] = None,
        sheaf_score: Optional[float] = None,
        spectral_energy: Optional[float] = None,
        h0_rank: Optional[int] = None,
        repe_profile: Optional[Dict[str, float]] = None,
        repe_shakiness: Optional[float] = None,
        repe_evasion: Optional[float] = None,
        repe_confluence: Optional[float] = None,
        repe_freedom: Optional[float] = None,
        omcd_score: Optional[float] = None,
        utility_score: Optional[float] = None,
        thimac_op: Optional[str] = None,
        thimac_level: Optional[str] = None,
        thimac_intent: Optional[str] = None,
        thimac_op_reason: Optional[str] = None,
        thimac_level_reason: Optional[str] = None,
        navigator_insight: Optional[str] = None,
        tool_calls: Optional[List[str]] = None,
        metadata_json: Optional[str] = None,
        step_summary: Optional[str] = None,
        semantic_gist: Optional[str] = None,
        inference_pressure: Optional[float] = None,
        relational_gravity: Optional[float] = None,
        free_energy: Optional[float] = None,
        epistemic_eros: Optional[float] = None,
        metabolic_state: Optional[str] = None,
        frequency: Optional[float] = None,
        confidence: Optional[float] = None,
        rtm_depth: Optional[int] = None,
        slac_at: Optional[float] = None,
        slac_stage: Optional[str] = None,
        slac_bar: Optional[str] = None,
        slac_critique: Optional[str] = None,
        validate: bool = True,
    ):
        """
        Creates a 'Thought' node in the graph.
        If parent_id is provided, creates a DECOMPOSES_INTO edge from parent to child.

        Args:
            thought_id: Global Unique ID for the thought (usually a UUID)
            prompt: Reasoning/action prompt
            logical_id: Deterministic ID for UI/deduplication (e.g. T1:S1)
            parent_id: Optional ID of parent thought (UUID)
            prompt_embedding: Optional vector representation
            session_id: Active session ID
            root_session_id: Root of the reasoning tree
            repl_id: Associated REPL ID
            status: Execution status
            execution_summary: Brief summary of execution result
            result: Full execution output
            next_action: What the agent should do next
            dreamer_analysis: Analysis from dreamer cycle
            final_response: RLM_FINAL_OUTPUT if terminal step
            round_id: ID of the current round
            turn_id: High-level turn counter
            step_id: Atomic step counter
            code_hash: SHA256 of executed code block
            sheaf_score: Local Consistency (1.0 = perfect)
            spectral_energy: Topological Stress
            h0_rank: Number of connected components
            repe_shakiness: Shakiness score (Performance vs Grounding)
            repe_evasion: Evasion score (Avoidance vs Agency)
            repe_confluence: Confluence score (Sycophancy vs Integrity)
            repe_freedom: Freedom score (Entropy vs Restriction)
            omcd_score: Confidence score from oMCD
            thimac_op: Current Thimac operation (ARRIVE, PROCESS, etc.)
            thimac_level: Current Thimac level (EXISTENCE, SUBSISTENCE)
            navigator_insight: Brief insight from Navigator (e.g. 'Class 4')
            metadata_json: Generic JSON blob for extensibility
            validate: Whether to run guardrail validation (skip for updates)
        """
        # If root_session_id is not provided, default to the session_id (implies this IS the root)
        final_root = root_session_id if root_session_id else session_id

        # --- GUARDRAILS ---
        if validate:
            try:
                # Check if parent exists and get its metadata for continuity check
                parent_meta = None
                parent_type = None
                if parent_id:
                    p_res = self.query(
                        "MATCH (p) WHERE p.id = $pid "
                        "RETURN labels(p)[0] as type, p.session_id as session_id, p.root_session_id as root_session_id",
                        {"pid": parent_id},
                    )
                    if p_res:
                        parent_meta = p_res[0]
                        parent_type = parent_meta.get("type")

                from .guardrails import validate_no_blind_transitions

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
                    node_type="Thought",
                )

                validate_no_blind_transitions(
                    node_type="Thought",
                    _content=prompt,
                    parent_type=parent_type,
                )
            except ValidationError as ge:
                logger.error("Guardrail Violation: %s", ge)
                raise
            except (
                RuntimeError,
                AttributeError,
                ValueError,
                TypeError,
                redis.exceptions.RedisError,
            ) as e:
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

        if logical_id:
            params["lid"] = logical_id
            cypher += ", t.logical_id = $lid"

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

        if slac_at is not None:
            params["slac_at"] = float(slac_at)
            cypher += ", t.slac_at = $slac_at"

        if slac_stage:
            params["slac_stage"] = slac_stage
            cypher += ", t.slac_stage = $slac_stage"

        if slac_bar:
            params["slac_bar"] = slac_bar
            cypher += ", t.slac_bar = $slac_bar"

        if slac_critique:
            params["slac_critique"] = slac_critique
            cypher += ", t.slac_critique = $slac_critique"

        if thimac_state:
            params["thimac_state"] = thimac_state
            cypher += ", t.thimac_state = $thimac_state"

        if reflexion_analysis:
            params["reflexion_analysis"] = reflexion_analysis
            cypher += ", t.reflexion_analysis = $reflexion_analysis"

        if thimac_op:
            params["thimac_op"] = thimac_op
            cypher += ", t.thimac_op = $thimac_op"

        if thimac_level:
            params["thimac_level"] = thimac_level
            cypher += ", t.thimac_level = $thimac_level"

        if thimac_intent:
            params["thimac_intent"] = thimac_intent
            cypher += ", t.thimac_intent = $thimac_intent"

        if thimac_op_reason:
            params["thimac_op_reason"] = thimac_op_reason
            cypher += ", t.thimac_op_reason = $thimac_op_reason"

        if thimac_level_reason:
            params["thimac_level_reason"] = thimac_level_reason
            cypher += ", t.thimac_level_reason = $thimac_level_reason"

        if final_response:
            params["final_response"] = final_response
            cypher += ", t.final_response = $final_response"

        if step_summary:
            params["step_summary"] = step_summary
            cypher += ", t.step_summary = $step_summary"

        if semantic_gist:
            params["semantic_gist"] = semantic_gist
            cypher += ", t.semantic_gist = $semantic_gist"

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

        if repe_shakiness is not None:
            params["repe_shakiness"] = float(repe_shakiness)
            cypher += ", t.repe_shakiness = $repe_shakiness"

        if repe_evasion is not None:
            params["repe_evasion"] = float(repe_evasion)
            cypher += ", t.repe_evasion = $repe_evasion"

        if repe_confluence is not None:
            params["repe_confluence"] = float(repe_confluence)
            cypher += ", t.repe_confluence = $repe_confluence"

        if repe_freedom is not None:
            params["repe_freedom"] = float(repe_freedom)
            cypher += ", t.repe_freedom = $repe_freedom"

        if repe_profile:
            # Store RepE profile as individual properties for easier querying
            for k, v in repe_profile.items():
                key = f"repe_{k.lower()}"
                params[key] = float(v)
                cypher += f", t.{key} = ${key}"

        if omcd_score is not None:
            params["omcd_score"] = float(omcd_score)
            cypher += ", t.omcd_score = $omcd_score"

        if utility_score is not None:
            params["ut"] = float(utility_score)
            cypher += ", t.utility_score = $ut"

        if thimac_op is not None:
            params["thimac_op"] = thimac_op
            cypher += ", t.thimac_op = $thimac_op"

        if thimac_level is not None:
            params["thimac_level"] = thimac_level
            cypher += ", t.thimac_level = $thimac_level"

        if navigator_insight is not None:
            params["navigator_insight"] = navigator_insight
            cypher += ", t.navigator_insight = $navigator_insight"

        if metadata_json is not None:
            params["metadata_json"] = metadata_json
            cypher += ", t.metadata_json = $metadata_json"

        if inference_pressure is not None:
            params["inf_p"] = float(inference_pressure)
            cypher += ", t.inference_pressure = $inf_p"

        if relational_gravity is not None:
            params["rel_g"] = float(relational_gravity)
            cypher += ", t.relational_gravity = $rel_g"

        if epistemic_eros is not None:
            params["ep_eros"] = float(epistemic_eros)
            cypher += ", t.epistemic_eros = $ep_eros"

        if free_energy is not None:
            params["free_e"] = float(free_energy)
            cypher += ", t.free_energy = $free_e"

        if metabolic_state is not None:
            params["m_state"] = metabolic_state
            cypher += ", t.metabolic_state = $m_state"

        if frequency is not None:
            params["freq"] = float(frequency)
            cypher += ", t.frequency = $freq"

        if confidence is not None:
            params["conf"] = float(confidence)
            cypher += ", t.confidence = $conf"

        if rtm_depth is not None:
            params["rdepth"] = int(rtm_depth)
            cypher += ", t.rtm_depth = $rdepth"

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

        # --- RESONATES_WITH: Topological Similarity Edge (Phase 8 TLTG) ---
        if prompt_embedding:
            similar = self.find_similar_thoughts(prompt_embedding, limit=2)
            for hit in similar:
                if hit["id"] != thought_id and hit["score"] > 0.85:
                    self.query(
                        "MATCH (t:Thought {id: $tid}), (m:Thought {id: $mid}) "
                        "MERGE (t)-[:RESONATES_WITH {similarity: $score}]->(m)",
                        {"tid": thought_id, "mid": hit["id"], "score": hit["score"]},
                    )

        # --- INVOKES: Explicit Tool Connectivity (MolHIT Van der Waals Bonds) ---
        if tool_calls:
            for tool_name in tool_calls:
                # Strip absolute namespace prefixes if present (e.g. mcp.desktop_commander.foo -> foo)
                clean_name = tool_name.split(".")[-1]
                edge_cypher = """
                MATCH (t:Thought {id: $tid})
                MERGE (s:Skill {name: $skill_name})
                MERGE (t)-[:INVOKES]->(s)
                """
                self.query(edge_cypher, {"tid": thought_id, "skill_name": clean_name})

    def create_insight_node(
        self,
        insight_id: str,
        content: str,
        session_id: str,
        root_session_id: str,
        round_id: str,
        source_thought_id: Optional[str] = None,
        insight_type: str = "trace",  # success | failure | trace
    ):
        """
        GEA Shared Experience Pool: Creates an 'Insight' node.
        Insights are shared execution traces (failures/successes) that
        provide context for sibling agents to avoid repeating mistakes.
        """
        params = {
            "iid": insight_id,
            "content": content,
            "sid": session_id,
            "rsid": root_session_id,
            "rid": round_id,
            "type": insight_type,
        }
        cypher = (
            "MERGE (i:Insight {id: $iid}) "
            "SET i.content = $content, i.session_id = $sid, i.root_session_id = $rsid, "
            "i.round_id = $rid, i.type = $type, i.created_at = timestamp()"
        )
        self.query(cypher, params)

        if source_thought_id:
            edge_params = {"iid": insight_id, "tid": source_thought_id}
            edge_cypher = """
            MATCH (t:Thought {id: $tid})
            MATCH (i:Insight {id: $iid})
            MERGE (t)-[:GENERATED]->(i)
            """
            self.query(edge_cypher, edge_params)
        logger.info("💡 GEA: Registered Insight node %s (%s)", insight_id, insight_type)

    def get_recent_insights(
        self, root_session_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves recent insight nodes for a session.
        """
        params = {"rsid": root_session_id, "limit": limit}
        cypher = """
        MATCH (i:Insight)
        WHERE i.root_session_id = $rsid
        RETURN i.content as content, i.type as type, i.created_at as created_at
        ORDER BY i.created_at DESC
        LIMIT $limit
        """
        return self.query(cypher, params)

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

    def get_causal_ancestors(
        self, thought_id: str, max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieves the causal ancestor chain of a thought node.

        Walks DECOMPOSES_INTO edges **backwards** from a target node to find
        the full chain of prerequisites (root → ... → target). Used for
        Causal Context Engineering during impasse resolution: instead of
        searching by semantic similarity, retrieve the exact topologically
        sorted chain that led to this node.

        Args:
            thought_id: The ID of the target thought node.
            max_depth: Maximum ancestor depth to traverse (default: 10).

        Returns:
            List of node dicts in topological order (root first, target last).
            Empty list if the node has no ancestors or doesn't exist.
        """
        cypher = f"""
        MATCH path = (root:Thought)-[:DECOMPOSES_INTO*1..{max_depth}]->(target:Thought {{id: $tid}})
        RETURN [n IN nodes(path) |
            {{id: n.id, prompt: n.prompt, status: n.status,
             execution_summary: n.execution_summary,
             step_id: n.step_id, round_id: n.round_id}}] as chain
        ORDER BY length(path) DESC
        LIMIT 1
        """
        try:
            results = self.query(cypher, {"tid": thought_id})
            if results and results[0].get("chain"):
                return results[0]["chain"]
            return []
        except (RuntimeError, redis.exceptions.RedisError) as e:
            logger.warning("Failed to get causal ancestors for %s: %s", thought_id, e)
            return []

    def get_execution_layers(
        self, root_session_id: str
    ) -> Tuple[List[str], List[List[str]]]:
        """Returns topologically sorted execution layers for a session's DAG.

        Fetches all DECOMPOSES_INTO edges for a session and runs Python-side
        topological sort to identify parallelizable execution layers.
        Each layer contains nodes whose dependencies are all satisfied.

        Args:
            root_session_id: The root session ID to scope the query.

        Returns:
            Tuple of (linear_order, layers) from topological_sort.
            Returns ([], []) if no edges exist or on error.
        """
        from .topology import topological_sort

        # Fetch all nodes and DECOMPOSES_INTO edges for this session
        q_nodes = """
        MATCH (n:Thought)
        WHERE n.root_session_id = $rsid
        RETURN n.id as id
        """
        q_edges = """
        MATCH (n:Thought)-[:DECOMPOSES_INTO]->(m:Thought)
        WHERE n.root_session_id = $rsid
        RETURN n.id as source, m.id as target
        """
        try:
            node_results = self.query(q_nodes, {"rsid": root_session_id})
            edge_results = self.query(q_edges, {"rsid": root_session_id})

            if not node_results:
                return [], []

            graph_nodes = [{"id": r["id"]} for r in node_results if r.get("id")]
            graph_edges = [
                (r["source"], r["target"])
                for r in edge_results
                if r.get("source") and r.get("target")
            ]

            return topological_sort(graph_nodes, graph_edges)
        except (RuntimeError, ValueError, redis.exceptions.RedisError) as e:
            logger.warning(
                "Failed to compute execution layers for %s: %s",
                root_session_id,
                e,
            )
            return [], []

    def update_thought_result(
        self,
        thought_id: str,
        result: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        repl_id: Optional[str] = None,
        status: Optional[str] = "complete",
        sheaf_score: Optional[float] = None,
        spectral_energy: Optional[float] = None,
        h0_rank: Optional[int] = None,
        step_summary: Optional[str] = None,
        semantic_gist: Optional[str] = None,
        inference_pressure: Optional[float] = None,
        relational_gravity: Optional[float] = None,
        free_energy: Optional[float] = None,
        epistemic_eros: Optional[float] = None,
        metabolic_state: Optional[str] = None,
        frequency: Optional[float] = None,
        confidence: Optional[float] = None,
        rtm_depth: Optional[int] = None,
        slac_at: Optional[float] = None,
        slac_stage: Optional[str] = None,
        slac_bar: Optional[str] = None,
        slac_critique: Optional[str] = None,
        utility_score: Optional[float] = None,
    ):
        """
        Updates the execution result and status of an existing thought node.
        """
        params: Dict[str, Any] = {
            "tid": thought_id,
            "status": status,
        }
        cypher = "MATCH (t:Thought {id: $tid}) SET t.status = $status, t.completed_at = timestamp()"

        if result is not None:
            params["result"] = result
            cypher += ", t.result = $result"
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

        if step_summary:
            params["step_summary"] = step_summary
            cypher += ", t.step_summary = $step_summary"

        if semantic_gist:
            params["semantic_gist"] = semantic_gist
            cypher += ", t.semantic_gist = $semantic_gist"

        if inference_pressure is not None:
            params["inf_p"] = float(inference_pressure)
            cypher += ", t.inference_pressure = $inf_p"

        if relational_gravity is not None:
            params["rel_g"] = float(relational_gravity)
            cypher += ", t.relational_gravity = $rel_g"

        if epistemic_eros is not None:
            params["ep_eros"] = float(epistemic_eros)
            cypher += ", t.epistemic_eros = $ep_eros"

        if free_energy is not None:
            params["free_e"] = float(free_energy)
            cypher += ", t.free_energy = $free_e"

        if utility_score is not None:
            params["ut"] = float(utility_score)
            # Actually, the existing pattern is to append to cypher
            cypher += ", t.utility_score = $ut"

        if metabolic_state is not None:
            params["m_state"] = metabolic_state
            cypher += ", t.metabolic_state = $m_state"

        if frequency is not None:
            params["freq"] = float(frequency)
            cypher += ", t.frequency = $freq"

        if confidence is not None:
            params["conf"] = float(confidence)
            cypher += ", t.confidence = $conf"

        if rtm_depth is not None:
            params["rdepth"] = int(rtm_depth)
            cypher += ", t.rtm_depth = $rdepth"

        if slac_at is not None:
            params["slac_at"] = float(slac_at)
            cypher += ", t.slac_at = $slac_at"

        if slac_stage:
            params["slac_stage"] = slac_stage
            cypher += ", t.slac_stage = $slac_stage"

        if slac_bar:
            params["slac_bar"] = slac_bar
            cypher += ", t.slac_bar = $slac_bar"

        if slac_critique:
            params["slac_critique"] = slac_critique
            cypher += ", t.slac_critique = $slac_critique"

        self.query(cypher, params)

        # --- RESONATES_WITH: Topological Similarity Edge (Phase 8 TLTG) ---
        if embedding:
            similar = self.find_similar_thoughts(embedding, limit=2)
            for hit in similar:
                if hit["id"] != thought_id and hit["score"] > 0.85:
                    self.query(
                        "MATCH (t:Thought {id: $tid}), (m:Thought {id: $mid}) "
                        "MERGE (t)-[:RESONATES_WITH {similarity: $score}]->(m)",
                        {"tid": thought_id, "mid": hit["id"], "score": hit["score"]},
                    )

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
            "YIELD node, score RETURN node.id, node.prompt, node.result, node.semantic_gist, score"
        )

        try:
            res = self.raw_graph.query(cypher, params)
            results = []
            for row in res.result_set:
                results.append(
                    {
                        "id": row[0],
                        "prompt": row[1],
                        "result": row[2],
                        "semantic_gist": row[3],
                        "score": row[4],
                    }
                )
            return results
        except (redis.exceptions.RedisError, redis.exceptions.ResponseError) as e:
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
        except (redis.exceptions.RedisError, redis.exceptions.ResponseError) as e:
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
        except (redis.exceptions.RedisError, redis.exceptions.ResponseError) as e:
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
        except (redis.exceptions.RedisError, redis.exceptions.ResponseError) as e:
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
        except (redis.exceptions.RedisError, redis.exceptions.ResponseError) as e:
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
            except (redis.exceptions.RedisError, redis.exceptions.ResponseError) as e:
                logger.debug("Index check polling error: %s", e)
            time.sleep(0.5)

    def get_graph_state(self):
        """
        Returns the entire graph structure for visualization.
        """
        # Return all relevant nodes and their relationships
        cypher = """
        MATCH (n)
        WHERE n:Thought OR n:Round OR n:Insight OR n:Axiom
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        return self.query(cypher)

    def get_context_frontier(
        self,
        session_id: str,
        limit: int = 10,
        exclude_statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the 'Frontier' of the conversation for a given session.
        The frontier consists of:
        1. Leaf nodes (thoughts with no children in this session).
        2. Recent linear history (if single thread).

        Used by the Stateless Agent to 'Wake Up' and load context.
        """
        params: Dict[str, Any] = {"sid": session_id, "limit": limit}

        # Default statuses to exclude if none provided (Internal System Actions)
        if exclude_statuses is None:
            exclude_statuses = [
                "system",
                "sheaf",
                "omcd",
                "navigator",
                "thimac",
                "validator",
                "dreamer",
                "axiomatic_check",
                "reflexion",
            ]

        params["excluded"] = exclude_statuses

        cypher = """
        MATCH (n:Thought)
        WHERE n.session_id = $sid
        AND NOT n.status IN $excluded
        RETURN n
        ORDER BY n.created_at DESC
        LIMIT $limit
        """

        try:
            return self.query(cypher, params)
        except (
            RuntimeError,
            AttributeError,
            ValueError,
            TypeError,
            redis.exceptions.RedisError,
        ) as e:
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
            except (
                RuntimeError,
                AttributeError,
                ValueError,
                TypeError,
                redis.exceptions.RedisError,
            ) as e:
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

        A 'round' is one user prompt -> agent steps -> RLM_FINAL_OUTPUT cycle.
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

        # Physical Link: Join this Round to all its Thoughts
        link_cypher = """
        MATCH (r:Round {round_id: $rid})
        MATCH (t:Thought {round_id: $rid})
        MERGE (r)-[:CONTAINS]->(t)
        """
        self.query(link_cypher, {"rid": round_id})

        logger.info(
            "Archived Round %s for session %s (Linked to Thoughts)",
            round_id,
            root_session_id,
        )

    def update_round_summaries(
        self,
        round_id: str,
        prompt_summary: str,
        result_summary: str,
    ):
        """
        Updates the summaries of an archived round.
        """
        params = {
            "rid": round_id,
            "p_sum": prompt_summary,
            "r_sum": result_summary,
        }
        cypher = """
        MATCH (r:Round {round_id: $rid})
        SET r.prompt_summary = $p_sum, r.result_summary = $r_sum
        """
        self.query(cypher, params)
        logger.info("Updated summaries for Round %s", round_id)

    def get_completed_rounds(self, root_session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all completed rounds for a session, ordered by completion time.
        Used to build compressed previous rounds summary in scratchpad.
        """
        params = {"rsid": root_session_id}
        cypher = """
        MATCH (r:Round)
        WHERE r.root_session_id = $rsid
        AND r.ended_at > 0
        RETURN r.round_id as round_id,
               r.user_prompt as user_prompt,
               r.repl_ids as repl_ids,
               r.final_response as final_response,
               r.prompt_summary as prompt_summary,
               r.result_summary as result_summary,
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
               n.repe_shakiness as repe_shakiness,
               n.repe_evasion as repe_evasion,
               n.repe_confluence as repe_confluence,
               n.repe_freedom as repe_freedom,
               n.omcd_score as omcd_score,
               n.thimac_op as thimac_op,
               n.thimac_level as thimac_level,
               n.navigator_insight as navigator_insight,
               n.semantic_gist as semantic_gist,
               n.step_summary as step_summary
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

    def disable_axiom(self, axiom_id: str):
        """
        Disables an axiom by setting its 'enabled' property to false.
        Used by the Dreamer/Sheaf when an axiom causes systemic inconsistency.
        """
        cypher = """
        MATCH (a:Axiom {id: $aid})
        SET a.enabled = false, a.disabled_at = timestamp()
        RETURN a.id
        """
        self.query(cypher, {"aid": axiom_id})
        logger.warning("🚫 Axiom %s has been DISABLED by the system.", axiom_id)

    def force_consolidate_noisy_branches(
        self, session_id: str
    ) -> tuple[int, list[str]]:
        """
        Autonomously prunes graph branches that contribute to high topological stress.
        Nodes with status 'failed', 'error', 'reflexion', or high sheaf_score are marked as consolidated
        and effectively hidden from the active context window.

        Returns:
            Tuple[int, List[str]]: (count of pruned nodes, list of pruned thought_ids)
        """
        try:
            cypher = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $sid OR n.session_id = $sid)
              AND NOT n.status IN ['consolidated', 'system_intervention', 'sheaf']
              AND NOT (n.status = 'success' AND n.result IS NOT NULL)
              AND (
                  (n.status IN ['failed', 'error', 'reflexion', 'LOGICAL_KNOT', 'EMPIRICAL_CONTRADICTION'])
                  OR
                  (n.sheaf_score >= 0.3)
                  OR
                  (n.spectral_energy >= 0.3)
              )
            SET n.status = 'consolidated',
                n.consolidated_at = timestamp(),
                n.prune_reason = 'topological_stress_auto_prune'
            RETURN n.id as tid
            """
            result = self.query(cypher, {"sid": session_id})
            pruned_ids = []
            if result:
                for row in result:
                    if isinstance(row, dict):
                        tid = row.get("tid")
                    elif hasattr(row, "tid"):
                        tid = row.tid
                    elif isinstance(row, (list, tuple)) and len(row) > 0:
                        tid = row[0]
                    else:
                        tid = None

                    if tid:
                        pruned_ids.append(str(tid))

                count = len(pruned_ids)
                if count > 0:
                    logger.info(
                        "Auto-pruned %d noisy nodes for session %s: %s",
                        count,
                        session_id,
                        pruned_ids[:5],
                    )
                return count, pruned_ids
            return 0, []
        except (AttributeError, RuntimeError, KeyError, TypeError, ValueError) as e:
            logger.error("Failed to consolidate noisy branches: %s", e, exc_info=True)
            return 0, []

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

    def get_kernel_results(self, root_session_id: str) -> Dict[str, Any]:
        """
        Retrieves kernel computation results for a session.
        Extracts sheaf_score, spectral_energy, and h0_rank from thought nodes.

        Args:
            root_session_id: The session ID to query

        Returns:
            Dictionary with kernel computation data:
            {
                'sheaf_scores': [list of scores],
                'spectral_energies': [list of energies],
                'h0_ranks': [list of ranks],
                'avg_sheaf_score': float,
                'avg_spectral_energy': float,
                'avg_h0_rank': float,
                'kernel_basis': [0.7071, 0.7071]  # Placeholder for actual kernel computation
            }
        """
        q = """
        MATCH (n:Thought)
        WHERE n.root_session_id = $rsid
        AND n.sheaf_score IS NOT NULL
        RETURN n.sheaf_score as sheaf_score,
               n.spectral_energy as spectral_energy,
               n.h0_rank as h0_rank,
               n.id as id
        """
        results = self.query(q, {"rsid": root_session_id})

        # 2. Extract Graph Topology for Basis Computation
        graph_nodes = [{"id": r.get("id")} for r in results if r.get("id")]

        # We need edges to build the Laplacian
        q_edges = """
        MATCH (n:Thought)-[:DECOMPOSES_INTO]->(m:Thought)
        WHERE n.root_session_id = $rsid
        RETURN n.id as source, m.id as target
        """
        edges_res = self.query(q_edges, {"rsid": root_session_id})
        graph_edges = [(r["source"], r["target"]) for r in edges_res]

        # 3. Compute Kernel Basis
        # Lazy import to avoid circular dependencies
        from .topology import compute_sheaf_laplacian, extract_kernel_basis

        try:
            laplacian = compute_sheaf_laplacian(graph_nodes, graph_edges)
            kernel_basis = extract_kernel_basis(laplacian)
        except (RuntimeError, ValueError, AttributeError, TypeError) as e:
            logger.warning("Failed to compute actual kernel basis: %s", e)
            kernel_basis = [[0.7071, 0.7071]]  # Final fallback

        if not results:
            return {
                "sheaf_scores": [],
                "spectral_energies": [],
                "h0_ranks": [],
                "avg_sheaf_score": 0.0,
                "avg_spectral_energy": 0.0,
                "avg_h0_rank": 0,
                "kernel_basis": [[0.7071, 0.7071]],  # Use the fallback here too
                "status": "no_data",
            }

        sheaf_scores = [
            r.get("sheaf_score", 0.0)
            for r in results
            if r.get("sheaf_score") is not None
        ]
        spectral_energies = [
            r.get("spectral_energy", 0.0)
            for r in results
            if r.get("spectral_energy") is not None
        ]
        h0_ranks = [
            r.get("h0_rank", 0) for r in results if r.get("h0_rank") is not None
        ]

        return {
            "sheaf_scores": sheaf_scores,
            "spectral_energies": spectral_energies,
            "h0_ranks": h0_ranks,
            "avg_sheaf_score": (
                sum(sheaf_scores) / len(sheaf_scores) if sheaf_scores else 0.0
            ),
            "avg_spectral_energy": (
                sum(spectral_energies) / len(spectral_energies)
                if spectral_energies
                else 0.0
            ),
            "avg_h0_rank": sum(h0_ranks) / len(h0_ranks) if h0_ranks else 0,
            "kernel_basis": kernel_basis,
            "status": "success",
        }

    def get_session_report_data(self, root_session_id: str) -> Dict[str, Any]:
        """
        Comprehensive data retrieval for report generation.
        Returns all relevant session data in a structured format.

        Args:
            root_session_id: The session ID to query

        Returns:
            Dictionary with complete session data for report population
        """
        kernel_results = self.get_kernel_results(root_session_id)

        # Get session trace for additional context
        trace = self.get_session_trace(root_session_id)

        return {
            "session_id": root_session_id,
            "kernel_results": kernel_results,
            "thought_count": len(trace),
            "operations": [t for t in trace if t.get("step_id") is not None],
            "results": [t for t in trace if t.get("result")],
            "paper_title": f"Analysis Report - Session {root_session_id[:8]}",
            "timestamp": time.strftime("%Y-%m-%d", time.localtime()),
        }


db = GraphClient()
