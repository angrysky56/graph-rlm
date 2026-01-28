from typing import Any, Dict, List, Optional

from langchain_community.graphs import FalkorDBGraph

from .config import settings
from .logger import get_logger

logger = get_logger("graph_rlm.db")


class GraphClient:
    def __init__(self):
        self.graph = FalkorDBGraph(
            database=settings.GRAPH_NAME,
            host=settings.FALKOR_HOST,
            port=settings.FALKOR_PORT,
        )

    def query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        return self.graph.query(query, params if params else {})

    def create_thought_node(
        self,
        thought_id: str,
        prompt: str,
        parent_id: Optional[str] = None,
        prompt_embedding: Optional[List[float]] = None,
        session_id: str = "default",
        root_session_id: Optional[str] = None,
        repl_id: Optional[str] = None,
    ):
        """
        Creates a 'Thought' node in the graph.
        If parent_id is provided, creates a DECOMPOSES_INTO edge from parent to child.
        """
        # If root_session_id is not provided, default to the session_id (implies this IS the root)
        final_root = root_session_id if root_session_id else session_id

        params: Dict[str, Any] = {
            "tid": thought_id,
            "prompt": prompt,
            "sid": session_id,
            "rsid": final_root,
        }

        # Create the node
        cypher = """
        MERGE (t:Thought {id: $tid})
        SET t.prompt = $prompt, t.status = 'pending', t.created_at = timestamp(), t.session_id = $sid, t.root_session_id = $rsid
        """
        if prompt_embedding:
            params["vec"] = prompt_embedding
            cypher += ", t.embedding = vecf32($vec)"

        if repl_id:
            params["repl_id"] = repl_id
            cypher += ", t.repl_id = $repl_id"

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

    def update_thought_result(
        self,
        thought_id: str,
        result: str,
        embedding: Optional[List[float]] = None,
        repl_id: Optional[str] = None,
    ):
        params: Dict[str, Any] = {"tid": thought_id, "result": result}
        cypher = """
        MATCH (t:Thought {id: $tid})
        SET t.result = $result, t.status = 'complete', t.completed_at = timestamp()
        """
        if embedding:
            # Note: Storing vectors in FalkorDB enables vector search
            # We assume embedding is a list of floats
            params["vec"] = embedding
            cypher += ", t.embedding = vecf32($vec)"

        if repl_id:
            params["repl_id"] = repl_id
            cypher += ", t.repl_id = $repl_id"

        self.query(cypher, params)

    def find_similar_thoughts(self, query_embedding: list[float], limit: int = 5):
        """
        Finds thoughts with similar embeddings to the query.
        """
        params = {"vec": query_embedding}

        # Using FalkorDB vector search procedure
        # Using FalkorDB vector search procedure
        # Note: Depending on client version, syntax might vary.
        # We try the standard procedure call.
        cypher = f"CALL db.idx.vector.queryNodes('Thought', 'embedding', {limit}, vecf32($vec)) YIELD node, score RETURN node, score"

        try:
            return self.query(cypher, params)
        except Exception as e:
            # If default index search fails, we might just return empty or log error
            logger.warning(f"Vector search failed (index missing?): {e}")
            return []

    def create_vector_index(self):
        """
        Creates a vector index on Thought.embedding.
        """
        # FalkorDB Modern Syntax
        try:
            self.query(
                "CREATE VECTOR INDEX FOR (t:Thought) ON (t.embedding) OPTIONS {dimension:768, similarityFunction:'cosine'}"
            )
        except Exception as e:
            # Log as info because it likely already exists
            logger.info(f"Vector index creation skipped: {e}")
            pass

    def drop_vector_index(self):
        try:
            self.query("DROP VECTOR INDEX FOR (t:Thought) ON (t.embedding)")
        except Exception as e:
            logger.info(f"Vector index drop skipped: {e}")

    def wait_for_index(self, label: str):
        import time

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
            except Exception as e:
                logger.debug(f"Index check polling error: {e}")
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
        # Strategy: Find nodes in this session that are not PARENTS of any other node in this session.
        # This gives us the tips of the branches.
        # Also order by timestamp to get the most recent.

        # Note: We filter by repl_id (which maps to session_id in our logic usually,
        # or we might need to store repl_id on nodes explicitly if session_id != repl_id).
        # In agent.py currently: params["sid"] = session_id.
        # Let's assume repl_id passed here IS the session_id used in creation.

        # If we didn't store repl_id but session_id, we use that.
        params = {"sid": repl_id, "limit": limit}

        cypher = """
        MATCH (n:Thought {session_id: $sid})
        WHERE NOT (n)-[:DECOMPOSES_INTO]->(:Thought {session_id: $sid})
        RETURN n
        ORDER BY n.timestamp DESC
        LIMIT $limit
        """

        try:
            return self.query(cypher, params)
        except Exception as e:
            logger.error(f"Failed to get context frontier: {e}")
            return []


db = GraphClient()
