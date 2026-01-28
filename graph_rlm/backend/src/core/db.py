from typing import Any, Dict, List, Optional

from falkordb import FalkorDB
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
        self.client = FalkorDB(
            host=settings.FALKOR_HOST,
            port=settings.FALKOR_PORT,
        )
        self.raw_graph = self.client.select_graph(settings.GRAPH_NAME)

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
        status: str = "pending",
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
        params: Dict[str, Any] = {"vec": query_embedding}

        # Using FalkorDB vector search procedure
        # Using FalkorDB vector search procedure
        # Note: Depending on client version, syntax might vary.
        # We try the standard procedure call.
        cypher = f"CALL db.idx.vector.queryNodes('Thought', 'embedding', {limit}, vecf32($vec)) YIELD node, score RETURN node, score"

        try:
            res = self.raw_graph.query(cypher, params)
            return res.result_set
        except Exception as e:
            # If default index search fails, we might just return empty or log error
            logger.warning(f"Vector search failed (index missing?): {e}")
            return []

    def create_vector_index(self):
        """
        Creates a vector index on Thought.embedding with the correct dimension (3072).
        """
        # Note: google/gemini-embedding-001 (default) returns 3072.
        # If this doesn't match, we drop and recreate.
        dim = 3072
        try:
            # We use self.raw_graph to bypass LangChain's potential parser issues
            self.raw_graph.query(
                f"CREATE VECTOR INDEX FOR (t:Thought) ON (t.embedding) OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
            )
            logger.info(
                f"Created Vector Index on Thought(embedding) with dimension {dim}"
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "already indexed" in err_msg:
                # If already exists, we might still want to check dimension?
                # For now, we just stay silent to avoid startup noise.
                return

            # Log other legitimate failures
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
        # Simplified Strategy: Just get the most recent N thoughts in this session.
        # This works for both linear chains (A->B->C) and flat logs.
        # It ensures we always see the "Recent History".

        params = {"sid": repl_id, "limit": limit}

        cypher = f"""
        MATCH (n:Thought)
        WHERE n.session_id = $sid
        RETURN n
        ORDER BY n.created_at DESC
        LIMIT {limit}
        """

        try:
            return self.query(cypher, params)
        except Exception as e:
            logger.error(f"Failed to get context frontier: {e}")
            return []

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
            except Exception as e:
                logger.error(f"Failed to re-embed thought {node_id}: {e}")

        logger.info(f"Re-embedding complete. Updated {count} thoughts.")
        return count


db = GraphClient()
