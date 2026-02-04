from typing import Any, Dict, List, Optional
from falkordb import FalkorDB
from langchain_community.graphs import FalkorDBGraph

from ..config import settings
from ..logger import get_logger

logger = get_logger("graph_rlm.db.client")

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
            logger.error(f"FalkorDB Query Error: {e}\nQuery: {query}\nParams: {params}")
            import traceback

            logger.error(traceback.format_exc())
            return []

client = GraphClient()
