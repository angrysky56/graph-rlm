from typing import Any, Dict, List, Optional
try:
    from falkordb import FalkorDB
    from langchain_community.graphs import FalkorDBGraph
    FALKOR_AVAILABLE = True
except ImportError:
    FALKOR_AVAILABLE = False

from ..config import settings
from ..logger import get_logger
from .repository import NetworkXRepository, FalkorDBRepository

logger = get_logger("graph_rlm.db.client")

class GraphClient:
    def __init__(self):
        self.repo = None
        self.use_falkor = False
        self.raw_graph = None

        # Try connecting to FalkorDB
        if FALKOR_AVAILABLE:
            try:
                # We use a raw client to check connectivity first
                _test_client = FalkorDB(host=settings.FALKOR_HOST, port=settings.FALKOR_PORT)
                _test_client.info() # Throws if connection fails

                # If success, initialize fully
                self.client_lib = _test_client
                self.raw_graph = self.client_lib.select_graph(settings.GRAPH_NAME)

                # Initialize LangChain graph wrapper if needed (for other parts of the system?)
                self.graph = FalkorDBGraph(
                    database=settings.GRAPH_NAME,
                    host=settings.FALKOR_HOST,
                    port=settings.FALKOR_PORT,
                )

                self.repo = FalkorDBRepository(self)
                self.use_falkor = True
                logger.info(f"Connected to FalkorDB at {settings.FALKOR_HOST}:{settings.FALKOR_PORT}")
            except Exception as e:
                logger.warning(f"Could not connect to FalkorDB: {e}. Falling back to NetworkX.")
        else:
             logger.warning("FalkorDB library not found. Falling back to NetworkX.")

        if not self.repo:
            logger.info("Using embedded NetworkX repository.")
            self.repo = NetworkXRepository()
            self.graph = None # LangChain graph not available in this mode

    def query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query on FalkorDB.
        """
        if not self.use_falkor:
            logger.debug(f"Ignored Cypher query (NetworkX mode): {query[:50]}...")
            return []

        try:
            res = self.raw_graph.query(query, params if params else {})
            results = []

            # Defensive check
            if not getattr(res, "header", None) or not res.header:
                return []

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
