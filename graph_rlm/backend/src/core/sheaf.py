from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .db import GraphClient, db
from .logger import get_logger

logger = get_logger("graph_rlm.sheaf")


class SheafMonitor:
    """
    Monitor that models the reasoning graph as a Cellular Sheaf to detect logical inconsistencies.

    Refactored to use In-Database GraphBLAS (FalkorDB) for O(1) energy calculation
    instead of O(N) NetworkX memory loading.
    """

    def __init__(self):
        self.db: GraphClient = db

    def compute_sheaf_surprise_score(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Push the 'Surprise Score' calculation into FalkorDB's matrix engine.
        Surprise = (1 - cosine_similarity) + (1.0 if status='failed' else 0.0)
        """
        query = """
        MATCH (parent:Thought)-[r:DECOMPOSES_INTO]->(child:Thought)
        WHERE parent.embedding IS NOT NULL AND child.embedding IS NOT NULL
        WITH parent, child,
             vec.cosineDistance(parent.embedding, child.embedding) AS distance

        WITH parent, child, distance,
             distance + (CASE WHEN child.status = 'failed' THEN 1.0 ELSE 0.0 END) AS surprise_score

        WHERE surprise_score > 0.3  // Filter for non-trivial surprise
        RETURN parent.id AS source, child.id AS target, surprise_score, child.status AS status
        ORDER BY surprise_score DESC
        LIMIT $limit
        """
        try:
            results = self.db.query(query, {"limit": limit})
            # Normalize results
            edges = []
            for row in results:
                # Handle FalkorDB client return format (list or dict)
                if isinstance(row, list):
                    # [source, target, surprise_score, status]
                    edges.append(
                        {
                            "source": row[0],
                            "target": row[1],
                            "surprise_score": row[2],
                            "status": row[3],
                        }
                    )
                elif isinstance(row, dict):
                    edges.append(row)
            return edges
        except Exception as e:
            logger.error(f"Failed to compute sheaf energy in DB: {e}")
            return []

    def diagnose_trace(
        self,
        root_id: str,
        hypothetical_node: Optional[Dict[str, Any]] = None,
        hypothetical_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Diagnose the reasoning trace LOCALLY without loading the full graph.
        Checks the consistency of the hypothetical step against its parents.
        """
        total_energy = 0.0
        details = []

        # If we are proposing a new node, check its consistency with proposed parents
        if hypothetical_node and hypothetical_edges:
            # 1. Get embeddings of parents from DB
            parent_ids = [
                u for u, v in hypothetical_edges if v == hypothetical_node["id"]
            ]

            if not parent_ids:
                return {"status": "HEALTHY", "energy": 0.0, "critique": None}

            # Fetch parents
            parents = {}
            try:
                q = "MATCH (n:Thought) WHERE n.id IN $ids RETURN n.id, n.embedding"
                res = self.db.query(q, {"ids": parent_ids})

                for row in res:
                    if isinstance(row, list):
                        parents[row[0]] = row[1]
                    elif isinstance(row, dict):
                        # Handle node object or dict return
                        n = row.get("n", row)
                        # Check if n is object with properties or dict
                        props = n.properties if hasattr(n, "properties") else n
                        if isinstance(props, dict):
                            parents[props.get("id")] = props.get("embedding")
            except Exception as e:
                logger.error(f"Sheaf diagnostic: Failed to fetch parents from DB: {e}")
                # We proceed with empty parents, resulting in HEALTHY (0.0 energy) status.

            # 2. Compute local energy
            vec_child = hypothetical_node.get("embedding")
            if vec_child:
                vec_child = np.array(vec_child, dtype=float)
                norm_child = np.linalg.norm(vec_child)

                for pid, p_emb in parents.items():
                    if p_emb:
                        vec_p = np.array(p_emb, dtype=float)
                        norm_p = np.linalg.norm(vec_p)

                        energy = 1.0
                        if norm_child > 0 and norm_p > 0:
                            sim = np.dot(vec_child, vec_p) / (norm_child * norm_p)
                            energy = 1.0 - sim

                        total_energy += energy
                        details.append(
                            f"Edge {pid}->{hypothetical_node['id']} Energy: {energy:.4f}"
                        )

        # Check for Thresholds
        status = "HEALTHY"
        critique = None
        if total_energy > 0.8:  # Strict threshold for immediate next step
            status = "INCONSISTENT"
            critique = f"Proposed thought has high semantic distance (Energy {total_energy:.2f}) from context."

        return {
            "status": status,
            "energy": total_energy,
            "consistency_energy": total_energy,  # Alias for agent.py compatibility
            "details": details,
            "critique": critique,
        }

    def scan_and_log(self):
        """
        Main runner.
        Uses native DB computation avoids memory explosion.
        """
        logger.info("Starting Sheaf Scan (In-Database)...")
        high_surprise_score_edges = self.compute_sheaf_surprise_score(limit=10)

        for edge in high_surprise_score_edges:
            logger.warning(
                f"High Surprise Edge: {edge['source']}->{edge['target']} (Surprise={edge['surprise_score']:.4f}, Status={edge['status']})"
            )

        return high_surprise_score_edges


sheaf = SheafMonitor()
