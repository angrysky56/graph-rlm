from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .db import GraphClient, db
from .logger import get_logger

logger = get_logger("graph_rlm.sheaf")


class SheafMonitor:
    """
    Monitor that models the reasoning graph as a Cellular Sheaf to detect logical inconsistencies.

    Theory:
    - Nodes are thoughts.
    - Edges are dependencies (DECOMPOSES_INTO, DEPENDS_ON).
    - Data on nodes (stalks) are embeddings (vectors).
    - Restriction maps are identity (for now) or linear transforms.
    - Laplacian L measures local consistency.
    - Energy E = x^T L x.
    """

    def __init__(self):
        self.db: GraphClient = db

    def build_graph_from_db(self) -> nx.Graph:
        """
        Fetches the current thought graph from FalkorDB and converts to NetworkX.
        """
        # Fetch all Thoughts and relationships
        # Note: In a real large scale system, we'd only fetch the active subgraph.
        q = """
        MATCH (n:Thought)
        OPTIONAL MATCH (n)-[r]->(m:Thought)
        RETURN n, r, m
        """
        res = self.db.query(q)

        G = nx.DiGraph()

        # Helper to safely get ID/Embedding whether it's a Node object or Dict
        def get_props(entity):
            if hasattr(entity, "properties"):
                return entity.properties
            if isinstance(entity, dict):
                return entity
            return {}  # Fallback

        for row in res:
            # FalkorDB Python Client & LangChain wrapper usually return results as a list of values.
            # E.g. [Node(...), Relationship(...), Node(...)]
            # We must handle both list/tuple (driver native) and dict (if wrapper changes).
            node_n = None
            rel = None
            node_m = None
            if isinstance(row, (list, tuple)):
                # Expected format: [n, r, m]
                if len(row) >= 1:
                    node_n = row[0]
                if len(row) >= 2:
                    rel = row[1]
                if len(row) >= 3:
                    node_m = row[2]
            elif isinstance(row, dict):
                # Fallback format: {'n': ..., 'r': ..., 'm': ...}
                node_n = row.get("n")
                rel = row.get("r")
                node_m = row.get("m")

            props_n = get_props(node_n)
            if props_n and "id" in props_n:
                G.add_node(props_n["id"], embedding=props_n.get("embedding"))

            if node_m:
                props_m = get_props(node_m)
                if props_m and "id" in props_m:
                    G.add_node(props_m["id"], embedding=props_m.get("embedding"))
                    # Edge handling
                    rel_type = "RELATED"
                    if hasattr(rel, "relation"):
                        rel_type = rel.relation
                    elif isinstance(rel, dict) and "type" in rel:
                        rel_type = rel["type"]
                    elif hasattr(rel, "type"):
                        rel_type = rel.type

                    if "id" in props_n and "id" in props_m:
                        G.add_edge(props_n["id"], props_m["id"], type=rel_type)

        return G

    def compute_energy(self, G: nx.Graph) -> Dict[str, float]:
        """
        Calculates consistency energy for each node/edge.
        Simplified Sheaf Laplacian:
        If we assume identity restriction maps (consistency means vectors should be similar):
        Energy(u, v) = || x_u - x_v ||^2
        """
        energies = {}

        for u, v, _ in G.edges(data=True):
            emb_u = G.nodes[u].get("embedding")
            emb_v = G.nodes[v].get("embedding")

            if emb_u is not None and emb_v is not None:
                # FalkorDB returns embeddings as list of floats directly.
                vec_u = np.array(emb_u, dtype=float)
                vec_v = np.array(emb_v, dtype=float)

                # Metrics:
                # We use Cosine Distance (1 - Cosine Similarity) as the energy metric.
                # Theory: In the Sheaf framework, 'Restriction Maps' are currently Identity.
                # Thus, high energy (distance) implies the child thought has drifted from the parent's semantic context.
                # While 'Entailment' (subset) is a stronger property, Semantic Similarity is the robust proxy for RLM consistency.

                # Normalize vectors for Cosine Similarity
                norm_u = np.linalg.norm(vec_u)
                norm_v = np.linalg.norm(vec_v)

                if norm_u > 0 and norm_v > 0:
                    similarity = np.dot(vec_u, vec_v) / (norm_u * norm_v)
                    energy = 1.0 - similarity  # Distance
                else:
                    energy = 1.0  # Max distance if zero vector

                energies[f"{u}<->{v}"] = float(energy)

                # Flag high energy in logs
                logger.info(f"Edge {u}-{v} Energy: {energy:.4f}")

        return energies

    def find_logical_knots(self, G: nx.Graph) -> List[Dict[str, Any]]:
        """
        Detects 'Logical Knots' (Cycles) in the reasoning graph.
        According to the framework, cycles with high energy represent contradictions/circular reasoning.
        """
        knots = []
        try:
            from networkx.algorithms import simple_cycles

            cycles = list(simple_cycles(G))
            for cycle in cycles:
                # cycle is list of nodes [n1, n2, n3 ...]
                cycle_energy = 0.0
                path_str = " -> ".join(cycle)

                # Sum energy of edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]

                    # Compute local energy
                    emb_u = G.nodes[u].get("embedding")
                    emb_v = G.nodes[v].get("embedding")
                    if emb_u is not None and emb_v is not None:
                        # Dist sq
                        d = np.array(emb_u) - np.array(emb_v)
                        cycle_energy += np.dot(d, d)

                if cycle_energy > 0.5:
                    knots.append(
                        {
                            "nodes": cycle,
                            "energy": cycle_energy,
                            "description": f"Circular reasoning detected: {path_str}",
                        }
                    )

        except Exception as e:
            logger.warning(f"Cycle detection skipped/failed: {e}")

        return knots

    def diagnose_trace(
        self,
        root_id: str,
        hypothetical_node: Optional[Dict[str, Any]] = None,
        hypothetical_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Diagnose the reasoning trace ending or involving 'root_id'.
        Returns a critique if errors found.

        Args:
            root_id: The ID of the node to check (or the parent of the hypothetical node).
            hypothetical_node: Optional dict representing a NEW node not yet in DB.
                               Format: {"id": "...", "embedding": [...]}
            hypothetical_edges: Optional list of (source_id, target_id) tuples.
        """
        G = self.build_graph_from_db()

        # Inject hypothetical elements
        if hypothetical_node:
            G.add_node(
                hypothetical_node["id"], embedding=hypothetical_node.get("embedding")
            )

        if hypothetical_edges:
            for u, v in hypothetical_edges:
                # Ensure u exists (it should be in DB)
                if u in G.nodes and hypothetical_node and v == hypothetical_node["id"]:
                    G.add_edge(u, v)

        # 1. Compute Energy
        energies = self.compute_energy(G)
        total_energy = sum(energies.values())

        # 2. Find Knots
        knots = self.find_logical_knots(G)

        status = "HEALTHY"
        critique = None

        # Thresholds (from doc)
        THRESHOLD_GEO = 5.0

        if knots:
            status = "LOGICAL_KNOT"
            critique = f"Logical Knot detected (Energy {knots[0]['energy']:.2f}). {knots[0]['description']}. The chain of reasoning is circular."
        elif total_energy > THRESHOLD_GEO:
            status = "INCONSISTENT"
            critique = f"High Inconsistency Energy ({total_energy:.2f}). The reasoning steps do not logically follow from one another."

        return {
            "status": status,
            "energy": total_energy,
            "knots": knots,
            "critique": critique,
        }

    def scan_and_log(self):
        """
        Main runner.
        """
        logger.info("Starting Sheaf Scan...")
        G = self.build_graph_from_db()
        energies = self.compute_energy(G)

        # Check for knots
        knots = self.find_logical_knots(G)
        if knots:
            logger.warning(f"LOGICAL KNOTS FOUND: {knots}")

        return energies


sheaf = SheafMonitor()
