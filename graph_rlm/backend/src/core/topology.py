"""
Topological primitives for Graph-RLM.
Provides utilities for Laplacian computation and spectral analysis.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp  # type: ignore
import scipy.sparse.linalg as spla  # type: ignore


def compute_sheaf_laplacian(
    graph_nodes: List[Dict[str, Any]], graph_edges: List[Tuple[str, str]]
) -> np.ndarray:
    """
    Constructs the Sheaf Laplacian matrix (L = D - A).
    Edge weights are calculated dynamically using a semantic Restriction Map
    between node embeddings. A high distance (low similarity) reduces the edge weight,
    representing a topological defect or verification obstruction.

    Args:
        graph_nodes: List of node dictionaries (must have 'id' and optionally 'vec' or 'embedding')
        graph_edges: List of (source_id, target_id) tuples

    Returns:
        Dense numpy array representing the Laplacian.
    """
    num_nodes = len(graph_nodes)
    if num_nodes == 0:
        return np.zeros((0, 0))

    # ID map
    id_map = {n["id"]: i for i, n in enumerate(graph_nodes)}

    # Store vectors for fast lookup
    vec_map = {}
    for n in graph_nodes:
        vec = n.get("vec") or n.get("embedding")
        if vec is not None:
            vec = np.array(vec)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec_map[n["id"]] = vec / norm

    # Adjacency Matrix
    row = []
    col = []
    data = []

    for u_id, v_id in graph_edges:
        if u_id in id_map and v_id in id_map:
            u, v = id_map[u_id], id_map[v_id]

            # Semantic Restriction Map (Dynamic Edge Weight)
            weight = 1.0
            if u_id in vec_map and v_id in vec_map:
                # Cosine similarity between normalized embeddings
                sim = float(np.dot(vec_map[u_id], vec_map[v_id]))
                # Weight bounds: 0.01 (min threshold to avoid isolated nodes) to 1.0
                # A low similarity drops the weight dramatically, indicating a fragmented geodesic
                weight = max(0.01, sim)

            row.append(u)
            col.append(v)
            data.append(weight)
            # undirected
            row.append(v)
            col.append(u)
            data.append(weight)

    adj_matrix = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # Degree Matrix
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degree_matrix = sp.diags(degrees)

    laplacian = degree_matrix - adj_matrix
    return laplacian.toarray()


def extract_kernel_basis(
    laplacian: np.ndarray, tolerance: float = 1e-4
) -> List[List[float]]:
    """
    Extracts the basis for the kernel of the Laplacian (Global Sections).
    Returns eigenvectors corresponding to zero eigenvalues.
    """
    n = laplacian.shape[0]
    if n == 0:
        return []

    try:
        # We look for eigenvalues near 0
        k_eigen = min(n - 1, 10)  # Extract up to 10 components
        if k_eigen < 1:
            # Trivial case for single node: Basis is [1.0]
            return [[1.0]]

        # Use shift-invert to find small eigenvalues
        vals, vecs_raw = spla.eigsh(laplacian, k=k_eigen, which="SM", sigma=1e-5)
        vecs = np.array(vecs_raw)

        # Count effectively zero eigenvalues
        zero_indices = np.where(np.abs(vals) < tolerance)[0]

        if len(zero_indices) == 0:
            # If none are strictly 0, take the smallest one (Fiedler vector / H0 approx)
            zero_indices = np.array([np.argmin(np.abs(vals))])

        kernel_vecs = vecs[:, zero_indices]

        # Return as list of lists (for JSON serialization)
        return kernel_vecs.T.tolist()
    except (RuntimeError, AttributeError, ValueError):
        # Fallback to a simple ones vector (normalized)
        ones = np.ones(n)
        norm = float(np.linalg.norm(ones))
        return [(ones / norm if norm > 0 else ones).tolist()]
