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


def compute_incidence_matrix(
    graph_nodes: List[Dict[str, Any]], graph_edges: List[Tuple[str, str]]
) -> np.ndarray:
    """Constructs the signed incidence (coboundary) matrix B for a directed graph.

    For each directed edge e = (u → v):
        B[u, e] = -1  (source)
        B[v, e] = +1  (target)

    This is the discrete coboundary operator δ_0 that maps 0-cochains
    (node values) to 1-cochains (edge differences), respecting direction.

    Args:
        graph_nodes: List of node dicts (must have 'id').
        graph_edges: List of (source_id, target_id) tuples.

    Returns:
        Dense numpy array of shape (num_nodes, num_edges).
    """
    num_nodes = len(graph_nodes)
    if num_nodes == 0:
        return np.zeros((0, 0))

    id_map = {n["id"]: i for i, n in enumerate(graph_nodes)}

    # Filter to valid edges (both endpoints exist)
    valid_edges = [
        (u_id, v_id) for u_id, v_id in graph_edges if u_id in id_map and v_id in id_map
    ]

    num_edges = len(valid_edges)
    if num_edges == 0:
        return np.zeros((num_nodes, 0))

    B = np.zeros((num_nodes, num_edges))
    for e_idx, (u_id, v_id) in enumerate(valid_edges):
        u, v = id_map[u_id], id_map[v_id]
        B[u, e_idx] = -1.0  # Source
        B[v, e_idx] = +1.0  # Target

    return B


def compute_directed_laplacian(
    graph_nodes: List[Dict[str, Any]], graph_edges: List[Tuple[str, str]]
) -> np.ndarray:
    """Constructs the directed Sheaf Laplacian: L_dir = B · W · B^T.

    Where:
        B = signed incidence matrix (coboundary operator δ_0)
        W = diagonal edge weight matrix (cosine similarity of embeddings)

    Unlike the undirected Laplacian (which symmetrizes), this operator
    preserves the causal direction of DECOMPOSES_INTO edges. The eigenvalue
    structure reveals directional inconsistency: a large eigenvalue for an
    edge means the parent's intent was not carried through to the child.

    Args:
        graph_nodes: List of node dicts (must have 'id' and optionally 'vec'/'embedding').
        graph_edges: List of (source_id, target_id) tuples.

    Returns:
        Dense numpy array of shape (num_nodes, num_nodes).
    """
    num_nodes = len(graph_nodes)
    if num_nodes == 0:
        return np.zeros((0, 0))

    id_map = {n["id"]: i for i, n in enumerate(graph_nodes)}

    # Build normalized vector map
    vec_map: Dict[str, np.ndarray] = {}
    for n in graph_nodes:
        vec = n.get("vec") or n.get("embedding")
        if vec is not None:
            v = np.array(vec)
            norm = np.linalg.norm(v)
            if norm > 0:
                vec_map[n["id"]] = v / norm

    # Filter valid edges
    valid_edges = [
        (u_id, v_id) for u_id, v_id in graph_edges if u_id in id_map and v_id in id_map
    ]

    num_edges = len(valid_edges)
    if num_edges == 0:
        return np.zeros((num_nodes, num_nodes))

    # Build incidence matrix B
    B = compute_incidence_matrix(graph_nodes, valid_edges)

    # Build diagonal weight matrix W
    weights = np.zeros(num_edges)
    for e_idx, (u_id, v_id) in enumerate(valid_edges):
        weight = 1.0
        if u_id in vec_map and v_id in vec_map:
            sim = float(np.dot(vec_map[u_id], vec_map[v_id]))
            weight = max(0.01, sim)  # Clamp to avoid isolated nodes
        weights[e_idx] = weight

    W = np.diag(weights)

    # L_dir = B · W · B^T
    L = B @ W @ B.T

    return L


def topological_sort(
    graph_nodes: List[Dict[str, Any]], graph_edges: List[Tuple[str, str]]
) -> Tuple[List[str], List[List[str]]]:
    """Topological sort of a DAG using Kahn's algorithm.

    Returns both a linear ordering and parallelizable layers.
    Each layer contains nodes whose dependencies are all satisfied
    by previous layers — these can be executed in parallel.

    Args:
        graph_nodes: List of node dicts (must have 'id').
        graph_edges: List of (source_id, target_id) tuples.

    Returns:
        Tuple of:
            - order: List of node IDs in topological order.
            - layers: List of lists, where each inner list contains
              node IDs that can be executed in parallel.

    Raises:
        ValueError: If the graph contains a cycle (not a DAG).
    """
    if not graph_nodes:
        return [], []

    node_ids = {n["id"] for n in graph_nodes}

    # Build adjacency list and in-degree count
    in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
    children: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    for u_id, v_id in graph_edges:
        if u_id in node_ids and v_id in node_ids:
            children[u_id].append(v_id)
            in_degree[v_id] += 1

    # Kahn's algorithm with layer tracking
    # Start with all nodes that have no incoming edges (roots)
    queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])

    order: List[str] = []
    layers: List[List[str]] = []
    processed = 0

    while queue:
        # Current queue = one parallelizable layer
        layer = sorted(queue)  # Sort for determinism
        layers.append(layer)
        order.extend(layer)
        processed += len(layer)

        next_queue: List[str] = []
        for nid in layer:
            for child in children[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_queue.append(child)
        queue = next_queue

    if processed != len(node_ids):
        raise ValueError(
            f"Graph contains a cycle: processed {processed}/{len(node_ids)} nodes. "
            "Topological sort requires a DAG."
        )

    return order, layers


def compute_graphsage_embedding(
    target_node_id: str,
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Tuple[str, str]],
    k_hops: int = 2,
    aggregator: str = "mean",
) -> np.ndarray:
    """
    Computes a structural embedding for a target node by aggregating features
    from its k-hop neighborhood, simulating a GraphSAGE message passing step.

    Args:
        target_node_id: ID of the node to embed.
        graph_nodes: List of node dicts (must have 'id' and 'vec' or 'embedding').
        graph_edges: List of (source_id, target_id) abstract connections.
        k_hops: Number of neighborhood hops (default 2).
        aggregator: Aggregation strategy ('mean', 'max', 'sum').

    Returns:
        np.ndarray containing the aggregated structural embedding.
        Returns a zero vector if the node is not found or no initial features exist.
    """
    # 1. Build adjacency list (undirected for context mapping)
    adj: Dict[str, set] = {n["id"]: set() for n in graph_nodes}
    for u, v in graph_edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    # 2. Extract initial features
    features: Dict[str, np.ndarray] = {}
    dim = 0
    for n in graph_nodes:
        vec = n.get("vec") or n.get("embedding")
        if vec is not None:
            v_arr = np.array(vec)
            features[n["id"]] = v_arr
            dim = len(v_arr)

    if target_node_id not in features or dim == 0:
        return np.zeros(dim if dim > 0 else 3072)

    # 3. GraphSAGE Message Passing (Breadth-First k-hops)
    current_neighborhood = {target_node_id}
    visited = {target_node_id}
    struct_features = [features[target_node_id]]

    for _ in range(k_hops):
        next_neighborhood = set()
        for node in current_neighborhood:
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_neighborhood.add(neighbor)
                    if neighbor in features:
                        struct_features.append(features[neighbor])
        current_neighborhood = next_neighborhood

    # 4. Component Aggregation
    stacked = np.vstack(struct_features)
    if aggregator == "max":
        agg_vec = np.max(stacked, axis=0)
    elif aggregator == "sum":
        agg_vec = np.sum(stacked, axis=0)
    else:  # default to mean
        agg_vec = np.mean(stacked, axis=0)

    # L2 Normalize the final combined state
    norm = np.linalg.norm(agg_vec)
    if norm > 0:
        agg_vec = agg_vec / norm

    return agg_vec


def compute_attention_matched_embedding(
    nodes: List[Dict[str, Any]], dim: int = 3072
) -> np.ndarray:
    """
    Computes a Latent Space Compressed embedding preserving attention mass
    via Per-Intent Matching (ArXiv 2602.16284).

    Instead of blurring all vectors uniformly, this groups vectors by
    their ThimacIntention KV-head, computing independent attention centroids
    that are then superpositioned. This preserves the structural distinctness
    of goals (DISTAL) vs actions (MOTOR).
    """
    if not nodes:
        return np.zeros(dim)

    # Group embeddings by intent to preserve distinct KV attention masses
    intent_groups = {"DISTAL": [], "PROXIMAL": [], "MOTOR": [], "DEFAULT": []}

    for node in nodes:
        vec = node.get("vec") or node.get("embedding")
        if vec is None:
            continue

        v_arr = np.array(vec)
        if len(v_arr) == 0:
            continue

        # Extract intent
        intent = str(
            node.get("thimac_intent") or node.get("intent_type") or "DEFAULT"
        ).upper()

        # Fallback mapping
        matched_group = "DEFAULT"
        for group in ["DISTAL", "PROXIMAL", "MOTOR"]:
            if group in intent:
                matched_group = group
                break

        intent_groups[matched_group].append(v_arr)

    # Compute center of mass for each attention head
    head_masses = []
    for _, vectors in intent_groups.items():
        if vectors:
            stacked = np.vstack(vectors)
            # Center of mass for this specific intent group
            group_mean = np.mean(stacked, axis=0)
            # Normalize to unit length per-head
            norm = np.linalg.norm(group_mean)
            if norm > 0:
                head_masses.append(group_mean / norm)

    if not head_masses:
        return np.zeros(dim)

    # Superposition the preserved attention heads into a unified KV latent block
    compacted_latent = np.sum(np.vstack(head_masses), axis=0)

    # Final L2 normalization
    final_norm = np.linalg.norm(compacted_latent)
    if final_norm > 0:
        compacted_latent = compacted_latent / final_norm

    return compacted_latent
