"""
Block 1: Discrete Sheaf Computational Solver.
This module provides logic to construct a sheaf over a graph and compute
the discrete sheaf cohomology groups H0 and H1 using coboundary matrices.
"""

import numpy as np
from typing import Dict, List, Tuple


def discrete_sheaf_computational_solver(
    vertices: List[int],
    edges: List[Tuple[int, int]],
    restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, int]:
    """
    Computes the dimensions of the 0th and 1th cohomology groups (H0, H1).

    Args:
        vertices: List of vertex indices.
        edges: List of edges as (source, target) tuples.
        restrictions: Mapping from edge index to (res_v_source, res_v_target)
                      where each res is a numpy array (linear map).

    Returns:
        A dictionary containing the dimensions of H0 and H1.
    """
    # Determine stalks dimensions (assuming consistency in restriction maps)
    # H0 is the kernel of the coboundary matrix d0.
    # d0: \bigoplus_{v \in V} F(v) -> \bigoplus_{e \in E} F(e)

    stalk_dims = {}
    edge_dims = {}

    for i, edge in enumerate(edges):
        res_s, res_t = restrictions[edge]
        edge_dims[i] = res_s.shape[0]
        stalk_dims[edge[0]] = res_s.shape[1]
        stalk_dims[edge[1]] = res_t.shape[1]

    total_v_dim = sum(stalk_dims[v] for v in vertices)
    total_e_dim = sum(edge_dims[i] for i in range(len(edges)))

    # Construct the d0 coboundary matrix
    d0 = np.zeros((total_e_dim, total_v_dim))

    # Helpers to track offsets in the block matrix
    v_offsets = {v: sum(stalk_dims[vertices[j]] for j in range(i))
                 for i, v in enumerate(vertices)}
    e_offset = 0

    for i, edge in enumerate(edges):
        u, v = edge
        res_u, res_v = restrictions[edge]
        e_len = edge_dims[i]

        # d0|_edge = Res_{u->e}(stalk_u) - Res_{v->e}(stalk_v)
        d0[e_offset:e_offset + e_len, v_offsets[u]:v_offsets[u] + stalk_dims[u]] = res_u
        d0[e_offset:e_offset + e_len, v_offsets[v]:v_offsets[v] + stalk_dims[v]] = -res_v
        e_offset += e_len

    # Compute H0 as the nullity of d0
    rank_d0 = np.linalg.matrix_rank(d0)
    h0_dim = total_v_dim - rank_d0

    # For a graph (1D complex), H1 = coker(d0) = codomain_dim - rank(d0)
    h1_dim = total_e_dim - rank_d0

    return {"h0": int(h0_dim), "h1": int(h1_dim)}
