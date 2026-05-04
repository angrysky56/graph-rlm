"""
Tests for directed topology primitives: incidence matrix, directed Laplacian,
topological sort, and their integration with SheafMonitor.

TDD: These tests define the expected behavior BEFORE implementation.
"""

import pytest

from graph_rlm.backend.src.core.topology import (
    compute_directed_laplacian,
    compute_incidence_matrix,
    topological_sort,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(node_id: str, vec: list[float]) -> dict:
    """Create a graph node dict with normalized embedding."""
    return {"id": node_id, "vec": vec}


# ---------------------------------------------------------------------------
# 1. Incidence Matrix Tests
# ---------------------------------------------------------------------------


class TestIncidenceMatrix:
    """Verify the signed incidence (coboundary) matrix B for directed graphs."""

    def test_simple_chain(self):
        """A→B→C chain:  B should be (3 nodes × 2 edges).

        Edge 0: A→B  →  B[A,0]=-1, B[B,0]=+1
        Edge 1: B→C  →  B[B,1]=-1, B[C,1]=+1
        """
        nodes = [
            _make_node("A", [1, 0]),
            _make_node("B", [1, 0]),
            _make_node("C", [1, 0]),
        ]
        edges = [("A", "B"), ("B", "C")]

        B = compute_incidence_matrix(nodes, edges)

        assert B.shape == (3, 2)
        # Edge 0: A→B
        assert B[0, 0] == -1  # A (source)
        assert B[1, 0] == +1  # B (target)
        assert B[2, 0] == 0  # C (not involved)
        # Edge 1: B→C
        assert B[0, 1] == 0  # A (not involved)
        assert B[1, 1] == -1  # B (source)
        assert B[2, 1] == +1  # C (target)

    def test_diamond_dag(self):
        """Diamond: A→B, A→C, B→D, C→D  →  B is (4 × 4)."""
        nodes = [
            _make_node("A", [1, 0]),
            _make_node("B", [1, 0]),
            _make_node("C", [1, 0]),
            _make_node("D", [1, 0]),
        ]
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]

        B = compute_incidence_matrix(nodes, edges)

        assert B.shape == (4, 4)
        # Edge 0: A→B
        assert B[0, 0] == -1
        assert B[1, 0] == +1
        # Edge 1: A→C
        assert B[0, 1] == -1
        assert B[2, 1] == +1
        # Edge 2: B→D
        assert B[1, 2] == -1
        assert B[3, 2] == +1
        # Edge 3: C→D
        assert B[2, 3] == -1
        assert B[3, 3] == +1

    def test_empty_graph(self):
        """No nodes → (0, 0) matrix."""
        B = compute_incidence_matrix([], [])
        assert B.shape == (0, 0)

    def test_single_node_no_edges(self):
        """Single node, no edges → (1, 0) matrix."""
        nodes = [_make_node("A", [1, 0])]
        B = compute_incidence_matrix(nodes, [])
        assert B.shape == (1, 0)


# ---------------------------------------------------------------------------
# 2. Directed Laplacian Tests
# ---------------------------------------------------------------------------


class TestDirectedLaplacian:
    """Verify L_dir = B^T · W · B with semantic edge weights."""

    def test_consistent_embeddings(self):
        """Identical embeddings → all weights ≈ 1.0.

        For A→B with both vec=[1,0], cosine_sim=1.0, weight=1.0.
        L_dir = B^T · diag(1.0) · B = B^T·B.
        For a single edge A→B:
            B = [[-1],[+1]]
            B^T·B = [[1]]  (1×1 edge-space Laplacian)

        But L_dir as (node × node) = B·W·B^T:
            = [[-1],[+1]] · [1] · [[-1, +1]]
            = [[1, -1], [-1, 1]]
        """
        nodes = [_make_node("A", [1.0, 0.0]), _make_node("B", [1.0, 0.0])]
        edges = [("A", "B")]

        L = compute_directed_laplacian(nodes, edges)

        assert L.shape == (2, 2)
        # Diagonal should equal degree (both 1.0)
        assert L[0, 0] == pytest.approx(1.0)
        assert L[1, 1] == pytest.approx(1.0)
        # Off-diagonal should be -weight
        assert L[0, 1] == pytest.approx(-1.0)
        assert L[1, 0] == pytest.approx(-1.0)

    def test_inconsistent_embeddings(self):
        """Orthogonal embeddings → weight ≈ 0.01 (clamped minimum).

        The edge weight drops to the floor, so diagonal energy is low.
        """
        nodes = [_make_node("A", [1.0, 0.0]), _make_node("B", [0.0, 1.0])]
        edges = [("A", "B")]

        L = compute_directed_laplacian(nodes, edges)

        # Weight should be clamped to 0.01
        assert L[0, 0] == pytest.approx(0.01)
        assert L[1, 1] == pytest.approx(0.01)

    def test_asymmetric_structure(self):
        """The directed Laplacian for a chain A→B→C should NOT be symmetric
        in the same way the undirected one is.

        Actually, B·W·B^T IS symmetric (it's a Gram matrix). The asymmetry
        shows up in that the incidence matrix itself encodes direction — the
        energy flow from parent to child is captured. The eigenvalue structure
        differs from the undirected Laplacian because the weight matrix
        encodes directional consistency.
        """
        # A close to B, B far from C → directional inconsistency
        nodes = [
            _make_node("A", [1.0, 0.0]),
            _make_node("B", [0.9, 0.1]),
            _make_node("C", [0.0, 1.0]),
        ]
        edges = [("A", "B"), ("B", "C")]

        L = compute_directed_laplacian(nodes, edges)

        assert L.shape == (3, 3)
        # The trace should reflect that A→B is consistent but B→C is not
        # A→B weight ≈ cos(A,B) ≈ 0.99+, B→C weight ≈ 0.01 (clamped)
        # Trace = sum of all edge weights = w_AB + w_BC

        # A is only in edge 0 (weight ~1.0)
        assert L[0, 0] > 0.5
        # B is in edge 0 AND edge 1 (weight ~1.0 + ~0.01)
        assert L[1, 1] > L[0, 0]  # B has more connections
        # C is only in edge 1 (weight = cos([0.9,0.1], [0,1]) ≈ 0.11)
        assert L[2, 2] < 0.2

    def test_empty_graph(self):
        """Empty graph → (0, 0) Laplacian."""
        L = compute_directed_laplacian([], [])
        assert L.shape == (0, 0)


# ---------------------------------------------------------------------------
# 3. Topological Sort Tests
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Verify Kahn's algorithm for DAG ordering."""

    def test_linear_chain(self):
        """A→B→C → sorted order [A, B, C]."""
        nodes = [_make_node("B", []), _make_node("A", []), _make_node("C", [])]
        edges = [("A", "B"), ("B", "C")]

        order, layers = topological_sort(nodes, edges)

        assert order == ["A", "B", "C"]
        assert layers == [["A"], ["B"], ["C"]]

    def test_diamond_dag(self):
        """Diamond: A→B, A→C, B→D, C→D.

        Valid orders: [A, B, C, D] or [A, C, B, D].
        Layers: [[A], [B, C], [D]].
        """
        nodes = [
            _make_node("D", []),
            _make_node("B", []),
            _make_node("C", []),
            _make_node("A", []),
        ]
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]

        order, layers = topological_sort(nodes, edges)

        # A must come before B, C; B and C before D
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

        # Layers: A alone, then B&C (parallelizable), then D
        assert layers[0] == ["A"]
        assert set(layers[1]) == {"B", "C"}
        assert layers[2] == ["D"]

    def test_cycle_detection(self):
        """Graph with cycle A→B→C→A should raise ValueError."""
        nodes = [_make_node("A", []), _make_node("B", []), _make_node("C", [])]
        edges = [("A", "B"), ("B", "C"), ("C", "A")]

        with pytest.raises(ValueError, match="cycle"):
            topological_sort(nodes, edges)

    def test_disconnected_components(self):
        """Two disconnected chains: A→B and C→D.

        Both roots should appear in layer 0.
        """
        nodes = [
            _make_node("A", []),
            _make_node("B", []),
            _make_node("C", []),
            _make_node("D", []),
        ]
        edges = [("A", "B"), ("C", "D")]

        order, layers = topological_sort(nodes, edges)

        assert order.index("A") < order.index("B")
        assert order.index("C") < order.index("D")
        assert set(layers[0]) == {"A", "C"}  # Both roots in layer 0

    def test_empty_graph(self):
        """Empty graph → empty order and layers."""
        order, layers = topological_sort([], [])
        assert order == []
        assert layers == []

    def test_single_node(self):
        """Single node, no edges → one layer with that node."""
        nodes = [_make_node("A", [])]
        order, layers = topological_sort(nodes, [])
        assert order == ["A"]
        assert layers == [["A"]]


# ---------------------------------------------------------------------------
# 4. Integration: SheafMonitor with Directed Laplacian
# ---------------------------------------------------------------------------


class TestSheafDirectedIntegration:
    """Verify SheafMonitor methods use directed topology where appropriate."""

    def test_h1_obstruction_uses_directed_energy(self):
        """H1 obstruction should detect directional inconsistency.

        A→B→C where A and B are consistent but C diverges.
        The directed Laplacian should capture that the B→C edge
        has a topological defect, reflected in a higher H1 score.
        """
        from graph_rlm.backend.src.core.sheaf import SheafMonitor

        monitor = SheafMonitor()

        # Consistent chain: A→B similar embeddings
        consistent_path = [
            {"id": "A", "vec": [1.0, 0.0], "embedding": [1.0, 0.0]},
            {"id": "B", "vec": [0.95, 0.05], "embedding": [0.95, 0.05]},
        ]
        h1_consistent = monitor.calculate_h1_obstruction(consistent_path)

        # Inconsistent chain: A→B where B is orthogonal
        inconsistent_path = [
            {"id": "A", "vec": [1.0, 0.0], "embedding": [1.0, 0.0]},
            {"id": "B", "vec": [0.0, 1.0], "embedding": [0.0, 1.0]},
        ]
        h1_inconsistent = monitor.calculate_h1_obstruction(inconsistent_path)

        # Inconsistent path should have higher H1 score
        assert h1_inconsistent["score"] > h1_consistent["score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
