import unittest
import networkx as nx
import numpy as np
from core.sheaf import SheafMonitor

class TestSheaf(unittest.TestCase):
    def setUp(self):
        self.monitor = SheafMonitor()

    def test_consistency_energy(self):
        # Case 1: Consistent (Low Energy)
        # Vectors are identical or close
        G = nx.Graph()
        vec_a = [1.0, 0.0]
        vec_b = [0.99, 0.01] # Very close

        G.add_node("A", embedding=vec_a)
        G.add_node("B", embedding=vec_b)
        G.add_edge("A", "B")

        energies = self.monitor.compute_energy(G)
        energy_val = energies["A<->B"]
        print(f"Consistent Energy: {energy_val}")
        self.assertLess(energy_val, 0.1) # Threshold for "Consistent"

    def test_contradiction_energy(self):
        # Case 2: Contradiction (High Energy)
        # Vectors are orthogonal or opposite
        G = nx.Graph()
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0] # Orthogonal (Logic flip?)

        G.add_node("A", embedding=vec_a)
        G.add_node("B", embedding=vec_b)
        G.add_edge("A", "B")

        energies = self.monitor.compute_energy(G)
        energy_val = energies["A<->B"]
        print(f"Contradiction Energy: {energy_val}")
        # Expected: (1-0)^2 + (0-1)^2 = 2.0
        self.assertAlmostEqual(energy_val, 2.0, places=2)
        self.assertGreater(energy_val, 1.0)

if __name__ == '__main__':
    unittest.main()
