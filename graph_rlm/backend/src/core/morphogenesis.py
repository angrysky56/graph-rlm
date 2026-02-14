"""
Morphological Memory System (Neural Cellular Automata).
Treats memory as a 2D 'living' grid that resists decay and heals damage.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger("graph_rlm.morphogenesis")


class MorphologicalMemory:
    def __init__(self, size: int = 16, channel_dim: int = 64):
        self.size = size
        self.channel_dim = channel_dim
        # The grid: [Size, Size, Channels]
        # Channels 0-3: RGBA (Visualization)
        # Channels 4+: Semantic Embedding Dimensions
        self.grid = np.zeros((size, size, channel_dim), dtype=np.float32)

        # Local perception kernel (Sobel filter for edge detection)
        self.kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def seed(self, embedding: List[float]):
        """Plant a 'seed' thought in the center of the grid."""
        if not embedding:
            return

        center = self.size // 2

        # Pad or trim embedding to match channel_dim
        # We use the full channel_dim for embedding storage, overriding visual channels if needed
        # (visuals are emergent properties of the embedding in this simplified model)
        vec = np.array(embedding[: self.channel_dim])
        if len(vec) < self.channel_dim:
            vec = np.pad(vec, (0, self.channel_dim - len(vec)))

        # Place seed
        self.grid[center, center] = vec

        # Ensure Alpha channel (alive) is set to 1.0 if channel_dim allows
        if self.channel_dim > 3:
            self.grid[center, center, 3] = 1.0

    def perceive(self):
        """Cells look at their neighbors."""
        # Simple convolution using numpy rolling (simulating local perception)
        perception = np.zeros_like(self.grid)

        # Simplified: Average of neighbors for diffusion simulation
        # Using wrap mode for toroidal topology
        padded = np.pad(self.grid, ((1, 1), (1, 1), (0, 0)), mode="wrap")
        for i in range(self.size):
            for j in range(self.size):
                # 3x3 neighborhood
                neighborhood = padded[i : i + 3, j : j + 3]
                perception[i, j] = np.mean(neighborhood, axis=(0, 1))

        return perception

    def update(self, steps: int = 5):
        """
        Run the growth rule.
        Rule: "Alive cells diffuse info to neighbors."
        (A continuous version of Game of Life applied to Embeddings)
        """
        for _ in range(steps):
            perception = self.perceive()

            # Stochastic update (biological noise)
            # 10% chance of update per cell per step to simulate organic growth
            update_mask = (np.random.rand(self.size, self.size, 1) > 0.9).astype(
                np.float32
            )

            # Update Rule: Move towards the average of neighbors (Diffusion/Smoothing)
            # This "heals" gaps where data was deleted.
            delta = (perception - self.grid) * 0.1

            self.grid += delta * update_mask

            # Constraint: Alpha channel clipping if applicable
            if self.channel_dim > 3:
                self.grid[:, :, 3] = np.clip(self.grid[:, :, 3], 0, 1)

    def damage(self, fraction: float = 0.2):
        """Simulate Amnesia/Truncation (wipe out a chunk of the grid)."""
        mask = (np.random.rand(self.size, self.size, 1) > fraction).astype(np.float32)
        self.grid *= mask
        logger.warning("Morphological Memory damaged by %.2f%%", fraction * 100)

    def read_state(self) -> np.ndarray:
        """
        Pool the grid into a single vector for the LLM.
        This vector represents the 'Gestalt' of the memory.
        """
        # Global Average Pooling
        # Only pool 'alive' cells (Alpha > 0.1) if we have an alpha channel
        if self.channel_dim > 3:
            mask = self.grid[:, :, 3:4] > 0.1
            if np.sum(mask) == 0:
                return np.zeros(self.channel_dim)
            weighted_sum = np.sum(self.grid * mask, axis=(0, 1))
            count = np.sum(mask)
            return weighted_sum / count
        else:
            return np.mean(self.grid, axis=(0, 1))

    def get_gestalt_string(self) -> str:
        """
        Convert pooled state into a human-readable summary string for prompt injection.
        """
        vec = self.read_state()
        # Truncate to first 16 dimensions for brevity in logs/scratchpad
        formatted = ",".join([f"{x:.3f}" for x in vec[:16]])
        return f"MorphState[Dim:{self.channel_dim}]<{formatted}...>"
