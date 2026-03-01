"""
ACT-R Memory Activation for Graph-RLM.

Implements activation-based memory retrieval from ACT-R theory:
  A_i = B_i + Σ(W_j * S_ji) + ε

Where:
  B_i = Base-Level Activation (recency + frequency, power-law decay)
  W_j * S_ji = Associative Activation (context-driven spreading)
  ε = Noise (optional stochastic retrieval)

This replaces flat/linear memory retrieval with psychologically grounded
priority scoring, ensuring that recent, frequently accessed, and contextually
relevant memories surface first.

References:
    - Anderson, J.R. (2007). "How Can the Human Mind Occur in the Physical Universe?"
    - Anderson, J.R. et al. (2004). "An Integrated Theory of the Mind."
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .logger import get_logger

logger = get_logger("graph_rlm.activation")

# ACT-R default decay parameter (d in t^-d)
DEFAULT_DECAY: float = 0.5

# Associative learning parameter
DEFAULT_S_WEIGHT: float = 1.0

# Retrieval threshold (items below this won't be retrieved)
DEFAULT_THRESHOLD: float = -1.0

# Noise parameter (σ for logistic noise)
DEFAULT_NOISE: float = 0.25


@dataclass
class ActivationRecord:
    """Tracks access history for a single memory chunk.

    Attributes:
        chunk_id: Unique identifier for this memory.
        access_times: Unix timestamps of each access.
        content_embedding: Semantic embedding for associative activation.
        metadata: Arbitrary metadata (type, source, etc.).
    """

    chunk_id: str
    access_times: List[float] = field(default_factory=list)
    content_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_access(self) -> None:
        """Record an access at the current time."""
        self.access_times.append(time.time())


class ActivationEngine:
    """ACT-R Activation-Based Memory Retrieval Engine.

    Computes activation values for memory chunks and retrieves the
    highest-activation items given a context.

    Usage:
        engine = ActivationEngine()
        engine.register("axiom_1", embedding=[...])
        engine.access("axiom_1")  # record usage

        # Later, retrieve top-K by activation:
        results = engine.retrieve(
            context_embedding=[...],
            top_k=5,
        )
    """

    def __init__(
        self,
        decay: float = DEFAULT_DECAY,
        noise: float = DEFAULT_NOISE,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.decay = decay
        self.noise = noise
        self.threshold = threshold
        self._chunks: Dict[str, ActivationRecord] = {}

    def register(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a memory chunk for activation tracking.

        Args:
            chunk_id: Unique identifier.
            embedding: Semantic embedding vector.
            metadata: Arbitrary metadata.
        """
        if chunk_id not in self._chunks:
            self._chunks[chunk_id] = ActivationRecord(
                chunk_id=chunk_id,
                content_embedding=embedding,
                metadata=metadata or {},
            )
        else:
            # Update embedding/metadata if provided
            record = self._chunks[chunk_id]
            if embedding is not None:
                record.content_embedding = embedding
            if metadata:
                record.metadata.update(metadata)

    def access(self, chunk_id: str) -> None:
        """Record an access to a memory chunk.

        Creates the chunk if it doesn't exist.

        Args:
            chunk_id: The chunk being accessed.
        """
        if chunk_id not in self._chunks:
            self._chunks[chunk_id] = ActivationRecord(chunk_id=chunk_id)
        self._chunks[chunk_id].record_access()

    def base_level_activation(
        self,
        chunk_id: str,
        current_time: Optional[float] = None,
    ) -> float:
        """Compute base-level activation B_i.

        B_i = ln(Σ t_j^(-d))

        Where t_j is the time since the j-th access.

        Args:
            chunk_id: The memory chunk.
            current_time: Reference time (defaults to now).

        Returns:
            Base-level activation value (log scale).
        """
        record = self._chunks.get(chunk_id)
        if not record or not record.access_times:
            return -float("inf")

        now = current_time or time.time()
        total = 0.0
        for t_access in record.access_times:
            age = max(now - t_access, 0.001)  # Avoid division by zero
            total += age ** (-self.decay)

        if total <= 0:
            return -float("inf")
        return math.log(total)

    def associative_activation(
        self,
        chunk_id: str,
        context_embedding: List[float],
    ) -> float:
        """Compute associative activation S_ji for a chunk given context.

        Uses cosine similarity between the chunk's embedding and
        the context embedding as a proxy for associative strength.

        Args:
            chunk_id: The memory chunk.
            context_embedding: Embedding of the current context/goal.

        Returns:
            Associative activation value (0.0 if no embedding).
        """
        record = self._chunks.get(chunk_id)
        if not record or record.content_embedding is None:
            return 0.0

        chunk_vec = np.array(record.content_embedding)
        ctx_vec = np.array(context_embedding)

        norm_c = np.linalg.norm(chunk_vec)
        norm_x = np.linalg.norm(ctx_vec)
        if norm_c == 0 or norm_x == 0:
            return 0.0

        similarity = float(np.dot(chunk_vec, ctx_vec) / (norm_c * norm_x))
        return DEFAULT_S_WEIGHT * max(0.0, similarity)

    def compute_activation(
        self,
        chunk_id: str,
        context_embedding: Optional[List[float]] = None,
        current_time: Optional[float] = None,
        add_noise: bool = False,
    ) -> float:
        """Compute total activation A_i = B_i + S_i + ε.

        Args:
            chunk_id: The memory chunk.
            context_embedding: Current context for associative activation.
            current_time: Reference time.
            add_noise: Whether to add stochastic noise.

        Returns:
            Total activation value.
        """
        b_i = self.base_level_activation(chunk_id, current_time)
        if b_i == -float("inf"):
            return -float("inf")

        s_i = 0.0
        if context_embedding:
            s_i = self.associative_activation(chunk_id, context_embedding)

        epsilon = 0.0
        if add_noise and self.noise > 0:
            # Logistic noise (ACT-R standard)
            epsilon = np.random.logistic(0, self.noise)

        activation = b_i + s_i + epsilon
        return float(activation)

    def retrieve(
        self,
        context_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        current_time: Optional[float] = None,
        add_noise: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K memory chunks by activation.

        Only chunks above the retrieval threshold are returned.

        Args:
            context_embedding: Current context for associative activation.
            top_k: Maximum number of chunks to retrieve.
            current_time: Reference time.
            add_noise: Whether to add stochastic noise.

        Returns:
            List of dicts with chunk_id, activation, and metadata,
            sorted by activation (descending).
        """
        scored = []
        for chunk_id, record in self._chunks.items():
            activation = self.compute_activation(
                chunk_id, context_embedding, current_time, add_noise
            )
            if activation > self.threshold:
                scored.append(
                    {
                        "chunk_id": chunk_id,
                        "activation": activation,
                        "access_count": len(record.access_times),
                        "metadata": record.metadata,
                    }
                )

        scored.sort(key=lambda x: x["activation"], reverse=True)
        return scored[:top_k]

    def decay_prune(
        self,
        min_activation: float = -2.0,
        current_time: Optional[float] = None,
    ) -> List[str]:
        """Remove chunks whose activation has fallen below threshold.

        Used for memory management — chunks that haven't been accessed
        recently and aren't contextually relevant get pruned.

        Args:
            min_activation: Minimum activation to keep.
            current_time: Reference time.

        Returns:
            List of pruned chunk_ids.
        """
        pruned = []
        for chunk_id in list(self._chunks.keys()):
            activation = self.base_level_activation(chunk_id, current_time)
            if activation < min_activation:
                del self._chunks[chunk_id]
                pruned.append(chunk_id)

        if pruned:
            logger.info(
                "🧹 [ACT-R] Pruned %d low-activation chunks: %s",
                len(pruned),
                pruned[:5],
            )
        return pruned

    @property
    def chunk_count(self) -> int:
        """Number of tracked memory chunks."""
        return len(self._chunks)
