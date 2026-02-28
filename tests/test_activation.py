"""Tests for ACT-R memory activation engine."""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_rlm.backend.src.core.activation import ActivationEngine


def test_register_and_access():
    """Registering and accessing chunks works."""
    engine = ActivationEngine()
    engine.register("chunk_1", metadata={"type": "axiom"})
    assert engine.chunk_count == 1

    engine.access("chunk_1")
    record = engine._chunks["chunk_1"]
    assert len(record.access_times) == 1


def test_base_level_recency():
    """More recent accesses yield higher base-level activation."""
    engine = ActivationEngine()
    now = time.time()

    engine.register("old_chunk")
    engine._chunks["old_chunk"].access_times = [now - 1000]

    engine.register("new_chunk")
    engine._chunks["new_chunk"].access_times = [now - 1]

    old_act = engine.base_level_activation("old_chunk", current_time=now)
    new_act = engine.base_level_activation("new_chunk", current_time=now)

    assert new_act > old_act, f"Recent ({new_act}) should be > old ({old_act})"


def test_base_level_frequency():
    """More frequent accesses yield higher base-level activation."""
    engine = ActivationEngine()
    now = time.time()

    engine.register("rare")
    engine._chunks["rare"].access_times = [now - 10]

    engine.register("frequent")
    engine._chunks["frequent"].access_times = [
        now - 10, now - 8, now - 5, now - 3, now - 1
    ]

    rare_act = engine.base_level_activation("rare", current_time=now)
    freq_act = engine.base_level_activation("frequent", current_time=now)

    assert freq_act > rare_act, f"Frequent ({freq_act}) should be > rare ({rare_act})"


def test_associative_activation():
    """Chunks with similar embeddings get associative boost."""
    engine = ActivationEngine()

    engine.register("relevant", embedding=[1.0, 0.0, 0.0])
    engine.register("irrelevant", embedding=[0.0, 0.0, 1.0])

    context = [1.0, 0.0, 0.0]

    rel_assoc = engine.associative_activation("relevant", context)
    irr_assoc = engine.associative_activation("irrelevant", context)

    assert rel_assoc > irr_assoc


def test_retrieve_top_k():
    """Retrieve returns top-K chunks by activation."""
    engine = ActivationEngine()
    now = time.time()

    for i in range(10):
        chunk_id = f"chunk_{i}"
        engine.register(chunk_id, embedding=[float(i), 0.0, 0.0])
        # More recent chunks have higher activation
        engine._chunks[chunk_id].access_times = [now - (10 - i)]

    results = engine.retrieve(top_k=3, current_time=now)
    assert len(results) <= 3
    # Results should be ordered by activation (highest first)
    activations = [r["activation"] for r in results]
    assert activations == sorted(activations, reverse=True)


def test_retrieve_with_context():
    """Context embedding biases retrieval toward similar chunks."""
    engine = ActivationEngine()
    now = time.time()

    # Two chunks accessed equally recently
    engine.register("math_chunk", embedding=[1.0, 0.0, 0.0])
    engine._chunks["math_chunk"].access_times = [now - 5]

    engine.register("code_chunk", embedding=[0.0, 1.0, 0.0])
    engine._chunks["code_chunk"].access_times = [now - 5]

    # Search with math-related context
    math_context = [1.0, 0.0, 0.0]
    results = engine.retrieve(context_embedding=math_context, top_k=2, current_time=now)

    assert results[0]["chunk_id"] == "math_chunk"


def test_decay_prune():
    """Prune removes chunks with low base-level activation."""
    engine = ActivationEngine()
    now = time.time()

    engine.register("ancient")
    engine._chunks["ancient"].access_times = [now - 100000]  # Very old

    engine.register("recent")
    engine._chunks["recent"].access_times = [now - 1]

    pruned = engine.decay_prune(min_activation=-2.0, current_time=now)
    assert "ancient" in pruned
    assert "recent" not in pruned
    assert engine.chunk_count == 1


def test_unaccessed_chunk():
    """Chunk with no accesses returns -inf activation."""
    engine = ActivationEngine()
    engine.register("no_access")
    act = engine.base_level_activation("no_access")
    assert act == -float("inf")


def test_access_creates_chunk():
    """Accessing a non-existent chunk auto-creates it."""
    engine = ActivationEngine()
    engine.access("auto_created")
    assert "auto_created" in engine._chunks
    assert len(engine._chunks["auto_created"].access_times) == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test_fn.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
