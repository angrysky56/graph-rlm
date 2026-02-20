import numpy as np

from graph_rlm.backend.src.core.morphogenesis import MorphologicalMemory


def test_morph_memory_init():
    size = 8
    channel_dim = 16
    memory = MorphologicalMemory(size=size, channel_dim=channel_dim)
    assert memory.grid.shape == (size, size, channel_dim)
    assert np.all(memory.grid == 0)


def test_morph_memory_seed():
    memory = MorphologicalMemory(size=10, channel_dim=8)
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    memory.seed(embedding)

    center = 5
    assert memory.grid[center, center, 0] == 0.1
    assert memory.grid[center, center, 3] == 1.0  # Alpha channel


def test_morph_memory_update():
    memory = MorphologicalMemory(size=10, channel_dim=8)
    embedding = [1.0] * 8
    memory.seed(embedding)

    # Run a lot of steps to ensure diffusion
    for _ in range(20):
        memory.update(steps=10)

    # Check that more than 1 cell has some value
    mask = memory.grid[:, :, 3] > 0.0001
    assert np.sum(mask) > 1

    # Total sum should be non-zero
    assert np.sum(memory.grid) > 0


def test_morph_memory_damage_and_read():
    memory = MorphologicalMemory(size=10, channel_dim=8)
    embedding = [1.0] * 8
    memory.seed(embedding)
    # Ensure it spreads significantly
    for _ in range(20):
        memory.update(steps=10)

    mask_before = (memory.grid[:, :, 3] > 0.0001).astype(float)
    count_before = np.sum(mask_before)
    assert count_before > 1

    # Damage a chunk
    memory.damage(fraction=0.5)

    mask_after = (memory.grid[:, :, 3] > 0.0001).astype(float)
    count_after = np.sum(mask_after)

    # Verify damage
    assert count_after < count_before

    # Heal
    for _ in range(30):
        memory.update(steps=10)

    mask_healed = (memory.grid[:, :, 3] > 0.0001).astype(float)
    count_healed = np.sum(mask_healed)
    assert count_healed > count_after


def test_gestalt_string():
    memory = MorphologicalMemory(size=4, channel_dim=64)
    memory.seed([1.0] * 64)
    gestalt = memory.get_gestalt_string()
    assert "MorphState[Dim:64]" in gestalt
    assert "1.000" in gestalt
