
import asyncio

import numpy as np


async def test_centroid_failure():
    print("Testing centroid shape mismatch...")

    # Simulate a successful centroid (e.g. 1536 dims)
    grounded_vec = np.random.rand(1536)

    # Simulate a failed centroid (returns np.zeros(1) from current code)
    neurotic_vec = np.zeros(1)

    print(f"Grounded shape: {grounded_vec.shape}")
    print(f"Neurotic shape: {neurotic_vec.shape}")

    if neurotic_vec.any() and grounded_vec.any():
        print("Both have .any() = True, attempting subtraction...")
        try:
            axis = grounded_vec - neurotic_vec
            print("Subtraction succeeded")
        except Exception as e:
            print(f"Subtraction FAILED: {e}")
    else:
        print("One or both have .any() = False, skipping subtraction as expected.")

if __name__ == "__main__":
    asyncio.run(test_centroid_failure())
