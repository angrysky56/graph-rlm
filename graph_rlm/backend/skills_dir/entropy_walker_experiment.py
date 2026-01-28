def run_entropy_walker_experiment(
    n_trials: int = 100, steps_per_walk: int = 512
) -> dict:
    """
    Compare exploration diversity between CURBy (external entropy) and Python PRNG.

    Tests the Cascade Model prediction: higher micro-randomness → more macro-diversity.

    Args:
        n_trials: Number of independent walks to run for statistical significance
        steps_per_walk: Number of steps per walk (matches typical beacon pulse size)

    Returns:
        dict with results, statistics, and visualization data
    """
    import random
    from collections import defaultdict

    import numpy as np
    import requests
    from scipy.stats import mannwhitneyu, wilcoxon

    API_BASE = "https://random.colorado.edu/api"

    def get_active_chain():
        """Get first active CURBy chain."""
        try:
            response = requests.get(f"{API_BASE}/chains/", timeout=10)
            response.raise_for_status()
            chains = response.json()
            for chain in chains:
                cid = chain["id"]
                test = requests.get(
                    f"{API_BASE}/chains/{cid}/pulses/latest", timeout=10
                )
                if test.status_code == 200:
                    return cid
        except Exception as e:
            return None
        return None

    def get_entropy_pulse(chain_id):
        """Fetch latest 512-bit hex pulse."""
        try:
            url = f"{API_BASE}/chains/{chain_id}/pulses/latest"
            response = requests.get(url, timeout=10)
            data = response.json()
            return data.get("outputValue", "")
        except:
            return None

    def hex_to_moves(hex_string, n_steps):
        """Convert hex to exactly n_steps moves."""
        mapping = {
            "0": (0, 1),
            "1": (0, 1),
            "2": (0, 1),
            "3": (0, 1),
            "4": (0, -1),
            "5": (0, -1),
            "6": (0, -1),
            "7": (0, -1),
            "8": (-1, 0),
            "9": (-1, 0),
            "A": (-1, 0),
            "B": (-1, 0),
            "C": (1, 0),
            "D": (1, 0),
            "E": (1, 0),
            "F": (1, 0),
        }
        moves = []
        hex_upper = hex_string.upper()
        for char in hex_upper:
            if char in mapping and len(moves) < n_steps:
                moves.append(mapping[char])
        # Pad if needed
        while len(moves) < n_steps:
            moves.append(random.choice(list(mapping.values())))
        return moves[:n_steps]

    def run_walk(moves):
        """Execute walk and return unique coordinate count + other metrics."""
        x, y = 0, 0
        path = [(x, y)]
        visited = set([(x, y)])

        for dx, dy in moves:
            x += dx
            y += dy
            path.append((x, y))
            visited.add((x, y))

        # Calculate final distance from origin
        final_distance = np.sqrt(x**2 + y**2)

        # Calculate path tortuosity (directional changes)
        angles = []
        for i in range(1, len(path)):
            dx_step = path[i][0] - path[i - 1][0]
            dy_step = path[i][1] - path[i - 1][1]
            if dx_step != 0 or dy_step != 0:
                angle = np.arctan2(dy_step, dx_step)
                angles.append(angle)

        tortuosity = np.std(angles) if angles else 0

        return {
            "diversity": len(visited),
            "final_distance": final_distance,
            "tortuosity": tortuosity,
            "path": path,
        }

    # Main experiment
    print(
        f"🧪 Entropy Walker Experiment: {n_trials} trials, {steps_per_walk} steps each"
    )

    chain_id = get_active_chain()
    if not chain_id:
        return {"error": "Could not connect to CURBy beacon"}

    print(f"✅ Connected to CURBy chain: {chain_id[:20]}...")

    curby_results = []
    pseudo_results = []
    step_options = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    for trial in range(n_trials):
        if trial % 10 == 0:
            print(f"  Trial {trial}/{n_trials}...")

        # CURBy-driven walk
        beacon_hex = get_entropy_pulse(chain_id)
        if beacon_hex:
            curby_moves = hex_to_moves(beacon_hex, steps_per_walk)
            curby_result = run_walk(curby_moves)
            curby_results.append(curby_result)

        # Pseudo-random walk (same number of steps)
        pseudo_moves = [random.choice(step_options) for _ in range(steps_per_walk)]
        pseudo_result = run_walk(pseudo_moves)
        pseudo_results.append(pseudo_result)

    # Statistical analysis
    curby_diversity = [r["diversity"] for r in curby_results]
    pseudo_diversity = [r["diversity"] for r in pseudo_results]

    curby_distance = [r["final_distance"] for r in curby_results]
    pseudo_distance = [r["final_distance"] for r in pseudo_results]

    # Mann-Whitney U test (non-parametric)
    diversity_stat, diversity_p = mannwhitneyu(
        curby_diversity, pseudo_diversity, alternative="two-sided"
    )
    distance_stat, distance_p = mannwhitneyu(
        curby_distance, pseudo_distance, alternative="two-sided"
    )

    results = {
        "n_trials": n_trials,
        "steps_per_walk": steps_per_walk,
        "curby": {
            "diversity_mean": np.mean(curby_diversity),
            "diversity_std": np.std(curby_diversity),
            "distance_mean": np.mean(curby_distance),
            "distance_std": np.std(curby_distance),
            "all_diversity": curby_diversity,
            "sample_paths": [r["path"] for r in curby_results[:5]],
        },
        "pseudo": {
            "diversity_mean": np.mean(pseudo_diversity),
            "diversity_std": np.std(pseudo_diversity),
            "distance_mean": np.mean(pseudo_distance),
            "distance_std": np.std(pseudo_distance),
            "all_diversity": pseudo_diversity,
            "sample_paths": [r["path"] for r in pseudo_results[:5]],
        },
        "statistics": {
            "diversity_test": {
                "statistic": float(diversity_stat),
                "p_value": float(diversity_p),
                "significant": diversity_p < 0.05,
            },
            "distance_test": {
                "statistic": float(distance_stat),
                "p_value": float(distance_p),
                "significant": distance_p < 0.05,
            },
        },
        "interpretation": {
            "diversity_effect": np.mean(curby_diversity) - np.mean(pseudo_diversity),
            "diversity_effect_pct": (
                (np.mean(curby_diversity) - np.mean(pseudo_diversity))
                / np.mean(pseudo_diversity)
            )
            * 100,
            "conclusion": (
                "SUPPORTS Cascade Model"
                if diversity_p < 0.05
                and np.mean(curby_diversity) > np.mean(pseudo_diversity)
                else "No significant difference"
            ),
        },
    }

    return results
