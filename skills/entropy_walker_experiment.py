"""
Experiment: Entropy Walker vs Pseudo-Random Control.

Tests the hypothesis that higher micro-randomness (external entropy) leads to
higher macro-diversity in random walks.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.stats import mannwhitneyu

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://random.colorado.edu/api"


def get_active_chain(api_base: str = DEFAULT_API_BASE) -> Optional[str]:
    """Get first active CURBy chain."""
    try:
        response = requests.get(f"{api_base}/chains/", timeout=10)
        response.raise_for_status()
        chains = response.json()
        for chain in chains:
            cid = chain["id"]
            if not cid:
                continue
            try:
                test = requests.get(f"{api_base}/chains/{cid}/pulses/latest", timeout=5)
                if test.status_code == 200:
                    return cid
            except requests.RequestException:
                continue
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch chains from {api_base}: {e}")
        return None
    return None


def get_entropy_pulse(chain_id: str, api_base: str = DEFAULT_API_BASE) -> Optional[str]:
    """Fetch latest 512-bit hex pulse."""
    try:
        url = f"{api_base}/chains/{chain_id}/pulses/latest"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("outputValue", "")
    except requests.RequestException:
        return None


def hex_to_moves(hex_string: str, n_steps: int) -> List[Tuple[int, int]]:
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

    # Use the hex string to generate moves
    for char in hex_upper:
        if char in mapping and len(moves) < n_steps:
            moves.append(mapping[char])

    # If we need more steps, cycle through the hex string again
    # This prevents using pseudo-randomness filler which invalidates the experiment
    if len(moves) < n_steps:
        extended_hex = hex_upper * ((n_steps // len(hex_upper)) + 2)
        for char in extended_hex:
            if len(moves) >= n_steps:
                break
            if char in mapping:
                moves.append(mapping[char])

    return moves[:n_steps]


def run_walk(moves: List[Tuple[int, int]]) -> Dict[str, float]:
    """Execute walk and return unique coordinate count + other metrics."""
    x, y = 0, 0
    path = [(x, y)]
    visited = {(x, y)}

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

    tortuosity = np.std(angles) if angles else 0.0

    return {
        "diversity": float(len(visited)),
        "final_distance": float(final_distance),
        "tortuosity": float(tortuosity),
        # path omitted to keep return dict small for stats, can be added if needed
    }


def run_entropy_walker_experiment(
    n_trials: int = 100, steps_per_walk: int = 128
) -> dict:
    """
    Compare exploration diversity between CURBy (external entropy) and Python PRNG.

    Tests the Cascade Model prediction: higher micro-randomness -> more macro-diversity.

    Args:
        n_trials: Number of independent walks to run for statistical significance
        steps_per_walk: Number of steps per walk. Defaults to 128 (matches 512-bit hex).

    Returns:
        dict with results, statistics, and visualization data
    """
    logger.info(
        f"🧪 Entropy Walker Experiment: {n_trials} trials, {steps_per_walk} steps each"
    )

    chain_id = get_active_chain()
    if not chain_id:
        return {"error": "Could not connect to CURBy beacon"}

    logger.info(f"✅ Connected to CURBy chain: {chain_id[:20]}...")

    curby_results = []
    pseudo_results = []
    step_options = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    for trial in range(n_trials):
        if trial % 10 == 0:
            logger.info(f"  Trial {trial}/{n_trials}...")

        # CURBy-driven walk
        # Only add result if network fetch succeeds
        beacon_hex = get_entropy_pulse(chain_id)
        if beacon_hex:
            curby_moves = hex_to_moves(beacon_hex, steps_per_walk)
            curby_result = run_walk(curby_moves)
            curby_results.append(curby_result)

            # Paired Pseudo-random walk (same number of steps)
            # Only run if curby one succeeded to keep samples paired
            pseudo_moves = [random.choice(step_options) for _ in range(steps_per_walk)]
            pseudo_result = run_walk(pseudo_moves)
            pseudo_results.append(pseudo_result)

    if not curby_results:
        return {"error": "All network requests failed."}

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
        "n_trials": len(curby_results),
        "steps_per_walk": steps_per_walk,
        "curby": {
            "diversity_mean": float(np.mean(curby_diversity)),
            "diversity_std": float(np.std(curby_diversity)),
            "distance_mean": float(np.mean(curby_distance)),
            "distance_std": float(np.std(curby_distance)),
            "all_diversity": curby_diversity,
        },
        "pseudo": {
            "diversity_mean": float(np.mean(pseudo_diversity)),
            "diversity_std": float(np.std(pseudo_diversity)),
            "distance_mean": float(np.mean(pseudo_distance)),
            "distance_std": float(np.std(pseudo_distance)),
            "all_diversity": pseudo_diversity,
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
            "diversity_effect_pct": float(
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
