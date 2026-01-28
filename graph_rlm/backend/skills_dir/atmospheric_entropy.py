"""
Atmospheric Entropy Beacon - Physical Randomness Harvester
"""

import hashlib
import logging
import secrets
import asyncio
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# Global Sensor Mesh: Diverse geographic locations to minimize weather correlation
GLOBAL_SENSOR_MESH = [
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Anaconda_MT", "lat": 46.1285, "lon": -112.9423},
    {"name": "Antarctica", "lat": -82.8628, "lon": 135.0000},
    {"name": "Sahara_Desert", "lat": 23.4162, "lon": 25.6628},
    {"name": "Amazon_Rainforest", "lat": -3.4653, "lon": -62.2159},
    {"name": "Himalayas", "lat": 27.9881, "lon": 86.9250},
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"name": "Reykjavik", "lat": 64.1265, "lon": -21.8174},
    {"name": "Cape_Town", "lat": -33.9249, "lon": 18.4241},
    {"name": "Moscow", "lat": 55.7558, "lon": 37.6173},
    {"name": "Hawaii", "lat": 19.8968, "lon": -155.5828},
    {"name": "Easter_Island", "lat": -27.1127, "lon": -109.3497},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"name": "Death_Valley", "lat": 36.5323, "lon": -116.9325},
    {"name": "Siberia", "lat": 66.4167, "lon": 112.4000},
    {"name": "Patagonia", "lat": -48.3556, "lon": -72.3786},
    {"name": "Nullarbor_Plain", "lat": -31.4284, "lon": 129.8821},
    {"name": "Greenland_Ice_Sheet", "lat": 71.7069, "lon": -42.6043},
    {"name": "Mariana_Trench_Surface", "lat": 11.3493, "lon": 142.4329},
]


async def run_atmospheric_beacon(
    num_pulses: int = 10,
    store: bool = True,
    selection_seed: str | None = None,
    subset_size: int = 4,
) -> dict:
    """
    Harvest entropy from global atmospheric chaos and optionally store in ChatDAG.

    Args:
        num_pulses: Number of entropy samples to collect (default: 10)
        store: Whether to store results in ChatDAG (default: True)
        selection_seed: Optional string seed to deterministically select sensors (for chaining)
        subset_size: Number of sensors to sample per pulse (default: 4)

    Returns:
        dict with entropy data and storage status
    """

    base_url = "https://api.open-meteo.com/v1/forecast"

    # Dynamic Sensor Selection
    if selection_seed:
        def get_hash(sensor: dict) -> str:
            return hashlib.sha256(
                f"{selection_seed}:{sensor['name']}".encode()
            ).hexdigest()

        shuffled = sorted(GLOBAL_SENSOR_MESH, key=get_hash)
        sensors = shuffled[: min(subset_size, len(GLOBAL_SENSOR_MESH))]
        logger.info(f"Selection Seed '{selection_seed}' -> Selected: {[s['name'] for s in sensors]}")
    else:
        indices = secrets.SystemRandom().sample(
            range(len(GLOBAL_SENSOR_MESH)), min(subset_size, len(GLOBAL_SENSOR_MESH))
        )
        sensors = [GLOBAL_SENSOR_MESH[i] for i in indices]
        logger.info(f"Chaos Selection -> Selected: {[s['name'] for s in sensors]}")

    def fetch_pulse(sensor: dict) -> str:
        try:
            params = {
                "latitude": sensor["lat"],
                "longitude": sensor["lon"],
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "surface_pressure",
                ],
                "timezone": "auto",
            }
            # Note: requests is blocking, but for this scale it's fine. 
            # In a high-perf environment, use httpx.
            import time
            response = requests.get(base_url, params=params, timeout=5)
            data = response.json()
            current = data.get("current", {})
            return (
                f"{sensor['name']}:{current.get('temperature_2m')}|"
                f"{current.get('relative_humidity_2m')}|"
                f"{current.get('wind_speed_10m')}|"
                f"{current.get('surface_pressure')}|{current.get('time')}"
            )
        except Exception:
            import time
            return f"FAIL:{sensor['name']}:{time.time()}"

    logger.info("=" * 60)
    logger.info("ATMOSPHERIC ENTROPY BEACON")
    logger.info("=" * 60)
    logger.info(f"Harvesting {num_pulses} atmospheric entropy pulses...")

    pulses = []
    import time
    master_entropy_pool = hashlib.sha512(
        (selection_seed or str(time.time())).encode()
    ).hexdigest()

    for pulse_num in range(num_pulses):
        pulse_data = {
            "pulse_number": pulse_num,
            "timestamp": datetime.now().isoformat(),
            "signals": [],
        }

        for sensor in sensors:
            signal = fetch_pulse(sensor)
            pulse_data["signals"].append(
                {"location": sensor["name"], "raw_signal": signal}
            )
            current_state = master_entropy_pool + signal
            master_entropy_pool = hashlib.sha512(current_state.encode()).hexdigest()

        pulse_data["cumulative_hash"] = master_entropy_pool
        pulses.append(pulse_data)

        if (pulse_num + 1) % 5 == 0:
            logger.info(f"  Collected {pulse_num + 1}/{num_pulses} pulses...")

        await asyncio.sleep(0.5)

    result = {
        "total_pulses": num_pulses,
        "collection_start": pulses[0]["timestamp"],
        "collection_end": pulses[-1]["timestamp"],
        "pulses": pulses,
        "final_seed": master_entropy_pool,
        "seed_preview": master_entropy_pool[:16],
        "float_normalized": int(master_entropy_pool[:16], 16) / 0xFFFFFFFFFFFFFFFF,
    }

    logger.info(f"Harvested {num_pulses} pulses")
    logger.info(f"Final entropy seed: {result['seed_preview']}")
    logger.info(f"Normalized float: {result['float_normalized']:.10f}")

    if store:
        try:
            from graph_rlm.backend.mcp_tools.chatdag import feed_data
            summary = (
                f"# Atmospheric Entropy Collection\n"
                f"Date: {result['collection_start']}\n"
                f"Pulses: {result['total_pulses']}\n"
                f"Final Seed: {result['final_seed']}\n\n"
                "Purpose: Physical entropy harvesting for algorithmically irreducible randomness.\n"
                "Alternative to PRNG - injects outside-the-system chaos into computational logic."
            )

            source_id = f"atmospheric_entropy/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await feed_data(
                content=summary,
                source_id=source_id,
                metadata={
                    "type": "entropy_harvest",
                    "pulse_count": result["total_pulses"],
                },
            )
            logger.info("Stored in ChatDAG")
            return {"stored": True, "source_id": source_id, "entropy_data": result}
        except ImportError:
            logger.warning("ChatDAG not available, returning data only")

    return {"stored": False, "entropy_data": result}
