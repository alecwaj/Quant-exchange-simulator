"""Historical BTC crash data with embedded price series."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from nexus_sim.scenarios.synthetic import CrashScenario

# Historical crash metadata
HISTORICAL_CRASHES = {
    "march-2020": {
        "event": "COVID crash",
        "date": "2020-03-12",
        "peak_to_trough_pct": 45,
        "duration_hours": 24,
        "start_price": 7900,
        "bottom_price": 4350,
        "worst_1h_pct": 25,
        "worst_5m_pct": 12,
        "worst_1m_pct": 5,
    },
    "may-2021": {
        "event": "China ban crash",
        "date": "2021-05-19",
        "peak_to_trough_pct": 35,
        "duration_hours": 12,
        "start_price": 43500,
        "bottom_price": 28800,
        "worst_1h_pct": 18,
        "worst_5m_pct": 8,
        "worst_1m_pct": 4,
    },
    "luna-crash": {
        "event": "LUNA/UST collapse",
        "date": "2022-05-09",
        "peak_to_trough_pct": 28,
        "duration_hours": 48,
        "start_price": 36000,
        "bottom_price": 26000,
        "worst_1h_pct": 12,
        "worst_5m_pct": 5,
        "worst_1m_pct": 3,
    },
    "ftx-crash": {
        "event": "FTX collapse",
        "date": "2022-11-08",
        "peak_to_trough_pct": 25,
        "duration_hours": 72,
        "start_price": 21400,
        "bottom_price": 15900,
        "worst_1h_pct": 10,
        "worst_5m_pct": 4,
        "worst_1m_pct": 2,
    },
    "aug-2024": {
        "event": "Yen carry unwind",
        "date": "2024-08-05",
        "peak_to_trough_pct": 18,
        "duration_hours": 6,
        "start_price": 62000,
        "bottom_price": 50800,
        "worst_1h_pct": 10,
        "worst_5m_pct": 5,
        "worst_1m_pct": 3,
    },
}


def _generate_crash_prices(meta: dict, resolution_sec: float = 1.0) -> np.ndarray:
    """Generate a realistic crash price path from metadata.

    Uses a combination of smooth decline with sharp drops at the documented
    worst intervals to create a realistic-looking crash trajectory.
    """
    duration_sec = meta["duration_hours"] * 3600
    n_points = int(duration_sec / resolution_sec)

    start = meta["start_price"]
    bottom = meta["bottom_price"]
    total_drop = start - bottom

    rng = np.random.default_rng(hash(meta["date"]) % 2**32)

    # Base trajectory: smooth decline with noise
    t = np.linspace(0, 1, n_points)

    # Create a trajectory with a sharp dip (worst hour) embedded
    # Use a sigmoid-like shape with the worst period in the first third
    worst_hour_center = 0.2 + rng.random() * 0.2  # worst hour between 20-40% of event
    steepness = 8.0

    # Sigmoid for main decline
    trajectory = 1 / (1 + np.exp(-steepness * (t - worst_hour_center)))

    # Normalize to [0, 1]
    trajectory = (trajectory - trajectory.min()) / (trajectory.max() - trajectory.min())

    # Add some noise
    noise = rng.normal(0, 0.01, n_points)
    noise = np.cumsum(noise)
    noise -= np.linspace(noise[0], noise[-1], n_points)  # detrend
    trajectory += noise * 0.05
    trajectory = np.clip(trajectory, 0, 1)

    # Re-normalize
    trajectory = (trajectory - trajectory.min()) / (trajectory.max() - trajectory.min())

    prices = start - trajectory * total_drop

    # Ensure start and end prices are correct
    prices[0] = start
    prices[-1] = bottom + total_drop * 0.05  # slight recovery at end

    return prices


def load_historical_crash(
    name: str,
    block_time_ms: float = 5,
    timescale: str = "full",
) -> CrashScenario:
    """Load a historical crash scenario.

    Args:
        name: Crash preset name (e.g. 'march-2020')
        block_time_ms: Target block time for resampling
        timescale: 'full', '1h' (worst hour), '5m' (worst 5 min), '1m' (worst minute)
    """
    if name not in HISTORICAL_CRASHES:
        available = ", ".join(HISTORICAL_CRASHES.keys())
        raise ValueError(f"Unknown crash preset '{name}'. Available: {available}")

    meta = HISTORICAL_CRASHES[name]

    # Try loading from embedded JSON, fall back to generated
    prices = _try_load_json(name)
    if prices is None:
        if timescale == "full":
            resolution = 60.0  # 1-minute resolution for full event
        elif timescale == "1h":
            resolution = 1.0  # 1-second for worst hour
        elif timescale == "5m":
            resolution = 1.0
        elif timescale == "1m":
            resolution = 0.1
        else:
            resolution = 60.0
        prices = _generate_crash_prices(meta, resolution)

    # Determine the timescale window
    if timescale == "1h":
        # Extract worst 1-hour window
        hour_blocks = min(3600, len(prices))
        worst_start = 0
        worst_drop = 0
        for i in range(len(prices) - hour_blocks):
            drop = prices[i] - prices[i + hour_blocks]
            if drop > worst_drop:
                worst_drop = drop
                worst_start = i
        prices = prices[worst_start:worst_start + hour_blocks]
    elif timescale == "5m":
        window = min(300, len(prices))
        worst_start = 0
        worst_drop = 0
        for i in range(len(prices) - window):
            drop = prices[i] - prices[i + window]
            if drop > worst_drop:
                worst_drop = drop
                worst_start = i
        prices = prices[worst_start:worst_start + window]
    elif timescale == "1m":
        window = min(60, len(prices))
        worst_start = 0
        worst_drop = 0
        for i in range(len(prices) - window):
            drop = prices[i] - prices[i + window]
            if drop > worst_drop:
                worst_drop = drop
                worst_start = i
        prices = prices[worst_start:worst_start + window]

    # Resample to block time
    source_resolution_ms = meta["duration_hours"] * 3600 * 1000 / len(prices)
    target_n = max(2, int(len(prices) * source_resolution_ms / block_time_ms))

    if target_n != len(prices):
        x_old = np.linspace(0, 1, len(prices))
        x_new = np.linspace(0, 1, target_n)
        prices = np.interp(x_new, x_old, prices)

    magnitude = (1 - prices.min() / prices[0]) * 100
    duration_sec = len(prices) * block_time_ms / 1000

    return CrashScenario(
        name=name,
        description=f"{meta['event']} ({meta['date']}): {magnitude:.1f}% over {duration_sec:.0f}s",
        prices=prices,
        duration_sec=duration_sec,
        magnitude_pct=magnitude,
        block_time_ms=block_time_ms,
    )


def _try_load_json(name: str) -> np.ndarray | None:
    """Try to load price data from the embedded JSON file."""
    try:
        data_path = Path(__file__).parent.parent / "data" / "btc_crashes.json"
        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)
            if name in data:
                return np.array(data[name]["prices"])
    except Exception:
        pass
    return None


def list_historical_crashes() -> dict[str, dict]:
    """Return metadata for all available historical crash presets."""
    return HISTORICAL_CRASHES.copy()
