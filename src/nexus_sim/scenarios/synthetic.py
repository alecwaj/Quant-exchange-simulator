"""Parametric crash generators for synthetic scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CrashScenario:
    name: str
    description: str
    prices: np.ndarray  # price series (absolute)
    duration_sec: float
    magnitude_pct: float
    block_time_ms: float


def generate_instant_crash(
    start_price: float,
    magnitude_pct: float,
    block_time_ms: float = 5,
    settle_duration_sec: float = 60,
) -> CrashScenario:
    """Instantaneous drop — price falls magnitude_pct in one block, then stays flat."""
    n_blocks = max(2, int(settle_duration_sec * 1000 / block_time_ms))
    prices = np.full(n_blocks, start_price * (1 - magnitude_pct / 100))
    prices[0] = start_price
    return CrashScenario(
        name=f"instant_{magnitude_pct}pct",
        description=f"Instantaneous {magnitude_pct}% drop",
        prices=prices,
        duration_sec=settle_duration_sec,
        magnitude_pct=magnitude_pct,
        block_time_ms=block_time_ms,
    )


def generate_linear_crash(
    start_price: float,
    magnitude_pct: float,
    duration_sec: float,
    block_time_ms: float = 5,
) -> CrashScenario:
    """Linear price decline over specified duration."""
    n_blocks = max(2, int(duration_sec * 1000 / block_time_ms))
    end_price = start_price * (1 - magnitude_pct / 100)
    prices = np.linspace(start_price, end_price, n_blocks)
    return CrashScenario(
        name=f"linear_{magnitude_pct}pct_{duration_sec}s",
        description=f"Linear {magnitude_pct}% drop over {duration_sec}s",
        prices=prices,
        duration_sec=duration_sec,
        magnitude_pct=magnitude_pct,
        block_time_ms=block_time_ms,
    )


def generate_vshape_crash(
    start_price: float,
    magnitude_pct: float,
    recovery_sec: float,
    block_time_ms: float = 5,
) -> CrashScenario:
    """V-shape: instant drop then linear recovery."""
    total_duration = recovery_sec + 1
    n_blocks = max(3, int(total_duration * 1000 / block_time_ms))
    bottom_price = start_price * (1 - magnitude_pct / 100)

    prices = np.empty(n_blocks)
    prices[0] = start_price
    # Instant drop at block 1
    n_recovery = n_blocks - 1
    prices[1:] = np.linspace(bottom_price, start_price, n_recovery)
    return CrashScenario(
        name=f"vshape_{magnitude_pct}pct_{recovery_sec}s",
        description=f"V-shape: {magnitude_pct}% drop, recover over {recovery_sec}s",
        prices=prices,
        duration_sec=total_duration,
        magnitude_pct=magnitude_pct,
        block_time_ms=block_time_ms,
    )


def generate_cascade_crash(
    start_price: float,
    initial_drop_pct: float,
    steps: int,
    interval_sec: float,
    block_time_ms: float = 5,
) -> CrashScenario:
    """Staircase crash: repeated drops at intervals."""
    blocks_per_step = max(1, int(interval_sec * 1000 / block_time_ms))
    n_blocks = blocks_per_step * steps
    prices = np.empty(n_blocks)

    current_price = start_price
    for step in range(steps):
        current_price *= (1 - initial_drop_pct / 100)
        start_idx = step * blocks_per_step
        end_idx = (step + 1) * blocks_per_step
        prices[start_idx:end_idx] = current_price

    total_mag = (1 - prices[-1] / start_price) * 100
    total_dur = n_blocks * block_time_ms / 1000
    return CrashScenario(
        name=f"cascade_{initial_drop_pct}pct_x{steps}",
        description=f"Staircase: {steps} drops of {initial_drop_pct}% every {interval_sec}s (total {total_mag:.1f}%)",
        prices=prices,
        duration_sec=total_dur,
        magnitude_pct=total_mag,
        block_time_ms=block_time_ms,
    )


def generate_volatility_scenario(
    start_price: float,
    annualized_vol: float,
    duration_sec: float,
    block_time_ms: float = 5,
    seed: int = 42,
) -> CrashScenario:
    """Geometric Brownian Motion with specified annualized volatility."""
    rng = np.random.default_rng(seed)
    n_blocks = max(2, int(duration_sec * 1000 / block_time_ms))

    # Convert annualized vol to per-block vol
    seconds_per_year = 365.25 * 24 * 3600
    dt = block_time_ms / 1000 / seconds_per_year
    block_vol = annualized_vol / 100 * np.sqrt(dt)

    # GBM: dS/S = sigma * dW (zero drift for crash simulation)
    returns = rng.normal(0, block_vol, n_blocks - 1)
    log_prices = np.cumsum(np.concatenate([[0], returns]))
    prices = start_price * np.exp(log_prices)

    actual_mag = (1 - prices.min() / start_price) * 100
    return CrashScenario(
        name=f"volatility_{annualized_vol}vol_{duration_sec}s",
        description=f"GBM: {annualized_vol}% annualized vol over {duration_sec}s",
        prices=prices,
        duration_sec=duration_sec,
        magnitude_pct=actual_mag,
        block_time_ms=block_time_ms,
    )
