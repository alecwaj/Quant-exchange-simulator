"""MAU → OI/position distribution model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GrowthModelConfig:
    active_trader_pct: float = 0.20  # 15-25% of MAU are active traders
    positions_per_trader: float = 1.5
    median_size_retail: float = 20_000
    median_size_whale: float = 500_000
    whale_pct: float = 0.05  # top 5% of users
    whale_oi_share: float = 0.40  # hold 40% of OI
    leverage_dist: dict[int, float] | None = None
    side_bias: float = 0.6

    def __post_init__(self) -> None:
        if self.leverage_dist is None:
            self.leverage_dist = {5: 0.2, 10: 0.4, 20: 0.3, 50: 0.1}


def mau_to_position_count(mau: int, config: GrowthModelConfig | None = None) -> int:
    """Estimate number of open positions from MAU."""
    if config is None:
        config = GrowthModelConfig()
    active = int(mau * config.active_trader_pct)
    return max(1, int(active * config.positions_per_trader))


def mau_to_total_oi(mau: int, config: GrowthModelConfig | None = None) -> float:
    """Estimate total OI from MAU."""
    if config is None:
        config = GrowthModelConfig()
    n_positions = mau_to_position_count(mau, config)
    n_whales = max(1, int(n_positions * config.whale_pct))
    n_retail = n_positions - n_whales

    retail_oi = n_retail * config.median_size_retail
    whale_oi = retail_oi * config.whale_oi_share / (1 - config.whale_oi_share)
    return retail_oi + whale_oi


def estimate_avg_position_size(mau: int, config: GrowthModelConfig | None = None) -> float:
    """Estimate average position size from MAU."""
    if config is None:
        config = GrowthModelConfig()
    total_oi = mau_to_total_oi(mau, config)
    n_positions = mau_to_position_count(mau, config)
    return total_oi / n_positions if n_positions > 0 else 0


def project_growth(
    mau_schedule: list[tuple[int, int]],  # list of (month, mau)
    config: GrowthModelConfig | None = None,
) -> list[dict]:
    """Project OI and position metrics across a growth schedule."""
    if config is None:
        config = GrowthModelConfig()

    projections = []
    for month, mau in mau_schedule:
        n_pos = mau_to_position_count(mau, config)
        total_oi = mau_to_total_oi(mau, config)
        avg_size = total_oi / n_pos if n_pos > 0 else 0
        projections.append({
            "month": month,
            "mau": mau,
            "active_traders": int(mau * config.active_trader_pct),
            "positions": n_pos,
            "total_oi": total_oi,
            "avg_position_size": avg_size,
        })
    return projections
