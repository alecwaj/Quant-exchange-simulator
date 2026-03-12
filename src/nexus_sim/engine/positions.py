"""Position generator with direct specification and MAU-based inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Position:
    id: int
    side: str  # "long" or "short"
    size_usd: float
    leverage: float
    entry_price: float
    margin: float = 0.0  # computed from size / leverage
    liquidation_price: float = 0.0
    bankruptcy_price: float = 0.0
    is_liquidated: bool = False

    def __post_init__(self) -> None:
        if self.margin == 0.0:
            self.margin = self.size_usd / self.leverage

    def compute_prices(self, mm_ratio: float) -> None:
        """Compute liquidation and bankruptcy prices."""
        if self.side == "long":
            self.liquidation_price = self.entry_price * (1 - (1 / self.leverage) + mm_ratio)
            self.bankruptcy_price = self.entry_price * (1 - 1 / self.leverage)
        else:
            self.liquidation_price = self.entry_price * (1 + (1 / self.leverage) - mm_ratio)
            self.bankruptcy_price = self.entry_price * (1 + 1 / self.leverage)

    def margin_ratio(self, mark_price: float) -> float:
        """Current margin ratio given a mark price."""
        if self.size_usd <= 0:
            return 1.0
        notional = self.size_usd  # fixed notional for simplicity
        if self.side == "long":
            unrealized_pnl = notional * (mark_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = notional * (self.entry_price - mark_price) / self.entry_price
        current_margin = self.margin + unrealized_pnl
        return current_margin / notional if notional > 0 else 0.0

    def residual_at_fill(self, fill_price: float) -> float:
        """Compute residual (positive) or deficit (negative) at fill price."""
        if self.side == "long":
            pnl = self.size_usd * (fill_price - self.entry_price) / self.entry_price
        else:
            pnl = self.size_usd * (self.entry_price - fill_price) / self.entry_price
        return self.margin + pnl


@dataclass
class PositionConfig:
    count: int = 500
    avg_size_usd: float = 50_000
    leverage_dist: dict[int, float] = field(default_factory=lambda: {5: 0.2, 10: 0.4, 20: 0.3, 50: 0.1})
    side_bias: float = 0.6  # fraction that are long


def parse_leverage_dist(spec: str) -> dict[int, float]:
    """Parse '5x:20%,10x:40%,20x:30%,50x:10%' into dict."""
    result = {}
    for part in spec.split(","):
        lev_str, pct_str = part.strip().split(":")
        lev = int(lev_str.replace("x", ""))
        pct = float(pct_str.replace("%", "")) / 100
        result[lev] = pct
    return result


def generate_positions(
    config: PositionConfig,
    entry_price: float,
    mm_ratio: float,
    rng: np.random.Generator | None = None,
) -> list[Position]:
    """Generate a set of positions based on config."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions = []
    leverages = list(config.leverage_dist.keys())
    weights = list(config.leverage_dist.values())
    # Normalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # Generate sizes from lognormal
    sizes = rng.lognormal(
        mean=np.log(config.avg_size_usd),
        sigma=0.8,
        size=config.count,
    )

    for i in range(config.count):
        side = "long" if rng.random() < config.side_bias else "short"
        leverage = rng.choice(leverages, p=weights)
        pos = Position(
            id=i,
            side=side,
            size_usd=float(sizes[i]),
            leverage=float(leverage),
            entry_price=entry_price,
        )
        pos.compute_prices(mm_ratio)
        positions.append(pos)

    return positions


def generate_positions_from_mau(
    mau: int,
    entry_price: float,
    mm_ratio: float,
    active_trader_pct: float = 0.20,
    positions_per_trader: float = 1.5,
    median_size_retail: float = 20_000,
    median_size_whale: float = 500_000,
    whale_pct: float = 0.05,
    whale_oi_share: float = 0.40,
    leverage_dist: dict[int, float] | None = None,
    side_bias: float = 0.6,
    rng: np.random.Generator | None = None,
) -> list[Position]:
    """Generate positions from MAU count using growth model assumptions."""
    if rng is None:
        rng = np.random.default_rng(42)
    if leverage_dist is None:
        leverage_dist = {5: 0.2, 10: 0.4, 20: 0.3, 50: 0.1}

    active_traders = int(mau * active_trader_pct)
    total_positions = max(1, int(active_traders * positions_per_trader))

    # Split into retail and whale segments
    n_whales = max(1, int(total_positions * whale_pct))
    n_retail = total_positions - n_whales

    retail_sizes = rng.lognormal(np.log(median_size_retail), 0.7, n_retail)
    whale_sizes = rng.lognormal(np.log(median_size_whale), 0.5, n_whales)

    # Scale whale sizes to hit target OI share
    total_retail_oi = retail_sizes.sum()
    target_whale_oi = total_retail_oi * whale_oi_share / (1 - whale_oi_share)
    if whale_sizes.sum() > 0:
        whale_sizes *= target_whale_oi / whale_sizes.sum()

    all_sizes = np.concatenate([retail_sizes, whale_sizes])
    rng.shuffle(all_sizes)

    leverages = list(leverage_dist.keys())
    weights = list(leverage_dist.values())
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    positions = []
    for i in range(len(all_sizes)):
        side = "long" if rng.random() < side_bias else "short"
        leverage = rng.choice(leverages, p=weights)
        pos = Position(
            id=i,
            side=side,
            size_usd=float(all_sizes[i]),
            leverage=float(leverage),
            entry_price=entry_price,
        )
        pos.compute_prices(mm_ratio)
        positions.append(pos)

    return positions
