"""Synthetic order book generator with liquidation impact modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrderBookConfig:
    depth_per_side_usd: float = 3_000_000
    shape: str = "realistic"  # "linear", "realistic", "top_heavy"
    num_levels: int = 50
    spread_pct: float = 0.0002  # 0.02%
    refresh_rate_ms: float = 200


@dataclass
class OrderBookLevel:
    price: float
    size_usd: float


@dataclass
class OrderBook:
    """Synthetic order book that responds to liquidation pressure."""

    config: OrderBookConfig
    mid_price: float = 0.0
    bids: list[OrderBookLevel] = None  # type: ignore[assignment]
    asks: list[OrderBookLevel] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.bids is None:
            self.bids = []
        if self.asks is None:
            self.asks = []

    def generate(self, mid_price: float) -> None:
        """Generate a fresh order book around the given mid price."""
        self.mid_price = mid_price
        half_spread = mid_price * self.config.spread_pct / 2
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread

        depth_profile = self._get_depth_profile()

        self.bids = []
        self.asks = []
        n = self.config.num_levels

        for i in range(n):
            frac = (i + 1) / n
            bid_price = best_bid * (1 - frac * 0.01)  # levels span 1% below best bid
            ask_price = best_ask * (1 + frac * 0.01)
            size = self.config.depth_per_side_usd * depth_profile[i]
            self.bids.append(OrderBookLevel(bid_price, size))
            self.asks.append(OrderBookLevel(ask_price, size))

    def _get_depth_profile(self) -> np.ndarray:
        """Return normalized depth weights for each level."""
        n = self.config.num_levels
        if self.config.shape == "linear":
            weights = np.ones(n)
        elif self.config.shape == "top_heavy":
            weights = np.exp(-np.arange(n) * 0.1)
        else:  # realistic
            weights = np.exp(-np.arange(n) * 0.05) + 0.2
        return weights / weights.sum()

    def execute_sell(self, size_usd: float) -> tuple[float, float]:
        """Execute a sell order against the bid side.

        Returns (avg_fill_price, new_mid_price).
        Depletes bid levels and moves the mid price down.
        """
        if not self.bids or size_usd <= 0:
            return self.mid_price, self.mid_price

        remaining = size_usd
        total_filled_value = 0.0
        total_filled_size = 0.0

        for level in self.bids:
            if remaining <= 0:
                break
            fill = min(remaining, level.size_usd)
            total_filled_value += fill
            total_filled_size += fill / level.price
            level.size_usd -= fill
            remaining -= fill

        if remaining > 0:
            # Book depleted — fill rest at worst bid
            worst_price = self.bids[-1].price * 0.99
            total_filled_value += remaining
            total_filled_size += remaining / worst_price

        avg_fill_price = total_filled_value / total_filled_size if total_filled_size > 0 else self.mid_price

        # Find new best bid
        new_best_bid = self.mid_price
        for level in self.bids:
            if level.size_usd > 0:
                new_best_bid = level.price
                break

        # Update mid price
        best_ask = self.asks[0].price if self.asks else self.mid_price
        self.mid_price = (new_best_bid + best_ask) / 2

        return avg_fill_price, self.mid_price

    def execute_buy(self, size_usd: float) -> tuple[float, float]:
        """Execute a buy order against the ask side.

        Returns (avg_fill_price, new_mid_price).
        """
        if not self.asks or size_usd <= 0:
            return self.mid_price, self.mid_price

        remaining = size_usd
        total_filled_value = 0.0
        total_filled_size = 0.0

        for level in self.asks:
            if remaining <= 0:
                break
            fill = min(remaining, level.size_usd)
            total_filled_value += fill
            total_filled_size += fill / level.price
            level.size_usd -= fill
            remaining -= fill

        if remaining > 0:
            worst_price = self.asks[-1].price * 1.01
            total_filled_value += remaining
            total_filled_size += remaining / worst_price

        avg_fill_price = total_filled_value / total_filled_size if total_filled_size > 0 else self.mid_price

        new_best_ask = self.mid_price
        for level in self.asks:
            if level.size_usd > 0:
                new_best_ask = level.price
                break

        best_bid = self.bids[0].price if self.bids else self.mid_price
        self.mid_price = (best_bid + new_best_ask) / 2

        return avg_fill_price, self.mid_price

    def refresh(self, mid_price: float) -> None:
        """Regenerate the book around a new mid price (MM refresh)."""
        self.generate(mid_price)

    def total_bid_depth(self) -> float:
        return sum(l.size_usd for l in self.bids)

    def total_ask_depth(self) -> float:
        return sum(l.size_usd for l in self.asks)

    def get_slippage_pct(self, size_usd: float, side: str = "sell") -> float:
        """Estimate slippage for a given order size without executing."""
        if side == "sell":
            fill_price, _ = self._simulate_sell(size_usd)
        else:
            fill_price, _ = self._simulate_buy(size_usd)
        return abs(fill_price - self.mid_price) / self.mid_price if self.mid_price > 0 else 0.0

    def _simulate_sell(self, size_usd: float) -> tuple[float, float]:
        """Simulate sell without modifying state."""
        if not self.bids:
            return self.mid_price, self.mid_price
        remaining = size_usd
        total_value = 0.0
        total_size = 0.0
        for level in self.bids:
            if remaining <= 0:
                break
            fill = min(remaining, level.size_usd)
            total_value += fill
            total_size += fill / level.price
            remaining -= fill
        if remaining > 0:
            worst = self.bids[-1].price * 0.99
            total_value += remaining
            total_size += remaining / worst
        avg = total_value / total_size if total_size > 0 else self.mid_price
        return avg, self.mid_price

    def _simulate_buy(self, size_usd: float) -> tuple[float, float]:
        if not self.asks:
            return self.mid_price, self.mid_price
        remaining = size_usd
        total_value = 0.0
        total_size = 0.0
        for level in self.asks:
            if remaining <= 0:
                break
            fill = min(remaining, level.size_usd)
            total_value += fill
            total_size += fill / level.price
            remaining -= fill
        if remaining > 0:
            worst = self.asks[-1].price * 1.01
            total_value += remaining
            total_size += remaining / worst
        avg = total_value / total_size if total_size > 0 else self.mid_price
        return avg, self.mid_price
