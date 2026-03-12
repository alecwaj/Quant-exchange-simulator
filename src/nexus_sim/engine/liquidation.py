"""Liquidation engine with full execution model and cascade tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from nexus_sim.engine.insurance import InsuranceFund
from nexus_sim.engine.orderbook import OrderBook
from nexus_sim.engine.positions import Position


class LiquidationEvent(NamedTuple):
    block: int
    position_id: int
    side: str
    size_usd: float
    leverage: float
    entry_price: float
    mark_price: float
    bankruptcy_price: float
    fill_price: float
    residual: float  # positive = profit to fund, negative = deficit
    slippage_pct: float
    cascade_depth: int


@dataclass
class LiquidationConfig:
    mm_ratio: float = 0.0625
    max_leverage: int = 16
    max_liquidations_per_block: int = 100
    use_dual_mark_price: bool = False
    same_block_execution: bool = True
    cascade_dampening: bool = False


@dataclass
class LiquidationResult:
    total_liquidated: int = 0
    total_positions: int = 0
    max_cascade_depth: int = 0
    events: list[LiquidationEvent] = field(default_factory=list)
    total_residual: float = 0.0
    total_deficit: float = 0.0
    unfilled: int = 0
    worst_fill_ratio: float = 1.0  # worst fill_price / bankruptcy_price


@dataclass
class LiquidationEngine:
    """Engine that scans positions and executes liquidations each block."""

    config: LiquidationConfig
    result: LiquidationResult = field(default_factory=LiquidationResult)

    def reset(self) -> None:
        self.result = LiquidationResult()

    def process_block(
        self,
        block: int,
        mark_price: float,
        positions: list[Position],
        orderbook: OrderBook,
        insurance: InsuranceFund,
    ) -> int:
        """Process liquidations for a single block.

        Returns number of liquidations executed.
        """
        # Find positions needing liquidation, sorted by margin ratio (worst first)
        candidates = []
        for pos in positions:
            if pos.is_liquidated:
                continue
            mr = pos.margin_ratio(mark_price)
            if mr <= self.config.mm_ratio:
                candidates.append((mr, pos))

        candidates.sort(key=lambda x: x[0])

        liq_count = 0
        cascade_depth = 0
        current_mark = mark_price

        for _, pos in candidates[:self.config.max_liquidations_per_block]:
            # Execute liquidation via limit order at bankruptcy price
            if pos.side == "long":
                fill_price, new_mid = orderbook.execute_sell(pos.size_usd)
                fill_price = min(fill_price, pos.bankruptcy_price)  # can't be better than bankruptcy
                fill_price = max(fill_price, current_mark * 0.9)  # floor at 10% below mark
            else:
                fill_price, new_mid = orderbook.execute_buy(pos.size_usd)
                fill_price = max(fill_price, pos.bankruptcy_price)
                fill_price = min(fill_price, current_mark * 1.1)

            # Compute residual
            residual = pos.residual_at_fill(fill_price)

            # Slippage
            if pos.bankruptcy_price > 0:
                slippage_pct = abs(fill_price - pos.bankruptcy_price) / pos.bankruptcy_price
            else:
                slippage_pct = 0.0

            # Fill ratio (how close fill is to bankruptcy)
            if pos.side == "long":
                fill_ratio = fill_price / pos.bankruptcy_price if pos.bankruptcy_price > 0 else 1.0
            else:
                fill_ratio = pos.bankruptcy_price / fill_price if fill_price > 0 else 1.0

            # Update insurance fund
            if residual >= 0:
                insurance.add_residual(block, residual)
                self.result.total_residual += residual
            else:
                insurance.cover_deficit(block, abs(residual))
                self.result.total_deficit += abs(residual)

            # Track worst fill
            self.result.worst_fill_ratio = min(self.result.worst_fill_ratio, fill_ratio)

            # Record event
            event = LiquidationEvent(
                block=block,
                position_id=pos.id,
                side=pos.side,
                size_usd=pos.size_usd,
                leverage=pos.leverage,
                entry_price=pos.entry_price,
                mark_price=current_mark,
                bankruptcy_price=pos.bankruptcy_price,
                fill_price=fill_price,
                residual=residual,
                slippage_pct=slippage_pct,
                cascade_depth=cascade_depth,
            )
            self.result.events.append(event)

            pos.is_liquidated = True
            liq_count += 1

            # Cascade: if liquidation moved the mid, check for new liquidations
            if self.config.cascade_dampening and new_mid != current_mark:
                cascade_depth += 1
                current_mark = new_mid

        self.result.total_liquidated += liq_count
        self.result.max_cascade_depth = max(self.result.max_cascade_depth, cascade_depth)

        return liq_count


def run_simulation(
    mark_prices: np.ndarray,
    positions: list[Position],
    orderbook: OrderBook,
    insurance: InsuranceFund,
    liq_config: LiquidationConfig,
    book_refresh_blocks: int = 40,  # refresh every N blocks
) -> LiquidationResult:
    """Run a full simulation over a mark price series.

    This is the main simulation loop. For performance, inner oracle→mark
    price computation should be done with run_mark_price_vectorized before
    calling this function.
    """
    engine = LiquidationEngine(config=liq_config)
    engine.result.total_positions = len(positions)
    n_blocks = len(mark_prices)

    for block in range(n_blocks):
        mp = float(mark_prices[block])

        # Periodic book refresh
        if block % book_refresh_blocks == 0:
            orderbook.refresh(mp)

        # Process liquidations
        engine.process_block(block, mp, positions, orderbook, insurance)

        # Record insurance fund state
        insurance.record_block(block)

    return engine.result
