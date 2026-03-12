"""Cascading liquidation analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus_sim.engine.insurance import InsuranceFund, InsuranceFundConfig
from nexus_sim.engine.liquidation import LiquidationConfig, LiquidationResult, run_simulation
from nexus_sim.engine.mark_price import MarkPriceConfig, run_mark_price_vectorized
from nexus_sim.engine.orderbook import OrderBook, OrderBookConfig
from nexus_sim.engine.positions import PositionConfig, generate_positions
from nexus_sim.scenarios.presets import SimulationState


@dataclass
class CascadeAnalysisResult:
    """Results of cascading liquidation analysis."""
    scenario_name: str
    max_cascade_depth: int
    avg_cascade_depth: float
    cascade_events: list[dict]  # list of {block, depth, size_usd, price_impact_pct}
    total_cascade_volume: float
    peak_block_liquidations: int
    recovery_blocks: int  # blocks until market stabilizes


def analyze_cascade_risk(
    state: SimulationState,
    oracle_prices: np.ndarray,
    dampening: bool = False,
) -> CascadeAnalysisResult:
    """Analyze cascading liquidation dynamics for a given scenario."""
    mp_config = MarkPriceConfig(
        block_time_ms=state.exchange.block_time_ms,
        max_price_change_pct=state.exchange.max_price_change_pct,
        phase=state.exchange.phase,
    )
    mark_prices, _ = run_mark_price_vectorized(mp_config, oracle_prices)

    pos_config = PositionConfig(
        count=min(state.users.position_count, 500),
        avg_size_usd=state.users.avg_size_usd,
    )
    positions = generate_positions(pos_config, state.entry_price, state.exchange.mm_ratio)

    ob_config = OrderBookConfig(depth_per_side_usd=state.market.book_depth_usd)
    orderbook = OrderBook(config=ob_config)
    orderbook.generate(state.entry_price)

    ins_config = InsuranceFundConfig(initial_balance=state.insurance.initial_balance)
    insurance = InsuranceFund(config=ins_config)

    liq_config = LiquidationConfig(
        mm_ratio=state.exchange.mm_ratio,
        max_leverage=state.exchange.max_leverage,
        cascade_dampening=dampening,
    )

    result = run_simulation(mark_prices, positions, orderbook, insurance, liq_config)

    # Analyze cascade patterns
    cascade_events = []
    block_liq_counts: dict[int, int] = {}
    block_liq_volume: dict[int, float] = {}

    for event in result.events:
        block_liq_counts[event.block] = block_liq_counts.get(event.block, 0) + 1
        block_liq_volume[event.block] = block_liq_volume.get(event.block, 0) + event.size_usd
        if event.cascade_depth > 0:
            cascade_events.append({
                "block": event.block,
                "depth": event.cascade_depth,
                "size_usd": event.size_usd,
                "price_impact_pct": event.slippage_pct * 100,
            })

    peak_block_liqs = max(block_liq_counts.values()) if block_liq_counts else 0

    # Estimate recovery: blocks after last liquidation
    if result.events:
        last_liq_block = max(e.block for e in result.events)
        recovery_blocks = len(mark_prices) - last_liq_block
    else:
        recovery_blocks = len(mark_prices)

    avg_cascade = (
        sum(e["depth"] for e in cascade_events) / len(cascade_events)
        if cascade_events else 0
    )

    return CascadeAnalysisResult(
        scenario_name="cascade_analysis",
        max_cascade_depth=result.max_cascade_depth,
        avg_cascade_depth=avg_cascade,
        cascade_events=cascade_events,
        total_cascade_volume=sum(e["size_usd"] for e in cascade_events),
        peak_block_liquidations=peak_block_liqs,
        recovery_blocks=recovery_blocks,
    )


def compare_dampening(
    state: SimulationState,
    oracle_prices: np.ndarray,
) -> dict[str, CascadeAnalysisResult]:
    """Compare cascade behavior with and without dampening."""
    return {
        "without_dampening": analyze_cascade_risk(state, oracle_prices, dampening=False),
        "with_dampening": analyze_cascade_risk(state, oracle_prices, dampening=True),
    }
