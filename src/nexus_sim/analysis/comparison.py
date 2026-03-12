"""Competitor benchmarking — Hyperliquid, dYdX, Binance."""

from __future__ import annotations

from typing import Any

import numpy as np

from nexus_sim.engine.insurance import InsuranceFund, InsuranceFundConfig
from nexus_sim.engine.liquidation import LiquidationConfig, run_simulation, LiquidationResult
from nexus_sim.engine.mark_price import MarkPriceConfig, run_mark_price_vectorized
from nexus_sim.engine.orderbook import OrderBook, OrderBookConfig
from nexus_sim.engine.positions import generate_positions, PositionConfig
from nexus_sim.scenarios.presets import SimulationState

COMPETITORS: dict[str, dict[str, Any]] = {
    "hyperliquid": {
        "mm_ratio": 0.05,
        "max_leverage": 50,
        "block_time_ms": 200,
        "book_depth_usd": 7_000_000,
        "insurance_fund": 50_000_000,
        "mark_price": "oracle + ema_basis",
        "liquidation": "engine_native",
        "phase": 2,
    },
    "dydx_v4": {
        "mm_ratio": 0.05,
        "max_leverage": 20,
        "block_time_ms": 500,
        "book_depth_usd": 3_000_000,
        "insurance_fund": 20_000_000,
        "mark_price": "oracle",
        "liquidation": "keeper_auction",
        "phase": 1,
    },
    "binance": {
        "mm_ratio": 0.025,
        "max_leverage": 125,
        "block_time_ms": 1,
        "book_depth_usd": 50_000_000,
        "insurance_fund": 1_000_000_000,
        "mark_price": "oracle + basis",
        "liquidation": "engine_native",
        "phase": 2,
    },
}


def run_comparison(
    oracle_prices: np.ndarray,
    nexus_state: SimulationState,
    competitors: list[str] | None = None,
    entry_price: float = 60_000,
) -> dict[str, dict[str, Any]]:
    """Run the same crash scenario across Nexus and competitor configurations.

    Returns dict of exchange_name → result metrics.
    """
    if competitors is None:
        competitors = list(COMPETITORS.keys())

    results: dict[str, dict[str, Any]] = {}

    # Run Nexus
    nexus_result = _run_for_config(
        oracle_prices=oracle_prices,
        block_time_ms=nexus_state.exchange.block_time_ms,
        mm_ratio=nexus_state.exchange.mm_ratio,
        max_leverage=nexus_state.exchange.max_leverage,
        phase=nexus_state.exchange.phase,
        book_depth=nexus_state.market.book_depth_usd,
        insurance_balance=nexus_state.insurance.initial_balance,
        position_count=nexus_state.users.position_count,
        entry_price=entry_price,
        max_price_change_pct=nexus_state.exchange.max_price_change_pct,
    )
    results["Nexus"] = nexus_result

    # Run competitors
    for name in competitors:
        if name not in COMPETITORS:
            continue
        comp = COMPETITORS[name]
        comp_result = _run_for_config(
            oracle_prices=oracle_prices,
            block_time_ms=comp["block_time_ms"],
            mm_ratio=comp["mm_ratio"],
            max_leverage=comp["max_leverage"],
            phase=comp.get("phase", 1),
            book_depth=comp["book_depth_usd"],
            insurance_balance=comp["insurance_fund"],
            position_count=nexus_state.users.position_count,
            entry_price=entry_price,
            max_price_change_pct=0.05,
        )
        results[name.title()] = comp_result

    return results


def _run_for_config(
    oracle_prices: np.ndarray,
    block_time_ms: float,
    mm_ratio: float,
    max_leverage: int,
    phase: int,
    book_depth: float,
    insurance_balance: float,
    position_count: int,
    entry_price: float,
    max_price_change_pct: float,
) -> dict[str, Any]:
    """Run a simulation with specific exchange parameters."""
    # Resample oracle prices to block time
    source_n = len(oracle_prices)
    # Assume source is at some standard resolution, map to block time
    target_n = max(2, source_n)

    mp_config = MarkPriceConfig(
        block_time_ms=block_time_ms,
        max_price_change_pct=max_price_change_pct,
        phase=phase,
    )
    mark_prices, _ = run_mark_price_vectorized(mp_config, oracle_prices)

    # Generate positions
    pos_config = PositionConfig(count=min(position_count, 1000), avg_size_usd=50_000)
    positions = generate_positions(pos_config, entry_price, mm_ratio)

    # Order book
    ob_config = OrderBookConfig(depth_per_side_usd=book_depth)
    orderbook = OrderBook(config=ob_config)
    orderbook.generate(entry_price)

    # Insurance
    ins_config = InsuranceFundConfig(initial_balance=insurance_balance)
    insurance = InsuranceFund(config=ins_config)

    # Liquidation
    liq_config = LiquidationConfig(mm_ratio=mm_ratio, max_leverage=max_leverage)

    result = run_simulation(mark_prices, positions, orderbook, insurance, liq_config)

    total = result.total_positions
    liqd = result.total_liquidated
    pct = liqd / total * 100 if total > 0 else 0

    return {
        "Block Time": f"{block_time_ms}ms",
        "MM Ratio": f"{mm_ratio * 100:.1f}%",
        "Max Leverage": f"{max_leverage}x",
        "Liquidated": f"{liqd}/{total} ({pct:.0f}%)",
        "Insurance Delta": f"${insurance.balance - insurance.config.initial_balance:+,.0f}",
        "Max Drawdown": f"${insurance.max_drawdown:,.0f}",
        "Cascade Depth": str(result.max_cascade_depth),
        "Safety": _safety_rating(pct, insurance.balance, insurance.config.initial_balance),
    }


def _safety_rating(liq_pct: float, final_balance: float, initial_balance: float) -> str:
    delta_pct = (final_balance - initial_balance) / initial_balance * 100 if initial_balance > 0 else 0
    if liq_pct < 50 and delta_pct > 0:
        return "SAFE"
    elif liq_pct < 75 and delta_pct > -20:
        return "MODERATE"
    elif final_balance > 0:
        return "RISKY"
    else:
        return "CRITICAL"


def list_competitors() -> dict[str, dict[str, Any]]:
    return COMPETITORS.copy()
