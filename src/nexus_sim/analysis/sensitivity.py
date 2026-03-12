"""Parameter sweep and heatmap generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from rich.progress import Progress

from nexus_sim.engine.insurance import InsuranceFund, InsuranceFundConfig
from nexus_sim.engine.liquidation import LiquidationConfig, LiquidationResult, run_simulation
from nexus_sim.engine.mark_price import MarkPriceConfig, run_mark_price_vectorized
from nexus_sim.engine.orderbook import OrderBook, OrderBookConfig
from nexus_sim.engine.positions import PositionConfig, generate_positions
from nexus_sim.scenarios.presets import SimulationState


@dataclass
class SweepResult:
    param_name: str
    param_values: list[float]
    metric_values: list[float]
    metric_name: str


@dataclass
class HeatmapResult:
    x_name: str
    y_name: str
    x_values: np.ndarray
    y_values: np.ndarray
    z_matrix: np.ndarray
    z_name: str


def run_parameter_sweep(
    state: SimulationState,
    oracle_prices: np.ndarray,
    param_name: str,
    min_val: float,
    max_val: float,
    steps: int = 10,
    metric: str = "liquidation_pct",
) -> SweepResult:
    """Sweep a single parameter and measure a metric."""
    values = np.linspace(min_val, max_val, steps)
    metrics = []

    for val in values:
        result = _run_with_override(state, oracle_prices, param_name, float(val))
        metrics.append(_extract_metric(result, metric))

    return SweepResult(
        param_name=param_name,
        param_values=values.tolist(),
        metric_values=metrics,
        metric_name=metric,
    )


def run_heatmap(
    state: SimulationState,
    oracle_prices: np.ndarray,
    x_param: str,
    y_param: str,
    x_range: tuple[float, float, int] = (0.02, 0.10, 8),
    y_range: tuple[float, float, int] = (5, 50, 8),
    metric: str = "liquidation_pct",
) -> HeatmapResult:
    """Generate a 2D heatmap by sweeping two parameters."""
    x_vals = np.linspace(x_range[0], x_range[1], x_range[2])
    y_vals = np.linspace(y_range[0], y_range[1], y_range[2])
    z_matrix = np.zeros((len(y_vals), len(x_vals)))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            result = _run_with_overrides(state, oracle_prices, {x_param: float(x), y_param: float(y)})
            z_matrix[i, j] = _extract_metric(result, metric)

    return HeatmapResult(
        x_name=x_param,
        y_name=y_param,
        x_values=x_vals,
        y_values=y_vals,
        z_matrix=z_matrix,
        z_name=metric,
    )


def buffer_zone_analysis(
    mm_ratios: list[float] | None = None,
    leverages: list[int] | None = None,
) -> HeatmapResult:
    """Compute buffer zone (IM - MM) for leverage/MM ratio grid."""
    if mm_ratios is None:
        mm_ratios = [0.02, 0.03, 0.04, 0.05, 0.0625, 0.08, 0.10]
    if leverages is None:
        leverages = [5, 10, 15, 20, 25, 30, 40, 50]

    x_vals = np.array(mm_ratios)
    y_vals = np.array(leverages, dtype=float)
    z = np.zeros((len(y_vals), len(x_vals)))

    for i, lev in enumerate(leverages):
        im = 1.0 / lev
        for j, mm in enumerate(mm_ratios):
            buffer = (im - mm) * 100
            z[i, j] = max(0, buffer)

    return HeatmapResult(
        x_name="MM Ratio",
        y_name="Max Leverage",
        x_values=x_vals,
        y_values=y_vals,
        z_matrix=z,
        z_name="Buffer Zone (%)",
    )


def insurance_sizing_analysis(
    state: SimulationState,
    oracle_prices: np.ndarray,
    balance_range: tuple[float, float, int] = (10_000, 1_000_000, 10),
) -> SweepResult:
    """Sweep insurance fund balance and measure deficit coverage."""
    return run_parameter_sweep(
        state, oracle_prices,
        param_name="insurance_balance",
        min_val=balance_range[0],
        max_val=balance_range[1],
        steps=balance_range[2],
        metric="insurance_delta_pct",
    )


def find_worst_case(
    state: SimulationState,
    oracle_prices: np.ndarray,
    n_samples: int = 50,
) -> dict[str, Any]:
    """Monte Carlo search for parameter combo that depletes the fund fastest."""
    rng = np.random.default_rng(42)
    worst_balance = float("inf")
    worst_params: dict[str, float] = {}

    for _ in range(n_samples):
        overrides = {
            "mm_ratio": float(rng.uniform(0.02, 0.10)),
            "max_leverage": float(rng.integers(5, 50)),
            "book_depth": float(rng.uniform(100_000, 5_000_000)),
        }
        result = _run_with_overrides(state, oracle_prices, overrides)
        if result["final_insurance"] < worst_balance:
            worst_balance = result["final_insurance"]
            worst_params = overrides.copy()

    return {
        "worst_balance": worst_balance,
        "params": worst_params,
    }


def _run_with_override(
    state: SimulationState,
    oracle_prices: np.ndarray,
    param_name: str,
    value: float,
) -> dict[str, Any]:
    return _run_with_overrides(state, oracle_prices, {param_name: value})


def _run_with_overrides(
    state: SimulationState,
    oracle_prices: np.ndarray,
    overrides: dict[str, float],
) -> dict[str, Any]:
    """Run simulation with parameter overrides."""
    # Apply overrides
    block_time = overrides.get("block_time", state.exchange.block_time_ms)
    mm_ratio = overrides.get("mm_ratio", state.exchange.mm_ratio)
    max_leverage = int(overrides.get("max_leverage", state.exchange.max_leverage))
    book_depth = overrides.get("book_depth", state.market.book_depth_usd)
    ins_balance = overrides.get("insurance_balance", state.insurance.initial_balance)
    phase = int(overrides.get("phase", state.exchange.phase))

    # Mark price
    mp_config = MarkPriceConfig(
        block_time_ms=block_time,
        max_price_change_pct=state.exchange.max_price_change_pct,
        phase=phase,
    )
    mark_prices, _ = run_mark_price_vectorized(mp_config, oracle_prices)

    # Positions
    pos_config = PositionConfig(
        count=min(state.users.position_count, 500),
        avg_size_usd=state.users.avg_size_usd,
    )
    positions = generate_positions(pos_config, state.entry_price, mm_ratio)

    # Order book
    ob_config = OrderBookConfig(depth_per_side_usd=book_depth)
    orderbook = OrderBook(config=ob_config)
    orderbook.generate(state.entry_price)

    # Insurance
    ins_config = InsuranceFundConfig(initial_balance=ins_balance)
    insurance = InsuranceFund(config=ins_config)

    # Liquidation
    liq_config = LiquidationConfig(mm_ratio=mm_ratio, max_leverage=max_leverage)
    result = run_simulation(mark_prices, positions, orderbook, insurance, liq_config)

    return {
        "result": result,
        "insurance": insurance,
        "final_insurance": insurance.balance,
        "liquidation_pct": result.total_liquidated / result.total_positions * 100 if result.total_positions > 0 else 0,
        "insurance_delta_pct": (insurance.balance - ins_balance) / ins_balance * 100 if ins_balance > 0 else 0,
        "max_drawdown": insurance.max_drawdown,
        "cascade_depth": result.max_cascade_depth,
    }


def _extract_metric(result: dict[str, Any], metric: str) -> float:
    return float(result.get(metric, 0))
