"""Plotext inline terminal charts."""

from __future__ import annotations

import numpy as np
import plotext as plt

from nexus_sim.engine.liquidation import LiquidationResult
from nexus_sim.engine.insurance import InsuranceFund


def chart_mark_price(
    mark_prices: np.ndarray,
    oracle_prices: np.ndarray,
    block_time_ms: float = 5,
    title: str = "Mark Price vs Oracle Price",
) -> None:
    """Plot mark price and oracle price as terminal line chart."""
    n = len(mark_prices)
    # Downsample for terminal display
    max_points = 200
    if n > max_points:
        step = n // max_points
        mark_prices = mark_prices[::step]
        oracle_prices = oracle_prices[::step]
        n = len(mark_prices)

    times = np.arange(n) * block_time_ms * (len(mark_prices) / n if n > 0 else 1) / 1000

    plt.clear_figure()
    plt.plot(times.tolist(), oracle_prices.tolist(), label="Oracle", color="blue")
    plt.plot(times.tolist(), mark_prices.tolist(), label="Mark", color="red")
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Price ($)")
    plt.theme("dark")
    plt.show()


def chart_liquidations(result: LiquidationResult, block_time_ms: float = 5) -> None:
    """Plot liquidation events as a bar chart by block."""
    if not result.events:
        print("No liquidation events to chart.")
        return

    # Aggregate liquidations by block ranges
    blocks = [e.block for e in result.events]
    if not blocks:
        return

    min_block = min(blocks)
    max_block = max(blocks)
    n_bins = min(100, max_block - min_block + 1)
    bin_edges = np.linspace(min_block, max_block + 1, n_bins + 1)

    counts, _ = np.histogram(blocks, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    times = bin_centers * block_time_ms / 1000

    plt.clear_figure()
    plt.bar(times.tolist(), counts.tolist(), width=0.8, color="red")
    plt.title("Liquidations Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Liquidation Count")
    plt.theme("dark")
    plt.show()


def chart_insurance(insurance: InsuranceFund, block_time_ms: float = 5) -> None:
    """Plot insurance fund balance over time."""
    if not insurance.history:
        print("No insurance fund history to chart.")
        return

    n = len(insurance.history)
    max_points = 200
    if n > max_points:
        step = n // max_points
        history = insurance.history[::step]
    else:
        history = insurance.history

    times = [i * block_time_ms * (n / len(history)) / 1000 for i in range(len(history))]

    plt.clear_figure()
    plt.plot(times, history, label="Balance", color="green")
    plt.title("Insurance Fund Balance")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Balance ($)")
    plt.theme("dark")
    plt.show()


def chart_heatmap(
    x_values: list[float],
    y_values: list[float],
    z_matrix: list[list[float]],
    title: str = "Heatmap",
    xlabel: str = "X",
    ylabel: str = "Y",
) -> None:
    """Plot a heatmap using plotext matrix plot."""
    plt.clear_figure()
    plt.matrix_plot(z_matrix)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.theme("dark")
    plt.show()


def chart_position_distribution(positions: list, field: str = "leverage") -> None:
    """Plot position distribution histogram."""
    if not positions:
        print("No positions to chart.")
        return

    if field == "leverage":
        values = [p.leverage for p in positions]
        title = "Leverage Distribution"
        xlabel = "Leverage (x)"
    elif field == "size":
        values = [p.size_usd for p in positions]
        title = "Position Size Distribution"
        xlabel = "Size (USD)"
    else:
        values = [p.leverage for p in positions]
        title = "Distribution"
        xlabel = field

    plt.clear_figure()
    plt.hist(values, bins=20, color="cyan")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.theme("dark")
    plt.show()


def chart_scatter_fill_prices(result: LiquidationResult) -> None:
    """Scatter plot of fill price vs bankruptcy price for each liquidation."""
    if not result.events:
        print("No liquidation events to chart.")
        return

    fill_prices = [e.fill_price for e in result.events]
    bankruptcy_prices = [e.bankruptcy_price for e in result.events]

    plt.clear_figure()
    plt.scatter(bankruptcy_prices, fill_prices, label="Liquidations", color="red", marker="dot")

    # Add reference line (y=x)
    min_p = min(min(fill_prices), min(bankruptcy_prices))
    max_p = max(max(fill_prices), max(bankruptcy_prices))
    plt.plot([min_p, max_p], [min_p, max_p], label="Break-even", color="green")

    plt.title("Fill Price vs Bankruptcy Price")
    plt.xlabel("Bankruptcy Price ($)")
    plt.ylabel("Fill Price ($)")
    plt.theme("dark")
    plt.show()
