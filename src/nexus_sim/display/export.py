"""Matplotlib PNG/PDF export for publication-quality charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from nexus_sim.engine.liquidation import LiquidationResult
from nexus_sim.engine.insurance import InsuranceFund

# Use non-interactive backend
matplotlib.use("Agg")

EXPORT_DIR = Path("exports")
COLORS = {
    "primary": "#2196F3",
    "danger": "#F44336",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "secondary": "#9C27B0",
    "bg": "#1a1a2e",
    "text": "#e0e0e0",
    "grid": "#333355",
}


def _setup_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["grid"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "font.size": 11,
    })


def ensure_export_dir() -> Path:
    EXPORT_DIR.mkdir(exist_ok=True)
    return EXPORT_DIR


def export_simulation_results(
    mark_prices: np.ndarray,
    oracle_prices: np.ndarray,
    result: LiquidationResult,
    insurance: InsuranceFund,
    scenario_name: str,
    block_time_ms: float,
    mm_ratio: float,
    book_depth: float,
    filename: str | None = None,
) -> str:
    """Export a comprehensive simulation results chart."""
    _setup_style()
    export_dir = ensure_export_dir()

    if filename is None:
        filename = f"{scenario_name}_{block_time_ms:.0f}ms_{mm_ratio * 100:.1f}mm_{book_depth:.0f}.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Nexus Risk Simulation — {scenario_name}", fontsize=16, color=COLORS["text"])

    n = len(mark_prices)
    times = np.arange(n) * block_time_ms / 1000

    # 1. Mark price vs oracle
    ax = axes[0, 0]
    _downsample_plot(ax, times, oracle_prices, label="Oracle", color=COLORS["primary"], alpha=0.7)
    _downsample_plot(ax, times, mark_prices, label="Mark", color=COLORS["danger"])
    ax.set_title("Mark Price vs Oracle Price")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Price ($)")
    ax.legend()

    # 2. Liquidation events
    ax = axes[0, 1]
    if result.events:
        liq_blocks = [e.block for e in result.events]
        liq_times = [b * block_time_ms / 1000 for b in liq_blocks]
        liq_sizes = [e.size_usd / 1000 for e in result.events]
        colors = [COLORS["danger"] if e.residual < 0 else COLORS["success"] for e in result.events]
        ax.scatter(liq_times, liq_sizes, c=colors, s=10, alpha=0.7)
    ax.set_title("Liquidation Events")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Size ($K)")

    # 3. Insurance fund balance
    ax = axes[1, 0]
    if insurance.history:
        ins_n = len(insurance.history)
        ins_times = np.arange(ins_n) * block_time_ms / 1000
        _downsample_plot(ax, ins_times, np.array(insurance.history), label="Balance", color=COLORS["success"])
        ax.axhline(y=0, color=COLORS["danger"], linestyle="--", alpha=0.5, label="Zero")
    ax.set_title("Insurance Fund Balance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Balance ($)")
    ax.legend()

    # 4. Summary stats
    ax = axes[1, 1]
    ax.axis("off")
    total = result.total_positions
    liqd = result.total_liquidated
    pct = liqd / total * 100 if total > 0 else 0
    initial_bal = insurance.config.initial_balance
    final_bal = insurance.balance
    delta = final_bal - initial_bal

    stats = [
        f"Positions Liquidated: {liqd} / {total} ({pct:.0f}%)",
        f"Max Cascade Depth: {result.max_cascade_depth}",
        f"Insurance: ${initial_bal:,.0f} → ${final_bal:,.0f} ({'+' if delta >= 0 else ''}{delta:,.0f})",
        f"Max Drawdown: -${insurance.max_drawdown:,.0f}",
        f"Worst Fill Ratio: {result.worst_fill_ratio * 100:.1f}%",
        f"Total Residual: ${result.total_residual:,.0f}",
        f"Total Deficit: ${result.total_deficit:,.0f}",
    ]
    for i, s in enumerate(stats):
        ax.text(0.05, 0.9 - i * 0.12, s, fontsize=12, transform=ax.transAxes,
                color=COLORS["text"], family="monospace")

    plt.tight_layout()
    path = export_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def export_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_matrix: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str = "",
    filename: str = "heatmap.png",
) -> str:
    """Export a heatmap as PNG."""
    _setup_style()
    export_dir = ensure_export_dir()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(x_values, y_values, z_matrix, cmap="RdYlGn_r", shading="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax)
    if zlabel:
        cbar.set_label(zlabel)

    plt.tight_layout()
    path = export_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def export_comparison_table(
    data: dict[str, dict],
    filename: str = "comparison.png",
) -> str:
    """Export a comparison table as PNG."""
    _setup_style()
    export_dir = ensure_export_dir()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    exchanges = list(data.keys())
    metrics = list(next(iter(data.values())).keys()) if data else []

    cell_text = []
    for exchange in exchanges:
        row = [str(data[exchange].get(m, "N/A")) for m in metrics]
        cell_text.append(row)

    if cell_text and metrics:
        table = ax.table(
            cellText=cell_text,
            rowLabels=exchanges,
            colLabels=metrics,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        for key, cell in table.get_celld().items():
            cell.set_edgecolor(COLORS["grid"])
            cell.set_facecolor(COLORS["bg"])
            cell.set_text_props(color=COLORS["text"])

    ax.set_title("Exchange Parameter Comparison", fontsize=14, color=COLORS["text"], pad=20)
    plt.tight_layout()
    path = export_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _downsample_plot(ax, x: np.ndarray, y: np.ndarray, max_points: int = 2000, **kwargs) -> None:
    """Plot with downsampling for large datasets."""
    if len(x) > max_points:
        step = len(x) // max_points
        x = x[::step]
        y = y[::step]
    ax.plot(x, y, **kwargs)
