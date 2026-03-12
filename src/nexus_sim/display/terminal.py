"""Rich-based terminal output for status panels and results tables."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

from nexus_sim.engine.liquidation import LiquidationResult
from nexus_sim.engine.insurance import InsuranceFund
from nexus_sim.scenarios.presets import SimulationState

console = Console()


def print_banner() -> None:
    console.print(Panel(
        "[bold cyan]Nexus Exchange Risk Simulator v0.1[/bold cyan]\n"
        "Type 'help' for commands, 'preset <name>' to load a configuration, 'run' to simulate.",
        border_style="cyan",
    ))


def print_status(state: SimulationState) -> None:
    """Print the full status panel with current configuration."""
    exchange_table = Table(show_header=False, box=None, padding=(0, 1))
    exchange_table.add_column(style="bold")
    exchange_table.add_column()
    exchange_table.add_row("Block time:", f"{state.exchange.block_time_ms}ms")
    exchange_table.add_row("MM ratio:", f"{state.exchange.mm_ratio * 100:.2f}%")
    exchange_table.add_row("Max leverage:", f"{state.exchange.max_leverage}x")
    exchange_table.add_row("Mark price:", f"Phase {state.exchange.phase}")
    exchange_table.add_row("Change bound:", f"{state.exchange.max_price_change_pct * 100:.1f}% / block")

    market_table = Table(show_header=False, box=None, padding=(0, 1))
    market_table.add_column(style="bold")
    market_table.add_column()
    market_table.add_row("Book depth:", f"${state.market.book_depth_usd:,.0f} / side")
    market_table.add_row("Spread:", f"{state.market.spread_pct * 100:.2f}%")
    market_table.add_row("Markets:", "BTC/USDX")
    market_table.add_row("Oracle:", "Pyth Pro")

    user_table = Table(show_header=False, box=None, padding=(0, 1))
    user_table.add_column(style="bold")
    user_table.add_column()
    user_table.add_row("MAU:", f"{state.users.mau:,}")
    user_table.add_row("Positions:", f"{state.users.position_count:,}")
    total_oi = state.total_oi()
    user_table.add_row("Total OI:", f"${total_oi:,.0f}")
    avg_lev = _estimate_avg_leverage(state)
    user_table.add_row("Avg leverage:", f"{avg_lev:.1f}x")

    ins_table = Table(show_header=False, box=None, padding=(0, 1))
    ins_table.add_column(style="bold")
    ins_table.add_column()
    ins_table.add_row("Balance:", f"${state.insurance.initial_balance:,.0f}")
    ins_table.add_row("Target:", f"{state.insurance.target_ratio * 100:.0f}% of OI")
    coverage = state.insurance.initial_balance / total_oi * 100 if total_oi > 0 else 0
    ins_table.add_row("Coverage:", f"{coverage:.2f}%")
    ins_table.add_row("Treasury cap:", f"${state.insurance.treasury_backstop:,.0f}")

    left_col = Columns([
        Panel(exchange_table, title="Exchange Config", border_style="blue"),
        Panel(user_table, title="Users & Positions", border_style="green"),
    ])
    right_col = Columns([
        Panel(market_table, title="Market State", border_style="yellow"),
        Panel(ins_table, title="Insurance Fund", border_style="red"),
    ])

    console.print(left_col)
    console.print(right_col)


def print_results(
    result: LiquidationResult,
    insurance: InsuranceFund,
    scenario_name: str = "",
    export_path: str | None = None,
) -> None:
    """Print simulation results in a rich panel."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()

    pct = result.total_liquidated / result.total_positions * 100 if result.total_positions > 0 else 0
    table.add_row(
        "Positions liquidated:",
        f"{result.total_liquidated} / {result.total_positions} ({pct:.0f}%)",
    )
    table.add_row("Cascade depth:", f"{result.max_cascade_depth} (max chain of sequential liquidations)")

    initial = insurance.config.initial_balance
    final = insurance.balance
    delta = final - initial
    sign = "+" if delta >= 0 else ""
    color = "green" if delta >= 0 else "red"
    table.add_row(
        "Insurance fund:",
        f"${initial:,.0f} → ${final:,.0f} ([{color}]{sign}${delta:,.0f}[/{color}])",
    )
    table.add_row("Fund was profitable:", "Yes" if delta >= 0 else "[red]No[/red]")
    table.add_row("Max drawdown:", f"-${insurance.max_drawdown:,.0f} (block {insurance.max_drawdown_block:,})")
    table.add_row("Unfilled liquidations:", str(result.unfilled))

    fill_pct = result.worst_fill_ratio * 100
    fill_color = "green" if fill_pct > 97 else "yellow" if fill_pct > 95 else "red"
    slippage = (1 - result.worst_fill_ratio) * 100
    table.add_row(
        "Worst fill vs bankruptcy:",
        f"[{fill_color}]{fill_pct:.1f}% ({slippage:.1f}% slippage)[/{fill_color}]",
    )

    console.print(Panel(table, title=f"Results — {scenario_name}", border_style="cyan"))

    if export_path:
        console.print(f"\nExported: [bold]{export_path}[/bold]")


def print_warning(param_name: str, value: str, message: str) -> None:
    """Print a parameter change warning."""
    console.print(f"[green]✓[/green] {param_name} → {value}  [yellow][WARNING: {message}][/yellow]")


def print_success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    console.print(f"[red]✗[/red] {message}")


def print_help() -> None:
    """Print command reference."""
    sections = [
        ("Configuration", [
            ("preset <name>", "Load named preset (conservative, competitive, stress_test)"),
            ("config load <path>", "Load YAML config file"),
            ("config save <path>", "Save current config to YAML"),
            ("status", "Show current configuration summary"),
            ("reset", "Reset to defaults"),
        ]),
        ("Parameters", [
            ("set block_time <ms>", "Set block time (5-2000)"),
            ("set mm_ratio <pct>", "Set maintenance margin ratio (1-20%)"),
            ("set leverage_max <n>", "Set max leverage (2-100)"),
            ("set book_depth <usd>", "Set book depth per side in USD"),
            ("set spread <pct>", "Set bid-ask spread"),
            ("set phase <1|2>", "Mark price phase"),
            ("set change_bound <pct>", "MAX_PRICE_CHANGE_PCT"),
            ("set insurance_balance <usd>", "Insurance fund starting balance"),
            ("set treasury_cap <usd>", "Protocol treasury backstop limit"),
            ("set oracle_latency <ms>", "Pyth delivery latency"),
            ("set oracle_staleness <ms>", "MAX_ORACLE_AGE_MS"),
            ("set opt.dual_mark on|off", "Dual mark price for liquidation"),
            ("set opt.same_block on|off", "Same-block liquidation pipeline"),
            ("set opt.cascade_dampen on|off", "Cascade dampening"),
        ]),
        ("Users / Positions", [
            ("set mau <count>", "Set MAU, auto-generate positions"),
            ("set positions <count>", "Set position count directly"),
            ("set oi <usd>", "Set total OI directly"),
            ("set leverage_dist '<spec>'", "e.g. '5x:20%,10x:40%,20x:30%,50x:10%'"),
            ("set side_bias <ratio>", "Long ratio, e.g. 0.6 = 60% long"),
            ("regen", "Regenerate position set with current params"),
            ("positions show", "Show position distribution summary"),
            ("positions list", "List all positions (paginated)"),
        ]),
        ("Simulation", [
            ("run crash <preset>", "Run crash scenario (march-2020, instant, linear, etc.)"),
            ("run normal <duration>", "Run normal market simulation"),
            ("run stress", "Run all historical crashes sequentially"),
            ("run sweep <param> <min> <max> <steps>", "Parameter sweep"),
            ("run compare <exchange>", "Compare with competitor (hyperliquid, dydx, all)"),
        ]),
        ("Analysis", [
            ("analyze sensitivity", "Generate full sensitivity heatmaps"),
            ("analyze insurance", "Insurance fund sizing analysis"),
            ("analyze cascade", "Cascading liquidation depth analysis"),
            ("analyze buffer", "Buffer zone analysis by leverage/MM"),
            ("analyze worst_case", "Find parameter combo that depletes fund"),
        ]),
        ("Charts & Export", [
            ("chart mark_price", "Show mark price behavior (terminal)"),
            ("chart liquidations", "Show liquidation cascade timeline"),
            ("chart insurance", "Show insurance fund balance over time"),
            ("chart heatmap <type>", "Show heatmap: mm_leverage, blocktime_leverage, crash_mm"),
            ("chart compare", "Side-by-side competitor comparison"),
            ("export <filename>", "Export last chart as PNG"),
            ("export report <filename>", "Export full PDF report"),
        ]),
    ]

    for section_name, commands in sections:
        table = Table(title=section_name, show_header=True, header_style="bold")
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        console.print(table)
        console.print()


def print_positions_summary(positions: list) -> None:
    """Print a summary of the current position set."""
    if not positions:
        print_error("No positions generated. Use 'set mau <n>' or 'set positions <n>' first.")
        return

    total_oi = sum(p.size_usd for p in positions)
    long_count = sum(1 for p in positions if p.side == "long")
    short_count = len(positions) - long_count
    long_oi = sum(p.size_usd for p in positions if p.side == "long")
    avg_leverage = sum(p.leverage for p in positions) / len(positions)
    max_leverage = max(p.leverage for p in positions)
    avg_size = total_oi / len(positions)

    table = Table(title="Position Summary", show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Total positions:", f"{len(positions):,}")
    table.add_row("Total OI:", f"${total_oi:,.0f}")
    table.add_row("Long/Short:", f"{long_count} / {short_count} ({long_count / len(positions) * 100:.0f}% / {short_count / len(positions) * 100:.0f}%)")
    table.add_row("Long OI:", f"${long_oi:,.0f} ({long_oi / total_oi * 100:.0f}%)")
    table.add_row("Avg size:", f"${avg_size:,.0f}")
    table.add_row("Avg leverage:", f"{avg_leverage:.1f}x")
    table.add_row("Max leverage:", f"{max_leverage:.0f}x")

    # Leverage distribution
    lev_buckets = {}
    for p in positions:
        bucket = int(p.leverage)
        lev_buckets[bucket] = lev_buckets.get(bucket, 0) + 1

    console.print(Panel(table, border_style="green"))

    lev_table = Table(title="Leverage Distribution")
    lev_table.add_column("Leverage", style="cyan")
    lev_table.add_column("Count")
    lev_table.add_column("Pct")
    for lev in sorted(lev_buckets.keys()):
        cnt = lev_buckets[lev]
        lev_table.add_row(f"{lev}x", str(cnt), f"{cnt / len(positions) * 100:.1f}%")
    console.print(lev_table)


def _estimate_avg_leverage(state: SimulationState) -> float:
    """Estimate average leverage from the leverage distribution."""
    dist = state.users.leverage_dist or {10: 1.0}
    total_w = sum(dist.values())
    return sum(lev * w / total_w for lev, w in dist.items())
