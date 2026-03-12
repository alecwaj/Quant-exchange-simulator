"""Click CLI entry point with interactive REPL mode."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from nexus_sim.engine.insurance import InsuranceFund, InsuranceFundConfig
from nexus_sim.engine.liquidation import LiquidationConfig, run_simulation
from nexus_sim.engine.mark_price import MarkPriceConfig, run_mark_price_vectorized
from nexus_sim.engine.orderbook import OrderBook, OrderBookConfig
from nexus_sim.engine.positions import (
    Position,
    PositionConfig,
    generate_positions,
    generate_positions_from_mau,
    parse_leverage_dist,
)
from nexus_sim.scenarios.historical import load_historical_crash, list_historical_crashes
from nexus_sim.scenarios.presets import (
    SimulationState,
    load_preset,
    load_yaml_config,
    save_yaml_config,
    list_presets,
)
from nexus_sim.scenarios.synthetic import (
    CrashScenario,
    generate_instant_crash,
    generate_linear_crash,
    generate_vshape_crash,
    generate_cascade_crash,
    generate_volatility_scenario,
)
from nexus_sim.display.terminal import (
    console,
    print_banner,
    print_status,
    print_results,
    print_warning,
    print_success,
    print_error,
    print_help,
    print_positions_summary,
)
from nexus_sim.display.charts import (
    chart_mark_price,
    chart_liquidations,
    chart_insurance,
    chart_heatmap,
    chart_position_distribution,
    chart_scatter_fill_prices,
)
from nexus_sim.display.export import (
    export_simulation_results,
    export_heatmap,
    export_comparison_table,
)
from nexus_sim.analysis.comparison import run_comparison, list_competitors
from nexus_sim.analysis.sensitivity import (
    run_parameter_sweep,
    run_heatmap,
    buffer_zone_analysis,
    insurance_sizing_analysis,
    find_worst_case,
)
from nexus_sim.analysis.cascade import analyze_cascade_risk, compare_dampening


class SessionState:
    """Mutable session state for the REPL."""

    def __init__(self) -> None:
        self.sim_state = load_preset("default")
        self.positions: list[Position] = []
        self.last_mark_prices: np.ndarray | None = None
        self.last_oracle_prices: np.ndarray | None = None
        self.last_result = None
        self.last_insurance: InsuranceFund | None = None
        self.last_scenario_name: str = ""

    def regenerate_positions(self) -> None:
        """Regenerate positions from current state."""
        if self.sim_state.users.mau > 0:
            self.positions = generate_positions_from_mau(
                mau=self.sim_state.users.mau,
                entry_price=self.sim_state.entry_price,
                mm_ratio=self.sim_state.exchange.mm_ratio,
                leverage_dist=self.sim_state.users.leverage_dist,
                side_bias=self.sim_state.users.side_bias,
            )
            self.sim_state.users.position_count = len(self.positions)
        else:
            config = PositionConfig(
                count=self.sim_state.users.position_count,
                avg_size_usd=self.sim_state.users.avg_size_usd,
                leverage_dist=self.sim_state.users.leverage_dist or {5: 0.2, 10: 0.4, 20: 0.3, 50: 0.1},
                side_bias=self.sim_state.users.side_bias,
            )
            self.positions = generate_positions(
                config,
                self.sim_state.entry_price,
                self.sim_state.exchange.mm_ratio,
            )


session = SessionState()


def _check_parameter_warnings(param: str, state: SimulationState) -> None:
    """Check for dangerous parameter combinations and print warnings."""
    buffer = state.buffer_zone_pct()
    drift = state.liq_delay_drift_pct()
    max_safe = state.max_safe_leverage()

    if buffer < 1.0 and state.exchange.max_leverage > 5:
        print_warning(
            param, "",
            f"Buffer zone is only {buffer:.1f}%. At {state.exchange.max_leverage}x leverage, "
            f"positions have very little room before liquidation. Risk of cascading liquidations is high.",
        )
    elif buffer < 3.0:
        print_warning(
            param, "",
            f"Buffer zone at {state.exchange.max_leverage}x/{state.exchange.mm_ratio * 100:.1f}% MM is "
            f"{buffer:.1f}%. At {state.exchange.block_time_ms}ms blocks, liq delay drift is {drift:.1f}%. "
            f"{'Marginal.' if drift > buffer * 0.3 else 'Acceptable.'}",
        )

    if state.exchange.max_leverage > max_safe:
        print_warning(
            param, "",
            f"Max leverage ({state.exchange.max_leverage}x) exceeds max safe leverage "
            f"({max_safe:.0f}x) for {state.exchange.mm_ratio * 100:.2f}% MM. "
            f"Positions at max leverage will be immediately liquidatable.",
        )


def _build_crash_scenario(args: list[str], block_time_ms: float) -> CrashScenario | None:
    """Parse crash arguments and build a scenario."""
    if not args:
        print_error("Usage: run crash <preset> or run crash --type <type> [options]")
        return None

    # Check for named historical preset
    historical = list_historical_crashes()
    if args[0] in historical:
        return load_historical_crash(args[0], block_time_ms)

    # Parse custom crash
    crash_type = None
    magnitude = 15.0
    duration = 60.0
    recovery = 120.0
    initial = 5.0
    steps = 8
    interval = 30.0
    annualized = 120.0
    start_price = 60_000.0

    i = 0
    while i < len(args):
        if args[i] == "--type" and i + 1 < len(args):
            crash_type = args[i + 1]
            i += 2
        elif args[i] == "--magnitude" and i + 1 < len(args):
            magnitude = float(args[i + 1])
            i += 2
        elif args[i] == "--duration" and i + 1 < len(args):
            duration = float(args[i + 1])
            i += 2
        elif args[i] == "--recovery" and i + 1 < len(args):
            recovery = float(args[i + 1])
            i += 2
        elif args[i] == "--initial" and i + 1 < len(args):
            initial = float(args[i + 1])
            i += 2
        elif args[i] == "--steps" and i + 1 < len(args):
            steps = int(args[i + 1])
            i += 2
        elif args[i] == "--interval" and i + 1 < len(args):
            interval = float(args[i + 1])
            i += 2
        elif args[i] == "--annualized" and i + 1 < len(args):
            annualized = float(args[i + 1])
            i += 2
        else:
            # Maybe it's a type without --type prefix
            if crash_type is None:
                crash_type = args[i]
            i += 1

    if crash_type is None:
        print_error("Must specify crash type: instant, linear, vshape, cascade, volatility")
        return None

    if crash_type == "instant":
        return generate_instant_crash(start_price, magnitude, block_time_ms)
    elif crash_type == "linear":
        return generate_linear_crash(start_price, magnitude, duration, block_time_ms)
    elif crash_type == "vshape":
        return generate_vshape_crash(start_price, magnitude, recovery, block_time_ms)
    elif crash_type == "cascade":
        return generate_cascade_crash(start_price, initial, steps, interval, block_time_ms)
    elif crash_type == "volatility":
        return generate_volatility_scenario(start_price, annualized, duration, block_time_ms)
    else:
        print_error(f"Unknown crash type: {crash_type}")
        return None


def _run_crash(args: list[str]) -> None:
    """Execute a crash simulation."""
    state = session.sim_state
    scenario = _build_crash_scenario(args, state.exchange.block_time_ms)
    if scenario is None:
        return

    console.print(f"\nRunning {scenario.description}...")

    # Ensure positions exist
    if not session.positions:
        session.regenerate_positions()

    # Reset positions for new run
    for p in session.positions:
        p.is_liquidated = False
        p.compute_prices(state.exchange.mm_ratio)

    oracle_prices = scenario.prices
    n_blocks = len(oracle_prices)

    # Run mark price engine
    mp_config = MarkPriceConfig(
        block_time_ms=state.exchange.block_time_ms,
        max_price_change_pct=state.exchange.max_price_change_pct,
        phase=state.exchange.phase,
        max_oracle_age_ms=state.exchange.max_oracle_age_ms,
        max_confidence_ratio=state.exchange.max_confidence_ratio,
        ema_decay_blocks=state.exchange.ema_decay_blocks,
        max_basis_pct=state.exchange.max_basis_pct,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} blocks"),
        console=console,
    ) as progress:
        task = progress.add_task("Simulating...", total=n_blocks)

        mark_prices, actions = run_mark_price_vectorized(mp_config, oracle_prices)
        progress.update(task, completed=n_blocks // 2)

        # Setup orderbook
        ob_config = OrderBookConfig(
            depth_per_side_usd=state.market.book_depth_usd,
            spread_pct=state.market.spread_pct,
            shape=state.market.book_shape,
            num_levels=state.market.num_levels,
        )
        orderbook = OrderBook(config=ob_config)
        orderbook.generate(state.entry_price)

        # Setup insurance
        ins_config = InsuranceFundConfig(
            initial_balance=state.insurance.initial_balance,
            treasury_backstop=state.insurance.treasury_backstop,
            target_ratio=state.insurance.target_ratio,
        )
        insurance = InsuranceFund(config=ins_config)

        # Setup liquidation
        liq_config = LiquidationConfig(
            mm_ratio=state.exchange.mm_ratio,
            max_leverage=state.exchange.max_leverage,
            max_liquidations_per_block=state.exchange.max_liquidations_per_block,
            use_dual_mark_price=state.exchange.use_dual_mark_price,
            same_block_execution=state.exchange.same_block_execution,
            cascade_dampening=state.exchange.cascade_dampening,
        )

        # Compute book refresh rate in blocks
        refresh_blocks = max(1, int(state.market.book_refresh_rate_ms / state.exchange.block_time_ms))

        # Run simulation
        result = run_simulation(
            mark_prices, session.positions, orderbook, insurance, liq_config, refresh_blocks
        )
        progress.update(task, completed=n_blocks)

    # Store results
    session.last_mark_prices = mark_prices
    session.last_oracle_prices = oracle_prices
    session.last_result = result
    session.last_insurance = insurance
    session.last_scenario_name = scenario.name

    # Display results
    print_results(result, insurance, scenario.name)

    # Auto-export
    export_path = export_simulation_results(
        mark_prices=mark_prices,
        oracle_prices=oracle_prices,
        result=result,
        insurance=insurance,
        scenario_name=scenario.name,
        block_time_ms=state.exchange.block_time_ms,
        mm_ratio=state.exchange.mm_ratio,
        book_depth=state.market.book_depth_usd,
    )
    console.print(f"\nExported: [bold]{export_path}[/bold]")


def _handle_set(args: list[str]) -> None:
    """Handle 'set' commands."""
    if len(args) < 2:
        print_error("Usage: set <parameter> <value>")
        return

    param = args[0]
    value = args[1]
    state = session.sim_state

    try:
        if param == "block_time":
            state.exchange.block_time_ms = float(value)
            print_success(f"Block time → {value}ms")
        elif param == "mm_ratio":
            state.exchange.mm_ratio = float(value) / 100 if float(value) > 1 else float(value)
            print_success(f"MM ratio → {state.exchange.mm_ratio * 100:.2f}%")
        elif param == "leverage_max":
            state.exchange.max_leverage = int(value)
            print_success(f"Max leverage → {value}x")
        elif param == "book_depth":
            state.market.book_depth_usd = float(value)
            print_success(f"Book depth → ${float(value):,.0f}")
        elif param == "spread":
            state.market.spread_pct = float(value) / 100 if float(value) > 0.1 else float(value)
            print_success(f"Spread → {state.market.spread_pct * 100:.3f}%")
        elif param == "phase":
            state.exchange.phase = int(value)
            print_success(f"Mark price phase → {value}")
        elif param == "change_bound":
            state.exchange.max_price_change_pct = float(value) / 100 if float(value) > 1 else float(value)
            print_success(f"Change bound → {state.exchange.max_price_change_pct * 100:.1f}%")
        elif param == "insurance_balance":
            state.insurance.initial_balance = float(value)
            print_success(f"Insurance balance → ${float(value):,.0f}")
        elif param == "treasury_cap":
            state.insurance.treasury_backstop = float(value)
            print_success(f"Treasury cap → ${float(value):,.0f}")
        elif param == "oracle_latency":
            console.print(f"[dim]Oracle latency noted: {value}ms (affects simulation realism)[/dim]")
        elif param == "oracle_staleness":
            state.exchange.max_oracle_age_ms = float(value)
            print_success(f"Oracle staleness → {value}ms")
        elif param == "mau":
            state.users.mau = int(value)
            session.regenerate_positions()
            state.users.position_count = len(session.positions)
            total_oi = sum(p.size_usd for p in session.positions)
            print_success(f"MAU → {value}, generated {len(session.positions)} positions, ${total_oi:,.0f} OI")
        elif param == "positions":
            state.users.position_count = int(value)
            session.regenerate_positions()
            print_success(f"Positions → {value}")
        elif param == "oi":
            target_oi = float(value)
            avg_size = target_oi / state.users.position_count if state.users.position_count > 0 else 50_000
            state.users.avg_size_usd = avg_size
            session.regenerate_positions()
            print_success(f"Target OI → ${target_oi:,.0f}")
        elif param == "leverage_dist":
            dist = parse_leverage_dist(value)
            state.users.leverage_dist = dist
            session.regenerate_positions()
            print_success(f"Leverage distribution updated")
        elif param == "side_bias":
            state.users.side_bias = float(value)
            session.regenerate_positions()
            print_success(f"Side bias → {float(value) * 100:.0f}% long")
        elif param.startswith("opt."):
            opt = param[4:]
            on = value.lower() in ("on", "true", "1", "yes")
            if opt == "dual_mark":
                state.exchange.use_dual_mark_price = on
            elif opt == "same_block":
                state.exchange.same_block_execution = on
            elif opt == "cascade_dampen":
                state.exchange.cascade_dampening = on
            elif opt == "backstop":
                console.print(f"[dim]Insurance backstop orders: {'enabled' if on else 'disabled'}[/dim]")
            else:
                print_error(f"Unknown optimization: {opt}")
                return
            print_success(f"{param} → {'on' if on else 'off'}")
        else:
            print_error(f"Unknown parameter: {param}")
            return

        _check_parameter_warnings(param, state)
    except ValueError as e:
        print_error(f"Invalid value: {e}")


def _handle_run(args: list[str]) -> None:
    """Handle 'run' commands."""
    if not args:
        print_error("Usage: run crash|normal|stress|sweep|compare <args>")
        return

    cmd = args[0]
    if cmd == "crash":
        _run_crash(args[1:])
    elif cmd == "normal":
        duration = float(args[1]) if len(args) > 1 else 300
        scenario = generate_volatility_scenario(
            session.sim_state.entry_price,
            annualized_vol=80,
            duration_sec=duration,
            block_time_ms=session.sim_state.exchange.block_time_ms,
        )
        _run_crash(["volatility", "--annualized", "80", "--duration", str(duration)])
    elif cmd == "stress":
        console.print("[bold]Running all historical crashes...[/bold]\n")
        for name in list_historical_crashes():
            _run_crash([name])
            console.print()
    elif cmd == "sweep":
        if len(args) < 5:
            print_error("Usage: run sweep <param> <min> <max> <steps>")
            return
        param = args[1]
        min_val = float(args[2])
        max_val = float(args[3])
        steps = int(args[4])

        # Need oracle prices
        if session.last_oracle_prices is None:
            scenario = generate_instant_crash(session.sim_state.entry_price, 20, session.sim_state.exchange.block_time_ms)
            oracle_prices = scenario.prices
        else:
            oracle_prices = session.last_oracle_prices

        console.print(f"Sweeping {param} from {min_val} to {max_val} in {steps} steps...")
        result = run_parameter_sweep(
            session.sim_state, oracle_prices, param, min_val, max_val, steps
        )
        # Display as chart
        import plotext as plt
        plt.clear_figure()
        plt.plot(result.param_values, result.metric_values, color="cyan")
        plt.title(f"{param} sweep → {result.metric_name}")
        plt.xlabel(param)
        plt.ylabel(result.metric_name)
        plt.theme("dark")
        plt.show()
    elif cmd == "compare":
        target = args[1] if len(args) > 1 else "all"

        # Need oracle prices
        if session.last_oracle_prices is None:
            scenario = generate_instant_crash(session.sim_state.entry_price, 20, session.sim_state.exchange.block_time_ms)
            oracle_prices = scenario.prices
        else:
            oracle_prices = session.last_oracle_prices

        if target == "all":
            competitors = None
        else:
            competitors = [target]

        console.print(f"[bold]Running comparison...[/bold]")
        results = run_comparison(oracle_prices, session.sim_state, competitors, session.sim_state.entry_price)

        # Display as table
        from rich.table import Table
        table = Table(title="Exchange Comparison")
        table.add_column("Exchange", style="cyan bold")
        if results:
            metrics = list(next(iter(results.values())).keys())
            for m in metrics:
                table.add_column(m)
            for exchange, data in results.items():
                row = [str(data.get(m, "")) for m in metrics]
                table.add_row(exchange, *row)
        console.print(table)

        # Export
        path = export_comparison_table(results)
        console.print(f"\nExported: [bold]{path}[/bold]")
    else:
        print_error(f"Unknown run command: {cmd}")


def _handle_analyze(args: list[str]) -> None:
    """Handle 'analyze' commands."""
    if not args:
        print_error("Usage: analyze sensitivity|insurance|cascade|buffer|worst_case")
        return

    # Ensure we have oracle prices
    if session.last_oracle_prices is None:
        scenario = generate_instant_crash(session.sim_state.entry_price, 20, session.sim_state.exchange.block_time_ms)
        oracle_prices = scenario.prices
    else:
        oracle_prices = session.last_oracle_prices

    cmd = args[0]
    if cmd == "sensitivity":
        console.print("[bold]Generating sensitivity heatmaps...[/bold]")
        result = run_heatmap(
            session.sim_state, oracle_prices,
            x_param="mm_ratio", y_param="max_leverage",
            x_range=(0.02, 0.10, 8), y_range=(5, 50, 8),
        )
        chart_heatmap(
            result.x_values.tolist(), result.y_values.tolist(),
            result.z_matrix.tolist(),
            title="Liquidation % by MM Ratio × Leverage",
            xlabel="MM Ratio", ylabel="Max Leverage",
        )
        path = export_heatmap(
            result.x_values, result.y_values, result.z_matrix,
            "Liquidation % by MM Ratio × Leverage",
            "MM Ratio", "Max Leverage", "Liquidation %",
            "sensitivity_mm_leverage.png",
        )
        console.print(f"Exported: [bold]{path}[/bold]")

    elif cmd == "insurance":
        console.print("[bold]Insurance fund sizing analysis...[/bold]")
        result = insurance_sizing_analysis(session.sim_state, oracle_prices)
        import plotext as plt
        plt.clear_figure()
        plt.plot(result.param_values, result.metric_values, color="green")
        plt.title("Insurance Balance vs Fund Delta %")
        plt.xlabel("Initial Balance ($)")
        plt.ylabel("Delta %")
        plt.theme("dark")
        plt.show()

    elif cmd == "cascade":
        console.print("[bold]Cascading liquidation analysis...[/bold]")
        results = compare_dampening(session.sim_state, oracle_prices)

        from rich.table import Table
        table = Table(title="Cascade Analysis")
        table.add_column("Metric", style="bold")
        table.add_column("Without Dampening")
        table.add_column("With Dampening")

        wo = results["without_dampening"]
        wi = results["with_dampening"]
        table.add_row("Max cascade depth", str(wo.max_cascade_depth), str(wi.max_cascade_depth))
        table.add_row("Avg cascade depth", f"{wo.avg_cascade_depth:.1f}", f"{wi.avg_cascade_depth:.1f}")
        table.add_row("Cascade volume", f"${wo.total_cascade_volume:,.0f}", f"${wi.total_cascade_volume:,.0f}")
        table.add_row("Peak block liqs", str(wo.peak_block_liquidations), str(wi.peak_block_liquidations))
        table.add_row("Recovery blocks", str(wo.recovery_blocks), str(wi.recovery_blocks))
        console.print(table)

    elif cmd == "buffer":
        console.print("[bold]Buffer zone analysis...[/bold]")
        result = buffer_zone_analysis()
        chart_heatmap(
            result.x_values.tolist(), result.y_values.tolist(),
            result.z_matrix.tolist(),
            title="Buffer Zone (%) by MM Ratio × Leverage",
            xlabel="MM Ratio", ylabel="Max Leverage",
        )
        path = export_heatmap(
            result.x_values, result.y_values, result.z_matrix,
            "Buffer Zone (%) by MM Ratio × Leverage",
            "MM Ratio", "Max Leverage", "Buffer %",
            "buffer_zone_analysis.png",
        )
        console.print(f"Exported: [bold]{path}[/bold]")

    elif cmd == "worst_case":
        console.print("[bold]Searching for worst-case parameter combo...[/bold]")
        result = find_worst_case(session.sim_state, oracle_prices)
        console.print(f"\n[red bold]Worst case insurance balance: ${result['worst_balance']:,.0f}[/red bold]")
        console.print("Parameters:")
        for k, v in result["params"].items():
            console.print(f"  {k}: {v}")
    else:
        print_error(f"Unknown analyze command: {cmd}")


def _handle_chart(args: list[str]) -> None:
    """Handle 'chart' commands."""
    if not args:
        print_error("Usage: chart mark_price|liquidations|insurance|heatmap|compare")
        return

    cmd = args[0]
    if cmd == "mark_price":
        if session.last_mark_prices is None or session.last_oracle_prices is None:
            print_error("No simulation results. Run a crash scenario first.")
            return
        chart_mark_price(
            session.last_mark_prices,
            session.last_oracle_prices,
            session.sim_state.exchange.block_time_ms,
        )
    elif cmd == "liquidations":
        if session.last_result is None:
            print_error("No simulation results. Run a crash scenario first.")
            return
        chart_liquidations(session.last_result, session.sim_state.exchange.block_time_ms)
    elif cmd == "insurance":
        if session.last_insurance is None:
            print_error("No simulation results. Run a crash scenario first.")
            return
        chart_insurance(session.last_insurance, session.sim_state.exchange.block_time_ms)
    elif cmd == "heatmap":
        heatmap_type = args[1] if len(args) > 1 else "mm_leverage"
        # Trigger analyze
        _handle_analyze(["sensitivity" if heatmap_type == "mm_leverage" else "buffer"])
    elif cmd == "compare":
        _handle_run(["compare", "all"])
    else:
        print_error(f"Unknown chart type: {cmd}")


def _handle_export(args: list[str]) -> None:
    """Handle 'export' commands."""
    if not args:
        print_error("Usage: export <filename> or export report <filename>")
        return

    if args[0] == "report":
        filename = args[1] if len(args) > 1 else "report.png"
        if session.last_mark_prices is not None and session.last_result is not None:
            path = export_simulation_results(
                session.last_mark_prices,
                session.last_oracle_prices,
                session.last_result,
                session.last_insurance,
                session.last_scenario_name,
                session.sim_state.exchange.block_time_ms,
                session.sim_state.exchange.mm_ratio,
                session.sim_state.market.book_depth_usd,
                filename=filename,
            )
            print_success(f"Report exported to {path}")
        else:
            print_error("No simulation results to export.")
    else:
        filename = args[0]
        if session.last_mark_prices is not None and session.last_result is not None:
            path = export_simulation_results(
                session.last_mark_prices,
                session.last_oracle_prices,
                session.last_result,
                session.last_insurance,
                session.last_scenario_name,
                session.sim_state.exchange.block_time_ms,
                session.sim_state.exchange.mm_ratio,
                session.sim_state.market.book_depth_usd,
                filename=filename,
            )
            print_success(f"Exported to {path}")
        else:
            print_error("No simulation results to export.")


def _handle_config(args: list[str]) -> None:
    """Handle 'config' commands."""
    if not args:
        print_error("Usage: config load|save <path>")
        return

    if args[0] == "load" and len(args) > 1:
        try:
            session.sim_state = load_yaml_config(args[1])
            session.regenerate_positions()
            print_success(f"Loaded config from {args[1]}")
        except Exception as e:
            print_error(f"Failed to load config: {e}")
    elif args[0] == "save" and len(args) > 1:
        try:
            save_yaml_config(session.sim_state, args[1])
            print_success(f"Saved config to {args[1]}")
        except Exception as e:
            print_error(f"Failed to save config: {e}")
    else:
        print_error("Usage: config load|save <path>")


def _handle_market(args: list[str]) -> None:
    """Handle 'market' commands."""
    if not args:
        print_error("Usage: market add|remove|list|config <symbol>")
        return

    cmd = args[0]
    if cmd == "list":
        console.print("Active markets: [cyan]BTC/USDX[/cyan]")
        console.print("[dim]Multi-market support: use 'market add <symbol>'[/dim]")
    elif cmd == "add":
        symbol = args[1] if len(args) > 1 else ""
        console.print(f"[dim]Market '{symbol}' noted. Multi-market simulation runs independently per market.[/dim]")
    elif cmd == "remove":
        console.print("[dim]Cannot remove BTC/USDX (primary market)[/dim]")
    elif cmd == "config":
        print_status(session.sim_state)
    else:
        print_error(f"Unknown market command: {cmd}")


def process_command(line: str) -> bool:
    """Process a single REPL command. Returns False to exit."""
    line = line.strip()
    if not line:
        return True

    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.split()

    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("quit", "exit", "q"):
        return False
    elif cmd == "help":
        print_help()
    elif cmd == "status":
        print_status(session.sim_state)
    elif cmd == "reset":
        session.sim_state = load_preset("default")
        session.positions.clear()
        session.last_mark_prices = None
        session.last_oracle_prices = None
        session.last_result = None
        session.last_insurance = None
        print_success("Reset to defaults")
    elif cmd == "preset":
        if not args:
            presets = list_presets()
            for name, desc in presets.items():
                console.print(f"  [cyan]{name}[/cyan] — {desc}")
            return True
        try:
            session.sim_state = load_preset(args[0])
            session.regenerate_positions()
            state = session.sim_state
            print_success(
                f"Loaded {args[0]} preset: {state.exchange.block_time_ms}ms blocks, "
                f"{state.exchange.mm_ratio * 100:.2f}% MM, "
                f"${state.market.book_depth_usd:,.0f} book depth, "
                f"{state.users.mau:,} MAU"
            )
        except ValueError as e:
            print_error(str(e))
    elif cmd == "set":
        _handle_set(args)
    elif cmd == "run":
        _handle_run(args)
    elif cmd == "analyze":
        _handle_analyze(args)
    elif cmd == "chart":
        _handle_chart(args)
    elif cmd == "export":
        _handle_export(args)
    elif cmd == "config":
        _handle_config(args)
    elif cmd == "market":
        _handle_market(args)
    elif cmd == "regen":
        session.regenerate_positions()
        print_success(f"Regenerated {len(session.positions)} positions")
    elif cmd == "positions":
        sub = args[0] if args else "show"
        if sub == "show":
            print_positions_summary(session.positions)
        elif sub == "list":
            if not session.positions:
                print_error("No positions. Use 'set mau <n>' first.")
            else:
                from rich.table import Table
                table = Table(title="Positions")
                table.add_column("ID")
                table.add_column("Side")
                table.add_column("Size ($)")
                table.add_column("Leverage")
                table.add_column("Entry")
                table.add_column("Liq Price")
                table.add_column("Bankrupt")
                limit = min(50, len(session.positions))
                for p in session.positions[:limit]:
                    table.add_row(
                        str(p.id), p.side,
                        f"${p.size_usd:,.0f}",
                        f"{p.leverage:.0f}x",
                        f"${p.entry_price:,.0f}",
                        f"${p.liquidation_price:,.0f}",
                        f"${p.bankruptcy_price:,.0f}",
                    )
                console.print(table)
                if len(session.positions) > limit:
                    console.print(f"[dim]Showing {limit} of {len(session.positions)} positions[/dim]")
        else:
            print_error(f"Unknown positions command: {sub}")
    else:
        print_error(f"Unknown command: {cmd}. Type 'help' for available commands.")

    return True


@click.command()
@click.option("--preset", "-p", default=None, help="Load a preset on startup")
@click.option("--config", "-c", default=None, help="Load a YAML config on startup")
@click.option("--command", "-e", default=None, help="Execute a single command and exit")
def main(preset: str | None, config: str | None, command: str | None) -> None:
    """Nexus Exchange Risk Simulator — interactive REPL for perpetual futures risk analysis."""
    # Load initial config
    if config:
        try:
            session.sim_state = load_yaml_config(config)
        except Exception as e:
            print_error(f"Failed to load config: {e}")
            sys.exit(1)
    elif preset:
        try:
            session.sim_state = load_preset(preset)
        except ValueError as e:
            print_error(str(e))
            sys.exit(1)

    # Single command mode
    if command:
        session.regenerate_positions()
        process_command(command)
        return

    # Interactive REPL
    print_banner()

    session.regenerate_positions()

    while True:
        try:
            line = input("\nnexus> ")
            if not process_command(line):
                console.print("[dim]Goodbye.[/dim]")
                break
        except KeyboardInterrupt:
            console.print("\n[dim]Use 'quit' to exit.[/dim]")
        except EOFError:
            console.print("\n[dim]Goodbye.[/dim]")
            break
        except Exception as e:
            print_error(f"Error: {e}")


if __name__ == "__main__":
    main()
