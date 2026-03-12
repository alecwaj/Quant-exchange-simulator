# Nexus Exchange Risk Simulator

Terminal-based risk simulation CLI for the Nexus perpetual futures exchange. Built for technical executives to explore how exchange parameters affect risk outcomes — liquidation safety, insurance fund requirements, and cascading liquidation dynamics.

## Requirements

- Python 3.11+
- pip

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Quant-exchange-simulator

# Install in editable mode
pip install -e .

# For development (includes pytest)
pip install -e ".[dev]"
```

## Quick Start

```bash
# Launch the interactive REPL
nexus-sim

# Or load a preset on startup
nexus-sim --preset conservative

# Or run a single command
nexus-sim --command "run crash --type instant --magnitude 20"
```

## Interactive REPL

The primary interface is an interactive REPL that maintains state between commands. Change any parameter, then run simulations — the tool remembers your settings.

```
$ nexus-sim

╭──────────────────────────────────────────────────╮
│  Nexus Exchange Risk Simulator v0.1              │
│  Type 'help' for commands, 'preset <name>' to    │
│  load a configuration, 'run' to simulate.        │
╰──────────────────────────────────────────────────╯

nexus> preset conservative
✓ Loaded conservative preset: 200ms blocks, 6.25% MM, $1M book depth, 100 MAU

nexus> status
╭─ Exchange Config ──────────────────╮  ╭─ Market State ──────────────────╮
│ Block time:     200ms              │  │ Book depth:    $1.0M / side     │
│ MM ratio:       6.25%              │  │ Spread:        0.05%            │
│ Max leverage:   16x                │  │ Markets:       BTC/USDX         │
│ Mark price:     Phase 1            │  │ Oracle:        Pyth Pro          │
╰────────────────────────────────────╯  ╰─────────────────────────────────╯

nexus> run crash march-2020
Running March 2020 crash scenario...
████████████████████████████████ 100%

╭─ Results ──────────────────────────────────────────────────────────╮
│ Positions liquidated:     38 / 50 (76%)                           │
│ Cascade depth:            4                                       │
│ Insurance fund:           $50,000 → $62,340 (+$12,340)            │
│ Worst fill vs bankruptcy: 98.2% (1.8% slippage)                  │
╰────────────────────────────────────────────────────────────────────╯
```

## Command Reference

### Configuration

| Command | Description |
|---------|-------------|
| `preset <name>` | Load a named preset (`conservative`, `competitive`, `stress_test`, `default`) |
| `config load <path>` | Load a YAML config file |
| `config save <path>` | Save current config to YAML |
| `status` | Show current configuration summary |
| `reset` | Reset all parameters to defaults |

### Parameter Tuning

Set individual parameters to drill down from preset configurations.

| Command | Description | Range |
|---------|-------------|-------|
| `set block_time <ms>` | Block time | 5–2000 |
| `set mm_ratio <pct>` | Maintenance margin ratio | 1–20% |
| `set leverage_max <n>` | Max leverage multiplier | 2–100 |
| `set book_depth <usd>` | Order book depth per side | any |
| `set spread <pct>` | Bid-ask spread | any |
| `set phase <1\|2>` | Mark price engine phase | 1 or 2 |
| `set change_bound <pct>` | Max price change per block | any |
| `set insurance_balance <usd>` | Insurance fund starting balance | any |
| `set treasury_cap <usd>` | Protocol treasury backstop limit | any |
| `set oracle_staleness <ms>` | Max oracle data age | any |

**Optimization flags:**

| Command | Description |
|---------|-------------|
| `set opt.dual_mark on\|off` | Use separate unbounded mark price for liquidation triggers |
| `set opt.same_block on\|off` | Execute liquidation in same block as trigger |
| `set opt.cascade_dampen on\|off` | Re-evaluate after each liquidation (reduces cascades) |

### Users & Positions

| Command | Description |
|---------|-------------|
| `set mau <count>` | Set MAU, auto-generate realistic position distribution |
| `set positions <count>` | Set position count directly |
| `set oi <usd>` | Set total open interest |
| `set leverage_dist "<spec>"` | e.g. `"5x:20%,10x:40%,20x:30%,50x:10%"` |
| `set side_bias <ratio>` | Long ratio (0.6 = 60% long, 40% short) |
| `regen` | Regenerate position set with current params |
| `positions show` | Show position distribution summary |
| `positions list` | List all positions (paginated, max 50) |

### Running Simulations

**Historical crash replays:**

```
run crash march-2020       # COVID crash, -45% over 24h
run crash may-2021         # China ban, -35% over 12h
run crash luna-crash       # LUNA/UST collapse, -28% over 48h
run crash ftx-crash        # FTX collapse, -25% over 72h
run crash aug-2024         # Yen carry unwind, -18% over 6h
run stress                 # Run ALL historical crashes sequentially
```

**Synthetic crash generators:**

```
run crash --type instant --magnitude 15           # Instantaneous 15% drop
run crash --type linear --magnitude 20 --duration 60    # 20% over 60 seconds
run crash --type vshape --magnitude 15 --recovery 120   # 15% drop, recover over 120s
run crash --type cascade --initial 5 --steps 8 --interval 30   # Staircase: 5% drops every 30s
run crash --type volatility --annualized 120 --duration 3600   # 1hr of 120% annualized vol
```

**Other simulation modes:**

```
run normal <duration_sec>                          # Normal market conditions
run sweep <param> <min> <max> <steps>              # Parameter sweep with chart
run compare hyperliquid                            # Compare with Hyperliquid params
run compare dydx                                   # Compare with dYdX v4 params
run compare all                                    # Side-by-side all competitors
```

### Analysis

| Command | Description |
|---------|-------------|
| `analyze sensitivity` | Generate MM ratio x leverage heatmap |
| `analyze insurance` | Insurance fund sizing analysis |
| `analyze cascade` | Compare cascade behavior with/without dampening |
| `analyze buffer` | Buffer zone analysis by leverage and MM ratio |
| `analyze worst_case` | Find parameter combo that depletes the fund |

### Charts

| Command | Description |
|---------|-------------|
| `chart mark_price` | Mark price vs oracle price over time (inline terminal) |
| `chart liquidations` | Liquidation cascade timeline |
| `chart insurance` | Insurance fund balance over time |
| `chart heatmap <type>` | Heatmap: `mm_leverage` or `buffer` |
| `chart compare` | Side-by-side competitor comparison |

### Export

| Command | Description |
|---------|-------------|
| `export <filename>` | Export last simulation chart as PNG |
| `export report <filename>` | Export full report PNG |

Every `run` command auto-exports a PNG to the `exports/` directory with a filename encoding the key parameters.

## Presets

| Preset | Use Case | Block Time | MM Ratio | Max Leverage | Book Depth | Insurance |
|--------|----------|-----------|----------|-------------|-----------|-----------|
| `conservative` | M1 testnet | 200ms | 6.25% | 16x | $1M | $50K |
| `competitive` | Target state (match HL) | 5ms | 5.0% | 20x | $5M | $5M |
| `stress_test` | Worst-case testing | 500ms | 3.0% | 33x | $500K | $100K |
| `default` | Balanced defaults | 200ms | 6.25% | 16x | $3M | $100K |

## Architecture

```
src/nexus_sim/
├── cli.py              # Interactive REPL with all commands
├── engine/
│   ├── mark_price.py   # Phase 1 (oracle) + Phase 2 (oracle + EMA basis)
│   ├── liquidation.py  # Full liquidation pipeline with cascade tracking
│   ├── insurance.py    # Insurance fund with treasury backstop
│   ├── orderbook.py    # Synthetic order book with depth profiles
│   └── positions.py    # Position generation (direct + MAU-based)
├── scenarios/
│   ├── historical.py   # 5 real BTC crash replays
│   ├── synthetic.py    # 5 parametric crash generators
│   └── presets.py      # Named configuration presets
├── analysis/
│   ├── sensitivity.py  # Parameter sweeps and heatmaps
│   ├── comparison.py   # Competitor benchmarking (HL, dYdX, Binance)
│   └── cascade.py      # Cascading liquidation analysis
├── display/
│   ├── terminal.py     # Rich panels, tables, progress bars
│   ├── charts.py       # plotext inline terminal charts
│   └── export.py       # matplotlib PNG export
├── models/
│   └── user_growth.py  # MAU → position distribution model
└── data/
    └── btc_crashes.json
```

## Key Concepts

**Buffer Zone:** The gap between initial margin (1/leverage) and maintenance margin. A larger buffer gives positions more room before liquidation. The tool warns you when buffer zones are dangerously thin.

**Cascade Depth:** When a liquidation sell order moves the mark price down, it can trigger additional liquidations. The cascade depth tracks how many sequential liquidations are triggered by a single price movement.

**Mark Price Phases:**
- Phase 1: Pure oracle price with change bounding and staleness protection
- Phase 2: Oracle price + EMA basis adjustment from the exchange's own order book

**Insurance Fund:** Collects residual margin from clean liquidations (where fill price > bankruptcy price) and pays deficits from underwater liquidations. A treasury backstop provides additional protection.

## Competitor Comparison

The `run compare` command benchmarks against real exchange parameters:

| Exchange | MM Ratio | Max Leverage | Block Time | Book Depth | Insurance |
|----------|---------|-------------|-----------|-----------|-----------|
| Hyperliquid | 5% | 50x | ~200ms | $7M | $50M |
| dYdX v4 | 5% | 20x | ~500ms | $3M | $20M |
| Binance | 2.5% | 125x | ~1ms | $50M | $1B |

## Configuration Files

Custom YAML configs can be loaded and saved:

```yaml
# configs/conservative.yaml
exchange:
  block_time_ms: 200
  mm_ratio: 0.0625
  max_leverage: 16
  phase: 1
market:
  book_depth_usd: 1000000
  spread_pct: 0.0005
users:
  mau: 100
  position_count: 50
insurance:
  initial_balance: 50000
  treasury_backstop: 1000000
  target_ratio: 0.01
```

```
nexus> config load configs/conservative.yaml
nexus> config save my_config.yaml
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test module
pytest tests/test_mark_price.py
pytest tests/test_liquidation.py
pytest tests/test_insurance.py
pytest tests/test_orderbook.py
pytest tests/test_positions.py
pytest tests/test_scenarios.py
```

## Examples

**Explore how block time affects liquidation safety:**

```
nexus> preset conservative
nexus> set block_time 5
nexus> run crash --type instant --magnitude 20
nexus> set block_time 500
nexus> run crash --type instant --magnitude 20
```

**Find the insurance fund size needed for a March 2020 style crash:**

```
nexus> preset competitive
nexus> run crash march-2020
nexus> analyze insurance
```

**Compare your exchange against competitors under stress:**

```
nexus> preset competitive
nexus> run crash --type cascade --initial 5 --steps 8 --interval 30
nexus> run compare all
```

**Sweep leverage to find the safe maximum:**

```
nexus> preset conservative
nexus> run sweep leverage_max 5 50 10
```
