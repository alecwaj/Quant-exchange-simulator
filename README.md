# Nexus Exchange Risk Simulator

Terminal-based risk simulation CLI for the Nexus perpetual futures exchange. Explore how exchange parameters (block time, leverage, maintenance margin, book depth, user growth) affect risk outcomes (liquidation safety, insurance fund requirements, cascading liquidation dynamics).

## Quick Start

```bash
pip install -e .
nexus-sim
```

## Usage

```
nexus> preset conservative
nexus> status
nexus> run crash march-2020
nexus> chart mark_price
nexus> analyze sensitivity
nexus> run compare all
```

## Architecture

- **Engine** — Mark price, liquidation, insurance fund, order book simulation
- **Scenarios** — Historical BTC crashes + synthetic crash generators
- **Analysis** — Sensitivity sweeps, competitor comparison, cascade analysis
- **Display** — Rich terminal UI, plotext inline charts, matplotlib PNG export

## Presets

| Preset | Block Time | MM Ratio | Max Leverage | Book Depth |
|--------|-----------|----------|-------------|-----------|
| conservative | 200ms | 6.25% | 16x | $1M |
| competitive | 5ms | 5% | 20x | $5M |
| stress_test | 500ms | 3% | 33x | $500K |

## Development

```bash
pip install -e ".[dev]"
pytest
```
