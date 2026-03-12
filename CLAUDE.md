# nexus-risk-sim

## Project
Terminal-based risk simulation CLI for Nexus perpetual futures exchange.
Python 3.11+, click CLI, rich terminal UI, plotext inline charts, matplotlib export.

## Architecture
- `src/nexus_sim/engine/` — core simulation engines (mark price, liquidation, insurance, orderbook)
- `src/nexus_sim/scenarios/` — crash scenario generators (historical + synthetic)
- `src/nexus_sim/analysis/` — sensitivity analysis, competitor comparison
- `src/nexus_sim/display/` — terminal UI and chart export
- `src/nexus_sim/models/` — MAU → position distribution model

## Style
- Python 3.11+, type hints on all functions
- dataclasses for config, numpy for vectorized computation
- No global mutable state — all state lives in SimulationState passed explicitly
- Tests in tests/ using pytest

## Key Invariants
- Insurance fund balance tracked to the cent
- Liquidation order price is always at or better than bankruptcy price
- Positions sorted by margin ratio for liquidation priority
- Mark price never uses stale oracle data (> MAX_ORACLE_AGE_MS)

## Running
pip install -e .
nexus-sim
