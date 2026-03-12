"""Named scenario presets that configure multiple parameters at once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExchangeParams:
    block_time_ms: float = 200
    mm_ratio: float = 0.0625
    max_leverage: int = 16
    phase: int = 1
    max_price_change_pct: float = 0.05
    max_oracle_age_ms: float = 2000
    max_confidence_ratio: float = 0.02
    ema_decay_blocks: int = 600
    max_basis_pct: float = 0.01
    max_liquidations_per_block: int = 100
    use_dual_mark_price: bool = False
    same_block_execution: bool = True
    cascade_dampening: bool = False


@dataclass
class MarketParams:
    book_depth_usd: float = 3_000_000
    spread_pct: float = 0.0002
    book_shape: str = "realistic"
    num_levels: int = 50
    book_refresh_rate_ms: float = 200


@dataclass
class UserParams:
    mau: int = 1000
    position_count: int = 500
    avg_size_usd: float = 50_000
    leverage_dist: dict[int, float] | None = None
    side_bias: float = 0.6

    def __post_init__(self) -> None:
        if self.leverage_dist is None:
            self.leverage_dist = {5: 0.2, 10: 0.4, 20: 0.3, 50: 0.1}


@dataclass
class InsuranceParams:
    initial_balance: float = 100_000
    treasury_backstop: float = 1_000_000
    target_ratio: float = 0.01


@dataclass
class SimulationState:
    """Central state object passed through all simulation components."""

    exchange: ExchangeParams
    market: MarketParams
    users: UserParams
    insurance: InsuranceParams
    entry_price: float = 60_000  # BTC reference price

    def total_oi(self) -> float:
        return self.users.position_count * self.users.avg_size_usd

    def buffer_zone_pct(self) -> float:
        """Buffer between maintenance margin and liquidation trigger."""
        if self.exchange.max_leverage <= 0:
            return 0.0
        im_ratio = 1.0 / self.exchange.max_leverage
        return (im_ratio - self.exchange.mm_ratio) * 100

    def max_safe_leverage(self) -> float:
        """Maximum leverage with positive buffer zone."""
        if self.exchange.mm_ratio <= 0:
            return 0.0
        return 1.0 / self.exchange.mm_ratio

    def liq_delay_drift_pct(self) -> float:
        """Estimated mark price drift during one block at current volatility."""
        # Rough estimate: 120% annualized vol → per-ms vol
        ann_vol = 1.2
        ms_vol = ann_vol / (365.25 * 24 * 3600 * 1000) ** 0.5
        return ms_vol * self.exchange.block_time_ms ** 0.5 * 100


# Built-in presets
PRESETS: dict[str, dict[str, Any]] = {
    "conservative": {
        "description": "M1 testnet — safe defaults",
        "exchange": {"block_time_ms": 200, "mm_ratio": 0.0625, "max_leverage": 16, "phase": 1},
        "market": {"book_depth_usd": 1_000_000, "spread_pct": 0.0005},
        "users": {"mau": 100, "position_count": 50},
        "insurance": {"initial_balance": 50_000},
    },
    "competitive": {
        "description": "Target state — matching Hyperliquid",
        "exchange": {"block_time_ms": 5, "mm_ratio": 0.05, "max_leverage": 20, "phase": 2},
        "market": {"book_depth_usd": 5_000_000, "spread_pct": 0.0002},
        "users": {"mau": 50_000, "position_count": 25_000},
        "insurance": {"initial_balance": 5_000_000},
    },
    "stress_test": {
        "description": "Worst-case stress test parameters",
        "exchange": {"block_time_ms": 500, "mm_ratio": 0.03, "max_leverage": 33, "phase": 1},
        "market": {"book_depth_usd": 500_000, "spread_pct": 0.001},
        "users": {"mau": 10_000, "position_count": 5_000},
        "insurance": {"initial_balance": 100_000},
    },
    "default": {
        "description": "Balanced defaults",
        "exchange": {},
        "market": {},
        "users": {},
        "insurance": {},
    },
}


def load_preset(name: str) -> SimulationState:
    """Load a named preset into a SimulationState."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    preset = PRESETS[name]
    return _build_state(preset)


def load_yaml_config(path: str | Path) -> SimulationState:
    """Load a YAML config file into a SimulationState."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return _build_state(data)


def save_yaml_config(state: SimulationState, path: str | Path) -> None:
    """Save current state to a YAML config file."""
    data = {
        "exchange": {
            "block_time_ms": state.exchange.block_time_ms,
            "mm_ratio": state.exchange.mm_ratio,
            "max_leverage": state.exchange.max_leverage,
            "phase": state.exchange.phase,
            "max_price_change_pct": state.exchange.max_price_change_pct,
            "max_oracle_age_ms": state.exchange.max_oracle_age_ms,
        },
        "market": {
            "book_depth_usd": state.market.book_depth_usd,
            "spread_pct": state.market.spread_pct,
        },
        "users": {
            "mau": state.users.mau,
            "position_count": state.users.position_count,
            "avg_size_usd": state.users.avg_size_usd,
            "side_bias": state.users.side_bias,
        },
        "insurance": {
            "initial_balance": state.insurance.initial_balance,
            "treasury_backstop": state.insurance.treasury_backstop,
            "target_ratio": state.insurance.target_ratio,
        },
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def _build_state(data: dict) -> SimulationState:
    """Build a SimulationState from a dict (preset or YAML)."""
    exchange_data = data.get("exchange", {})
    market_data = data.get("market", {})
    user_data = data.get("users", {})
    insurance_data = data.get("insurance", {})

    exchange = ExchangeParams(**{k: v for k, v in exchange_data.items() if k in ExchangeParams.__dataclass_fields__})
    market = MarketParams(**{k: v for k, v in market_data.items() if k in MarketParams.__dataclass_fields__})
    users = UserParams(**{k: v for k, v in user_data.items() if k in UserParams.__dataclass_fields__})
    insurance = InsuranceParams(**{k: v for k, v in insurance_data.items() if k in InsuranceParams.__dataclass_fields__})

    return SimulationState(
        exchange=exchange,
        market=market,
        users=users,
        insurance=insurance,
    )


def list_presets() -> dict[str, str]:
    """Return dict of preset name → description."""
    return {name: p["description"] for name, p in PRESETS.items()}
