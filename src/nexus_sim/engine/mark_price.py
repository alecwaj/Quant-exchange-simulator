"""Mark price engine — Phase 1 (pure oracle) and Phase 2 (oracle + EMA basis)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np


class OracleStatus(Enum):
    OK = "ok"
    STALE = "stale"
    LOW_CONFIDENCE = "low_confidence"


class MarkAction(Enum):
    TRACKING = "tracking"
    BOUNDED = "bounded"
    STALE_HOLD = "stale_hold"
    CONFIDENCE_HOLD = "confidence_hold"


class MarkPriceTick(NamedTuple):
    block: int
    mark_price: float
    oracle_price: float
    action: MarkAction
    oracle_status: OracleStatus
    ema_basis: float  # Phase 2 only


@dataclass
class MarkPriceConfig:
    max_oracle_age_ms: float = 2000
    max_confidence_ratio: float = 0.02
    max_price_change_pct: float = 0.05
    block_time_ms: float = 5
    # Phase 2
    ema_decay_blocks: int = 600
    max_basis_pct: float = 0.01
    phase: int = 1  # 1 or 2


@dataclass
class MarkPriceEngine:
    """Stateful mark price engine that processes oracle updates block by block."""

    config: MarkPriceConfig
    current_mark: float = 0.0
    ema_basis: float = 0.0
    last_oracle_time_ms: float = 0.0
    history: list[MarkPriceTick] = field(default_factory=list)
    _initialized: bool = False

    def reset(self) -> None:
        self.current_mark = 0.0
        self.ema_basis = 0.0
        self.last_oracle_time_ms = 0.0
        self.history.clear()
        self._initialized = False

    def process_block(
        self,
        block: int,
        oracle_price: float,
        oracle_age_ms: float = 0.0,
        oracle_confidence: float = 0.0,
        exchange_mid: float | None = None,
    ) -> MarkPriceTick:
        """Process a single block and return the mark price tick."""
        oracle_status = self._check_oracle(oracle_age_ms, oracle_confidence, oracle_price)

        if not self._initialized:
            self.current_mark = oracle_price
            self._initialized = True
            tick = MarkPriceTick(block, oracle_price, oracle_price, MarkAction.TRACKING, oracle_status, 0.0)
            self.history.append(tick)
            return tick

        if oracle_status == OracleStatus.STALE:
            tick = MarkPriceTick(block, self.current_mark, oracle_price, MarkAction.STALE_HOLD, oracle_status, self.ema_basis)
            self.history.append(tick)
            return tick

        if oracle_status == OracleStatus.LOW_CONFIDENCE:
            tick = MarkPriceTick(block, self.current_mark, oracle_price, MarkAction.CONFIDENCE_HOLD, oracle_status, self.ema_basis)
            self.history.append(tick)
            return tick

        if self.config.phase == 2 and exchange_mid is not None:
            self._update_ema_basis(oracle_price, exchange_mid)

        target = oracle_price
        if self.config.phase == 2:
            clamped_basis = np.clip(self.ema_basis, -self.config.max_basis_pct, self.config.max_basis_pct)
            target = oracle_price * (1 + clamped_basis)

        max_change = self.current_mark * self.config.max_price_change_pct
        if abs(target - self.current_mark) <= max_change:
            self.current_mark = target
            action = MarkAction.TRACKING
        else:
            direction = 1.0 if target > self.current_mark else -1.0
            self.current_mark += direction * max_change
            action = MarkAction.BOUNDED

        tick = MarkPriceTick(block, self.current_mark, oracle_price, action, oracle_status, self.ema_basis)
        self.history.append(tick)
        return tick

    def _check_oracle(self, age_ms: float, confidence: float, price: float) -> OracleStatus:
        if age_ms > self.config.max_oracle_age_ms:
            return OracleStatus.STALE
        if price > 0 and confidence / price > self.config.max_confidence_ratio:
            return OracleStatus.LOW_CONFIDENCE
        return OracleStatus.OK

    def _update_ema_basis(self, oracle_price: float, exchange_mid: float) -> None:
        if oracle_price <= 0:
            return
        basis = (exchange_mid - oracle_price) / oracle_price
        alpha = 2.0 / (self.config.ema_decay_blocks + 1)
        self.ema_basis = alpha * basis + (1 - alpha) * self.ema_basis


def run_mark_price_vectorized(
    config: MarkPriceConfig,
    oracle_prices: np.ndarray,
    oracle_ages_ms: np.ndarray | None = None,
    oracle_confidences: np.ndarray | None = None,
    exchange_mids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized mark price computation for performance.

    Returns (mark_prices, actions) where actions are integer codes:
        0=tracking, 1=bounded, 2=stale_hold, 3=confidence_hold
    """
    n = len(oracle_prices)
    mark_prices = np.empty(n)
    actions = np.zeros(n, dtype=np.int8)

    if oracle_ages_ms is None:
        oracle_ages_ms = np.zeros(n)
    if oracle_confidences is None:
        oracle_confidences = np.zeros(n)

    max_change_pct = config.max_price_change_pct
    max_oracle_age = config.max_oracle_age_ms
    max_conf_ratio = config.max_confidence_ratio

    mark = oracle_prices[0]
    mark_prices[0] = mark
    ema_basis = 0.0
    alpha = 2.0 / (config.ema_decay_blocks + 1)

    for i in range(1, n):
        op = oracle_prices[i]

        # Staleness check
        if oracle_ages_ms[i] > max_oracle_age:
            mark_prices[i] = mark
            actions[i] = 2
            continue

        # Confidence check
        if op > 0 and oracle_confidences[i] / op > max_conf_ratio:
            mark_prices[i] = mark
            actions[i] = 3
            continue

        # Phase 2: EMA basis
        target = op
        if config.phase == 2 and exchange_mids is not None:
            basis = (exchange_mids[i] - op) / op if op > 0 else 0.0
            ema_basis = alpha * basis + (1 - alpha) * ema_basis
            clamped = max(-config.max_basis_pct, min(config.max_basis_pct, ema_basis))
            target = op * (1 + clamped)

        # Change bounding
        max_change = mark * max_change_pct
        diff = target - mark
        if abs(diff) <= max_change:
            mark = target
            actions[i] = 0  # tracking
        else:
            mark += (1.0 if diff > 0 else -1.0) * max_change
            actions[i] = 1  # bounded

        mark_prices[i] = mark

    return mark_prices, actions
