"""Insurance fund model with treasury backstop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


class FundEvent(NamedTuple):
    block: int
    amount: float  # positive = inflow, negative = outflow
    balance_after: float
    event_type: str  # "residual", "deficit", "backstop"


@dataclass
class InsuranceFundConfig:
    initial_balance: float = 100_000
    treasury_backstop: float = 1_000_000
    target_ratio: float = 0.01  # 1% of total OI


@dataclass
class InsuranceFund:
    """Insurance fund that absorbs liquidation residuals and deficits."""

    config: InsuranceFundConfig
    balance: float = 0.0
    treasury_used: float = 0.0
    history: list[float] = field(default_factory=list)
    events: list[FundEvent] = field(default_factory=list)
    peak_balance: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_block: int = 0
    total_residual: float = 0.0
    total_deficit: float = 0.0

    def __post_init__(self) -> None:
        self.balance = self.config.initial_balance
        self.peak_balance = self.balance

    def reset(self) -> None:
        self.balance = self.config.initial_balance
        self.treasury_used = 0.0
        self.history.clear()
        self.events.clear()
        self.peak_balance = self.balance
        self.max_drawdown = 0.0
        self.max_drawdown_block = 0
        self.total_residual = 0.0
        self.total_deficit = 0.0

    def record_block(self, block: int) -> None:
        """Record balance snapshot for current block."""
        self.history.append(self.balance)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = self.peak_balance - self.balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_block = block

    def add_residual(self, block: int, amount: float) -> None:
        """Add residual margin from a profitable liquidation."""
        self.balance += amount
        self.total_residual += amount
        self.events.append(FundEvent(block, amount, self.balance, "residual"))

    def cover_deficit(self, block: int, amount: float) -> float:
        """Cover a deficit from an underwater liquidation.

        Returns the actual amount covered. If fund is depleted,
        uses treasury backstop up to the configured limit.
        """
        covered = 0.0

        # First draw from fund
        fund_draw = min(amount, max(0, self.balance))
        self.balance -= fund_draw
        covered += fund_draw
        self.total_deficit += fund_draw

        remaining = amount - fund_draw
        if remaining > 0:
            # Draw from treasury backstop
            backstop_available = self.config.treasury_backstop - self.treasury_used
            backstop_draw = min(remaining, backstop_available)
            self.treasury_used += backstop_draw
            covered += backstop_draw
            self.total_deficit += backstop_draw
            if backstop_draw > 0:
                self.events.append(FundEvent(block, -backstop_draw, self.balance, "backstop"))
            # If still remaining, fund goes negative
            uncovered = remaining - backstop_draw
            if uncovered > 0:
                self.balance -= uncovered
                self.total_deficit += uncovered

        self.events.append(FundEvent(block, -amount, self.balance, "deficit"))
        return covered

    def coverage_ratio(self, total_oi: float) -> float:
        """Current fund balance as percentage of total OI."""
        if total_oi <= 0:
            return 0.0
        return self.balance / total_oi

    def is_healthy(self, total_oi: float) -> bool:
        """Check if fund meets target ratio."""
        return self.coverage_ratio(total_oi) >= self.config.target_ratio

    def depletion_probability(self, avg_deficit_per_crash: float, crashes_per_year: float = 4) -> float:
        """Rough estimate of annual depletion probability."""
        if avg_deficit_per_crash <= 0:
            return 0.0
        crashes_to_deplete = self.balance / avg_deficit_per_crash
        if crashes_to_deplete >= crashes_per_year * 3:
            return 0.01
        elif crashes_to_deplete >= crashes_per_year:
            return 0.1
        elif crashes_to_deplete >= 1:
            return 0.5
        else:
            return 0.9
