"""Tests for the insurance fund model."""

import pytest

from nexus_sim.engine.insurance import InsuranceFund, InsuranceFundConfig


class TestInsuranceFund:
    def test_initial_balance(self):
        fund = InsuranceFund(config=InsuranceFundConfig(initial_balance=100_000))
        assert fund.balance == 100_000

    def test_residual_increases_balance(self):
        fund = InsuranceFund(config=InsuranceFundConfig(initial_balance=100_000))
        fund.add_residual(0, 5000)
        assert fund.balance == 105_000
        assert fund.total_residual == 5000

    def test_deficit_decreases_balance(self):
        fund = InsuranceFund(config=InsuranceFundConfig(initial_balance=100_000))
        covered = fund.cover_deficit(0, 30_000)
        assert covered == 30_000
        assert fund.balance == 70_000

    def test_treasury_backstop(self):
        config = InsuranceFundConfig(initial_balance=10_000, treasury_backstop=50_000)
        fund = InsuranceFund(config=config)
        # Deficit larger than fund balance
        covered = fund.cover_deficit(0, 20_000)
        assert covered == 20_000
        assert fund.treasury_used == 10_000  # drew 10k from treasury

    def test_backstop_limit(self):
        config = InsuranceFundConfig(initial_balance=1000, treasury_backstop=5000)
        fund = InsuranceFund(config=config)
        # Deficit way larger than fund + backstop
        covered = fund.cover_deficit(0, 100_000)
        assert fund.treasury_used == 5000  # maxed out backstop
        assert fund.balance < 0  # fund went negative

    def test_drawdown_tracking(self):
        fund = InsuranceFund(config=InsuranceFundConfig(initial_balance=100_000))
        fund.record_block(0)
        fund.add_residual(1, 20_000)
        fund.record_block(1)
        fund.cover_deficit(2, 50_000)
        fund.record_block(2)

        assert fund.peak_balance == 120_000
        assert fund.max_drawdown == 50_000

    def test_coverage_ratio(self):
        fund = InsuranceFund(config=InsuranceFundConfig(initial_balance=100_000))
        assert fund.coverage_ratio(10_000_000) == pytest.approx(0.01)
        assert fund.is_healthy(10_000_000)
        assert not fund.is_healthy(100_000_000)  # 0.1% < 1% target

    def test_reset(self):
        fund = InsuranceFund(config=InsuranceFundConfig(initial_balance=100_000))
        fund.add_residual(0, 5000)
        fund.record_block(0)
        fund.reset()
        assert fund.balance == 100_000
        assert fund.total_residual == 0
        assert len(fund.history) == 0
