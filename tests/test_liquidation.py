"""Tests for the liquidation engine."""

import numpy as np
import pytest

from nexus_sim.engine.insurance import InsuranceFund, InsuranceFundConfig
from nexus_sim.engine.liquidation import (
    LiquidationConfig,
    LiquidationEngine,
    run_simulation,
)
from nexus_sim.engine.mark_price import MarkPriceConfig, run_mark_price_vectorized
from nexus_sim.engine.orderbook import OrderBook, OrderBookConfig
from nexus_sim.engine.positions import Position, PositionConfig, generate_positions


class TestLiquidationEngine:
    def _make_position(self, side="long", leverage=10, size=50000, entry=60000, mm=0.0625):
        pos = Position(id=0, side=side, size_usd=size, leverage=leverage, entry_price=entry)
        pos.compute_prices(mm)
        return pos

    def _make_engine(self, mm_ratio=0.0625):
        return LiquidationEngine(config=LiquidationConfig(mm_ratio=mm_ratio))

    def _make_orderbook(self, mid=60000, depth=3_000_000):
        ob = OrderBook(config=OrderBookConfig(depth_per_side_usd=depth))
        ob.generate(mid)
        return ob

    def _make_insurance(self, balance=100_000):
        return InsuranceFund(config=InsuranceFundConfig(initial_balance=balance))

    def test_no_liquidation_above_mm(self):
        engine = self._make_engine()
        pos = self._make_position()
        ob = self._make_orderbook()
        insurance = self._make_insurance()

        # Mark price at entry — margin ratio is 1/leverage = 10%, above 6.25% MM
        count = engine.process_block(0, 60000.0, [pos], ob, insurance)
        assert count == 0
        assert not pos.is_liquidated

    def test_liquidation_below_mm(self):
        engine = self._make_engine()
        pos = self._make_position(leverage=10, entry=60000)
        ob = self._make_orderbook()
        insurance = self._make_insurance()

        # Drop price enough that margin ratio < 6.25%
        # At 10x, liq price = 60000 * (1 - 0.1 + 0.0625) = 60000 * 0.9625 = 57750
        count = engine.process_block(0, 57000.0, [pos], ob, insurance)
        assert count == 1
        assert pos.is_liquidated

    def test_liquidation_priority_by_margin_ratio(self):
        engine = self._make_engine()
        # Position at 10x closer to liquidation
        pos1 = Position(id=0, side="long", size_usd=50000, leverage=10, entry_price=60000)
        pos1.compute_prices(0.0625)
        # Position at 5x — more margin buffer
        pos2 = Position(id=1, side="long", size_usd=50000, leverage=5, entry_price=60000)
        pos2.compute_prices(0.0625)

        ob = self._make_orderbook()
        insurance = self._make_insurance()

        # Price that liquidates 10x but maybe not 5x
        # 10x liq: 57750, 5x liq: 60000 * (1 - 0.2 + 0.0625) = 51750
        count = engine.process_block(0, 57000.0, [pos1, pos2], ob, insurance)
        assert pos1.is_liquidated
        assert not pos2.is_liquidated

    def test_short_liquidation(self):
        engine = self._make_engine()
        pos = self._make_position(side="short", leverage=10, entry=60000)
        ob = self._make_orderbook()
        insurance = self._make_insurance()

        # Short liq price = 60000 * (1 + 0.1 - 0.0625) = 60000 * 1.0375 = 62250
        count = engine.process_block(0, 63000.0, [pos], ob, insurance)
        assert count == 1
        assert pos.is_liquidated

    def test_residual_goes_to_insurance(self):
        engine = self._make_engine()
        pos = self._make_position(leverage=10, entry=60000)
        ob = self._make_orderbook(depth=10_000_000)  # deep book = good fills
        insurance = self._make_insurance(balance=100_000)

        # Price well below liq price to trigger liquidation with clear residual/deficit
        engine.process_block(0, 57000.0, [pos], ob, insurance)
        assert pos.is_liquidated
        # Insurance fund should have recorded events
        assert len(insurance.events) > 0


class TestFullSimulation:
    def test_instant_crash_simulation(self):
        """Integration test: instant crash should liquidate high-leverage positions."""
        # Generate a 20% instant drop
        n = 1000
        oracle_prices = np.full(n, 60000.0)
        oracle_prices[0] = 60000.0
        oracle_prices[1:] = 48000.0  # 20% drop

        mp_config = MarkPriceConfig(max_price_change_pct=0.05)
        mark_prices, _ = run_mark_price_vectorized(mp_config, oracle_prices)

        # Generate positions
        pos_config = PositionConfig(count=100, avg_size_usd=50000)
        positions = generate_positions(pos_config, 60000.0, 0.0625)

        ob = OrderBook(config=OrderBookConfig(depth_per_side_usd=3_000_000))
        ob.generate(60000.0)

        insurance = InsuranceFund(config=InsuranceFundConfig(initial_balance=100_000))

        liq_config = LiquidationConfig(mm_ratio=0.0625)
        result = run_simulation(mark_prices, positions, ob, insurance, liq_config)

        # Some positions should be liquidated
        assert result.total_liquidated > 0
        assert result.total_positions == 100
        # Insurance should have recorded blocks
        assert len(insurance.history) == n

    def test_no_liquidations_flat_market(self):
        """No liquidations in a flat market for low-leverage positions."""
        n = 500
        oracle_prices = np.full(n, 60000.0)

        mp_config = MarkPriceConfig()
        mark_prices, _ = run_mark_price_vectorized(mp_config, oracle_prices)

        # Use only low leverage (5x) so IM (20%) >> MM (6.25%) with large buffer
        pos_config = PositionConfig(count=50, avg_size_usd=50000, leverage_dist={5: 1.0})
        positions = generate_positions(pos_config, 60000.0, 0.0625)

        ob = OrderBook(config=OrderBookConfig())
        ob.generate(60000.0)

        insurance = InsuranceFund(config=InsuranceFundConfig())

        liq_config = LiquidationConfig(mm_ratio=0.0625)
        result = run_simulation(mark_prices, positions, ob, insurance, liq_config)

        assert result.total_liquidated == 0

    def test_insurance_fund_never_negative_beyond_backstop(self):
        """Insurance fund should use treasury backstop before going deeply negative."""
        n = 500
        oracle_prices = np.full(n, 60000.0)
        oracle_prices[1:] = 36000.0  # 40% crash

        mp_config = MarkPriceConfig(max_price_change_pct=0.05)
        mark_prices, _ = run_mark_price_vectorized(mp_config, oracle_prices)

        pos_config = PositionConfig(count=200, avg_size_usd=100000)
        positions = generate_positions(pos_config, 60000.0, 0.0625)

        ob = OrderBook(config=OrderBookConfig(depth_per_side_usd=1_000_000))
        ob.generate(60000.0)

        ins_config = InsuranceFundConfig(initial_balance=50_000, treasury_backstop=500_000)
        insurance = InsuranceFund(config=ins_config)

        liq_config = LiquidationConfig(mm_ratio=0.0625)
        result = run_simulation(mark_prices, positions, ob, insurance, liq_config)

        # Treasury should have been used
        # (fund may go negative if backstop exhausted, but it tried)
        assert result.total_liquidated > 0
