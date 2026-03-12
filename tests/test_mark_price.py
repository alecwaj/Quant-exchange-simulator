"""Tests for the mark price engine."""

import numpy as np
import pytest

from nexus_sim.engine.mark_price import (
    MarkPriceConfig,
    MarkPriceEngine,
    MarkAction,
    OracleStatus,
    run_mark_price_vectorized,
)


class TestMarkPriceEngine:
    def test_initial_price_tracks_oracle(self):
        config = MarkPriceConfig()
        engine = MarkPriceEngine(config=config)
        tick = engine.process_block(0, 60000.0)
        assert tick.mark_price == 60000.0
        assert tick.action == MarkAction.TRACKING

    def test_tracking_within_bound(self):
        config = MarkPriceConfig(max_price_change_pct=0.05)
        engine = MarkPriceEngine(config=config)
        engine.process_block(0, 60000.0)
        # 1% change is within 5% bound
        tick = engine.process_block(1, 60600.0)
        assert tick.mark_price == 60600.0
        assert tick.action == MarkAction.TRACKING

    def test_bounded_when_exceeding_max_change(self):
        config = MarkPriceConfig(max_price_change_pct=0.01)  # 1% max change
        engine = MarkPriceEngine(config=config)
        engine.process_block(0, 60000.0)
        # 10% drop exceeds 1% bound
        tick = engine.process_block(1, 54000.0)
        assert tick.action == MarkAction.BOUNDED
        # Should have moved only 1% toward target
        assert tick.mark_price == pytest.approx(60000.0 - 600.0)

    def test_stale_oracle_holds_price(self):
        config = MarkPriceConfig(max_oracle_age_ms=2000)
        engine = MarkPriceEngine(config=config)
        engine.process_block(0, 60000.0)
        tick = engine.process_block(1, 59000.0, oracle_age_ms=3000)
        assert tick.action == MarkAction.STALE_HOLD
        assert tick.mark_price == 60000.0
        assert tick.oracle_status == OracleStatus.STALE

    def test_low_confidence_holds_price(self):
        config = MarkPriceConfig(max_confidence_ratio=0.02)
        engine = MarkPriceEngine(config=config)
        engine.process_block(0, 60000.0)
        # Confidence of 2000 / 60000 = 0.033 > 0.02
        tick = engine.process_block(1, 59000.0, oracle_confidence=2000)
        assert tick.action == MarkAction.CONFIDENCE_HOLD
        assert tick.mark_price == 60000.0

    def test_phase2_ema_basis(self):
        config = MarkPriceConfig(phase=2, ema_decay_blocks=10, max_basis_pct=0.01)
        engine = MarkPriceEngine(config=config)
        engine.process_block(0, 60000.0, exchange_mid=60000.0)
        # Exchange mid higher than oracle → positive basis
        tick = engine.process_block(1, 60000.0, exchange_mid=60300.0)
        assert tick.ema_basis > 0

    def test_convergence_after_bounded(self):
        config = MarkPriceConfig(max_price_change_pct=0.01)
        engine = MarkPriceEngine(config=config)
        engine.process_block(0, 60000.0)
        # Keep feeding same price — mark should converge
        target = 57000.0  # 5% drop
        for i in range(1, 20):
            tick = engine.process_block(i, target)
        # After many blocks, should be close to target
        assert abs(tick.mark_price - target) < target * 0.01


class TestMarkPriceVectorized:
    def test_basic_vectorized(self):
        config = MarkPriceConfig(max_price_change_pct=0.05)
        prices = np.array([60000, 59700, 59400, 59100, 58800], dtype=float)
        mark, actions = run_mark_price_vectorized(config, prices)
        assert len(mark) == 5
        assert mark[0] == 60000.0
        # All changes are small (0.5%), should track
        assert all(a == 0 for a in actions[1:])

    def test_vectorized_bounding(self):
        config = MarkPriceConfig(max_price_change_pct=0.01)
        # Instant 10% drop
        prices = np.array([60000, 54000, 54000, 54000, 54000], dtype=float)
        mark, actions = run_mark_price_vectorized(config, prices)
        assert actions[1] == 1  # bounded
        assert mark[1] > 54000  # didn't fully track

    def test_vectorized_staleness(self):
        config = MarkPriceConfig(max_oracle_age_ms=1000)
        prices = np.array([60000, 59000, 58000], dtype=float)
        ages = np.array([0, 500, 2000], dtype=float)
        mark, actions = run_mark_price_vectorized(config, prices, oracle_ages_ms=ages)
        assert actions[2] == 2  # stale hold
        assert mark[2] == mark[1]  # held previous price

    def test_performance_large_series(self):
        """Ensure vectorized version handles large price series."""
        config = MarkPriceConfig()
        n = 100_000
        prices = 60000 + np.cumsum(np.random.randn(n) * 10)
        mark, actions = run_mark_price_vectorized(config, prices)
        assert len(mark) == n
        assert mark[0] == prices[0]
