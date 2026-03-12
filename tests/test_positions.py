"""Tests for position generation."""

import pytest
import numpy as np

from nexus_sim.engine.positions import (
    Position,
    PositionConfig,
    generate_positions,
    generate_positions_from_mau,
    parse_leverage_dist,
)


class TestPosition:
    def test_margin_computed(self):
        pos = Position(id=0, side="long", size_usd=50000, leverage=10, entry_price=60000)
        assert pos.margin == 5000

    def test_long_liquidation_price(self):
        pos = Position(id=0, side="long", size_usd=50000, leverage=10, entry_price=60000)
        pos.compute_prices(0.0625)
        # liq = entry * (1 - 1/lev + mm) = 60000 * 0.9625 = 57750
        assert pos.liquidation_price == pytest.approx(57750)
        # bankruptcy = entry * (1 - 1/lev) = 60000 * 0.9 = 54000
        assert pos.bankruptcy_price == pytest.approx(54000)

    def test_short_liquidation_price(self):
        pos = Position(id=0, side="short", size_usd=50000, leverage=10, entry_price=60000)
        pos.compute_prices(0.0625)
        # liq = entry * (1 + 1/lev - mm) = 60000 * 1.0375 = 62250
        assert pos.liquidation_price == pytest.approx(62250)
        # bankruptcy = entry * (1 + 1/lev) = 60000 * 1.1 = 66000
        assert pos.bankruptcy_price == pytest.approx(66000)

    def test_margin_ratio_at_entry(self):
        pos = Position(id=0, side="long", size_usd=50000, leverage=10, entry_price=60000)
        # At entry, margin ratio = 1/leverage = 0.1
        assert pos.margin_ratio(60000) == pytest.approx(0.1)

    def test_residual_at_bankruptcy_is_zero(self):
        pos = Position(id=0, side="long", size_usd=50000, leverage=10, entry_price=60000)
        pos.compute_prices(0.0625)
        # At bankruptcy price, residual should be ~0
        residual = pos.residual_at_fill(pos.bankruptcy_price)
        assert abs(residual) < 1.0  # within $1


class TestPositionGeneration:
    def test_generate_correct_count(self):
        config = PositionConfig(count=100)
        positions = generate_positions(config, 60000, 0.0625)
        assert len(positions) == 100

    def test_leverage_distribution(self):
        config = PositionConfig(count=10000, leverage_dist={10: 0.5, 20: 0.5})
        positions = generate_positions(config, 60000, 0.0625)
        lev_10 = sum(1 for p in positions if p.leverage == 10)
        lev_20 = sum(1 for p in positions if p.leverage == 20)
        # Should be roughly 50/50
        assert 4000 < lev_10 < 6000
        assert 4000 < lev_20 < 6000

    def test_side_bias(self):
        config = PositionConfig(count=10000, side_bias=0.7)
        positions = generate_positions(config, 60000, 0.0625)
        long_count = sum(1 for p in positions if p.side == "long")
        assert 6000 < long_count < 8000  # ~70% should be long

    def test_all_positions_have_liq_prices(self):
        config = PositionConfig(count=50)
        positions = generate_positions(config, 60000, 0.0625)
        for p in positions:
            assert p.liquidation_price > 0
            assert p.bankruptcy_price > 0

    def test_parse_leverage_dist(self):
        dist = parse_leverage_dist("5x:20%,10x:40%,20x:30%,50x:10%")
        assert dist == {5: 0.2, 10: 0.4, 20: 0.3, 50: 0.1}


class TestMAUGeneration:
    def test_mau_generates_positions(self):
        positions = generate_positions_from_mau(1000, 60000, 0.0625)
        assert len(positions) > 0

    def test_more_mau_means_more_positions(self):
        pos_100 = generate_positions_from_mau(100, 60000, 0.0625)
        pos_1000 = generate_positions_from_mau(1000, 60000, 0.0625)
        assert len(pos_1000) > len(pos_100)

    def test_whale_concentration(self):
        positions = generate_positions_from_mau(10000, 60000, 0.0625)
        # Sort by size
        sorted_pos = sorted(positions, key=lambda p: p.size_usd, reverse=True)
        top_5_pct = sorted_pos[:max(1, len(sorted_pos) // 20)]
        top_oi = sum(p.size_usd for p in top_5_pct)
        total_oi = sum(p.size_usd for p in positions)
        # Top 5% should hold roughly 40% of OI (with tolerance)
        assert top_oi / total_oi > 0.2  # at least 20%
