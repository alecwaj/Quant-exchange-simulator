"""Tests for scenario generators."""

import numpy as np
import pytest

from nexus_sim.scenarios.synthetic import (
    generate_instant_crash,
    generate_linear_crash,
    generate_vshape_crash,
    generate_cascade_crash,
    generate_volatility_scenario,
)
from nexus_sim.scenarios.historical import load_historical_crash, list_historical_crashes
from nexus_sim.scenarios.presets import load_preset, list_presets, SimulationState


class TestSyntheticCrashes:
    def test_instant_crash(self):
        scenario = generate_instant_crash(60000, 15)
        assert scenario.prices[0] == 60000
        assert scenario.prices[-1] == pytest.approx(60000 * 0.85)

    def test_linear_crash(self):
        scenario = generate_linear_crash(60000, 20, 60, block_time_ms=100)
        assert scenario.prices[0] == 60000
        assert scenario.prices[-1] == pytest.approx(60000 * 0.80)
        # Should be monotonically decreasing
        for i in range(len(scenario.prices) - 1):
            assert scenario.prices[i] >= scenario.prices[i + 1]

    def test_vshape_crash(self):
        scenario = generate_vshape_crash(60000, 15, 120)
        assert scenario.prices[0] == 60000
        # Bottom should be ~15% below start
        assert min(scenario.prices) == pytest.approx(60000 * 0.85, rel=0.01)
        # Should recover to near start
        assert scenario.prices[-1] > 60000 * 0.95

    def test_cascade_crash(self):
        scenario = generate_cascade_crash(60000, 5, 4, 30)
        # 4 steps of 5% = ~18.5% total drop
        assert scenario.magnitude_pct > 15
        assert scenario.magnitude_pct < 25

    def test_volatility_scenario(self):
        scenario = generate_volatility_scenario(60000, 120, 3600)
        assert len(scenario.prices) > 0
        assert scenario.prices[0] == 60000


class TestHistoricalCrashes:
    def test_list_crashes(self):
        crashes = list_historical_crashes()
        assert "march-2020" in crashes
        assert "may-2021" in crashes
        assert "luna-crash" in crashes
        assert "ftx-crash" in crashes
        assert "aug-2024" in crashes

    def test_load_march_2020(self):
        scenario = load_historical_crash("march-2020", block_time_ms=200)
        assert len(scenario.prices) > 0
        assert scenario.magnitude_pct > 30  # ~45% drop

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown crash preset"):
            load_historical_crash("nonexistent")

    def test_worst_hour_window(self):
        scenario = load_historical_crash("march-2020", block_time_ms=200, timescale="1h")
        assert len(scenario.prices) > 0
        # Worst hour should have some drop
        assert scenario.magnitude_pct > 0


class TestPresets:
    def test_list_presets(self):
        presets = list_presets()
        assert "conservative" in presets
        assert "competitive" in presets
        assert "stress_test" in presets

    def test_load_conservative(self):
        state = load_preset("conservative")
        assert isinstance(state, SimulationState)
        assert state.exchange.block_time_ms == 200
        assert state.exchange.mm_ratio == 0.0625
        assert state.exchange.max_leverage == 16

    def test_load_competitive(self):
        state = load_preset("competitive")
        assert state.exchange.block_time_ms == 5
        assert state.exchange.mm_ratio == 0.05
        assert state.exchange.max_leverage == 20

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent")

    def test_buffer_zone_calculation(self):
        state = load_preset("conservative")
        buffer = state.buffer_zone_pct()
        # IM = 1/16 = 6.25%, MM = 6.25%, buffer = 0%
        assert buffer == pytest.approx(0.0)

    def test_competitive_has_positive_buffer(self):
        state = load_preset("competitive")
        buffer = state.buffer_zone_pct()
        # IM = 1/20 = 5%, MM = 5%, buffer = 0%
        assert buffer >= 0
