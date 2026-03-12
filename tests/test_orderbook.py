"""Tests for the synthetic order book."""

import pytest

from nexus_sim.engine.orderbook import OrderBook, OrderBookConfig


class TestOrderBook:
    def test_generate_creates_levels(self):
        ob = OrderBook(config=OrderBookConfig(num_levels=50))
        ob.generate(60000.0)
        assert len(ob.bids) == 50
        assert len(ob.asks) == 50
        assert ob.mid_price == 60000.0

    def test_bid_prices_descending(self):
        ob = OrderBook(config=OrderBookConfig(num_levels=10))
        ob.generate(60000.0)
        for i in range(len(ob.bids) - 1):
            assert ob.bids[i].price > ob.bids[i + 1].price

    def test_ask_prices_ascending(self):
        ob = OrderBook(config=OrderBookConfig(num_levels=10))
        ob.generate(60000.0)
        for i in range(len(ob.asks) - 1):
            assert ob.asks[i].price < ob.asks[i + 1].price

    def test_total_depth(self):
        config = OrderBookConfig(depth_per_side_usd=1_000_000, num_levels=20)
        ob = OrderBook(config=config)
        ob.generate(60000.0)
        assert ob.total_bid_depth() == pytest.approx(1_000_000, rel=0.01)
        assert ob.total_ask_depth() == pytest.approx(1_000_000, rel=0.01)

    def test_sell_depletes_bids_and_moves_mid(self):
        ob = OrderBook(config=OrderBookConfig(depth_per_side_usd=1_000_000))
        ob.generate(60000.0)
        original_mid = ob.mid_price

        fill_price, new_mid = ob.execute_sell(500_000)
        assert fill_price > 0
        assert new_mid < original_mid  # selling pushes price down

    def test_buy_depletes_asks_and_moves_mid(self):
        ob = OrderBook(config=OrderBookConfig(depth_per_side_usd=1_000_000))
        ob.generate(60000.0)
        original_mid = ob.mid_price

        fill_price, new_mid = ob.execute_buy(500_000)
        assert fill_price > 0
        assert new_mid > original_mid  # buying pushes price up

    def test_large_sell_causes_more_slippage(self):
        ob = OrderBook(config=OrderBookConfig(depth_per_side_usd=1_000_000))
        ob.generate(60000.0)
        slippage_small = ob.get_slippage_pct(10_000, "sell")

        ob.generate(60000.0)  # reset
        slippage_large = ob.get_slippage_pct(900_000, "sell")
        assert slippage_large > slippage_small

    def test_refresh_restores_depth(self):
        ob = OrderBook(config=OrderBookConfig(depth_per_side_usd=1_000_000))
        ob.generate(60000.0)
        ob.execute_sell(800_000)  # deplete most bids
        depleted_depth = ob.total_bid_depth()

        ob.refresh(59000.0)
        assert ob.total_bid_depth() > depleted_depth

    def test_different_shapes(self):
        for shape in ["linear", "realistic", "top_heavy"]:
            ob = OrderBook(config=OrderBookConfig(shape=shape, depth_per_side_usd=1_000_000))
            ob.generate(60000.0)
            assert ob.total_bid_depth() == pytest.approx(1_000_000, rel=0.01)
