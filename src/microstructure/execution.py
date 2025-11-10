"""Execution analytics built on top of the microstructure toolkit."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .lob import LimitOrderBook, Order, OrderFlowSimulator, Side


@dataclass
class ExecutionResult:
    fill_probability: float
    expected_fill: float
    expected_cost: float


def _clone_book(book: LimitOrderBook) -> LimitOrderBook:
    clone = LimitOrderBook()
    for side in (Side.BUY, Side.SELL):
        src = book.bids if side is Side.BUY else book.asks
        dst = clone.bids if side is Side.BUY else clone.asks
        for price, orders in src.items():
            dst[price] = deque(
                Order(
                    order_id=order.order_id,
                    side=order.side,
                    quantity=order.quantity,
                    price=order.price,
                    timestamp=order.timestamp,
                    order_type=order.order_type,
                )
                for order in orders
            )
    clone._order_lookup = dict(book._order_lookup)
    return clone


class ExecutionAnalytics:
    """Monte Carlo analytics to evaluate execution tactics."""

    def __init__(
        self,
        book: LimitOrderBook,
        simulator: Optional[OrderFlowSimulator] = None,
    ) -> None:
        self.book = book
        self.simulator = simulator

    def estimate_fill_probability(
        self,
        side: Side,
        quantity: float,
        horizon: float,
        n_paths: int = 100,
    ) -> ExecutionResult:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if self.simulator is None:
            depth = self.book.depth(side.opposite, levels=10)
            cumulative = 0.0
            cost = 0.0
            remaining = quantity
            mid = self.book.mid_price() or 0.0
            for price, level_qty in depth:
                trade_qty = min(remaining, level_qty)
                cost += trade_qty * price
                cumulative += trade_qty
                remaining -= trade_qty
                if remaining <= 0:
                    break
            fill_prob = 1.0 if remaining <= 0 else cumulative / quantity
            expected_cost = cost if cumulative > 0 else quantity * mid
            return ExecutionResult(fill_prob, cumulative, expected_cost)

        fills = []
        costs = []
        for _ in range(n_paths):
            scenario_book = _clone_book(self.book)
            sim = self.simulator
            scenario = sim.simulate(horizon, book=scenario_book)
            filled = scenario.add_market_order(side, quantity, timestamp=horizon)
            fills.append(filled)
            cost = 0.0
            if filled > 0:
                # Estimate cost as average execution price using fills recorded in event log.
                prices = []
                for _, message in scenario.event_log:
                    if message.startswith("FILL") and f"{side}" in message:
                        price_part = message.split("@")[-1].split()[0]
                        try:
                            prices.append(float(price_part))
                        except ValueError:
                            continue
                if prices:
                    avg_price = float(np.mean(prices))
                else:
                    # Fall back to mid price if we could not infer fills.
                    mid = scenario.mid_price() or 0.0
                    avg_price = mid
                cost = avg_price * filled
            costs.append(cost)
        fills_arr = np.array(fills)
        costs_arr = np.array(costs)
        fill_prob = float(np.mean(fills_arr >= quantity))
        expected_fill = float(np.mean(fills_arr))
        expected_cost = float(np.mean(costs_arr))
        return ExecutionResult(fill_prob, expected_fill, expected_cost)

    def estimate_price_impact(
        self,
        side: Side,
        quantity: float,
        depth_levels: int = 5,
    ) -> float:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        depth = self.book.depth(side.opposite, levels=depth_levels)
        if not depth:
            return 0.0
        remaining = quantity
        weighted_price = 0.0
        total = 0.0
        for price, level_qty in depth:
            trade_qty = min(remaining, level_qty)
            weighted_price += trade_qty * price
            total += trade_qty
            remaining -= trade_qty
            if remaining <= 0:
                break
        if total == 0:
            return 0.0
        avg_exec_price = weighted_price / total
        mid = self.book.mid_price()
        if mid is None:
            return 0.0
        direction = 1 if side is Side.BUY else -1
        impact = direction * (avg_exec_price - mid)
        return float(impact)

    def simulate_execution_paths(
        self,
        side: Side,
        quantity: float,
        horizon: float,
        n_paths: int = 50,
    ) -> np.ndarray:
        if self.simulator is None:
            raise RuntimeError("simulate_execution_paths requires a simulator")
        fills = []
        for _ in range(n_paths):
            scenario_book = _clone_book(self.book)
            scenario = self.simulator.simulate(horizon, book=scenario_book)
            fills.append(scenario.add_market_order(side, quantity, horizon))
        return np.asarray(fills, dtype=float)


__all__ = ["ExecutionAnalytics", "ExecutionResult"]
