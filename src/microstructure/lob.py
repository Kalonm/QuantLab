"""Limit order book simulation toolkit."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    CANCEL = "CANCEL"


@dataclass
class Order:
    order_id: str
    side: Side
    quantity: float
    price: Optional[float]
    timestamp: float
    order_type: OrderType = OrderType.LIMIT


class LimitOrderBook:
    """Simplified price-time priority limit order book."""

    def __init__(self):
        self.bids: Dict[float, Deque[Order]] = defaultdict(deque)
        self.asks: Dict[float, Deque[Order]] = defaultdict(deque)
        self.event_log: List[Tuple[float, str]] = []
        self._order_lookup: Dict[str, Tuple[Side, float]] = {}

    def _book_for(self, side: Side) -> Dict[float, Deque[Order]]:
        return self.bids if side is Side.BUY else self.asks

    def add_limit_order(self, order: Order) -> None:
        if order.price is None:
            raise ValueError("Limit orders require a price")
        book = self._book_for(order.side)
        book[order.price].append(order)
        self._order_lookup[order.order_id] = (order.side, order.price)
        self.event_log.append((order.timestamp, f"ADD {order.side} {order.quantity}@{order.price}"))

    def _match_against_book(self, side: Side, quantity: float) -> float:
        book = self.asks if side is Side.BUY else self.bids
        filled = 0.0
        level_prices = sorted(book.keys()) if side is Side.BUY else sorted(book.keys(), reverse=True)
        for price in level_prices:
            level = book[price]
            while level and quantity > 0:
                resting = level[0]
                trade_qty = min(quantity, resting.quantity)
                quantity -= trade_qty
                resting.quantity -= trade_qty
                filled += trade_qty
                self.event_log.append(
                    (
                        resting.timestamp,
                        f"FILL {side} {trade_qty}@{price} against {resting.order_id}",
                    )
                )
                if resting.quantity <= 0:
                    level.popleft()
                    self._order_lookup.pop(resting.order_id, None)
            if not level:
                book.pop(price)
            if quantity <= 0:
                break
        return filled

    def add_market_order(self, side: Side, quantity: float, timestamp: float) -> float:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        filled = self._match_against_book(side, quantity)
        self.event_log.append((timestamp, f"MARKET {side} {quantity} FILLED {filled}"))
        return filled

    def cancel_order(self, order_id: str, timestamp: float) -> bool:
        info = self._order_lookup.get(order_id)
        if not info:
            return False
        side, price = info
        book = self._book_for(side)
        level = book.get(price)
        if not level:
            return False
        for idx, existing in enumerate(level):
            if existing.order_id == order_id:
                level.rotate(-idx)
                level.popleft()
                level.rotate(idx)
                if not level:
                    book.pop(price)
                self._order_lookup.pop(order_id, None)
                self.event_log.append((timestamp, f"CANCEL {order_id}"))
                return True
        return False

    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        if not self.bids:
            return None
        price = max(self.bids.keys())
        quantity = sum(order.quantity for order in self.bids[price])
        return price, quantity

    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        if not self.asks:
            return None
        price = min(self.asks.keys())
        quantity = sum(order.quantity for order in self.asks[price])
        return price, quantity

    def mid_price(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if not best_bid or not best_ask:
            return None
        return (best_bid[0] + best_ask[0]) / 2.0

    def spread(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if not best_bid or not best_ask:
            return None
        return best_ask[0] - best_bid[0]

    def depth(self, side: Side, levels: int = 5) -> List[Tuple[float, float]]:
        book = self._book_for(side)
        if not book:
            return []
        prices = sorted(book.keys(), reverse=True) if side is Side.BUY else sorted(book.keys())
        result = []
        for price in prices[:levels]:
            qty = sum(order.quantity for order in book[price])
            result.append((price, qty))
        return result

    def snapshot(self) -> Dict[str, List[Tuple[float, float]]]:
        return {
            "bids": self.depth(Side.BUY, levels=len(self.bids)),
            "asks": self.depth(Side.SELL, levels=len(self.asks)),
        }


class OrderFlowSimulator:
    """Generate random order flow for scenario testing."""

    def __init__(
        self,
        limit_rate: float,
        market_rate: float,
        cancel_rate: float,
        price_grid: Iterable[float],
        seed: Optional[int] = None,
    ):
        if min(limit_rate, market_rate, cancel_rate) < 0:
            raise ValueError("Rates must be non-negative")
        self.limit_rate = limit_rate
        self.market_rate = market_rate
        self.cancel_rate = cancel_rate
        self.price_grid = list(price_grid)
        if not self.price_grid:
            raise ValueError("price_grid cannot be empty")
        self.rng = np.random.default_rng(seed)
        self._order_counter = 0

    def _new_order_id(self) -> str:
        self._order_counter += 1
        return f"O{self._order_counter:06d}"

    def step(self, book: LimitOrderBook, timestamp: float) -> None:
        total_rate = self.limit_rate + self.market_rate + self.cancel_rate
        if total_rate <= 0:
            return
        u = self.rng.uniform()
        threshold_limit = self.limit_rate / total_rate
        threshold_market = (self.limit_rate + self.market_rate) / total_rate
        if u < threshold_limit:
            side = Side.BUY if self.rng.uniform() < 0.5 else Side.SELL
            price = self.rng.choice(self.price_grid)
            quantity = float(self.rng.integers(1, 5))
            order = Order(
                order_id=self._new_order_id(),
                side=side,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
            )
            book.add_limit_order(order)
        elif u < threshold_market:
            side = Side.BUY if self.rng.uniform() < 0.5 else Side.SELL
            quantity = float(self.rng.integers(1, 5))
            book.add_market_order(side, quantity, timestamp)
        else:
            if not book._order_lookup:
                return
            order_id = self.rng.choice(list(book._order_lookup.keys()))
            book.cancel_order(order_id, timestamp)

    def simulate(self, horizon: float, book: Optional[LimitOrderBook] = None) -> LimitOrderBook:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if book is None:
            book = LimitOrderBook()
        t = 0.0
        while t < horizon:
            total_rate = self.limit_rate + self.market_rate + self.cancel_rate
            if total_rate <= 0:
                break
            wait = self.rng.exponential(1.0 / total_rate)
            t += wait
            if t > horizon:
                break
            self.step(book, t)
        return book


__all__ = [
    "LimitOrderBook",
    "Order",
    "OrderType",
    "OrderFlowSimulator",
    "Side",
]
