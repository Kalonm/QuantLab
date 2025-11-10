from src.microstructure.execution import ExecutionAnalytics
from src.microstructure.lob import LimitOrderBook, Order, OrderFlowSimulator, Side


def build_book():
    book = LimitOrderBook()
    book.add_limit_order(Order(order_id="B1", side=Side.BUY, quantity=5, price=99.5, timestamp=0.0))
    book.add_limit_order(Order(order_id="B2", side=Side.BUY, quantity=5, price=99.0, timestamp=0.1))
    book.add_limit_order(Order(order_id="S1", side=Side.SELL, quantity=4, price=100.5, timestamp=0.2))
    book.add_limit_order(Order(order_id="S2", side=Side.SELL, quantity=3, price=101.0, timestamp=0.3))
    return book


def test_depth_and_spread():
    book = build_book()
    best_bid = book.get_best_bid()
    best_ask = book.get_best_ask()
    assert best_bid[0] == 99.5
    assert best_ask[0] == 100.5
    assert book.spread() == 1.0


def test_execution_analytics_fill_probability():
    book = build_book()
    simulator = OrderFlowSimulator(
        limit_rate=4.0,
        market_rate=1.0,
        cancel_rate=0.5,
        price_grid=[99.0, 99.5, 100.5, 101.0],
        seed=123,
    )
    analytics = ExecutionAnalytics(book, simulator)
    result = analytics.estimate_fill_probability(Side.BUY, quantity=4, horizon=2.0, n_paths=10)
    assert 0.0 <= result.fill_probability <= 1.0
    impact = analytics.estimate_price_impact(Side.BUY, quantity=4)
    assert isinstance(impact, float)
