"""
Tests for backtest.py

Run with: pytest -v

These import directly from backtest.py, so they test the real functions
your pipeline uses — not a copy of the logic.
"""

import pandas as pd
from strategy import build_signal, run_backtest, metrics

def test_no_lookahead():
    """
    Position on day t must equal the SIGNAL from day t-1, never day t's own
    signal. This is the core anti-cheating property of the whole project:
    you can't trade on a signal you couldn't have known yet.
    """
    df = pd.DataFrame({
        "symbol": ["AAPL"] * 4,
        "date": pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
        ),
        "sma_fast": [1, 2, 2, 1],
        "sma_slow": [2, 1, 1, 2],   # trend_up -> False, True, True, False
        "sentiment": [0, 0, 0, 0],  # sentiment_ok -> True every day
        "daily_return": [0.01, 0.01, 0.01, 0.01],
    })

    out = run_backtest(build_signal(df))

    # signal should follow trend_up exactly: 0, 1, 1, 0
    assert list(out["signal"]) == [0, 1, 1, 0]

    # position must be the signal shifted forward by ONE day:
    # day1 position = 0 (no prior day, filled with 0)
    # day2 position = day1's signal = 0
    # day3 position = day2's signal = 1
    # day4 position = day3's signal = 1
    assert list(out["position"]) == [0, 0, 1, 1]


def test_costs_charged_on_position_change():
    """
    Every time the position flips (0->1 or 1->0), a transaction cost must
    be subtracted. With zero market returns, any negative strategy_return
    can only be coming from that cost.
    """
    df = pd.DataFrame({
        "symbol": ["AAPL"] * 3,
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "sma_fast": [2, 1, 2],
        "sma_slow": [1, 2, 1],   # signal flips: 1, 0, 1
        "sentiment": [0, 0, 0],
        "daily_return": [0.0, 0.0, 0.0],
    })

    out = run_backtest(build_signal(df))

    # market moved nothing, so any loss must be pure transaction cost
    assert out["strategy_return"].sum() <= 0
    # and since the position does flip at least once, some cost must exist
    assert out["strategy_return"].sum() < 0

def test_metrics_on_flat_returns():
    """A strategy that never trades (all-zero returns) has zero Sharpe,
    zero drawdown, and zero total return."""
    flat = pd.Series([0.0] * 10)

    result = metrics(flat)

    assert result["Total return"] == 0.0
    assert result["Sharpe"] == 0.0
    assert result["Max drawdown"] == 0.0


def test_metrics_on_empty_series():
    """metrics() must not crash on an empty input (e.g. a ticker with
    no trading days in some edge case)."""
    empty = pd.Series([], dtype=float)

    result = metrics(empty)

    assert result["Total return"] == 0.0
    assert result["Sharpe"] == 0.0
    assert result["Hit rate"] == 0.0