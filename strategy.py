"""
strategy.py

Single source of truth for the trading rule and the backtest math.
backtest.py and dashboard.py both import from here instead of keeping
their own copies -- this is the fix for the duplication flagged in the
README as known design debt.

Nothing in this file touches SQLite or Streamlit. It only takes a
DataFrame in and returns a DataFrame or numbers out, which makes it easy
to unit test (see test_backtest.py) and easy to reuse anywhere.
"""

import numpy as np
import pandas as pd

COST = 0.001          # transaction cost charged on every position change
TRADING_DAYS = 252    # used to annualize the Sharpe ratio


def build_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long (1) when the short-term trend is above the long-term trend AND
    sentiment isn't negative. Flat (0) otherwise. Deliberately simple.
    """
    trend_up = df["sma_fast"] > df["sma_slow"]
    sentiment_ok = df["sentiment"] >= 0
    df["signal"] = np.where(trend_up & sentiment_ok, 1, 0)
    return df


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns a signal into a position and a return, honestly:
    - position is YESTERDAY's signal (shift(1)), so nothing trades on
      information it couldn't have had yet.
    - every position change is charged COST.
    Shifting happens within each symbol's own group so one ticker's
    history never leaks into another's.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["position"] = df.groupby("symbol")["signal"].shift(1).fillna(0)
    prev_position = df.groupby("symbol")["position"].shift(1).fillna(0)
    df["position_change"] = (df["position"] - prev_position).abs()
    df["strategy_return"] = (
        df["position"] * df["daily_return"] - COST * df["position_change"]
    ).fillna(0)
    return df


def build_and_run(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: signal + backtest in one call."""
    return run_backtest(build_signal(df))


def metrics(returns: pd.Series) -> dict:
    """
    Total return, annualized Sharpe, max drawdown, hit rate.
    Returns a dict so callers can build a metrics table with pd.DataFrame(...).
    """
    returns = returns.fillna(0)

    if len(returns) == 0 or returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)

    equity = (1 + returns).cumprod()
    max_dd = (equity / equity.cummax() - 1).min() if len(equity) else 0.0
    total_return = equity.iloc[-1] - 1 if len(equity) else 0.0

    active = returns[returns != 0]
    hit_rate = (active > 0).mean() if len(active) else 0.0

    return {
        "Total return": total_return,
        "Sharpe": sharpe,
        "Max drawdown": max_dd,
        "Hit rate": hit_rate,
    }