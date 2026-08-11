"""
signal-lab dashboard
Run with: streamlit run dashboard.py
"""

import sqlite3
import pandas as pd
import streamlit as st

from strategy import build_and_run, metrics

DB_PATH = "data/market.db"

st.set_page_config(page_title="signal-lab", layout="wide")


@st.cache_data
def load(query):
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


try:
    features = load("SELECT * FROM features")
    equity = load("SELECT * FROM backtest_equity")
    llm = load("""
        SELECT l.ticker, l.event_type, l.directional_lean, l.rationale,
               a.title, a.time_published, t.ticker_sentiment_label AS av_label
        FROM news_llm_analysis l
        JOIN news_articles a ON l.url = a.url
        JOIN news_ticker_sentiment t ON l.url = t.url AND l.ticker = t.ticker
        WHERE l.rationale NOT IN ('error', 'parse error')
        ORDER BY a.time_published DESC
    """)
except Exception as e:
    st.error(f"Run the pipeline first (fetch → extract → features → backtest).\n\n{e}")
    st.stop()

features["date"] = pd.to_datetime(features["date"])
equity["date"] = pd.to_datetime(equity["date"])
features = build_and_run(features)

n_days = features["date"].nunique()
n_sent = int((features["sentiment"] != 0).sum())

st.title("signal-lab")
st.markdown(
    "An LLM-augmented news-to-signal research pipeline. Financial news and prices in, "
    "structured judgments out, backtested without lookahead bias and with transaction costs."
)
st.markdown(
    f"**This is a research framework, not a trading system.** {n_days} trading days, "
    f"{features['symbol'].nunique()} tickers, one market regime — far too little data to "
    "establish that any strategy works. No claim of alpha is made."
)

st.divider()

port = features.groupby("date")["strategy_return"].mean().fillna(0)
bench = features.groupby("date")["daily_return"].mean().fillna(0)

st.subheader("Strategy vs. buy-and-hold")
comparison = pd.DataFrame({"Strategy": metrics(port), "Buy and hold": metrics(bench)}).T
st.dataframe(
    comparison.style.format(
        {"Total return": "{:.2%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.2%}", "Hit rate": "{:.0%}"}
    ),
    use_container_width=True,
)

curve = equity.set_index("date")[["strategy_equity", "benchmark_equity"]]
curve.columns = ["Strategy", "Buy and hold"]
st.line_chart(curve)
st.caption(
    "Value of $1, equal-weighted across the five tickers. The strategy holds cash when "
    "the short-term trend is down or sentiment is negative — trading some return for less risk."
)

st.subheader("Per ticker")
per_ticker = pd.DataFrame(
    {sym: metrics(sub["strategy_return"]) for sym, sub in features.groupby("symbol")}
).T
st.dataframe(
    per_ticker.style.format(
        {"Total return": "{:.2%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.2%}", "Hit rate": "{:.0%}"}
    ),
    use_container_width=True,
)
st.caption(
    "The same rule produced very different outcomes across five large-cap tech stocks. "
    "On a sample this small, that spread is noise, not skill."
)

symbol = st.selectbox("Inspect a ticker", sorted(features["symbol"].unique()))
sub = features[features["symbol"] == symbol].set_index("date")
st.line_chart(sub[["close", "sma_fast", "sma_slow"]])
st.caption("Close price with 5-day and 20-day moving averages.")

st.subheader("LLM news extraction")
st.markdown(
    "Each article is passed to an LLM constrained to a strict JSON schema, which returns "
    "an event type, directional lean, and a one-line rationale. Alpha Vantage's own "
    "sentiment label is shown alongside for comparison — neither is ground truth."
)
if llm.empty:
    st.info("No LLM analysis yet. Run `python llm_extract.py`.")
else:
    st.dataframe(
        llm[["ticker", "title", "event_type", "directional_lean", "av_label", "rationale"]],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Methodology and limitations")
st.markdown(
    f"""
**Avoiding lookahead bias.** A signal from today's close cannot be traded today. The
position held on day *t* is the signal from day *t−1* (`signal.shift(1)`, applied within
each ticker). Every position change is charged 0.1%.

**What this does not show.**
- {n_days} trading days, one market regime. Illustrative, not evidence of edge.
- Only {n_sent} of {len(features)} rows carry non-zero sentiment — the free news API covers
  a recent window only. For most days the strategy runs on moving averages alone, so the
  sentiment layer demonstrates the pipeline rather than a statistically powered factor.
- Fixed transaction cost, no slippage, no shorting, no position sizing, daily closes only.
- Five hand-picked tickers, which is itself a selection bias.
"""
)

st.caption("Data: Alpha Vantage. LLM: open-weight model via Groq. Not investment advice.")