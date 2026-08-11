# Signal Lab

![Tests](https://github.com/arthika333/signal-lab/actions/workflows/tests.yml/badge.svg)

An LLM-augmented news-to-signal research pipeline for equities. It ingests market data and financial news, uses an LLM to extract structured judgments from unstructured headlines, combines those with technical indicators into a trading signal, and backtests the result **without lookahead bias and with transaction costs.**

**[Live dashboard →](http://localhost:8501/)**

<img width="889" height="266" alt="image" src="https://github.com/user-attachments/assets/37380028-d357-4342-9fdf-b127ba92473e" />
---

## Why this exists

Most student trading projects quietly cheat: they train on the future, ignore transaction costs, and present a beautiful equity curve that could never have been traded. This project is the opposite. **It makes no claim of alpha.** It is a demonstration of a rigorous research pipeline and an honest evaluation of what the data can and cannot support — see [Limitations](#limitations) before drawing any conclusions from the numbers below.

---

## Results

Equal-weight portfolio across five tickers (AAPL, MSFT, GOOGL, AMZN, META), ~102 trading days.

| | Strategy | Buy & Hold |
|---|---|---|
| Total return | +5.62% | **+6.18%** |
| Sharpe ratio | **1.34** | 0.75 |
| Max drawdown | **−6.40%** | −14.59% |
| Hit rate | 47% | — |

The strategy slightly underperformed buy-and-hold on raw return, but with roughly **half the drawdown** and nearly **double the risk-adjusted return**. That's the expected signature of a trend-following rule that sits in cash during weak stretches — it trades some upside for a lot less risk.

<details>
<summary>Per-ticker breakdown</summary>

| Ticker | Return | Sharpe | Max DD | Hit rate |
|---|---|---|---|---|
| GOOGL | +20.85% | 2.34 | −4.88% | 53% |
| AAPL | +12.16% | 1.79 | −8.59% | 52% |
| AMZN | +9.38% | 1.29 | −12.91% | 49% |
| MSFT | −2.50% | −0.20 | −13.81% | 40% |
| META | −11.73% | −1.23 | −17.49% | 45% |

The same rule produced wildly different outcomes across five similar large-cap tech stocks (Sharpe 2.34 to −1.23). On a sample this small, that spread is noise, not evidence of skill.
</details>

On 102 days and one market regime, none of this is statistically meaningful — see [Limitations](#limitations).

---

## Architecture

```
Alpha Vantage API ──┬──> daily_prices ─────────────┐
                     │                              │
                     └──> news_articles ────────────┤
                          news_ticker_sentiment ──┐  │
                                                  │  │
                     Groq (LLM) ──> news_llm_analysis
                                                  │  │
                                                  ▼  ▼
                                            build_features.py
                                                   │
                                                   ▼
                                               features
                                                   │
                                                   ▼
                                     strategy.py  ◄── backtest.py, dashboard.py
                                                   │
                                                   ▼
                                           backtest_equity
                                                   │
                                                   ▼
                                            dashboard.py (Streamlit)
```

Everything persists in a single SQLite file, `data/market.db`.

### Pipeline

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `fetch_prices.py` | Alpha Vantage `TIME_SERIES_DAILY` | `daily_prices` |
| 2 | `fetch_news.py` | Alpha Vantage `NEWS_SENTIMENT` | `news_articles`, `news_ticker_sentiment` |
| 3 | `llm_extract.py` | news tables | `news_llm_analysis` |
| 4 | `build_features.py` | prices + news | `features` |
| 5 | `backtest.py` | `features` (via `strategy.py`) | `backtest_equity` |
| 6 | `dashboard.py` | all of the above | Streamlit UI |

`strategy.py` holds the trading rule and backtest math (signal generation, position shifting, transaction costs, metrics). Both `backtest.py` and `dashboard.py` import from it, so the two can never silently disagree.

### Database schema

- **`daily_prices`** — `symbol, date, open, high, low, close, volume`. PK `(symbol, date)`.
- **`news_articles`** — one row per article. PK `url`.
- **`news_ticker_sentiment`** — one row per (article, ticker), since one article can mention several companies with different sentiment. PK `(url, ticker)`, FK → `news_articles`.
- **`news_llm_analysis`** — one row per (article, ticker): the LLM's structured judgment. PK `(url, ticker)`.
- **`features`** — derived: one row per (symbol, date) with returns, moving averages, and daily sentiment.
- **`backtest_equity`** — derived: the equity curve, strategy vs. benchmark.

---

## Avoiding lookahead bias

A signal computed from today's closing price cannot be traded today — you only know the close after the market shuts. So the position held on day *t* is the signal generated on day *t−1*:

```python
df["position"] = df.groupby("symbol")["signal"].shift(1).fillna(0)
```

The shift happens within each ticker's own group, so one stock's history never leaks into another's. Every position change is charged a 0.1% transaction cost. This property is machine-checked, not just asserted — see `test_backtest.py::test_no_lookahead`.

---

## Design decisions

- **Two news tables, not one.** An article-ticker relationship is one-to-many, so article-level fields and per-ticker sentiment live in separate tables joined on `url`.
- **Idempotent writes.** Composite primary keys + `INSERT OR REPLACE` mean re-running any script never duplicates rows.
- **LLM calls are skipped once done, not repeated.** `llm_extract.py` selects only unanalyzed (url, ticker) pairs, so re-runs cost nothing.
- **Structured LLM output.** The system prompt constrains the model to a strict JSON schema, so its output lands directly in a relational table. Parse failures fall back to a safe default row instead of crashing the run.
- **The dashboard reads; it never recomputes independently.** It calls the same `strategy.py` functions as the backtest — one source of truth.

---

## Limitations

- **The sample is far too small.** ~102 trading days, 5 tickers, one market regime. Results are illustrative, not evidence of edge.
- **Sentiment is sparse.** Only ~15 of 510 feature rows carry non-zero sentiment — the free-tier news API covers a recent window only, while price history spans ~100 days. For most days the strategy runs on the moving-average rule alone; the LLM/sentiment layer demonstrates the *pipeline*, not a statistically powered factor.
- **The per-ticker spread is noise**, not skill — see the table above.
- **Simplifications:** fixed transaction cost, no slippage, no shorting, no position sizing, daily closes only, no survivorship-bias handling, and five hand-picked tickers (itself a selection bias).
- **No claim of alpha is made.** The deliverable is an honest, lookahead-free, cost-aware research pipeline — not a profitable trading strategy.

---

## Setup

### Requirements
- Python 3.11+
- Free API keys: [Alpha Vantage](https://www.alphavantage.co/support/#api-key), [Groq](https://console.groq.com)

### Install

```bash
git clone https://github.com/arthika333/signal-lab.git
cd signal-lab

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then fill in your keys
```

### Run the pipeline

```bash
python fetch_prices.py
python fetch_news.py
python llm_extract.py
python build_features.py
python backtest.py
```

### Run the dashboard

```bash
streamlit run dashboard.py
```

### Run the tests

```bash
pytest -v
```

---

## Tech stack

Python · SQLite · pandas · NumPy · Streamlit · Groq (Llama 3.1 8B, OpenAI-compatible API) · pytest · GitHub Actions

---

## Roadmap / possible extensions

- Extract a shared evaluation harness for the LLM layer (hand-labeled accuracy check).
- Walk-forward / out-of-sample validation instead of a single in-sample backtest.
- Longer price history (`outputsize=full`) for a statistically meaningful sample.
- Scheduled daily refresh via GitHub Actions.

*More tickers, more indicators, and fancier models are deliberately **not** on this list — they'd add surface area without addressing the actual constraint (sample size and news coverage), which runs counter to the project's point.*

---

## Notes
Not investment advice.
