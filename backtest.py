import sqlite3
import pandas as pd

from strategy import build_and_run, metrics

DB_PATH = "data/market.db"


def load_features(conn):
    df = pd.read_sql_query("SELECT * FROM features", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def main():
    conn = sqlite3.connect(DB_PATH)
    df = build_and_run(load_features(conn))

    port = df.groupby("date")["strategy_return"].mean().fillna(0)
    bench = df.groupby("date")["daily_return"].mean().fillna(0)

    print("\nper ticker strategy:")
    for sym, sub in df.groupby("symbol"):
        m = metrics(sub["strategy_return"])
        print(
            f"{sym}: return={m['Total return']:+.2%}, "
            f"sharpe={m['Sharpe']:.2f}, "
            f"maxDD={m['Max drawdown']:.2%}, "
            f"hit={m['Hit rate']:.0%}"
        )

    s = metrics(port)
    b = metrics(bench)
    print("\nequal weight portfolio")
    print(
        f"strategy: return={s['Total return']:+.2%}, sharpe={s['Sharpe']:.2f}, "
        f"maxdd={s['Max drawdown']:.2%}, hit={s['Hit rate']:.0%}"
    )
    print(
        f"buy and hold: return={b['Total return']:+.2%}, sharpe={b['Sharpe']:.2f}, "
        f"maxdd={b['Max drawdown']:.2%}"
    )

    out = pd.DataFrame({
        "date": port.index,
        "strategy_equity": (1 + port).cumprod().values,
        "benchmark_equity": (1 + bench).cumprod().values,
    })
    out.to_sql("backtest_equity", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"\nsaved {len(out)} rows to backtest_equity")


if __name__ == "__main__":
    main()