import sqlite3
import pandas as pd
import numpy as np
DB_PATH="data/market.db"
COST=0.001
TRADING_DAYS=252

def load_features(conn):
    df=pd.read_sql_query("SELECT * FROM features",conn)
    df["date"]=pd.to_datetime(df["date"])
    return df.sort_values(["symbol","date"]).reset_index(drop=True)

def build_signal(df):
    trend_up=df["sma_fast"]>df["sma_slow"]
    sentiment_ok=df["sentiment"]>=0
    df["signal"]=np.where(trend_up & sentiment_ok,1,0)
    return df

def run_backtest(df):
    df["position"]=df.groupby("symbol")["signal"].shift(1).fillna(0)
    df["prev_position"]=df.groupby("symbol")["position"].shift(1).fillna(0)
    df["position_change"]=(df["position"]-df["prev_position"]).abs()
    df["strategy_return"]=(df["position"]*df["daily_return"]-(COST*df["position_change"])).fillna(0)
    return df

def metrics(returns):
    returns=returns.fillna(0)
    if len(returns)==0 or returns.std()==0:
        sharpe=0.0
    else:
        sharpe=(returns.mean()/returns.std())*np.sqrt(TRADING_DAYS)
    
    equity=(1+returns).cumprod()
    max_dd=(equity/equity.cummax()-1).min()
    total_return=equity.iloc[-1]-1 if len(equity) else 0.0
    active=returns[returns!=0]
    hit_rate=(active>0).mean() if len (active) else 0.0
    return total_return, sharpe, max_dd, hit_rate

def main():
    conn=sqlite3.connect(DB_PATH)
    df=run_backtest(build_signal(load_features(conn)))
    port=df.groupby("date")["strategy_return"].mean().fillna(0)
    bench=df.groupby("date")["daily_return"].mean().fillna(0)
    print("\nper ticker strategy:")
    for sym,sub in df.groupby("symbol"):
        tr,sh,dd,hr=metrics(sub["strategy_return"])
        print(f"{sym}: return={tr:+.2%}, sharpe={sh:.2f}, maxDD={dd:.2%}, hit={hr:.0%}")
    s=metrics(port)
    b=metrics(bench)
    print("\nequal weight portfolio")
    print(f"strategy: return={s[0]:+.2%}, sharpe={s[1]:.2f}, maxdd={s[2]:.2%}, hit={s[3]:.0%}")
    print(f"by and hold: return={b[0]:+.2%}, sharpe={b[1]:.2f}, maxdd={b[2]:.2%}")
    out=pd.DataFrame({
        "date":port.index,
        "strategy_equity":(1+port).cumprod().values,
        "benchmark_equity":(1+bench).cumprod().values
    })
    out.to_sql("backtest_equity",conn,if_exists="replace",index=False)
    conn.commit()
    conn.close()
    print(f"\nsaved {len(out)} rows to backtest_equity")

if __name__=="__main__":
    main()        