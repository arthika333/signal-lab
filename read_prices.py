import os
import sqlite3
import pandas as pd

conn=sqlite3.connect("data/market.db")

df=pd.read_sql_query("SELECT * FROM daily_prices",
                     conn)
print(df.head())
print(df.info())
df=df.sort_values(by=["symbol","date"])
print(df[["date","close"]].head(10))
df["return"]=df.groupby("symbol")["close"].pct_change()
print(df[["date","close","return"]].head(10))
df["range"]=df["high"]-df["low"]
df["m20"]=df.groupby("symbol")["close"].transform(lambda s: s.rolling(20).mean())