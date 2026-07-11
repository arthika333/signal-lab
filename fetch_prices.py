import os
import sqlite3
import requests
from dotenv import load_dotenv
import time

load_dotenv()
API_KEY=os.getenv("ALPHAVANTAGE_API_KEY")
TICKERS=["AAPL",
         "MSFT",
         "GOOGL",
         "AMZN",
         "META",
         ]
DB_PATH="data/market.db"
BASE_URL="https://www.alphavantage.co/query"

def fetch_daily(symbol):
    params={
        "function":"TIME_SERIES_DAILY",
        "symbol":symbol,
        "outputsize":"compact",
        "apikey":API_KEY
    }
    resp=requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data=resp.json()

    if "Time Series (Daily)" not in data:
        raise RuntimeError(f"unexpected response from api : {data}")
    
    return data["Time Series (Daily)"]

def save_to_sqlite(symbol,series):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn=sqlite3.connect(DB_PATH)
    cur=conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_prices(
        symbol text not null,
        date text not null,
        open real,
        high real,
        low real,
        close real,
        volume integer,
        primary key(symbol, date)
        )
        """
    )
    rows=[
        (
            symbol, 
            date,
            float(values["1. open"]),
            float(values["2. high"]),
            float(values["3. low"]),
            float(values["4. close"]),
            int(values["5. volume"])
        )
        for date,values in series.items()
    ]

    cur.executemany(
    """
    INSERT OR REPLACE INTO daily_prices
    (symbol, date, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    rows
    )
    conn.commit()
    conn.close()
    return len(rows)

def main():
    if not API_KEY:
        raise SystemExit("set ALPHAVANTAGE_API_KEY in .env file")
    for symbol in TICKERS:
        series=fetch_daily(symbol)
        count=save_to_sqlite(symbol, series)
        print(f"saved {count} rows for {symbol} to {DB_PATH}")
        time.sleep(15)  
        
if __name__=="__main__":
    main()