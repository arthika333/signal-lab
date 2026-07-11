import sqlite3
import pandas as pd

DB_PATH="data/market.db"
TICKERS=["AAPL","MSFT","GOOGL","AMZN","META"]
SMA_FAST=5
SMA_SLOW=20

def load_prices(conn):
    df=pd.read_sql_query("SELECT symbol,date,open,high,low,close,volume FROM daily_prices",
    conn)
    df["date"]=pd.to_datetime(df["date"])
    return df.sort_values(by=["symbol","date"]).reset_index(drop=True)


def add_price_features(df):
    g=df.groupby(by=["symbol"])
    df["daily_return"]=g["close"].pct_change()
    df["sma_fast"]=g["close"].transform(lambda s: s.rolling(SMA_FAST).mean())
    df["sma_slow"]=g["close"].transform(lambda s: s.rolling(SMA_SLOW).mean())
    return df

def load_daily_sentiment(conn):
    placeholders=",".join(f"'{t}'" for t in TICKERS)
    news=pd.read_sql_query(f"""
                           SELECT a.time_published, t.ticker, t.ticker_sentiment_score, t.relevance_score
                           FROM news_articles a
                           JOIN news_ticker_sentiment t ON a.url=t.url
                           WHERE t.ticker IN ({placeholders})
                           """, conn)
    if news.empty:
        return pd.DataFrame(columns=["time_published","ticker","ticker_sentiment_score","relevance_score"])
    
    news["date"]=pd.to_datetime(news["time_published"].str[:8],format="%Y%m%d")
    news["weighted"]=news["ticker_sentiment_score"]*news["relevance_score"]
    daily=news.groupby(by=["ticker","date"])["weighted"].mean().reset_index().rename(columns={"ticker":"symbol","weighted":"sentiment"})
    return daily

def main():
    conn=sqlite3.connect(DB_PATH)
    df=add_price_features(load_prices(conn))
    sentiment=load_daily_sentiment(conn)
    df=pd.merge(df,sentiment,on=["symbol","date"],how="left")
    df["sentiment"]=df["sentiment"].fillna(0.0)
    df.to_sql("features",conn,if_exists="replace",index=False)
    conn.commit()
    conn.close()
    print(f"features: {len(df)} rows, {df['symbol'].nunique()} tickers")
    print(df.tail()[["symbol","date","close","daily_return","sma_fast","sma_slow","sentiment"]])

if __name__=="__main__":
    main()