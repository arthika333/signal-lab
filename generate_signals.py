import sqlite3
import pandas as pd

DB_PATH="data/market.db"
conn=sqlite3.connect(DB_PATH)
query="""SELECT ticker, avg(ticker_sentiment_score*relevance_score) as weighted_sentiment, count(*) as mentions
FROM news_ticker_sentiment
WHERE ticker IN ('AAPL','MSFT','GOOGL','AMZN','META')
group by ticker order by mentions desc
"""
df=pd.read_sql_query(query, conn)
print("sentiment signal ranking\n",df)