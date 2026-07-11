import os
import json
import sqlite3
from datetime import datetime,timezone
import requests
from dotenv import load_dotenv
import time

load_dotenv()
API_KEY=os.getenv("ALPHAVANTAGE_API_KEY")
TICKERS=["AAPL",
         "MSFT",
         "GOOGL",
         "AMZN",
         "META"
        ]
DB_PATH="data/market.db"
BASE_URL="https://www.alphavantage.co/query"
NEWS_LIMIT=50

def fetch_news(symbol, retries=3):
    for i in range(retries):
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "sort": "LATEST",
                "limit": NEWS_LIMIT,
                "apikey": API_KEY,
            }

            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "feed" not in data:
                raise RuntimeError(data)

            return data["feed"]

        except Exception as e:
            print(f"Retry {i+1}/{retries} failed for {symbol}: {e}")
            time.sleep(25)

    return []

def create_tables(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS news_articles;
        DROP TABLE IF EXISTS news_ticker_sentiment;

        CREATE TABLE IF NOT EXISTS news_articles(
        url text primary key,
        title text,
        source text,
        source_domain text,
        time_published text,
        summary text,
        overall_sentiment_score real,
        overall_sentiment_label text,
        topics text, -- json string
        fetched_at text
        );

        CREATE TABLE IF NOT EXISTS news_ticker_sentiment(
        url text,
        ticker text,
        relevance_score real,
        ticker_sentiment_score real,
        ticker_sentiment_label text,
        primary key (url,ticker),
        foreign key (url) references news_articles(url)
        );
        
        """
    )

def save_news(conn,feed):
    fetched_at=datetime.now(timezone.utc).isoformat()
    article_rows,ticker_rows=[],[]
    for article in feed:
        url=article["url"]
        article_rows.append((
            url,
            article.get("title"),
            article.get("source"),      
            article.get("source_domain"),
            article.get("time_published"),
            article.get("summary"),
            float(article.get("overall_sentiment_score",0.0)),
            article.get("overall_sentiment_label"),
            json.dumps(article.get("topics",[])),
            fetched_at
        ))

        for ticker in article.get("ticker_sentiment",[]):
            ticker_rows.append((
                url,
                ticker.get("ticker"),
                float(ticker.get("relevance_score",0.0)),
                float(ticker.get("ticker_sentiment_score",0.0)),
                ticker.get("ticker_sentiment_label")
            ))

    conn.executemany(
            """
            INSERT OR REPLACE INTO news_articles (
                url,
                title,
                source,
                source_domain,
                time_published,
                summary,
                overall_sentiment_score,
                overall_sentiment_label,
                topics,
                fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            article_rows
        )

    conn.executemany(
            """
            INSERT OR REPLACE INTO news_ticker_sentiment(
                url,
                ticker,
                relevance_score,
                ticker_sentiment_score,
                ticker_sentiment_label
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ticker_rows
        )
    return len(article_rows),len(ticker_rows)
    
def main():
    if not API_KEY:
        raise SystemExit("set ALPHAVANTAGE_API_KEY in .env file")
    conn=sqlite3.connect(DB_PATH)
    create_tables(conn)
    total_articles,total_ticker_rows=0,0
    for symbol in TICKERS:
        feed=fetch_news(symbol)
        article_count,ticker_count=save_news(conn,feed)
        total_articles += article_count
        total_ticker_rows += ticker_count
        print(f"{symbol}: saved {article_count} articles")
        time.sleep(25)  

    conn.commit()
    conn.close()
    print(f"Total: saved {total_articles} articles and {total_ticker_rows} ticker sentiment rows")
    
if __name__=="__main__":
    main()