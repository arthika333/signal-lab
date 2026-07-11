import os
import sqlite3
import json
import time
from datetime import datetime,timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DB_PATH="data/market.db"
MODEL="llama-3.1-8b-instant"
API_KEY=os.getenv("GROQ_API_KEY")
client=OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")

SYSTEM_PROMPT = """
You are a financial news analyst.

Read one financial news article and return a structured assessment.

Respond with ONLY a valid JSON object.
Do not include markdown.
Do not include code fences.
Do not include explanations outside the JSON.

Use exactly this schema:

{
  "event_type": string,
  "theme": string,
  "directional_lean": "bullish" | "bearish" | "neutral",
  "confidence": number between 0 and 1,
  "rationale": string
}

Rules:

- event_type must be one of:
  earnings,
  product,
  regulatory,
  legal,
  analyst_rating,
  macro,
  partnership,
  other

- theme should be a short phrase (2-6 words).

- directional_lean should describe the likely impact on the mentioned company.

- confidence should reflect certainty in the classification.

- rationale should be exactly one sentence.

Return ONLY the JSON object.
"""

def get_unprocessed_articles(conn):
    return conn.execute("""
                        SELECT a.url, a.title, a.summary, t.ticker
                        FROM news_articles a
                        JOIN news_ticker_sentiment t ON a.url = t.url
                        WHERE t.ticker IN('AAPL','MSFT','GOOGL','AMZN','META')
                        AND NOT EXISTS (
                        SELECT 1
                        FROM news_llm_analysis l
                        WHERE l.url = a.url
                        AND l.ticker = t.ticker
                        )
                        LIMIT 50
                        """).fetchall()

def analyze(title, summary, ticker):
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"Company ticker: {ticker}\nHeadline: {title}\nSummary: {summary}"}
    ]
    try:
        response=client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=250, temperature=0.2)
        text=response.choices[0].message.content
        start=text.find("{"); end=text.rfind("}")+1
        return json.loads(text[start:end])
    except Exception as e:
        print(f"  analyze failed: {e}")
        return {"event_type":"other","theme":"unknown",
                "directional_lean":"neutral","confidence":0.0,"rationale":"error"}

def create_table(conn):
    conn.execute("""
                    CREATE TABLE IF NOT EXISTS news_llm_analysis (
                    url text,
                    ticker text,
                    model text,
                    event_type text,
                    theme text,
                    directional_lean text,
                    confidence real,
                    rationale text,
                    analyzed_at text,
                    PRIMARY KEY (url, ticker)
                    )
                 """)

def main():
    conn=sqlite3.connect(DB_PATH)
    create_table(conn)
    rows=get_unprocessed_articles(conn)
    for url,title,summary,ticker in rows:
        result=analyze(title or "", summary or "", ticker or "")
        conn.execute("""
                        INSERT OR REPLACE INTO news_llm_analysis
                        (url, ticker, model, event_type, theme, directional_lean, confidence, rationale, analyzed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
        (
        url,
        ticker,
        MODEL,
        result.get("event_type"),
        result.get("theme"),
        result.get("directional_lean"),
        float(result.get("confidence")),
        result.get("rationale"),
        datetime.now(timezone.utc).isoformat()
        ))

        time.sleep(5)

    conn.commit()    
    conn.close()
    print("LLM Analysis complete.")

if __name__=="__main__":
    main()