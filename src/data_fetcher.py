import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Path to the Nifty 50 lookup CSV
_NIFTY50_CSV = Path(__file__).parent.parent / "data" / "nifty50.csv"


def get_company_name(ticker: str) -> str:
    """Look up a company name from the Nifty 50 CSV. Returns ticker if not found."""
    try:
        df = pd.read_csv(_NIFTY50_CSV)
        match = df[df["ticker"] == ticker]
        if not match.empty:
            return match.iloc[0]["company_name"]
    except Exception:
        pass
    return ticker


def get_nifty50_list() -> pd.DataFrame:
    """Return the full Nifty 50 DataFrame for the dropdown."""
    try:
        return pd.read_csv(_NIFTY50_CSV)
    except Exception:
        return pd.DataFrame(columns=["ticker", "company_name", "sector"])

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_price_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Return OHLCV DataFrame for the given ticker and period."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def fetch_news_headlines(ticker: str, days_back: int = 30) -> list[dict]:
    """
    Fetch news headlines for a ticker from NewsAPI.
    Uses company name lookup for better Indian stock results.
    Returns a list of dicts with keys: title, description, publishedAt, url, source.
    """
    if not NEWS_API_KEY:
        return []

    # Look up the company name for a smarter search query
    company_name = get_company_name(ticker)
    
    # Build query: search for the company name OR the bare ticker without .NS suffix
    bare_ticker = ticker.replace(".NS", "").replace(".BO", "")
    query = f'"{company_name}" OR "{bare_ticker}"'

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 100,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "published_at": a.get("publishedAt", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", ""),
            }
            for a in articles
            if a.get("title")
        ]
    except requests.RequestException:
        return []


def get_ticker_info(ticker: str) -> dict:
    """Return basic company info for display."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "industry": "N/A",
                "market_cap": None, "current_price": None}
