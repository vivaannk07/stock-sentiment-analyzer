import os
import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta, timezone
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

# NewsAPI free tier only serves the last ~30 days. Set True whenever a caller
# requests a longer window than we can actually deliver, so the UI can warn.
NEWS_TRUNCATED = False


def fetch_price_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Return OHLCV DataFrame for the given ticker and period. Retries on rate limits."""
    import time
    
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                # Empty response — retry once after a short wait
                if attempt < 2:
                    time.sleep(2)
                    continue
                return df
            
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))  # Wait 3s, then 6s
                continue
            # Final attempt failed — return empty DataFrame instead of crashing
            print(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()


def fetch_news_headlines(ticker: str, days_back: int = 30) -> list[dict]:
    """
    Fetch news headlines for a ticker from NewsAPI.
    Uses company name lookup for better Indian stock results.
    Returns a list of dicts with keys: title, description, publishedAt, url, source.
    """
    if not NEWS_API_KEY:
        return []

    # Free tier caps the lookback at 30 days; clip and flag if more was asked for.
    global NEWS_TRUNCATED
    if days_back > 30:
        NEWS_TRUNCATED = True
        days_back = 30
    else:
        NEWS_TRUNCATED = False

    # Look up the company name for a smarter search query
    company_name = get_company_name(ticker)
    
    # Build query: search for the company name OR the bare ticker without .NS suffix
    bare_ticker = ticker.replace(".NS", "").replace(".BO", "")
    query = f'"{company_name}" OR "{bare_ticker}"'

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
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
        if response.status_code != 200:
            print(f"NewsAPI error {response.status_code}: {response.text[:200]}")
            return []
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


def get_yf_enrichment(ticker: str) -> dict:
    """
    Previously pulled the slow/rate-limited yf.Ticker(...).info call. That call
    was the main source of yfinance rate-limit stalls, so as of Phase 1 we no
    longer make it: industry/market cap are dropped and price comes solely from
    the already-fetched price_df. Kept as a stable no-op so the cached wrapper
    and call sites in app.py don't need to change.
    """
    return {"industry": "N/A", "market_cap": None, "fallback_price": None}


@st.cache_data(ttl=86400, show_spinner=False)  # 24h — fundamentals barely move intraday
def fetch_market_stats(ticker: str) -> dict:
    """
    Slow-moving fundamentals for the KPI row: market cap, trailing P/E,
    52-week range and latest volume.

    This is the one place we still hit the rate-limited yf.Ticker(...).info
    endpoint (get_yf_enrichment is a deliberate no-op — see above), so the whole
    thing is wrapped in try/except and every field defaults to "N/A". A failed or
    throttled call therefore degrades gracefully instead of crashing the view.
    The 24h TTL keeps us off that endpoint on repeat views of the same ticker.
    """
    stats = {
        "market_cap": "N/A",
        "pe_ratio": "N/A",
        "week52_low": "N/A",
        "week52_high": "N/A",
        "volume": "N/A",
    }
    try:
        info = yf.Ticker(ticker).info or {}

        market_cap = info.get("marketCap")
        if market_cap:
            stats["market_cap"] = int(market_cap)

        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe:
            stats["pe_ratio"] = round(float(pe), 2)

        low = info.get("fiftyTwoWeekLow")
        high = info.get("fiftyTwoWeekHigh")
        if low:
            stats["week52_low"] = float(low)
        if high:
            stats["week52_high"] = float(high)

        volume = info.get("volume") or info.get("regularMarketVolume") or info.get("averageVolume")
        if volume:
            stats["volume"] = int(volume)
    except Exception as e:
        # Network error, rate limit, or a missing/renamed field — keep the "N/A"
        # defaults so the KPI row still renders.
        print(f"Market stats fetch failed for {ticker}: {e}")

    return stats


def get_ticker_info(
    ticker: str,
    price_df: pd.DataFrame = None,
    enrichment: dict = None,
) -> dict:
    """
    Return basic company info for display.

    Reads name + sector from the local Nifty 50 CSV (instant, no network),
    current price from the already-fetched price_df, and industry / market cap
    from the optional `enrichment` dict (which the caller is responsible for
    caching — see app.cached_yf_enrichment).
    """
    company_name = get_company_name(ticker)

    sector = "N/A"
    try:
        df = pd.read_csv(_NIFTY50_CSV)
        match = df[df["ticker"] == ticker]
        if not match.empty:
            sector = match.iloc[0]["sector"]
    except Exception:
        pass

    current_price = None
    if price_df is not None and not price_df.empty:
        current_price = float(price_df["Close"].iloc[-1])

    enrichment = enrichment or {}
    industry = enrichment.get("industry", "N/A")
    market_cap = enrichment.get("market_cap")
    if current_price is None:
        current_price = enrichment.get("fallback_price")

    return {
        "name": company_name,
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
        "current_price": current_price,
    }