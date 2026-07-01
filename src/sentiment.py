import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_headline(text: str) -> float:
    """Return VADER compound score in [-1, 1] for a single string."""
    if not text or not text.strip():
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def score_articles(articles: list[dict]) -> pd.DataFrame:
    """
    Score a list of article dicts (from data_fetcher) and return a DataFrame
    with columns: date, title, source, url, compound, sentiment_label.
    """
    rows = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}".strip()
        compound = score_headline(text)
        label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
        rows.append({
            "date": pd.to_datetime(article.get("published_at", "")).normalize()
            if article.get("published_at") else pd.NaT,
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "compound": compound,
            "sentiment_label": label,
        })

    if not rows:
        return pd.DataFrame(columns=["date", "title", "source", "url", "compound", "sentiment_label"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def daily_sentiment(scored_df: pd.DataFrame, trading_days=None) -> pd.DataFrame:
    """
    Aggregate per-article scores into a daily average compound score.
    Returns DataFrame with columns: date, avg_compound, article_count.

    If `trading_days` is provided (typically the trading-day DatetimeIndex from
    yfinance), the result is reindexed against it so news-free days appear as
    rows with NaN avg_compound and zero article_count. This keeps sentiment bars
    on the chart aligned with the candlesticks.
    """
    empty_cols = ["date", "avg_compound", "article_count"]

    if scored_df.empty:
        if trading_days is not None:
            normalized = pd.to_datetime(trading_days).tz_localize(None).normalize()
            return pd.DataFrame({
                "date": normalized,
                "avg_compound": float("nan"),
                "article_count": 0,
            })
        return pd.DataFrame(columns=empty_cols)

    agg = (
        scored_df.dropna(subset=["date"])
        .groupby("date")
        .agg(avg_compound=("compound", "mean"), article_count=("compound", "count"))
        .reset_index()
        .sort_values("date")
    )

    if trading_days is not None:
        normalized = pd.to_datetime(trading_days).tz_localize(None).normalize().unique()
        agg = (
            agg.set_index("date")
            .reindex(normalized)
            .rename_axis("date")
            .reset_index()
        )
        # Missing trading days stay NaN for avg_compound (chart shows a gap) but
        # are explicitly zero-count.
        agg["article_count"] = agg["article_count"].fillna(0).astype(int)

    return agg


def sentiment_summary(scored_df: pd.DataFrame) -> dict:
    """Return high-level summary stats for the sidebar."""
    if scored_df.empty:
        return {"total": 0, "positive": 0, "neutral": 0, "negative": 0, "avg_compound": 0.0}

    counts = scored_df["sentiment_label"].value_counts()
    return {
        "total": len(scored_df),
        "positive": int(counts.get("positive", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "negative": int(counts.get("negative", 0)),
        "avg_compound": round(float(scored_df["compound"].mean()), 4),
    }
def classify_signal(avg_compound: float) -> dict:
    """
    Convert an average compound score into a Buy/Hold/Sell signal.
    Returns dict with 'signal', 'color', 'emoji' and 'icon' keys.

    Colours track the SENTRA palette (--buy / --hold / --sell); the thresholds
    themselves are unchanged.
    """
    if avg_compound >= 0.2:
        return {"signal": "BUY", "color": "#34E0A1", "emoji": "🟢", "icon": "↑"}
    elif avg_compound <= -0.2:
        return {"signal": "SELL", "color": "#FF5C5C", "emoji": "🔴", "icon": "↓"}
    else:
        return {"signal": "HOLD", "color": "#F4B740", "emoji": "🟡", "icon": "→"}


def conviction_score(summary: dict) -> int:
    """
    Derive a 0–100 conviction score for the verdict gauge from a sentiment
    summary (see `sentiment_summary`).

    Two ingredients, weighted evenly-ish:
      • strength  — how far the average compound is from neutral (capped at 0.5)
      • agreement — share of articles that back the signal's direction
    With no articles it returns 0. Because HOLD's dominant bucket is "neutral"
    and its strength is low by construction, conviction stays deliberately
    capped until news flow breaks one way — matching the reference behaviour.
    """
    total = summary.get("total", 0)
    if total == 0:
        return 0

    avg = summary.get("avg_compound", 0.0)
    if avg >= 0.2:
        dominant = summary.get("positive", 0)
    elif avg <= -0.2:
        dominant = summary.get("negative", 0)
    else:
        dominant = summary.get("neutral", 0)

    agreement = dominant / total
    strength = min(1.0, abs(avg) / 0.5)
    return int(round(100 * (0.55 * agreement + 0.45 * strength)))
