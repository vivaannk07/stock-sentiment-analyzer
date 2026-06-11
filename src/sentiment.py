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


def daily_sentiment(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-article scores into a daily average compound score.
    Returns DataFrame with columns: date, avg_compound, article_count.
    """
    if scored_df.empty:
        return pd.DataFrame(columns=["date", "avg_compound", "article_count"])

    agg = (
        scored_df.dropna(subset=["date"])
        .groupby("date")
        .agg(avg_compound=("compound", "mean"), article_count=("compound", "count"))
        .reset_index()
        .sort_values("date")
    )
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
    Convert an average compound score into a Buy/Hold/Sell signal with color.
    Returns dict with 'signal', 'color', and 'emoji' keys.
    """
    if avg_compound >= 0.2:
        return {"signal": "BUY", "color": "#26a69a", "emoji": "🟢"}
    elif avg_compound <= -0.2:
        return {"signal": "SELL", "color": "#ef5350", "emoji": "🔴"}
    else:
        return {"signal": "HOLD", "color": "#ffa726", "emoji": "🟡"}
