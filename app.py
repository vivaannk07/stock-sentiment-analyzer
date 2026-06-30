import streamlit as st
import pandas as pd

from src.data_fetcher import (
    fetch_price_history,
    fetch_news_headlines,
    get_ticker_info,
    get_yf_enrichment,
    get_nifty50_list,
)
from src.sentiment import score_articles, daily_sentiment, sentiment_summary, classify_signal
from src.visualizations import price_sentiment_chart, sentiment_distribution_chart, volume_chart

# NewsAPI free tier only returns articles from the last ~30 days. We cap the
# news lookback so longer price periods don't silently fetch the same 30 days
# while pretending to cover more.
NEWS_API_MAX_DAYS = 30

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Sentiment Analyzer",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached data wrappers (module scope so the cache survives reruns) ─────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_price(ticker: str, period: str) -> pd.DataFrame:
    return fetch_price_history(ticker, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_news(ticker: str, days_back: int) -> list:
    return fetch_news_headlines(ticker, days_back=days_back)


@st.cache_data(ttl=86400, show_spinner=False)  # 24h — sector / industry rarely change
def cached_yf_enrichment(ticker: str) -> dict:
    return get_yf_enrichment(ticker)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🇮🇳 Nifty 50 Sentiment Analyzer")
    st.markdown("---")

    # Load Nifty 50 list for dropdown
    nifty_df = get_nifty50_list()
    if nifty_df.empty:
        st.error("Could not load Nifty 50 list. Check data/nifty50.csv")
        st.stop()

    # Build display options: "Company Name (TICKER)"
    nifty_df["display"] = nifty_df["company_name"] + " (" + nifty_df["ticker"].str.replace(".NS", "") + ")"

    selected_display = st.selectbox(
        "Select Stock",
        options=nifty_df["display"].tolist(),
        index=0,  # Reliance by default
    )

    # Get the actual ticker from the selection
    ticker = nifty_df[nifty_df["display"] == selected_display].iloc[0]["ticker"]

    period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}
    selected_period_label = st.selectbox("Price History Period", list(period_options.keys()), index=1)
    period = period_options[selected_period_label]

    # Price chart shows the full selected period, but news sentiment is capped
    # at NEWS_API_MAX_DAYS because the free NewsAPI tier doesn't go further back.
    requested_days_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    requested_days = requested_days_map[selected_period_label]
    days_back = min(requested_days, NEWS_API_MAX_DAYS)
    news_capped = requested_days > NEWS_API_MAX_DAYS

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    if news_capped:
        st.caption(
            f"ℹ️ News sentiment is limited to the last {NEWS_API_MAX_DAYS} days "
            "(free NewsAPI tier). Price chart still covers the full period."
        )

    st.markdown("---")
    st.caption("Data: Yahoo Finance · NewsAPI · VADER NLP")
    st.caption("Built with Streamlit + Plotly")

# ── Main content ──────────────────────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar and click **Analyze**.")
    st.stop()

if analyze_btn or "last_ticker" in st.session_state:
    # Keep results visible after button press without forcing a re-run
    if analyze_btn:
        st.session_state["last_ticker"] = ticker
        st.session_state["last_period"] = period
        st.session_state["last_days_back"] = days_back

    active_ticker = st.session_state.get("last_ticker", ticker)
    active_period = st.session_state.get("last_period", period)
    active_days_back = st.session_state.get("last_days_back", days_back)

    # ── Data loading ─────────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for {active_ticker}…"):
        price_df = cached_price(active_ticker, active_period)
        articles = cached_news(active_ticker, active_days_back)
        enrichment = cached_yf_enrichment(active_ticker)
        info = get_ticker_info(active_ticker, price_df=price_df, enrichment=enrichment)

    scored_df = score_articles(articles)
    daily_df = daily_sentiment(
        scored_df,
        trading_days=price_df.index if not price_df.empty else None,
    )
    summary = sentiment_summary(scored_df)

    # ── Company header ────────────────────────────────────────────────────────
    col_name, col_price, col_sector = st.columns([3, 1, 2])
    with col_name:
        st.subheader(f"{info['name']} ({active_ticker})")
    with col_price:
        price = info.get("current_price")
        st.metric("Price", f"₹{price:,.2f}" if price else "N/A")
    with col_sector:
        st.caption(f"**Sector:** {info['sector']}")

    if selected_period_label != "1 Month":
        st.info(
            "ℹ️ News sentiment is limited to the last 30 days (NewsAPI free tier). "
            f"Price chart shows your full selected period of {selected_period_label}."
        )

    st.markdown("---")

    # ── Sentiment KPIs ────────────────────────────────────────────────────────
    if summary["total"] == 0:
        st.warning(
            "No news articles found. Make sure your `NEWS_API_KEY` is set in `.env` "
            "and the ticker is covered by NewsAPI."
        )
    else:
        if summary["total"] > 0:
            signal = classify_signal(summary["avg_compound"])
            st.markdown(
                f"""
                <div style="background-color:{signal['color']}; color:white; text-align:center;
                            border-radius:12px; padding:24px; margin-bottom:16px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.2);">
                    <div style="font-size:2.5rem; font-weight:bold;">{signal['emoji']} {signal['signal']}</div>
                    <div style="font-size:1rem;">Based on average sentiment: {summary['avg_compound']:+.3f} across {summary['total']} articles</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Articles", summary["total"])
        k2.metric("Positive", summary["positive"], delta=None)
        k3.metric("Neutral", summary["neutral"])
        k4.metric("Negative", summary["negative"])
        avg = summary["avg_compound"]
        k5.metric(
            "Avg Sentiment",
            f"{avg:+.3f}",
            delta="Bullish" if avg >= 0.05 else ("Bearish" if avg <= -0.05 else "Neutral"),
            delta_color="normal" if avg >= 0.05 else ("inverse" if avg <= -0.05 else "off"),
        )

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    if price_df.empty:
        st.error(
            f"⚠️ Could not fetch price data for **{active_ticker}** right now. "
            "Yahoo Finance may be temporarily rate-limiting requests. "
            "Please wait 30 seconds and click Analyze again."
        )
        st.info("💡 Tip: Try selecting a different stock — sometimes only certain tickers are blocked.")
    else:
        st.plotly_chart(
            price_sentiment_chart(price_df, daily_df, active_ticker),
            use_container_width=True,
        )

        col_vol, col_dist = st.columns([2, 1])
        with col_vol:
            st.subheader("Volume")
            st.plotly_chart(volume_chart(price_df), use_container_width=True)
        with col_dist:
            st.subheader("Sentiment Distribution")
            if scored_df.empty:
                st.caption("No articles to display.")
            else:
                st.plotly_chart(sentiment_distribution_chart(scored_df), use_container_width=True)

    # ── Recent headlines table ────────────────────────────────────────────────
    if not scored_df.empty:
        st.markdown("---")
        st.subheader("Recent Headlines")

        display_df = scored_df[["date", "title", "source", "compound", "sentiment_label"]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df.columns = ["Date", "Headline", "Source", "Score", "Sentiment"]

        st.dataframe(
            display_df.head(50),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn(format="%.3f"),
                "Sentiment": st.column_config.TextColumn(),
            },
        )

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV",
            data=csv,
            file_name=f"{active_ticker}_sentiment.csv",
            mime="text/csv",
        )

else:
    # Landing state
    st.markdown(
        """
## Welcome 🇮🇳

This dashboard combines **real-time news sentiment** with **Nifty 50 stock price history**
to help spot correlations between market narrative and price movement in Indian markets.

**How to use:**
1. Select a Nifty 50 stock from the sidebar (e.g. Reliance, TCS, Infosys)
2. Choose a time period
3. Click **Analyze**

**Built with:** yfinance · NewsAPI · VADER NLP · Streamlit · Plotly

> **Requires a NewsAPI key** — add it to `.env` as `NEWS_API_KEY`.
> Get a free key at [newsapi.org](https://newsapi.org).
        """
    )

st.markdown("---")
st.markdown("<div style='text-align:center; color:#888; font-size:0.85rem; padding:20px;'>Built by Vivaan Karnani · <a href='https://github.com/vivaannk07/stock-sentiment-analyzer' style='color:#26a69a;'>GitHub</a> · Educational tool only — not investment advice.</div>", unsafe_allow_html=True)
