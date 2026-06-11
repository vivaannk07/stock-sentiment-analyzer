import streamlit as st
import pandas as pd

from src.data_fetcher import fetch_price_history, fetch_news_headlines, get_ticker_info, get_nifty50_list
from src.sentiment import score_articles, daily_sentiment, sentiment_summary, classify_signal
from src.visualizations import price_sentiment_chart, sentiment_distribution_chart, volume_chart

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Sentiment Analyzer",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    days_back_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    days_back = days_back_map[selected_period_label]

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

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
    # Cached wrappers — avoid repeated API calls
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_price(ticker, period):
        return fetch_price_history(ticker, period=period)
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_news(ticker, days_back):
        return fetch_news_headlines(ticker, days_back=days_back)
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_info(ticker):
        return get_ticker_info(ticker)
    
    with st.spinner(f"Fetching data for {active_ticker}…"):
        price_df = cached_price(active_ticker, active_period)
        articles = cached_news(active_ticker, active_days_back)
        info = cached_info(active_ticker)

    scored_df = score_articles(articles)
    daily_df = daily_sentiment(scored_df)
    summary = sentiment_summary(scored_df)

    # ── Company header ────────────────────────────────────────────────────────
    col_name, col_price, col_sector = st.columns([3, 1, 2])
    with col_name:
        st.subheader(f"{info['name']} ({active_ticker})")
    with col_price:
        price = info.get("current_price")
        st.metric("Price", f"₹{price:,.2f}" if price else "N/A")
    with col_sector:
        st.caption(f"**Sector:** {info['sector']}  \n**Industry:** {info['industry']}")

    st.markdown("---")

    # ── Sentiment KPIs ────────────────────────────────────────────────────────
    if summary["total"] == 0:
        st.warning(
            "No news articles found. Make sure your `NEWS_API_KEY` is set in `.env` "
            "and the ticker is covered by NewsAPI."
        )
    else:
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
        st.error(f"Could not fetch price data for **{active_ticker}**. Check the ticker symbol.")
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
