from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

from src.data_fetcher import (
    fetch_price_history,
    fetch_news_headlines,
    fetch_market_stats,
    get_ticker_info,
    get_yf_enrichment,
    get_nifty50_list,
)
from src.sentiment import (
    score_articles,
    daily_sentiment,
    sentiment_summary,
    classify_signal,
    conviction_score,
)
from src.visualizations import (
    price_sentiment_chart,
    sentiment_distribution_chart,
    volume_chart,
    conviction_gauge,
)

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


# ── SENTRA styling ────────────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "assets" / "style.css"


def load_css() -> None:
    """Inject the SENTRA stylesheet once per session."""
    try:
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except OSError:
        pass  # missing stylesheet shouldn't take the app down


load_css()


# ── Formatting helpers ─────────────────────────────────────────────────────────
def fmt_inr_cr(value) -> str:
    """Format a rupee figure into the Indian crore convention (₹… Cr / L Cr)."""
    if not isinstance(value, (int, float)):
        return "N/A"
    crore = value / 1e7
    if crore >= 1e5:
        return f"₹{crore / 1e5:,.2f} L Cr"
    return f"₹{crore:,.0f} Cr"


def fmt_compact(value) -> str:
    """Human-readable large integers (e.g. volume): 1.2M, 3.4K."""
    if not isinstance(value, (int, float)):
        return "N/A"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= div:
            return f"{value / div:.1f}{suffix}"
    return f"{value:,.0f}"


def fmt_range(low, high) -> str:
    """52-week low–high as '₹lo – ₹hi', tolerating missing bounds."""
    lo = f"₹{low:,.0f}" if isinstance(low, (int, float)) else "N/A"
    hi = f"₹{high:,.0f}" if isinstance(high, (int, float)) else "N/A"
    return f"{lo} – {hi}"


def initials(name: str) -> str:
    """Two-letter monogram for the avatar badge."""
    parts = [p for p in str(name).split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def price_delta(price_df: pd.DataFrame):
    """Return (abs_change, pct_change) of the latest close vs. the prior close."""
    if price_df is None or price_df.empty or len(price_df) < 2:
        return None, None
    last = float(price_df["Close"].iloc[-1])
    prev = float(price_df["Close"].iloc[-2])
    if prev == 0:
        return None, None
    return last - prev, (last - prev) / prev * 100


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

    analyze_btn = st.button("Analyze", type="primary", width="stretch")

    if news_capped:
        st.caption(
            f"ℹ️ News sentiment is limited to the last {NEWS_API_MAX_DAYS} days "
            "(free NewsAPI tier). Price chart still covers the full period."
        )

    st.markdown("---")
    st.caption("Data: Yahoo Finance · NewsAPI · VADER NLP")
    st.caption("Built with Streamlit + Plotly")

# ── Main content ──────────────────────────────────────────────────────────────
# Cosmetic SENTRA-style top label. This single-page app only has the Markets view,
# so this is a plain static label (no tabs/routing). See .sentra-nav in
# assets/style.css. Rendered once here so it sits above both the hero and the
# analysis breadcrumb.
st.markdown(
    """
    <nav class="sentra-nav">
      <span class="sentra-nav-label">MARKETS</span>
    </nav>
    """,
    unsafe_allow_html=True,
)

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
        market_stats = fetch_market_stats(active_ticker)

    scored_df = score_articles(articles)
    daily_df = daily_sentiment(
        scored_df,
        trading_days=price_df.index if not price_df.empty else None,
    )
    summary = sentiment_summary(scored_df)

    # Signal + conviction drive the accent colours across header, verdict & chart.
    signal = classify_signal(summary["avg_compound"])
    sig_color = signal["color"]
    conviction = conviction_score(summary)
    bare_ticker = active_ticker.replace(".NS", "").replace(".BO", "")

    # ── Company header ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="sentra-breadcrumb">
          Discover <span class="sep">/</span> Nifty 50
          <span class="sep">/</span> <span class="cur">{bare_ticker}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    delta_abs, delta_pct = price_delta(price_df)
    price = info.get("current_price")
    price_str = f"₹{price:,.2f}" if isinstance(price, (int, float)) else "N/A"
    if delta_abs is not None:
        dcolor = "var(--buy)" if delta_abs >= 0 else "var(--sell)"
        arrow = "▲" if delta_abs >= 0 else "▼"
        delta_html = (
            f'<span class="sentra-delta" style="color:{dcolor}">'
            f'{arrow} ₹{abs(delta_abs):,.2f} ({delta_pct:+.2f}%)</span>'
        )
    else:
        delta_html = '<span class="sentra-delta" style="color:var(--tx3)">—</span>'

    st.markdown(
        f"""
        <div class="sentra-header">
          <div class="sentra-avatar"
               style="color:{sig_color};background:{sig_color}18;border:1px solid {sig_color}44;">
            {initials(info['name'])}
          </div>
          <div class="sentra-titlewrap">
            <div class="sentra-titlerow">
              <h1 class="sentra-ticker">{bare_ticker}</h1>
              <span class="sentra-tag">NSE</span>
              <span class="sentra-tag">{info['sector']}</span>
            </div>
            <div class="sentra-company">{info['name']}</div>
            <div class="sentra-pricerow">
              <span class="sentra-price">{price_str}</span>
              {delta_html}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="hr-line"/>', unsafe_allow_html=True)

    # ── KPI row (market fundamentals) ─────────────────────────────────────────
    pe = market_stats["pe_ratio"]
    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-tile">
            <div class="kpi-label">Market Cap</div>
            <div class="kpi-value">{fmt_inr_cr(market_stats['market_cap'])}</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">P / E (TTM)</div>
            <div class="kpi-value">{pe if isinstance(pe, (int, float)) else 'N/A'}</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">52-Week Range</div>
            <div class="kpi-value" style="font-size:15px">
              {fmt_range(market_stats['week52_low'], market_stats['week52_high'])}
            </div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">Volume</div>
            <div class="kpi-value">{fmt_compact(market_stats['volume'])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if news_capped:
        st.markdown(
            f'<div style="font-size:12px;color:var(--tx3);margin-top:12px">'
            f"News sentiment covers the last {NEWS_API_MAX_DAYS} days (NewsAPI free tier); "
            f"the price chart still spans your full {selected_period_label} selection.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="hr-line"/>', unsafe_allow_html=True)

    # ── Verdict card + conviction gauge ───────────────────────────────────────
    if summary["total"] == 0:
        st.warning(
            "No news articles found. Make sure your `NEWS_API_KEY` is set in `.env` "
            "and the ticker is covered by NewsAPI."
        )
    else:
        as_of = datetime.now().strftime("%d %b %Y").upper()
        sub_map = {
            "BUY": "Positive news flow — sentiment cleared the bullish threshold.",
            "SELL": "Negative news flow — sentiment fell below the bearish threshold.",
            "HOLD": "Mixed news flow — no decisive catalyst, verdict held at neutral.",
        }

        # A bordered container styled (via #verdict-anchor in style.css) into the
        # SENTRA signal card: gradient fill + coloured left accent bar.
        with st.container(border=True):
            # Set the accent colours on the container itself (custom properties
            # don't inherit upward from the anchor child). Selector must match the
            # one in style.css — Streamlit 1.58 renders bordered containers as a
            # plain stVerticalBlock, so we scope by the #verdict-anchor marker.
            st.markdown(
                f'<style>div[data-testid="stVerticalBlock"]:has('
                f'> div[data-testid="stElementContainer"] #verdict-anchor)'
                f'{{--vc-accent:{sig_color};--vc-border:{sig_color}44}}</style>'
                f'<div id="verdict-anchor"></div>'
                f'<div class="verdict-head"><span class="label">SENTRA VERDICT</span>'
                f'<span class="asof">{as_of}</span></div>',
                unsafe_allow_html=True,
            )
            c_text, c_gauge = st.columns([1.25, 1])
            with c_text:
                st.markdown(
                    f"""
                    <div class="verdict-main">
                      <div class="verdict-badge"
                           style="color:{sig_color};background:{sig_color}1c;border:1px solid {sig_color}44;">
                        {signal['icon']}
                      </div>
                      <div>
                        <div class="verdict-word" style="color:{sig_color}">{signal['signal']}</div>
                        <div class="verdict-sub">{sub_map[signal['signal']]}</div>
                      </div>
                    </div>
                    <div style="margin-top:10px">
                      <span class="sig-pill"
                            style="color:{sig_color};background:{sig_color}1a;border:1px solid {sig_color}40">
                        AVG {summary['avg_compound']:+.3f}
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_gauge:
                st.plotly_chart(
                    conviction_gauge(conviction, sig_color),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            st.markdown(
                f'<div class="verdict-foot">Conviction · driven by '
                f'<span class="n">{summary["total"]}</span> headlines</div>',
                unsafe_allow_html=True,
            )

        # Sentiment breakdown tiles beneath the verdict.
        avg = summary["avg_compound"]
        mood = "Bullish" if avg >= 0.05 else ("Bearish" if avg <= -0.05 else "Neutral")
        st.markdown(
            f"""
            <div class="kpi-grid" style="margin-top:14px">
              <div class="kpi-tile">
                <div class="kpi-label">Articles</div>
                <div class="kpi-value">{summary['total']}</div>
              </div>
              <div class="kpi-tile">
                <div class="kpi-label">Positive</div>
                <div class="kpi-value" style="color:var(--buy)">{summary['positive']}</div>
              </div>
              <div class="kpi-tile">
                <div class="kpi-label">Neutral</div>
                <div class="kpi-value" style="color:var(--hold)">{summary['neutral']}</div>
              </div>
              <div class="kpi-tile">
                <div class="kpi-label">Negative</div>
                <div class="kpi-value" style="color:var(--sell)">{summary['negative']}</div>
              </div>
            </div>
            <div style="font-family:var(--mono);font-size:11px;color:var(--tx3);
                        margin-top:10px;letter-spacing:.04em">
              AVG SENTIMENT {avg:+.3f} · {mood.upper()}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="hr-line"/>', unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    if price_df.empty:
        st.error(
            f"⚠️ Could not fetch price data for **{active_ticker}** right now. "
            "Yahoo Finance may be temporarily rate-limiting requests. "
            "Please wait 30 seconds and click Analyze again."
        )
        st.info("💡 Tip: Try selecting a different stock — sometimes only certain tickers are blocked.")
    else:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:11px;letter-spacing:.14em;'
            'text-transform:uppercase;color:var(--tx3);margin-bottom:6px">'
            'Price &amp; Sentiment</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            price_sentiment_chart(price_df, daily_df, active_ticker, signal_color=sig_color),
            width="stretch",
        )

        col_vol, col_dist = st.columns([2, 1])
        with col_vol:
            st.subheader("Volume")
            st.plotly_chart(volume_chart(price_df), width="stretch")
        with col_dist:
            st.subheader("Sentiment Distribution")
            if scored_df.empty:
                st.caption("No articles to display.")
            else:
                st.plotly_chart(sentiment_distribution_chart(scored_df), width="stretch")

    # ── Recent headlines table ────────────────────────────────────────────────
    if not scored_df.empty:
        st.markdown("---")
        st.subheader("Recent Headlines")

        display_df = scored_df[["date", "title", "source", "compound", "sentiment_label"]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df.columns = ["Date", "Headline", "Source", "Score", "Sentiment"]

        st.dataframe(
            display_df.head(50),
            width="stretch",
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
    # Landing state — SENTRA "Discover" hero styling (see assets/style.css).
    st.markdown(
        """
        <div class="sentra-hero">
          <div class="sentra-eyebrow">
            <span class="dot"></span>
            <span class="txt">Sentiment Engine · Nifty 50</span>
          </div>
          <h1 class="sentra-hero-title">
            The headline moves<br>
            before the <span class="accent">market</span> does.
          </h1>
          <p class="sentra-hero-sub">
            This dashboard combines <span class="hi">real-time news sentiment</span> with
            <span class="hi">Nifty 50 price history</span> to help spot correlations between
            market narrative and price movement across India's largest listed companies.
          </p>

          <div class="sentra-steps">
            <div class="sentra-step">
              <div class="num">01</div>
              <div class="title">Pick a stock</div>
              <div class="desc">Choose a Nifty 50 constituent from the sidebar — Reliance, TCS, Infosys and 47 more.</div>
            </div>
            <div class="sentra-step">
              <div class="num">02</div>
              <div class="title">Set the window</div>
              <div class="desc">Select a price-history period, from one month out to a full year.</div>
            </div>
            <div class="sentra-step">
              <div class="num">03</div>
              <div class="title">Analyze</div>
              <div class="desc">Hit Analyze for a BUY · HOLD · SELL verdict fused from news sentiment and price action.</div>
            </div>
          </div>

          <div class="sentra-note">
            <span class="ico">🔑</span>
            <div class="body">
              Requires a <b>NewsAPI key</b> — add it to <b>.env</b> as <b>NEWS_API_KEY</b>.
              Grab a free key at <a href="https://newsapi.org">newsapi.org</a>.
              Data: Yahoo Finance · NewsAPI · VADER NLP.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("<div style='text-align:center; color:#888; font-size:0.85rem; padding:20px;'>Built by Vivaan Karnani · <a href='https://github.com/vivaannk07/stock-sentiment-analyzer' style='color:#26a69a;'>GitHub</a> · Educational tool only — not investment advice.</div>", unsafe_allow_html=True)
