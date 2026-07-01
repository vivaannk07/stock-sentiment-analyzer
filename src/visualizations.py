import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── SENTRA palette (mirrors design-reference tokens) ─────────────────────────
BG = "#08090B"
S1 = "#0F1013"
TX = "#ECEDEF"
TX2 = "#969BA4"
TX3 = "#5C616B"
GRID = "rgba(255,255,255,0.05)"
BUY = "#34E0A1"
HOLD = "#F4B740"
SELL = "#FF5C5C"

UI_FONT = "Space Grotesk, sans-serif"
MONO_FONT = "IBM Plex Mono, monospace"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to an 'rgba(r,g,b,a)' string for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared dark SENTRA chrome to a figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=UI_FONT, color=TX2, size=12),
        height=height,
        margin=dict(l=44, r=24, t=30, b=24),
        legend=dict(
            orientation="h", y=1.06, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TX2, size=11),
        ),
        hoverlabel=dict(bgcolor=S1, font=dict(family=MONO_FONT, color=TX)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TX3, size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TX3, size=10))
    return fig


def price_sentiment_chart(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
    signal_color: str = BUY,
) -> go.Figure:
    """
    Price as a filled area/line with the daily sentiment score overlaid as a
    dotted line on a secondary axis — the SENTRA detail-chart style. The price
    line is tinted with the current signal colour.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    close = price_df["Close"]
    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=close,
            name="Price",
            mode="lines",
            line=dict(color=signal_color, width=2),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(signal_color, 0.08),
            hovertemplate="₹%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    if not sentiment_df.empty and sentiment_df["avg_compound"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=sentiment_df["date"],
                y=sentiment_df["avg_compound"],
                name="Sentiment",
                mode="lines",
                line=dict(color=HOLD, width=1.6, dash="dot"),
                connectgaps=True,
                hovertemplate="%{y:+.3f}<extra></extra>",
            ),
            secondary_y=True,
        )

    # Keep the price line floating mid-panel: clip the fill below the data range
    # rather than letting tozeroy stretch it from ₹0.
    if not close.empty:
        lo, hi = float(close.min()), float(close.max())
        pad = max((hi - lo) * 0.08, hi * 0.01)
        fig.update_yaxes(range=[lo - pad, hi + pad], secondary_y=False)

    _base_layout(fig, height=430)
    fig.update_yaxes(title_text="Price (₹)", secondary_y=False,
                     title_font=dict(color=TX3, size=11))
    fig.update_yaxes(title_text="Sentiment", secondary_y=True, range=[-1, 1],
                     showgrid=False, title_font=dict(color=TX3, size=11))
    return fig


def conviction_gauge(score: float, signal_color: str = HOLD, height: int = 200) -> go.Figure:
    """
    Semicircular conviction meter (0–100) built with go.Indicator. The value arc
    is coloured to match the current Buy/Hold/Sell signal; the track uses the
    SENTRA surface colour. Mirrors the reference's 180° radial gauge.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0, min(100, score)),
            number=dict(font=dict(family=MONO_FONT, size=34, color=TX), suffix=""),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100],
                    tickcolor=TX3,
                    tickfont=dict(color=TX3, size=9),
                ),
                bar=dict(color=signal_color, thickness=0.30),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[dict(range=[0, 100], color="#1C1F25")],  # track
            ),
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=UI_FONT, color=TX),
        height=height,
        margin=dict(l=24, r=24, t=18, b=8),
    )
    return fig


def sentiment_distribution_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Donut chart showing positive / neutral / negative article counts."""
    if scored_df.empty:
        return go.Figure()

    counts = scored_df["sentiment_label"].value_counts()
    colors = {"positive": BUY, "neutral": TX3, "negative": SELL}

    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.62,
            marker=dict(colors=[colors.get(l, TX2) for l in counts.index],
                        line=dict(color=BG, width=2)),
            textinfo="label+percent",
            textfont=dict(family=UI_FONT, color=TX, size=11),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family=UI_FONT, color=TX2),
        margin=dict(l=10, r=10, t=20, b=10),
        height=260,
    )
    return fig


def volume_chart(price_df: pd.DataFrame) -> go.Figure:
    """Volume bars tinted up/down by the day's close vs. open."""
    up = price_df["Close"] >= price_df["Open"]
    colors = [_hex_to_rgba(BUY, 0.65) if u else _hex_to_rgba(SELL, 0.65) for u in up]

    fig = go.Figure(
        go.Bar(
            x=price_df.index,
            y=price_df["Volume"],
            name="Volume",
            marker_color=colors,
        )
    )
    _base_layout(fig, height=200)
    fig.update_layout(margin=dict(l=44, r=24, t=20, b=24))
    fig.update_yaxes(title_text="Volume", title_font=dict(color=TX3, size=11))
    return fig
