import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def price_sentiment_chart(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Dual-axis chart: candlestick price + daily sentiment bar overlay.
    price_df must have OHLCV columns; sentiment_df must have date + avg_compound.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=(f"{ticker} Price", "Daily Avg Sentiment"),
    )

    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    if not sentiment_df.empty:
        colors = [
            "#26a69a" if v >= 0.05 else ("#ef5350" if v <= -0.05 else "#9e9e9e")
            for v in sentiment_df["avg_compound"]
        ]
        fig.add_trace(
            go.Bar(
                x=sentiment_df["date"],
                y=sentiment_df["avg_compound"],
                name="Sentiment",
                marker_color=colors,
                opacity=0.85,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=40, r=20, t=60, b=20),
        height=550,
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1)
    return fig

def sentiment_distribution_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Donut chart showing positive / neutral / negative article counts."""
    if scored_df.empty:
        return go.Figure()

    counts = scored_df["sentiment_label"].value_counts()
    colors = {"positive": "#26a69a", "neutral": "#9e9e9e", "negative": "#ef5350"}

    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.5,
            marker_colors=[colors.get(l, "#ccc") for l in counts.index],
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
    )
    return fig


def volume_chart(price_df: pd.DataFrame) -> go.Figure:
    """Simple volume bar chart."""
    fig = go.Figure(
        go.Bar(
            x=price_df.index,
            y=price_df["Volume"],
            name="Volume",
            marker_color="#5c6bc0",
            opacity=0.8,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=30, b=20),
        height=200,
        yaxis_title="Volume",
    )
    return fig
