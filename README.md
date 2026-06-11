# Stock Market Sentiment Analyzer

A Python-based tool that combines real-time financial news sentiment analysis with stock price data to help identify correlations between market sentiment and price movements. Built with Streamlit for an interactive web dashboard.

---

## Features

- **Real-time sentiment scoring** — fetches recent news headlines for any ticker and scores them using VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon-based model tuned for financial and social media text
- **Historical price overlay** — pulls OHLCV data via `yfinance` and plots it alongside the rolling sentiment score
- **Sentiment trend visualization** — interactive Plotly charts showing sentiment over time, volume, and price correlation
- **Ticker comparison** — compare sentiment scores across multiple tickers side by side
- **ML baseline** — a scikit-learn logistic regression model trained on sentiment + technical features to predict next-day price direction (up/down)
- **Exportable data** — download sentiment + price data as CSV for further analysis

---

## Tech Stack

| Layer | Library |
|---|---|
| UI / Dashboard | Streamlit |
| Sentiment Analysis | VADER (vaderSentiment) |
| Stock Data | yfinance |
| News Data | NewsAPI (requests) |
| Visualization | Plotly |
| Data Processing | pandas |
| ML Model | scikit-learn |
| Config | python-dotenv |

---

## Project Structure

```
stock-sentiment-analyzer/
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py      # yfinance price data + NewsAPI headlines
│   ├── sentiment.py         # VADER scoring + aggregation logic
│   └── visualizations.py   # Plotly chart builders
├── data/                    # Local cache for fetched data (gitignored)
├── notebooks/               # Exploratory analysis notebooks
├── app.py                   # Streamlit entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/stock-sentiment-analyzer.git
cd stock-sentiment-analyzer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your [NewsAPI](https://newsapi.org/) key:

```
NEWS_API_KEY=your_key_here
```

> **Note:** A free NewsAPI key gives 100 requests/day and up to 1 month of headlines — sufficient for personal use and demos.

### 5. Run the app

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## How It Works

1. **Data ingestion** — `data_fetcher.py` queries NewsAPI for headlines matching a ticker symbol and fetches the corresponding OHLCV history from Yahoo Finance via `yfinance`.
2. **Sentiment scoring** — `sentiment.py` runs each headline through VADER's `SentimentIntensityAnalyzer` to produce a compound score in `[-1, 1]`. These are aggregated into a daily average.
3. **Visualization** — `visualizations.py` builds dual-axis Plotly figures: closing price on the primary axis and rolling sentiment on the secondary axis.
4. **ML prediction** — A logistic regression model uses lagged sentiment scores and technical indicators (RSI, SMA crossover) as features to classify next-day direction.

---

## Screenshots

*Coming soon — add after first deployment.*

---

## Roadmap

- [ ] Reddit (r/wallstreetbets, r/stocks) sentiment via PRAW
- [ ] Twitter/X sentiment integration
- [ ] FinBERT fine-tuned transformer model as an alternative scorer
- [ ] Real-time streaming with WebSocket price feed
- [ ] Backtesting module: simulate trades driven by sentiment signals
- [ ] Docker + deployment to Streamlit Cloud

---

## License

MIT
