# 📈 Stock Analyzer — Phase 1

A production-grade stock analysis platform built with **FastAPI** (backend) + **Streamlit** (frontend), both deployed on **Railway**.

---

## Architecture

```
┌──────────────────────────┐      HTTP/REST     ┌──────────────────────────┐
│   Streamlit Frontend     │ ◄────────────────► │    FastAPI Backend       │
│   Railway Service #2     │                    │    Railway Service #1    │
│                          │                    │                          │
│  Tab 1: Overview         │                    │  GET  /info/{ticker}     │
│  Tab 2: Technical        │                    │  GET  /indicators/{tick} │
│  Tab 3: Sentiment        │                    │  GET  /sentiment/{tick}  │
│  Tab 4: Earnings         │                    │  GET  /earnings/{tick}   │
│  Tab 5: Screener         │                    │  POST /screen            │
└──────────────────────────┘                    └──────────────────────────┘
                                                          │
                                                          ▼
                                               yfinance (free, no key)
                                               VADER sentiment (free)
```

> **Why not Vercel?** Streamlit is a Python server process — Vercel only runs static files and JS/Python serverless functions. Railway runs full Docker/Nixpacks containers, making it perfect for both services.

---

## Features (Phase 1)

### 📉 Technical Indicator Engine
| Indicator | What it detects |
|---|---|
| RSI (14) | Overbought (>70) / Oversold (<30) |
| MACD (12,26,9) | Bullish / Bearish crossovers |
| Bollinger Bands (20,2) | Squeeze, breakout direction |
| Golden / Death Cross | SMA 50 vs SMA 200 trend confirmation |
| Stochastic (14,3) | Short-term momentum extremes |
| ATR, OBV | Volatility, volume confirmation |

Each indicator generates a **signal** (BUY / WEAK BUY / NEUTRAL / WEAK SELL / SELL) with a reason and strength level. An **Overall Signal** score aggregates all signals.

### 📰 News Sentiment Analyzer
- Fetches latest news via `yfinance` (no API key needed)
- Scores each headline using **VADER** (rule-based, tuned for financial text)
- Per-article sentiment + overall gauge visualization
- Positive / Negative / Neutral counts

### 💰 Earnings & Fundamentals
- Quarterly EPS (Actual vs Estimate) bar chart
- Revenue history
- Next earnings date
- Analyst consensus (Strong Buy → Strong Sell distribution)
- Recent analyst upgrade/downgrade actions

### 🔍 Stock Screener
- Preset watchlists: US Tech, US Finance, US Healthcare, India NSE, ETFs
- Custom ticker input
- Filter by RSI range and/or signal type
- Color-coded results table
- One-click drill-down into full analysis

---

## Local Development

```bash
# 1. Clone / download the project
cd stock-analyzer

# 2. Start the backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Start the frontend (new terminal)
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run app.py
# Open http://localhost:8501
```

---

## Deployment on Railway

### Step 1 — Deploy the Backend

1. Go to [railway.app](https://railway.app) → **New Project**
2. Click **Deploy from GitHub repo** → select your repo
3. Set the **Root Directory** to `backend`
4. Railway auto-detects `railway.toml` and starts with:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. Once deployed, go to **Settings → Networking → Generate Domain**
6. Copy your backend URL: `https://stock-api-xxxx.up.railway.app`
7. Test it: visit `https://stock-api-xxxx.up.railway.app/health` → should return `{"status":"ok"}`

### Step 2 — Deploy the Frontend

1. In the same Railway project → **New Service → GitHub repo**
2. Set **Root Directory** to `frontend`
3. Add environment variable:
   ```
   BACKEND_URL = https://stock-api-xxxx.up.railway.app
   ```
   *(use the URL from Step 1)*
4. Railway starts with:
   ```
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Generate a domain for the frontend service too
6. Open your Streamlit URL — done! 🎉

### Railway Free Tier Notes
- Each service gets **500 free hours/month** (enough for 24/7 for ~20 days)
- No credit card needed for Hobby plan
- Both services can run in the same project (shared billing)
- Cold starts take ~10s on free tier — consider upgrading to Developer ($5/mo) for always-on

---

## Supported Ticker Formats

| Market | Format | Example |
|---|---|---|
| US Stocks | Plain symbol | `AAPL`, `TSLA`, `NVDA` |
| India NSE | Append `.NS` | `RELIANCE.NS`, `TCS.NS` |
| India BSE | Append `.BO` | `RELIANCE.BO` |
| UK | Append `.L` | `HSBA.L` |
| Germany | Append `.DE` | `VOW3.DE` |
| Crypto | Append `-USD` | `BTC-USD`, `ETH-USD` |
| ETFs | Plain symbol | `SPY`, `QQQ`, `GLD` |

---

## Project Structure

```
stock-analyzer/
├── backend/
│   ├── main.py          # FastAPI app + all routes
│   ├── indicators.py    # RSI, MACD, BB, Stochastic, ATR, OBV + signal logic
│   ├── sentiment.py     # yfinance news + VADER sentiment scoring
│   ├── earnings.py      # Quarterly EPS, revenue, analyst recs
│   ├── screener.py      # Fast batch ticker analysis
│   ├── requirements.txt
│   └── railway.toml     # Railway deployment config
├── frontend/
│   ├── app.py           # Full Streamlit UI (5 tabs, Plotly charts)
│   ├── requirements.txt
│   └── railway.toml     # Railway deployment config
└── README.md
```

---

## Phase Roadmap

| Phase | Status | Features |
|---|---|---|
| **Phase 1** | ✅ Built | Screener, Technical Indicators, Sentiment, Earnings |
| **Phase 2** | 🔜 Next | Backtesting engine, walk-forward validation, portfolio optimization |
| **Phase 3** | 🔜 Future | Probabilistic forecasting, regime detection, ensemble methods |

---

## What's Better Than the Old App

| Old Approach | New Approach |
|---|---|
| LSTM point predictions (unreliable) | Signal-based analysis (transparent) |
| Single hardcoded ticker (VOW3.DE) | Any ticker, user-selectable |
| Fake confidence labels (HIGH/LOW) | Real indicator math with reasons |
| 6-hour refresh ignoring market hours | On-demand fetch, 5-min cache |
| Recursive error compounding | No compounding — each indicator independent |
| No sentiment or fundamentals | News sentiment + earnings + analyst recs |
| Render deployment (single service) | Railway (2 services, clean separation) |

---

## Disclaimer

> This tool is for **educational and research purposes only**.  
> It is **not financial advice**.  
> Past signals do not guarantee future performance.  
> Always consult a qualified financial advisor before making investment decisions.
