# Development Journey

> How this project evolved from a broken LSTM prediction model into a production-grade stock analysis platform.

**[← Back to README](./README.md)**

---

## Where It Started

The original project was an LSTM neural network (`.h5` model file) trained to predict stock prices for a single German stock (`VOW3.DE`). It had several fundamental problems:

```
 LSTM trained on raw prices (non-stationary data)
 Single hardcoded ticker — not extensible
 Recursive prediction compounded errors day by day
 Fake confidence labels (HIGH/MEDIUM/LOW) with no statistical basis
 No sentiment, no fundamentals, no risk management
 Deployed on Render — crashed constantly
```

**The decision:** Stop thinking "prediction model first." Build proper tooling instead.

---

## Phase 1 — Analysis Engine

**Goal:** Build the foundation. Make it work for any ticker, any market.

### What was built
- **Technical Indicator Engine** — RSI, MACD, Bollinger Bands, Golden/Death Cross, Stochastic, ATR, OBV
- **Signal Generator** — each indicator produces a BUY/SELL/NEUTRAL signal with a reason
- **News Sentiment** — Yahoo Finance RSS headlines scored with VADER (no API key needed)
- **Earnings Summarizer** — quarterly EPS actual vs estimate, revenue, analyst consensus
- **Stock Screener** — batch scan multiple tickers by RSI range and signal

### Key technical decisions
- **yfinance over a paid API** — free, covers 50,000+ tickers globally
- **VADER over ML sentiment** — rule-based, no training needed, works well for financial headlines
- **Yahoo Finance quoteSummary via cookie session** — bypasses 429 rate limiting that blocked naive requests
- **3-strategy data fetching fallback** — S1 (Ticker.history) → S2 (yf.download) → S3 (v8 JSON API with crumb)

### Problems solved
The biggest challenge was Yahoo Finance's aggressive rate limiting. The solution was building a cookie-authenticated session that mimics a real Chrome browser at the HTTP level, then calling the v8 chart API directly instead of relying on yfinance's internal requests.

---

## Phase 2 — Backtesting & Portfolio

**Goal:** Test whether strategies actually work historically. Optimize portfolios mathematically.

### What was built
- **6 Trading Strategies:**
  - RSI Mean Reversion
  - MACD Momentum
  - MA Cross (Golden/Death Cross)
  - Bollinger Band Mean Reversion
  - ATR Momentum (volatility-adaptive)
  - Multi-Factor Consensus (combines all 5 with weighted scoring)

- **Backtesting Engine:**
  - Zero lookahead bias (signal on day T executes at day T+1 open)
  - 0.1% commission per trade
  - 8 performance metrics: Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor
  - Walk-Forward Validation — splits data into 5 windows, tests on unseen data

- **Portfolio Optimizer:**
  - Monte Carlo simulation (3000 random portfolios)
  - Efficient frontier visualization
  - 4 optimal portfolios: Max Sharpe, Min Volatility, Risk Parity, Equal Weight
  - Correlation heatmap
  - scipy.optimize for precise weight calculation

### Key insight
Walk-forward validation was the most important addition. Without it, any strategy looks good — you can always find parameters that worked in the past. Walk-forward tests on data the strategy never saw during "training," giving honest out-of-sample performance.

---

## Phase 3 — AI & Decision Engine

**Goal:** Synthesize everything into one clear decision. Add global macro context.

### What was built

**Regime Detection (`regime.py`)**
- GaussianMixture model (sklearn) with 4 components
- Features: log returns, rolling volatility, RSI, volume ratio, ATR %
- Labels regimes by return/volatility characteristics: Bull Trend, Bear Trend, High Volatility, Consolidation
- Each regime changes which strategy to use

**Probabilistic Forecasting (`forecaster.py`)**
- EWMA-GARCH (RiskMetrics λ=0.94) for volatility estimation
- ARIMA(1,1,1) via statsmodels for trend drift
- Bootstrap Monte Carlo (500 simulations) — samples from actual historical returns
- Output: fan chart with p5/p25/p50/p75/p95 percentile bands
- Honest: gives probability ranges, not point predictions

**Ensemble Scoring (`ensemble.py`)**
- Regime-adaptive weighting: momentum signals weighted higher in trends, mean-reversion weighted higher in consolidation
- 7 components: RSI, MACD, MA Cross, Bollinger, Stochastic, Volume, ATR Trend
- Score 0–100: 75+ = Strong Buy, 25- = Strong Sell

**Global Macro Sentiment (`macro_sentiment.py`)**
- 7 free RSS feeds: Reuters (2), BBC (2), AP, CNBC, MarketWatch
- Sector keyword mapping: each stock's sector gets relevant keywords
  - Technology → "semiconductor", "chip export", "AI regulation", "Taiwan strait"
  - Energy → "OPEC", "oil embargo", "sanctions", "pipeline"
  - Consumer Defensive → "food shortage", "drought", "commodity prices"
- Relevance scoring 0–10 per article
- VADER sentiment on title + description
- Articles below 3.0 relevance filtered out

**Decision Engine (`decision_engine.py`)**
- Weighted combination: 35% Technical + 25% Macro + 15% Earnings + 15% Forecast + 10% Stock News
- Entry zone: forecast p25–p50 (buy on dip)
- Stop loss: entry − 1.5 × ATR
- Take profit: forecast p75
- Risk/reward ratio calculation
- Plain English explanation of the verdict

### The philosophy
The key insight from Phase 3: **regime-adaptive weighting** makes the ensemble much more useful than fixed weights. RSI at 70 means very different things in a bull trend vs consolidation. The regime context changes which signals to trust.

---

## Problems Encountered & How They Were Solved

| Problem | Root Cause | Solution |
|---|---|---|
| Yahoo Finance 429 errors | Bare requests without cookies | Chrome-impersonated session + crumb token |
| LSTM predictions useless | Non-stationary data, error compounding | Replaced with GARCH + bootstrap Monte Carlo |
| Streamlit duplicate element IDs | Streamlit 1.57 stricter widget IDs | Added unique `key=` to every `st.plotly_chart` |
| `use_container_width` warnings | Deprecated in Streamlit 1.57 | Replaced with `width='stretch'` |
| Market cap/P/E showing N/A | v8 chart API doesn't include fundamentals | Added `get_key_stats()` via quoteSummary |
| Irrelevant macro news (Modi/gold for AAPL) | Relevance threshold too low (0.5/10) | Raised threshold to 3.0, tightened keyword matching |
| Typing "Apple" gives error | Only ticker codes supported | Added `resolve_ticker()` name-to-ticker lookup |
| Session state KeyError (Streamlit 1.57) | Widget key conflicts with manual state writes | Removed manual session state writes for widget-bound keys |

---

## Architecture Decisions

### Why FastAPI over Flask
FastAPI provides automatic OpenAPI docs at `/docs`, async support, and Pydantic validation. The interactive docs make it easy to test every endpoint during development.

### Why Streamlit over React
Streamlit lets you build a full data visualization UI in pure Python without any JavaScript. For a data-heavy app with Plotly charts, it's the right tradeoff — much faster to build, and the final result looks professional.

### Why no .h5 or .pkl model files
Pre-trained models go stale. A GARCH model fit on 2022 data gives wrong volatility estimates in 2026. By fitting models on-demand with the latest data, the predictions are always current. The "cost" is a 1-2 second fit time per request, which is acceptable.

### Why RSS feeds over paid news APIs
- Reuters, BBC, AP, CNBC all provide free RSS feeds
- No API key, no rate limits, no credit card
- VADER sentiment works well on news headlines
- Total cost: $0

### Why Yahoo Finance over Alpha Vantage / Polygon
- 50,000+ tickers including India NSE, Germany, UK, Crypto
- No API key needed
- Free forever
- The 429 rate limiting was solvable (and we solved it)

---

## Deployment Journey

```
Local development
  ↓
Render (original LSTM project) — crashed, unreliable
  ↓  
Railway free tier — 500 hour/month limit, not suitable for public use
  ↓
AWS EC2 t3.micro (free tier, 12 months)
  + Streamlit Community Cloud (free forever)
  = Always-on, production-grade, $0/month
```

---

## What This Project Teaches

1. **Data quality beats model complexity** — reliable data fetching matters more than a fancy ML model
2. **Walk-forward validation is non-negotiable** — in-sample backtest results are always optimistic
3. **Honest uncertainty communication** — confidence intervals beat fake point predictions
4. **Regime-aware analysis** — the same signal means different things in different market conditions
5. **Free tools are enough** — VADER, sklearn, statsmodels, scipy, yfinance, RSS feeds cover 95% of what expensive Bloomberg terminals do

---

## Future Roadmap

- **Phase 4:** Options flow analysis, dark pool data integration
- **Phase 5:** Paper trading mode with simulated portfolio tracking
- **Phase 6:** Multi-user support with authentication and saved watchlists
- **Phase 7:** Mobile app (React Native + existing FastAPI backend)

---

*Built with Python 3.12, FastAPI, Streamlit, and a lot of debugging.*