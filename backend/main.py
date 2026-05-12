"""
main.py — FastAPI Backend (Phase 1 + 2)
New Phase 2 endpoints: /backtest, /portfolio/optimize
"""
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from utils       import fetch_history, get_info_v8
from indicators  import compute_all_indicators
from sentiment   import analyze_news_sentiment
from earnings    import get_earnings_data
from screener    import screen_tickers
from strategies  import STRATEGIES
from backtester  import run_backtest, walk_forward
from portfolio   import optimize_portfolio

app = FastAPI(title="Stock Analyzer API", version="3.0.0",
              description="Phase 1 + 2 — Indicators · Sentiment · Backtesting · Portfolio")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ── Health ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Stock Analyzer API", "version": "3.0", "phase": "1+2"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Phase 1 endpoints ─────────────────────────────────────────
@app.get("/info/{ticker_sym}")
def get_info(ticker_sym: str):
    sym = ticker_sym.upper()
    try:
        info  = get_info_v8(sym)
        price = info.get("last_price")
        if not price:
            raise HTTPException(status_code=404,
                detail=f"No data for '{sym}'. Check format: AAPL, RELIANCE.NS, BTC-USD")
        mcap = info.get("market_cap")
        return {
            "ticker": sym, "name": info.get("name", sym),
            "sector": info.get("sector","N/A"), "industry": info.get("industry","N/A"),
            "exchange": info.get("exchange","N/A"), "currency": info.get("currency","USD"),
            "market_cap": mcap,
            "market_cap_fmt": (f"${mcap/1e12:.2f}T" if mcap and mcap>=1e12 else
                               f"${mcap/1e9:.1f}B"   if mcap and mcap>=1e9  else
                               f"${mcap/1e6:.0f}M"   if mcap else "N/A"),
            "current_price":  round(float(price), 4),
            "previous_close": info.get("previous_close"),
            "52w_high": info.get("52w_high"), "52w_low": info.get("52w_low"),
            "pe_ratio": None, "forward_pe": None, "dividend_yield": None,
            "beta": None, "avg_volume": None, "description": info.get("description",""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Info error [{sym}]: {e}")

@app.get("/indicators/{ticker_sym}")
def get_indicators(ticker_sym: str, period: str = "6mo"):
    sym = ticker_sym.upper()
    try:
        df = fetch_history(sym, period=period)
        if len(df) < 30:
            raise HTTPException(status_code=400, detail=f"Only {len(df)} rows.")
        df.attrs["ticker"] = sym
        return compute_all_indicators(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indicators error [{sym}]: {e}")

@app.get("/sentiment/{ticker_sym}")
def get_sentiment(ticker_sym: str):
    try:
        return analyze_news_sentiment(ticker_sym.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/earnings/{ticker_sym}")
def get_earnings(ticker_sym: str):
    try:
        return get_earnings_data(ticker_sym.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ScreenRequest(BaseModel):
    tickers: List[str]
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    signal:  Optional[str]   = None

@app.post("/screen")
def screen(req: ScreenRequest):
    try:
        results = screen_tickers(tickers=[t.upper() for t in req.tickers],
                                 min_rsi=req.min_rsi, max_rsi=req.max_rsi,
                                 signal_filter=req.signal)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 2: Backtesting ──────────────────────────────────────
@app.get("/strategies")
def list_strategies():
    """List all available strategies and their default parameters."""
    return {
        name: {"params": info["params"]}
        for name, info in STRATEGIES.items()
    }

class BacktestRequest(BaseModel):
    ticker:           str
    strategy:         str
    period:           str   = "2y"
    initial_capital:  float = 10000.0
    commission:       float = 0.001
    params:           Optional[Dict[str, Any]] = None
    walk_forward:     bool  = True
    wf_windows:       int   = 5

@app.post("/backtest")
def backtest(req: BacktestRequest):
    sym = req.ticker.upper()
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400,
            detail=f"Unknown strategy '{req.strategy}'. Available: {list(STRATEGIES.keys())}")
    try:
        df  = fetch_history(sym, period=req.period)
        if len(df) < 60:
            raise HTTPException(status_code=400,
                detail=f"Only {len(df)} bars — need ≥60 for backtesting. Use period='2y'.")

        strat   = STRATEGIES[req.strategy]
        fn      = strat["fn"]
        params  = {**strat["params"], **(req.params or {})}

        result  = run_backtest(df, fn, params, req.initial_capital, req.commission)
        result["ticker"]   = sym
        result["strategy"] = req.strategy
        result["period"]   = req.period

        if req.walk_forward and len(df) >= 120:
            result["walk_forward"] = walk_forward(
                df, fn, params, n_windows=req.wf_windows,
                initial_capital=req.initial_capital, commission=req.commission
            )
        else:
            result["walk_forward"] = []

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error [{sym}]: {e}")


# ── Phase 2: Portfolio Optimizer ──────────────────────────────
class PortfolioRequest(BaseModel):
    tickers:      List[str]
    period:       str   = "2y"
    n_portfolios: int   = 3000

@app.post("/portfolio/optimize")
def portfolio_optimize(req: PortfolioRequest):
    if len(req.tickers) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tickers.")
    if len(req.tickers) > 15:
        raise HTTPException(status_code=400, detail="Max 15 tickers per optimization.")
    try:
        tickers = [t.upper() for t in req.tickers]
        prices  = {}
        for t in tickers:
            df        = fetch_history(t, period=req.period)
            prices[t] = df["Close"]

        price_df   = pd.DataFrame(prices).dropna()
        returns_df = price_df.pct_change().dropna()

        return optimize_portfolio(tickers, returns_df,
                                  n_portfolios=min(req.n_portfolios, 5000))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)), reload=False)


# ── Phase 3: AI Analysis (single combined endpoint) ───────────

def _tomorrow_signal(ensemble: dict, forecast: dict, regime: str) -> dict:
    """
    Derives a tomorrow action from ensemble score + forecast day-1 range + regime.
    Confidence is intentionally capped at MODERATE — honest about model limits.
    """
    score = ensemble.get("score", 50) or 50
    buy_c = ensemble.get("buy_components", 0)
    sel_c = ensemble.get("sell_components", 0)
    components = ensemble.get("components", {})

    # ── Action ────────────────────────────────────────────────
    if   score >= 72: action, color, arrow = "BUY",       "#3fb950", "▲"
    elif score >= 58: action, color, arrow = "WEAK BUY",  "#7ce08a", "↗"
    elif score >= 42: action, color, arrow = "HOLD",      "#8b949e", "→"
    elif score >= 28: action, color, arrow = "WEAK SELL", "#ffa657", "↘"
    else:             action, color, arrow = "SELL",      "#f85149", "▼"

    # ── Confidence (honest — never HIGH) ──────────────────────
    diff = abs(score - 50)
    if   diff >= 22: confidence, conf_color = "MODERATE", "#d29922"
    elif diff >= 12: confidence, conf_color = "LOW",      "#8b949e"
    else:            confidence, conf_color = "VERY LOW", "#6e7681"

    # ── Tomorrow price range from forecast day 1 ──────────────
    bands  = forecast.get("bands", {})
    last   = forecast.get("last_price") or 0
    t_p25  = (bands.get("p25") or [None])[0]
    t_p50  = (bands.get("p50") or [None])[0]
    t_p75  = (bands.get("p75") or [None])[0]
    t_p5   = (bands.get("p5")  or [None])[0]
    t_p95  = (bands.get("p95") or [None])[0]
    prob_gain = forecast.get("prob_gain", 50) or 50

    # ── Top reasons ───────────────────────────────────────────
    reasons = []
    # Most influential indicators (sorted by abs contribution)
    for name, data in sorted(components.items(),
                              key=lambda x: abs(x[1].get("score",0.5)-0.5),
                              reverse=True)[:3]:
        sig = data.get("signal","NEUTRAL")
        val = data.get("value","")
        reasons.append(f"{name}: {sig} ({val})")

    if buy_c > sel_c:
        summary = f"{buy_c} of 7 indicators are bullish vs {sel_c} bearish"
    elif sel_c > buy_c:
        summary = f"{sel_c} of 7 indicators are bearish vs {buy_c} bullish"
    else:
        summary = f"Indicators split equally — mixed signals, low conviction"

    return {
        "action":       action,
        "color":        color,
        "arrow":        arrow,
        "score":        score,
        "confidence":   confidence,
        "conf_color":   conf_color,
        "prob_gain":    round(prob_gain, 1),
        "regime":       regime,
        "tomorrow_range": {
            "p5":    round(t_p5,  2) if t_p5  else None,
            "p25":   round(t_p25, 2) if t_p25 else None,
            "p50":   round(t_p50, 2) if t_p50 else None,
            "p75":   round(t_p75, 2) if t_p75 else None,
            "p95":   round(t_p95, 2) if t_p95 else None,
            "last":  round(last,  2) if last  else None,
        },
        "summary":     summary,
        "top_reasons": reasons,
        "disclaimer":  (
            "Signal based on historical indicator patterns. "
            "Confidence is capped at MODERATE — no model predicts tomorrow reliably. "
            "Use as one input, not a trading instruction."
        ),
    }


from regime      import detect_regime
from forecaster  import probabilistic_forecast
from ensemble    import compute_ensemble

@app.get("/analysis/{ticker_sym}")
def get_analysis(ticker_sym: str, period: str = "1y",
                 forecast_days: int = 10, n_sim: int = 500):
    """
    Phase 3 combined endpoint — runs all 3 analyses in one call.
    Returns: regime detection + probabilistic forecast + ensemble score.
    Single call reduces API overhead on Railway free tier.
    """
    sym = ticker_sym.upper()
    try:
        df = fetch_history(sym, period=period)
        if len(df) < 60:
            raise HTTPException(status_code=400,
                detail=f"Only {len(df)} bars — need ≥60 for Phase 3 analysis.")

        # 1. Regime detection
        try:
            regime_data = detect_regime(df)
        except Exception as e:
            regime_data = {"error": str(e), "current_regime": "Unknown"}

        # 2. Probabilistic forecast
        try:
            forecast_data = probabilistic_forecast(
                df, days=min(forecast_days, 20), n_sim=min(n_sim, 500)
            )
        except Exception as e:
            forecast_data = {"error": str(e)}

        # 3. Ensemble score (uses regime for adaptive weighting)
        try:
            current_regime = regime_data.get("current_regime", "default")
            ensemble_data  = compute_ensemble(df, regime=current_regime)
        except Exception as e:
            ensemble_data = {"error": str(e)}

        # Derive tomorrow signal from all 3 components
        try:
            tomorrow = _tomorrow_signal(
                ensemble_data,
                forecast_data,
                regime_data.get("current_regime","default")
            )
        except Exception as e:
            tomorrow = {"error": str(e)}

        return {
            "ticker":   sym,
            "period":   period,
            "regime":   regime_data,
            "forecast": forecast_data,
            "ensemble": ensemble_data,
            "tomorrow": tomorrow,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error [{sym}]: {e}")
