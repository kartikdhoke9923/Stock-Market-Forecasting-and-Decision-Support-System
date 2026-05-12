"""
forecaster.py — Probabilistic Price Forecasting (Phase 3)
Method: ARIMA(1,1,1) trend + EWMA-GARCH volatility + Bootstrap Monte Carlo
Returns fan chart bands (p5/p25/p50/p75/p95) — honest uncertainty, not fake point predictions.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

def _safe(v, dec=4):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, dec)
    except Exception:
        return None


def _ewma_volatility(log_returns: np.ndarray, lam: float = 0.94) -> float:
    """
    RiskMetrics EWMA volatility — JP Morgan's standard, lightweight GARCH alternative.
    λ=0.94 is the industry standard for daily data.
    """
    vol = float(np.std(log_returns[-60:]))
    for r in log_returns[-120:]:
        vol = float(np.sqrt(lam * vol**2 + (1 - lam) * r**2))
    return vol


def _arima_drift(log_returns: np.ndarray) -> float:
    """
    ARIMA(1,1,1) simplified: estimate drift from recent returns using
    statsmodels SARIMAX. Falls back to rolling mean if statsmodels unavailable.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = SARIMAX(log_returns[-120:], order=(1, 0, 1),
                             trend="c", enforce_stationarity=False)
            result = model.fit(disp=False, maxiter=50)
            # One-step forecast drift
            return float(result.forecast(1)[0])
    except Exception:
        return float(np.mean(log_returns[-60:]))


def probabilistic_forecast(df: pd.DataFrame, days: int = 10,
                            n_sim: int = 500) -> dict:
    """
    Bootstrap Monte Carlo forecast with GARCH-style volatility scaling.
    
    n_sim=500 is enough for stable percentiles and light on Railway free tier memory.
    """
    c        = df["Close"]
    log_ret  = np.log(c / c.shift(1)).dropna().values

    if len(log_ret) < 60:
        raise ValueError("Need at least 60 bars for forecasting.")

    last_price   = float(c.iloc[-1])
    daily_vol    = _ewma_volatility(log_ret)
    ann_vol      = daily_vol * np.sqrt(252) * 100
    drift        = _arima_drift(log_ret)

    # ── Bootstrap Monte Carlo ─────────────────────────────────
    np.random.seed(42)
    paths = np.zeros((n_sim, days))

    for i in range(n_sim):
        # Sample historical returns (captures fat tails + skew automatically)
        sampled = np.random.choice(log_ret[-252:], size=days, replace=True)
        # Scale by current GARCH vol / historical vol ratio
        hist_vol = float(np.std(log_ret[-252:]))
        vol_scale = daily_vol / (hist_vol + 1e-9)
        scaled   = sampled * vol_scale + drift * 0.3  # small drift weight
        paths[i] = last_price * np.exp(np.cumsum(scaled))

    # ── Percentile bands ──────────────────────────────────────
    pcts = {
        "p5":  np.percentile(paths, 5,  axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }

    # ── Forecast dates ────────────────────────────────────────
    last_date      = df.index[-1]
    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=days
    )

    final_prices   = paths[:, -1]
    prob_gain      = float(np.mean(final_prices > last_price) * 100)
    exp_low        = float(np.percentile(final_prices, 10))
    exp_high       = float(np.percentile(final_prices, 90))

    # Historical context (last 30 bars) for chart
    hist_dates  = [str(d.date()) for d in df.index[-30:]]
    hist_prices = [_safe(v, 4) for v in c.tail(30)]

    return {
        "last_price":      _safe(last_price, 4),
        "forecast_days":   days,
        "n_simulations":   n_sim,
        "daily_vol":       _safe(daily_vol, 6),
        "annual_vol_pct":  _safe(ann_vol, 2),
        "drift_daily":     _safe(drift, 6),
        "prob_gain":       _safe(prob_gain, 1),
        "expected_range": {
            "low":      _safe(exp_low, 2),
            "high":     _safe(exp_high, 2),
            "low_pct":  _safe((exp_low  - last_price) / last_price * 100, 2),
            "high_pct": _safe((exp_high - last_price) / last_price * 100, 2),
        },
        "bands": {k: [_safe(v, 4) for v in arr] for k, arr in pcts.items()},
        "forecast_dates":  [str(d.date()) for d in forecast_dates],
        "hist_dates":       hist_dates,
        "hist_prices":      hist_prices,
        "methodology":     "EWMA-GARCH volatility + ARIMA(1,1,1) drift + Bootstrap Monte Carlo",
    }
