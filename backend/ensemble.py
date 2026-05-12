"""
ensemble.py — Ensemble Signal Engine (Phase 3)
Combines ALL Phase 1 indicators + Phase 2 regime context into a
single 0–100 confidence score with BUY/HOLD/SELL verdict.
Weights are regime-adaptive — momentum signals count more in trends,
mean-reversion signals count more in consolidation.
"""
import numpy as np
import pandas as pd
from indicators import (compute_rsi, compute_macd, compute_bollinger,
                        compute_stochastic, compute_atr)

# ── Regime-adaptive weights ────────────────────────────────────
# Each weight set is tuned for a specific market regime.
# Momentum signals (MACD, MA) dominate in trends.
# Mean-reversion signals (RSI, BB, Stoch) dominate in consolidation.
REGIME_WEIGHTS = {
    "Bull Trend": {
        "rsi": 0.10, "macd": 0.25, "ma_cross": 0.25,
        "bollinger": 0.10, "stochastic": 0.10, "volume": 0.10, "atr_trend": 0.10,
    },
    "Bear Trend": {
        "rsi": 0.10, "macd": 0.25, "ma_cross": 0.25,
        "bollinger": 0.10, "stochastic": 0.10, "volume": 0.10, "atr_trend": 0.10,
    },
    "High Volatility": {
        "rsi": 0.20, "macd": 0.15, "ma_cross": 0.10,
        "bollinger": 0.25, "stochastic": 0.15, "volume": 0.10, "atr_trend": 0.05,
    },
    "Consolidation": {
        "rsi": 0.22, "macd": 0.12, "ma_cross": 0.08,
        "bollinger": 0.25, "stochastic": 0.20, "volume": 0.08, "atr_trend": 0.05,
    },
    "default": {
        "rsi": 0.15, "macd": 0.20, "ma_cross": 0.20,
        "bollinger": 0.15, "stochastic": 0.15, "volume": 0.08, "atr_trend": 0.07,
    },
}

SCORE_LABELS = [
    (80, "STRONG BUY",  "#1a7f37"),
    (65, "BUY",         "#3fb950"),
    (55, "WEAK BUY",    "#7ce08a"),
    (45, "NEUTRAL",     "#8b949e"),
    (35, "WEAK SELL",   "#ffa657"),
    (20, "SELL",        "#f85149"),
    ( 0, "STRONG SELL", "#8b0000"),
]


def _norm(v, lo, hi):
    """Normalise a value to [0,1] between lo and hi."""
    return float(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1))


def compute_ensemble(df: pd.DataFrame, regime: str = "default") -> dict:
    c   = df["Close"]
    h   = df["High"]
    l   = df["Low"]
    v   = df["Volume"]
    w   = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["default"])

    component_scores = {}
    component_detail  = {}

    # ── RSI (0–1, higher = more bullish) ──────────────────────
    rsi     = compute_rsi(c)
    rsi_v   = float(rsi.iloc[-1])
    rsi_s   = _norm(rsi_v, 20, 80)          # 20→0 (very oversold→bullish) reversed
    rsi_s   = 1 - _norm(rsi_v, 20, 80) if rsi_v < 50 else _norm(rsi_v, 20, 80)
    # More intuitive: oversold (30) = high score (bullish), overbought (70) = low score
    if rsi_v <= 30:   rsi_s = 0.90
    elif rsi_v <= 45: rsi_s = 0.65
    elif rsi_v <= 55: rsi_s = 0.50
    elif rsi_v <= 70: rsi_s = 0.40
    else:             rsi_s = 0.15
    component_scores["rsi"]  = rsi_s
    component_detail["RSI"]  = {"value": round(rsi_v,1), "score": round(rsi_s,3),
                                  "signal": "BUY" if rsi_v<45 else "SELL" if rsi_v>65 else "NEUTRAL"}

    # ── MACD ──────────────────────────────────────────────────
    macd_l, macd_s, macd_h = compute_macd(c)
    hist_v   = float(macd_h.iloc[-1])
    hist_prev = float(macd_h.iloc[-2])
    # Crossover bonus
    if hist_v > 0 and hist_prev <= 0:   macd_score = 0.85
    elif hist_v < 0 and hist_prev >= 0: macd_score = 0.15
    elif hist_v > 0:                    macd_score = 0.65
    else:                               macd_score = 0.35
    component_scores["macd"]  = macd_score
    component_detail["MACD"]  = {"value": round(hist_v,4), "score": round(macd_score,3),
                                   "signal": "BUY" if macd_score>0.5 else "SELL"}

    # ── MA Cross ──────────────────────────────────────────────
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    price  = float(c.iloc[-1])
    s50    = float(sma50.iloc[-1])  if not pd.isna(sma50.iloc[-1])  else price
    s200   = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else price
    s50_p  = float(sma50.iloc[-2])  if not pd.isna(sma50.iloc[-2])  else s50
    s200_p = float(sma200.iloc[-2]) if not pd.isna(sma200.iloc[-2]) else s200
    if s50 > s200 and s50_p <= s200_p:    ma_score = 0.95  # Golden cross
    elif s50 < s200 and s50_p >= s200_p:  ma_score = 0.05  # Death cross
    elif s50 > s200:                       ma_score = 0.70
    else:                                  ma_score = 0.30
    component_scores["ma_cross"]  = ma_score
    component_detail["MA Cross"]  = {"value": f"SMA50={round(s50,2)}", "score": round(ma_score,3),
                                      "signal": "BUY" if ma_score>0.5 else "SELL"}

    # ── Bollinger Bands ────────────────────────────────────────
    bb_u, bb_m, bb_b = compute_bollinger(c)
    bbu = float(bb_u.iloc[-1])
    bbl = float(bb_b.iloc[-1])
    pct_b = (price - bbl) / (bbu - bbl + 1e-9)
    if   pct_b < 0:    bb_score = 0.88
    elif pct_b < 0.2:  bb_score = 0.72
    elif pct_b < 0.4:  bb_score = 0.58
    elif pct_b < 0.6:  bb_score = 0.50
    elif pct_b < 0.8:  bb_score = 0.38
    elif pct_b < 1.0:  bb_score = 0.22
    else:              bb_score = 0.12
    component_scores["bollinger"]  = bb_score
    component_detail["Bollinger"]  = {"value": round(pct_b,3), "score": round(bb_score,3),
                                       "signal": "BUY" if bb_score>0.55 else "SELL" if bb_score<0.45 else "NEUTRAL"}

    # ── Stochastic ────────────────────────────────────────────
    sk, sd = compute_stochastic(h, l, c)
    sk_v = float(sk.iloc[-1])
    if   sk_v < 20:  stoch_score = 0.85
    elif sk_v < 40:  stoch_score = 0.62
    elif sk_v < 60:  stoch_score = 0.50
    elif sk_v < 80:  stoch_score = 0.38
    else:            stoch_score = 0.15
    component_scores["stochastic"]  = stoch_score
    component_detail["Stochastic"]  = {"value": round(sk_v,1), "score": round(stoch_score,3),
                                        "signal": "BUY" if sk_v<30 else "SELL" if sk_v>70 else "NEUTRAL"}

    # ── Volume confirmation ────────────────────────────────────
    vol_ma  = v.rolling(20).mean()
    vol_r   = float(v.iloc[-1]) / (float(vol_ma.iloc[-1]) + 1e-9)
    price_r = float(c.iloc[-1]) > float(c.iloc[-2])
    if   vol_r > 1.5 and price_r:  vol_score = 0.80   # High volume + up
    elif vol_r > 1.5 and not price_r: vol_score = 0.20 # High volume + down
    elif vol_r < 0.7:               vol_score = 0.45   # Low volume = weak signal
    else:                           vol_score = 0.55   # Normal volume
    component_scores["volume"]  = vol_score
    component_detail["Volume"]  = {"value": round(vol_r,2), "score": round(vol_score,3),
                                    "signal": "BUY" if vol_score>0.6 else "SELL" if vol_score<0.4 else "NEUTRAL"}

    # ── ATR Trend ─────────────────────────────────────────────
    atr     = compute_atr(h, l, c)
    ema20   = c.ewm(span=20, adjust=False).mean()
    atr_v   = float(atr.iloc[-1])
    ema_v   = float(ema20.iloc[-1])
    upper   = ema_v + 2 * atr_v
    lower_b = ema_v - 2 * atr_v
    if   price > upper:  atr_score = 0.80
    elif price > ema_v:  atr_score = 0.62
    elif price < lower_b: atr_score = 0.20
    else:                atr_score = 0.38
    component_scores["atr_trend"]  = atr_score
    component_detail["ATR Trend"]  = {"value": round(atr_v,4), "score": round(atr_score,3),
                                       "signal": "BUY" if atr_score>0.55 else "SELL" if atr_score<0.45 else "NEUTRAL"}

    # ── Weighted final score ───────────────────────────────────
    final = sum(component_scores[k] * w[k] for k in component_scores)
    score_0_100 = round(final * 100, 1)

    # Label
    label, color = "NEUTRAL", "#8b949e"
    for threshold, lbl, clr in SCORE_LABELS:
        if score_0_100 >= threshold:
            label, color = lbl, clr
            break

    buy_components  = sum(1 for k,v in component_scores.items() if v > 0.55)
    sell_components = sum(1 for k,v in component_scores.items() if v < 0.45)

    return {
        "score":           score_0_100,
        "label":           label,
        "color":           color,
        "regime_used":     regime,
        "buy_components":  buy_components,
        "sell_components": sell_components,
        "components":      component_detail,
        "weights_used":    {k: round(v,3) for k,v in w.items()},
    }
