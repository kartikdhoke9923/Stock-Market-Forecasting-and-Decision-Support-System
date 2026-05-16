"""
decision_engine.py — Final Decision Engine
Synthesizes ALL signals into one clear BUY/HOLD/SELL verdict with
entry zone, stop loss, take profit and plain English explanation.
"""
import numpy as np
import pandas as pd
from indicators import compute_atr

def _safe(v, dec=2):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, dec)
    except: return None

# ── Signal weights ─────────────────────────────────────────────
WEIGHTS = {
    "technical":       0.35,   # ensemble score (all indicators)
    "stock_sentiment": 0.10,   # Yahoo Finance news
    "macro_sentiment": 0.25,   # Global news (Reuters/BBC/AP)
    "earnings":        0.15,   # Earnings quality
    "forecast":        0.15,   # Probabilistic forecast direction
}

VERDICT_MAP = [
    (75, "STRONG BUY",  "#1a7f37", "▲▲"),
    (62, "BUY",         "#3fb950", "▲"),
    (52, "WEAK BUY",    "#7ce08a", "↗"),
    (48, "HOLD",        "#8b949e", "→"),
    (38, "WEAK SELL",   "#ffa657", "↘"),
    (25, "SELL",        "#f85149", "▼"),
    ( 0, "STRONG SELL", "#8b0000", "▼▼"),
]

CONFIDENCE_MAP = [
    (22, "MODERATE", "#d29922"),
    (12, "LOW",      "#8b949e"),
    ( 0, "VERY LOW", "#6e7681"),
]


def _score_to_0_100(val, lo=0, hi=1):
    return float(np.clip((val - lo) / (hi - lo + 1e-9), 0, 1) * 100)


def _sentiment_to_score(compound: float) -> float:
    """Convert VADER compound (-1 to +1) to 0-100 score."""
    return _score_to_0_100(compound, -1, 1)


def _earnings_to_score(earnings: dict) -> float:
    """Score earnings quality 0-100 based on EPS beat/miss history."""
    if not earnings:
        return 50.0
    history = earnings.get("earnings_history", [])
    if not history:
        return 50.0
    surprises = [h.get("surprise") for h in history[-4:] if h.get("surprise") is not None]
    if not surprises:
        return 50.0
    avg_surprise = float(np.mean(surprises))
    # +10% surprise → score 80, -10% → score 20
    score = 50 + avg_surprise * 3
    return float(np.clip(score, 10, 90))


def _entry_stop_target(df: pd.DataFrame, forecast: dict, last_price: float):
    """Calculate entry zone, stop loss, take profit from ATR + forecast bands."""
    atr     = compute_atr(df["High"], df["Low"], df["Close"])
    atr_val = float(atr.iloc[-1])
    bands   = forecast.get("bands", {})

    p25 = (bands.get("p25") or [None])[0]
    p50 = (bands.get("p50") or [None])[0]
    p75 = (bands.get("p75") or [None])[0]

    entry_low  = _safe(p25 or last_price * 0.985)
    entry_high = _safe(p50 or last_price * 1.002)
    stop_loss  = _safe((entry_low or last_price) - 1.5 * atr_val)
    take_profit= _safe(p75 or last_price * 1.05)

    entry_mid  = ((entry_low or last_price) + (entry_high or last_price)) / 2
    risk       = entry_mid - (stop_loss or entry_mid * 0.97)
    reward     = (take_profit or entry_mid * 1.05) - entry_mid
    rr_ratio   = _safe(reward / risk) if risk > 0 else None

    return {
        "entry_low":   entry_low,
        "entry_high":  entry_high,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "atr":         _safe(atr_val, 4),
        "rr_ratio":    rr_ratio,
        "stop_pct":    _safe((stop_loss - entry_mid) / entry_mid * 100) if stop_loss else None,
        "target_pct":  _safe((take_profit - entry_mid) / entry_mid * 100) if take_profit else None,
    }


def _plain_english(verdict, score, macro, stock_sent, earnings_sc, tech_sc, risks):
    lines = []
    # Lead sentence
    if score >= 62:
        lines.append(f"Most signals agree this stock is worth considering for a BUY.")
    elif score <= 38:
        lines.append(f"Most signals suggest caution — more bearish signals than bullish.")
    else:
        lines.append(f"Signals are mixed. No strong directional edge right now.")

    # Technical
    if tech_sc >= 60:
        lines.append(f"Technical indicators are mostly bullish (score {tech_sc:.0f}/100).")
    elif tech_sc <= 40:
        lines.append(f"Technical indicators are mostly bearish (score {tech_sc:.0f}/100).")

    # Macro
    macro_sig = macro.get("macro_signal","NEUTRAL")
    risk_lv   = macro.get("risk_level","MODERATE")
    hi_count  = macro.get("high_impact_count", 0)
    if hi_count > 0:
        lines.append(f"⚠️ {hi_count} high-impact global event(s) affect this stock's sector.")
    if "BEARISH" in macro_sig:
        lines.append(f"Global macro news is negative — {macro.get('summary','')}")
    elif "BULLISH" in macro_sig:
        lines.append(f"Global macro news is supportive — {macro.get('summary','')}")

    # Earnings
    if earnings_sc >= 65:
        lines.append("Recent earnings have beaten estimates — company fundamentals are solid.")
    elif earnings_sc <= 35:
        lines.append("Recent earnings have missed estimates — fundamental weakness.")

    # Risks
    if risks:
        lines.append("Key risks: " + " | ".join(risks[:3]))

    return " ".join(lines)


def compute_decision(ticker, ensemble, stock_sentiment,
                     macro_sentiment, earnings, forecast, df) -> dict:
    last_price = forecast.get("last_price") or 0

    # ── Component scores (0-100) ──────────────────────────────
    tech_sc    = float(ensemble.get("score", 50) or 50)
    stock_sc   = _sentiment_to_score(stock_sentiment.get("overall_score", 0) or 0)
    macro_sc   = _sentiment_to_score(macro_sentiment.get("overall_score", 0) or 0)
    earn_sc    = _earnings_to_score(earnings)
    fcst_sc    = float(forecast.get("prob_gain", 50) or 50)

    # ── Weighted final score ──────────────────────────────────
    final = (
        WEIGHTS["technical"]       * tech_sc  +
        WEIGHTS["stock_sentiment"] * stock_sc +
        WEIGHTS["macro_sentiment"] * macro_sc +
        WEIGHTS["earnings"]        * earn_sc  +
        WEIGHTS["forecast"]        * fcst_sc
    )
    final = round(final, 1)

    # ── Verdict ───────────────────────────────────────────────
    verdict = color = arrow = ""
    for threshold, v, c, a in VERDICT_MAP:
        if final >= threshold:
            verdict, color, arrow = v, c, a
            break

    # ── Confidence ────────────────────────────────────────────
    diff = abs(final - 50)
    confidence = conf_color = ""
    for threshold, conf, cc in CONFIDENCE_MAP:
        if diff >= threshold:
            confidence, conf_color = conf, cc
            break

    # ── Entry / Stop / Target ─────────────────────────────────
    levels = {}
    if df is not None and last_price:
        try:
            levels = _entry_stop_target(df, forecast, last_price)
        except Exception:
            pass

    # ── Risk factors ──────────────────────────────────────────
    risks = []
    rsi_val = ensemble.get("components", {}).get("RSI", {}).get("value", 50)
    try:
        if float(rsi_val) > 68: risks.append("RSI near overbought")
        if float(rsi_val) < 32: risks.append("RSI near oversold (high volatility)")
    except: pass
    if macro_sentiment.get("risk_level") in ("HIGH","ELEVATED"):
        risks.append(f"Global macro risk: {macro_sentiment.get('risk_level')}")
    stoch = ensemble.get("components", {}).get("Stochastic", {}).get("value", 50)
    try:
        if float(stoch) > 85: risks.append("Stochastic extremely overbought")
    except: pass
    bb_val = ensemble.get("components", {}).get("Bollinger", {}).get("value", 0.5)
    try:
        if float(bb_val) > 0.95: risks.append("Price above Bollinger upper band")
    except: pass

    plain = _plain_english(verdict, final, macro_sentiment,
                           stock_sc, earn_sc, tech_sc, risks)

    return {
        "ticker":       ticker,
        "final_score":  final,
        "verdict":      verdict,
        "color":        color,
        "arrow":        arrow,
        "confidence":   confidence,
        "conf_color":   conf_color,
        "component_scores": {
            "technical":       round(tech_sc, 1),
            "stock_sentiment": round(stock_sc, 1),
            "macro_sentiment": round(macro_sc, 1),
            "earnings":        round(earn_sc, 1),
            "forecast":        round(fcst_sc, 1),
        },
        "weights": WEIGHTS,
        "levels":       levels,
        "risks":        risks,
        "last_price":   _safe(last_price, 4),
        "plain_english": plain,
        "macro_signal": macro_sentiment.get("macro_signal","NEUTRAL"),
        "macro_risk":   macro_sentiment.get("risk_level","UNKNOWN"),
    }
