"""
indicators.py — Technical Indicator Engine (Phase 1)
Computes RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV, Moving Averages
and derives actionable signals from each.
"""
import numpy as np
import pandas as pd


def _safe(val, decimals=4):
    """Convert NaN/inf → None for clean JSON serialization."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, decimals)
    except Exception:
        return None


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def compute_bollinger(close: pd.Series, period=20, width=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + width * std, mid, mid - width * std


def compute_stochastic(high, low, close, k=14, d=3):
    ll    = low.rolling(k).min()
    hh    = high.rolling(k).max()
    pct_k = 100 * (close - ll) / (hh - ll + 1e-9)
    return pct_k, pct_k.rolling(d).mean()


def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def _sig(indicator, signal, reason, strength):
    return {"indicator": indicator, "signal": signal, "reason": reason, "strength": strength}


def compute_all_indicators(df: pd.DataFrame) -> dict:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    rsi                    = compute_rsi(c)
    macd_l, macd_s, macd_h = compute_macd(c)
    bb_u,   bb_m,   bb_l  = compute_bollinger(c)
    stoch_k, stoch_d       = compute_stochastic(h, l, c)
    atr                    = compute_atr(h, l, c)
    obv                    = compute_obv(c, v)
    sma20                  = c.rolling(20).mean()
    sma50                  = c.rolling(50).mean()
    sma200                 = c.rolling(200).mean()
    ema9                   = c.ewm(span=9,  adjust=False).mean()
    ema21                  = c.ewm(span=21, adjust=False).mean()

    # Latest values
    price     = c.iloc[-1]
    rsi_v     = rsi.iloc[-1]
    mach_v    = macd_h.iloc[-1]
    mach_prev = macd_h.iloc[-2]
    bbu_v     = bb_u.iloc[-1]
    bbl_v     = bb_l.iloc[-1]
    bbm_v     = bb_m.iloc[-1]
    sk_v      = stoch_k.iloc[-1]
    sd_v      = stoch_d.iloc[-1]
    s50       = sma50.iloc[-1]
    s200      = sma200.iloc[-1]
    s50_p     = sma50.iloc[-2]
    s200_p    = sma200.iloc[-2]

    signals = []

    # ── RSI ─────────────────────────────────────────────────
    if rsi_v < 30:
        signals.append(_sig("RSI", "BUY",       f"RSI={rsi_v:.1f} — Oversold (<30)",           "STRONG"))
    elif rsi_v > 70:
        signals.append(_sig("RSI", "SELL",      f"RSI={rsi_v:.1f} — Overbought (>70)",          "STRONG"))
    elif rsi_v < 45:
        signals.append(_sig("RSI", "WEAK SELL", f"RSI={rsi_v:.1f} — Bearish territory",         "MODERATE"))
    elif rsi_v > 55:
        signals.append(_sig("RSI", "WEAK BUY",  f"RSI={rsi_v:.1f} — Bullish territory",         "MODERATE"))
    else:
        signals.append(_sig("RSI", "NEUTRAL",   f"RSI={rsi_v:.1f} — Mid-range",                 "WEAK"))

    # ── MACD ────────────────────────────────────────────────
    if mach_v > 0 and mach_prev <= 0:
        signals.append(_sig("MACD", "BUY",       "Bullish crossover — MACD crossed above signal", "STRONG"))
    elif mach_v < 0 and mach_prev >= 0:
        signals.append(_sig("MACD", "SELL",      "Bearish crossover — MACD crossed below signal", "STRONG"))
    elif mach_v > 0:
        signals.append(_sig("MACD", "WEAK BUY",  "MACD histogram positive",                       "MODERATE"))
    else:
        signals.append(_sig("MACD", "WEAK SELL", "MACD histogram negative",                       "MODERATE"))

    # ── Bollinger Bands ──────────────────────────────────────
    pct_b = (price - bbl_v) / (bbu_v - bbl_v + 1e-9) * 100
    if price < bbl_v:
        signals.append(_sig("Bollinger Bands", "BUY",     "Price below lower band — oversold",      "MODERATE"))
    elif price > bbu_v:
        signals.append(_sig("Bollinger Bands", "SELL",    "Price above upper band — overbought",     "MODERATE"))
    else:
        signals.append(_sig("Bollinger Bands", "NEUTRAL", f"%B = {pct_b:.1f}% — within bands",      "WEAK"))

    # ── Golden / Death Cross ─────────────────────────────────
    if not pd.isna(s50) and not pd.isna(s200):
        if s50 > s200 and s50_p <= s200_p:
            signals.append(_sig("MA Cross", "BUY",       "🌟 Golden Cross — 50 SMA crossed above 200 SMA", "STRONG"))
        elif s50 < s200 and s50_p >= s200_p:
            signals.append(_sig("MA Cross", "SELL",      "💀 Death Cross — 50 SMA crossed below 200 SMA",  "STRONG"))
        elif s50 > s200:
            signals.append(_sig("MA Cross", "WEAK BUY",  "50 SMA above 200 SMA — established uptrend",     "MODERATE"))
        else:
            signals.append(_sig("MA Cross", "WEAK SELL", "50 SMA below 200 SMA — established downtrend",   "MODERATE"))

    # ── Stochastic ───────────────────────────────────────────
    if sk_v < 20:
        signals.append(_sig("Stochastic", "BUY",     f"K={sk_v:.1f} D={sd_v:.1f} — Oversold",   "MODERATE"))
    elif sk_v > 80:
        signals.append(_sig("Stochastic", "SELL",    f"K={sk_v:.1f} D={sd_v:.1f} — Overbought", "MODERATE"))
    else:
        signals.append(_sig("Stochastic", "NEUTRAL", f"K={sk_v:.1f} D={sd_v:.1f}",              "WEAK"))

    # ── Overall score ────────────────────────────────────────
    buy_c  = sum(1 for s in signals if "BUY"  in s["signal"])
    sell_c = sum(1 for s in signals if "SELL" in s["signal"])
    if   buy_c  >= 4: overall = "STRONG BUY"
    elif buy_c  >  sell_c: overall = "BUY"
    elif sell_c >= 4: overall = "STRONG SELL"
    elif sell_c >  buy_c:  overall = "SELL"
    else:              overall = "NEUTRAL"

    # ── History (last 90 days for charts) ────────────────────
    n = 90
    def safe_list(s):
        return [_safe(x) for x in s.tail(n)]

    return {
        "ticker":         df.attrs.get("ticker", ""),
        "price":          _safe(price, 2),
        "overall_signal": overall,
        "buy_count":      buy_c,
        "sell_count":     sell_c,
        "signals":        signals,
        "indicators": {
            "rsi":            _safe(rsi_v,          2),
            "macd":           _safe(macd_l.iloc[-1]),
            "macd_signal":    _safe(macd_s.iloc[-1]),
            "macd_histogram": _safe(mach_v),
            "bb_upper":       _safe(bbu_v,           2),
            "bb_middle":      _safe(bbm_v,           2),
            "bb_lower":       _safe(bbl_v,           2),
            "sma_20":         _safe(sma20.iloc[-1],  2),
            "sma_50":         _safe(s50,             2),
            "sma_200":        _safe(s200,            2),
            "ema_9":          _safe(ema9.iloc[-1],   2),
            "ema_21":         _safe(ema21.iloc[-1],  2),
            "stoch_k":        _safe(sk_v,            2),
            "stoch_d":        _safe(sd_v,            2),
            "atr":            _safe(atr.iloc[-1],    4),
            "obv":            _safe(obv.iloc[-1],    0),
        },
        "history": {
            "dates":          [str(d.date()) for d in df.index[-n:]],
            "open":           safe_list(df["Open"]),
            "high":           safe_list(df["High"]),
            "low":            safe_list(df["Low"]),
            "close":          safe_list(c),
            "volume":         [int(x) if not pd.isna(x) else 0 for x in v.tail(n)],
            "sma20":          safe_list(sma20),
            "sma50":          safe_list(sma50),
            "sma200":         safe_list(sma200),
            "bb_upper":       safe_list(bb_u),
            "bb_middle":      safe_list(bb_m),
            "bb_lower":       safe_list(bb_l),
            "rsi":            safe_list(rsi),
            "macd_line":      safe_list(macd_l),
            "macd_signal":    safe_list(macd_s),
            "macd_histogram": safe_list(macd_h),
        },
    }
