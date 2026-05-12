"""
screener.py — Stock Screener (Phase 1)
Uses shared browser session from utils to avoid 429 errors.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from utils import fetch_history
from indicators import compute_rsi


def _quick(sym: str) -> Optional[dict]:
    try:
        df    = fetch_history(sym, period="3mo")
        if len(df) < 30:
            return None

        c     = df["Close"]
        v     = df["Volume"]
        price = float(c.iloc[-1])
        prev  = float(c.iloc[-2])
        chg   = (price - prev) / prev * 100

        rsi_s = compute_rsi(c)
        rsi_v = float(rsi_s.iloc[-1])
        sma20 = float(c.rolling(20).mean().iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else float("nan")
        vol_r = float(v.iloc[-1]) / float(v.rolling(20).mean().iloc[-1] + 1e-9)

        buy = sell = 0
        if rsi_v < 35:   buy  += 2
        elif rsi_v > 65: sell += 2
        if not np.isnan(sma20) and price > sma20: buy  += 1
        else:                                      sell += 1
        if not np.isnan(sma50) and price > sma50: buy  += 1
        else:                                      sell += 1

        signal = ("BUY"     if buy  > sell + 1 else
                  "SELL"    if sell > buy  + 1 else
                  "NEUTRAL")

        return {
            "ticker":      sym,
            "price":       round(price, 2),
            "change_1d":   round(chg, 2),
            "rsi":         round(rsi_v, 1),
            "above_sma20": bool(not np.isnan(sma20) and price > sma20),
            "above_sma50": bool(not np.isnan(sma50) and price > sma50),
            "vol_ratio":   round(vol_r, 2),
            "signal":      signal,
        }
    except Exception as e:
        return {"ticker": sym, "error": str(e), "signal": "ERROR"}


def screen_tickers(
    tickers: List[str],
    min_rsi: Optional[float] = None,
    max_rsi: Optional[float] = None,
    signal_filter: Optional[str] = None,
) -> List[dict]:
    results = []
    for t in tickers:
        data = _quick(t)
        if data is None or "error" in data:
            continue
        if min_rsi and data.get("rsi", 50) < min_rsi: continue
        if max_rsi and data.get("rsi", 50) > max_rsi: continue
        if signal_filter and data.get("signal") != signal_filter.upper(): continue
        results.append(data)
    return sorted(results, key=lambda x: x.get("rsi", 50))
