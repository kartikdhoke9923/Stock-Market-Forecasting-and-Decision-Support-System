"""
strategies.py — Trading Strategy Library (Phase 2)
Each function returns pd.Series: 1=BUY, -1=SELL, 0=HOLD
Signal on day T is acted on day T+1 open — zero lookahead bias.
"""
import numpy as np
import pandas as pd
from indicators import compute_rsi, compute_macd, compute_bollinger, compute_atr


def rsi_strategy(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """RSI Mean Reversion — buy on oversold cross, sell on overbought cross."""
    rsi    = compute_rsi(df["Close"])
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(rsi < oversold)  & (rsi.shift(1) >= oversold)]  = 1
    signal[(rsi > overbought) & (rsi.shift(1) <= overbought)] = -1
    return signal


def macd_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """MACD Momentum — buy on bullish histogram crossover, sell on bearish."""
    _, _, hist = compute_macd(df["Close"], fast=fast, slow=slow, signal=sig)
    signal     = pd.Series(0, index=df.index, dtype=int)
    signal[(hist > 0) & (hist.shift(1) <= 0)] = 1
    signal[(hist < 0) & (hist.shift(1) >= 0)] = -1
    return signal


def ma_cross_strategy(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Golden / Death Cross — 50 SMA vs 200 SMA crossover."""
    c      = df["Close"]
    maf    = c.rolling(fast).mean()
    mas    = c.rolling(slow).mean()
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(maf > mas) & (maf.shift(1) <= mas.shift(1))] = 1   # Golden Cross
    signal[(maf < mas) & (maf.shift(1) >= mas.shift(1))] = -1  # Death Cross
    return signal


def bollinger_strategy(df: pd.DataFrame, period: int = 20, width: float = 2.0) -> pd.Series:
    """Bollinger Band Mean Reversion — buy lower band touch, sell upper band touch."""
    c                  = df["Close"]
    bb_u, _, bb_l      = compute_bollinger(c, period=period, width=width)
    signal             = pd.Series(0, index=df.index, dtype=int)
    signal[(c < bb_l) & (c.shift(1) >= bb_l.shift(1))] = 1
    signal[(c > bb_u) & (c.shift(1) <= bb_u.shift(1))] = -1
    return signal


def atr_momentum_strategy(df: pd.DataFrame, ema_period: int = 20,
                           atr_period: int = 14, atr_mult: float = 2.0) -> pd.Series:
    """
    ATR Chandelier-style Trend Following (2024 favourite).
    Buy when price breaks above EMA + ATR band.
    Sell when price drops below EMA - ATR band.
    Adapts automatically to volatility — tighter in calm markets, wider in volatile ones.
    """
    c      = df["Close"]
    ema    = c.ewm(span=ema_period, adjust=False).mean()
    atr    = compute_atr(df["High"], df["Low"], c, period=atr_period)
    upper  = ema + atr_mult * atr
    lower  = ema - atr_mult * atr
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(c > upper) & (c.shift(1) <= upper.shift(1))] = 1
    signal[(c < lower) & (c.shift(1) >= lower.shift(1))] = -1
    return signal


def multi_factor_strategy(df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """
    Multi-Factor Consensus (most robust — no single indicator dominates).
    Scores each bar from -5 to +5 across RSI, MACD, MAs, Bollinger, Volume.
    Buy when consensus score >= threshold, Sell when <= -threshold.
    Ensemble approach reduces false signals significantly vs any single indicator.
    """
    c     = df["Close"]
    v     = df["Volume"]
    score = pd.Series(0.0, index=df.index)

    # RSI: ±1
    rsi = compute_rsi(c)
    score += (rsi < 35).astype(float)
    score -= (rsi > 65).astype(float)

    # MACD histogram: ±1
    _, _, hist = compute_macd(c)
    score += (hist > 0).astype(float)
    score -= (hist < 0).astype(float)

    # Price vs MAs: ±1
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    score += ((c > ma20) & (c > ma50)).astype(float)
    score -= ((c < ma20) & (c < ma50)).astype(float)

    # Bollinger %B: ±1
    bb_u, _, bb_l = compute_bollinger(c)
    pct_b         = (c - bb_l) / (bb_u - bb_l + 1e-9)
    score += (pct_b < 0.2).astype(float)
    score -= (pct_b > 0.8).astype(float)

    # Volume confirmation: ±0.5
    vol_ma = v.rolling(20).mean()
    score += ((v > vol_ma * 1.5) & (c > c.shift(1))).astype(float) * 0.5
    score -= ((v > vol_ma * 1.5) & (c < c.shift(1))).astype(float) * 0.5

    signal             = pd.Series(0, index=df.index, dtype=int)
    signal[score >=  threshold] = 1
    signal[score <= -threshold] = -1
    return signal


# Registry — used by API to look up strategy by name
STRATEGIES = {
    "RSI Mean Reversion":     {"fn": rsi_strategy,        "params": {"oversold": 30, "overbought": 70}},
    "MACD Momentum":          {"fn": macd_strategy,        "params": {"fast": 12, "slow": 26, "sig": 9}},
    "MA Cross (50/200)":      {"fn": ma_cross_strategy,    "params": {"fast": 50, "slow": 200}},
    "Bollinger Band":         {"fn": bollinger_strategy,   "params": {"period": 20, "width": 2.0}},
    "ATR Momentum":           {"fn": atr_momentum_strategy,"params": {"ema_period": 20, "atr_period": 14, "atr_mult": 2.0}},
    "Multi-Factor Consensus": {"fn": multi_factor_strategy,"params": {"threshold": 2.0}},
}
