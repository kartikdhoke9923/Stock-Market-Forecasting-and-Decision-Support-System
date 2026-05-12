"""
regime.py — Market Regime Detection (Phase 3)
Uses GaussianMixture (sklearn) to identify 4 market states:
Bull Trend / Bear Trend / High Volatility / Consolidation
Each regime carries a strategy recommendation.
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from indicators import compute_rsi, compute_atr

REGIME_COLORS = {
    "Bull Trend":       "#3fb950",
    "Bear Trend":       "#f85149",
    "High Volatility":  "#d29922",
    "Consolidation":    "#58a6ff",
}

REGIME_STRATEGIES = {
    "Bull Trend":      "MOMENTUM — Trend-following strategies work best. MA Cross, ATR Momentum. Stay long, let winners run.",
    "Bear Trend":      "DEFENSIVE — Reduce exposure. Cash is a position. Avoid dip-buying. Wait for regime change.",
    "High Volatility": "MEAN REVERSION — Smaller position sizes. RSI & Bollinger Band strategies. Volatility will compress.",
    "Consolidation":   "RANGE TRADING — RSI and Bollinger Bands effective. Set tight stops. Breakout incoming.",
}


def _label_regimes(labels: np.ndarray, features_df: pd.DataFrame) -> dict:
    """Label GMM components by their return/volatility characteristics."""
    mapping = {}
    used    = set()
    stats   = []

    for i in range(4):
        mask    = labels == i
        avg_ret = float(features_df.loc[mask, "ret"].mean()) * 252
        avg_vol = float(features_df.loc[mask, "vol"].mean())
        stats.append((i, avg_ret, avg_vol))

    vol_median = np.median([s[2] for s in stats])

    for idx, avg_ret, avg_vol in stats:
        if avg_vol > vol_median * 1.3:
            label = "High Volatility"
        elif avg_ret > 0.05:
            label = "Bull Trend"
        elif avg_ret < -0.05:
            label = "Bear Trend"
        else:
            label = "Consolidation"

        # Avoid duplicate labels
        if label in used:
            fallbacks = ["Bull Trend","Bear Trend","High Volatility","Consolidation"]
            label = next((l for l in fallbacks if l not in used), f"State {idx}")
        used.add(label)
        mapping[idx] = label

    return mapping


def detect_regime(df: pd.DataFrame) -> dict:
    c   = df["Close"]
    v   = df["Volume"]

    log_ret  = np.log(c / c.shift(1))
    roll_vol = log_ret.rolling(20).std() * np.sqrt(252)
    roll_ret = log_ret.rolling(20).mean() * 252
    rsi      = compute_rsi(c)
    vol_ratio = v / (v.rolling(20).mean() + 1e-9)
    atr       = compute_atr(df["High"], df["Low"], c)
    atr_ratio = atr / (c + 1e-9) * 100   # ATR as % of price

    features_df = pd.DataFrame({
        "ret":       log_ret,
        "vol":       roll_vol,
        "momentum":  roll_ret,
        "rsi_norm":  (rsi - 50) / 50,
        "vol_ratio": vol_ratio,
        "atr_pct":   atr_ratio,
    }).dropna()

    if len(features_df) < 60:
        raise ValueError("Need at least 60 bars for regime detection.")

    scaler = StandardScaler()
    X      = scaler.fit_transform(features_df)

    gmm = GaussianMixture(n_components=4, covariance_type="full",
                          random_state=42, n_init=5, max_iter=200)
    gmm.fit(X)

    labels     = gmm.predict(X)
    probs      = gmm.predict_proba(X)
    regime_map = _label_regimes(labels, features_df)

    history        = [regime_map[l] for l in labels]
    current_label  = history[-1]
    current_probs  = {regime_map[i]: round(float(probs[-1][i]), 3) for i in range(4)}

    # Regime duration (how many bars in current regime)
    duration = 1
    for h in reversed(history[:-1]):
        if h == current_label:
            duration += 1
        else:
            break

    # Transition stats — how often does each regime follow each other
    transitions = {}
    for i in range(1, len(history)):
        key = f"{history[i-1]}→{history[i]}"
        transitions[key] = transitions.get(key, 0) + 1

    # Regime distribution over full history
    regime_counts = {}
    for r in history:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    total = len(history)
    regime_pct = {k: round(v/total*100, 1) for k, v in regime_counts.items()}

    return {
        "current_regime":   current_label,
        "current_color":    REGIME_COLORS.get(current_label, "#8b949e"),
        "probabilities":    current_probs,
        "duration_bars":    duration,
        "regime_pct":       regime_pct,
        "strategy":         REGIME_STRATEGIES.get(current_label, ""),
        "history":          history[-90:],     # last 90 bars for chart
        "dates":            [str(d.date()) for d in features_df.index[-90:]],
        "color_map":        REGIME_COLORS,
    }
