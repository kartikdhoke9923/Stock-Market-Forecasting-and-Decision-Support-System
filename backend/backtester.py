"""
backtester.py — Backtesting Engine + Walk-Forward Validation (Phase 2)
Signal on day T executes on day T+1 open — zero lookahead bias.
Metrics: Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor.
"""
import numpy as np
import pandas as pd
from typing import Callable

RISK_FREE = 0.05  # 5% annual

def _safe(v, dec=4):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, dec)
    except Exception:
        return None


def run_backtest(df: pd.DataFrame, strategy_fn: Callable,
                 strategy_params: dict = {}, initial_capital: float = 10000.0,
                 commission: float = 0.001) -> dict:
    if len(df) < 50:
        raise ValueError("Need at least 50 bars of data.")

    signals     = strategy_fn(df, **strategy_params)
    cash        = initial_capital
    shares      = 0.0
    position    = 0
    entry_price = 0.0
    equity_curve = [initial_capital]
    trades       = []

    for i in range(1, len(df)):
        sig   = int(signals.iloc[i - 1])   # previous bar signal → no lookahead
        open_ = float(df["Open"].iloc[i])
        close = float(df["Close"].iloc[i])
        date  = df.index[i]

        if sig == 1 and position == 0:
            shares      = cash / (open_ * (1 + commission))
            cash        = 0.0
            position    = 1
            entry_price = open_
            trades.append({"date": str(date.date()), "action": "BUY",
                           "price": round(open_, 4), "pnl_pct": None})

        elif sig == -1 and position == 1:
            pnl_pct = (open_ - entry_price) / entry_price * 100
            cash    = shares * open_ * (1 - commission)
            shares  = 0.0
            position = 0
            trades.append({"date": str(date.date()), "action": "SELL",
                           "price": round(open_, 4), "pnl_pct": round(pnl_pct, 2)})

        equity_curve.append(cash + shares * close)

    # Close any open position at last bar
    if position == 1:
        fp      = float(df["Close"].iloc[-1])
        pnl_pct = (fp - entry_price) / entry_price * 100
        cash    = shares * fp * (1 - commission)
        equity_curve[-1] = cash
        trades.append({"date": str(df.index[-1].date()), "action": "CLOSE",
                       "price": round(fp, 4), "pnl_pct": round(pnl_pct, 2)})

    eq = np.array(equity_curve)

    # ── Metrics ────────────────────────────────────────────────
    total_ret    = (eq[-1] - initial_capital) / initial_capital * 100
    n_years      = max(len(df) / 252, 0.1)
    cagr         = ((eq[-1] / initial_capital) ** (1 / n_years) - 1) * 100
    daily_ret    = pd.Series(eq).pct_change().dropna()
    sharpe       = (daily_ret.mean() * 252 - RISK_FREE) / (daily_ret.std() * np.sqrt(252) + 1e-9)
    downside     = daily_ret[daily_ret < 0]
    sortino      = (daily_ret.mean() * 252 - RISK_FREE) / (downside.std() * np.sqrt(252) + 1e-9)
    roll_max     = pd.Series(eq).cummax()
    drawdown     = (pd.Series(eq) - roll_max) / roll_max * 100
    max_dd       = float(drawdown.min())
    calmar       = cagr / abs(max_dd) if max_dd != 0 else 0

    closed       = [t for t in trades if t["action"] in ("SELL","CLOSE") and t["pnl_pct"] is not None]
    wins         = [t for t in closed if t["pnl_pct"] > 0]
    losses       = [t for t in closed if t["pnl_pct"] <= 0]
    win_rate     = len(wins) / len(closed) * 100 if closed else 0
    avg_win      = np.mean([t["pnl_pct"] for t in wins])   if wins   else 0
    avg_loss     = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss   = abs(sum(t["pnl_pct"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

    bh_ret = (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])) / float(df["Close"].iloc[0]) * 100

    return {
        "initial_capital":  initial_capital,
        "final_value":      _safe(eq[-1], 2),
        "total_return":     _safe(total_ret, 2),
        "cagr":             _safe(cagr, 2),
        "sharpe":           _safe(sharpe, 3),
        "sortino":          _safe(sortino, 3),
        "max_drawdown":     _safe(max_dd, 2),
        "calmar":           _safe(calmar, 3),
        "win_rate":         _safe(win_rate, 1),
        "total_trades":     len(closed),
        "avg_win_pct":      _safe(avg_win, 2),
        "avg_loss_pct":     _safe(avg_loss, 2),
        "profit_factor":    _safe(profit_factor, 3),
        "benchmark_return": _safe(bh_ret, 2),
        "outperformance":   _safe(total_ret - bh_ret, 2),
        "equity_curve":     [_safe(v, 2) for v in eq],
        "drawdown_curve":   [_safe(v, 4) for v in drawdown],
        "dates":            [str(d.date()) for d in df.index],
        "trades":           trades,
    }


def walk_forward(df: pd.DataFrame, strategy_fn: Callable,
                 strategy_params: dict = {}, n_windows: int = 5,
                 test_pct: float = 0.3, initial_capital: float = 10000.0,
                 commission: float = 0.001) -> list:
    """
    Splits data into n non-overlapping windows.
    Runs backtest only on the test portion (out-of-sample) of each window.
    This reveals whether a strategy actually generalises or just curve-fits.
    """
    total = len(df)
    wsize = total // n_windows
    if wsize < 60:
        raise ValueError(f"Not enough data for {n_windows} windows. Use a longer period (2y+).")

    results = []
    for i in range(n_windows):
        s   = i * wsize
        mid = s + int(wsize * (1 - test_pct))
        e   = min(s + wsize, total)
        test_df = df.iloc[mid:e].copy()
        if len(test_df) < 20:
            continue
        try:
            r = run_backtest(test_df, strategy_fn, strategy_params, initial_capital, commission)
            results.append({
                "window":       i + 1,
                "start":        str(df.index[mid].date()),
                "end":          str(df.index[e - 1].date()),
                "total_return": r["total_return"],
                "sharpe":       r["sharpe"],
                "max_drawdown": r["max_drawdown"],
                "win_rate":     r["win_rate"],
                "trades":       r["total_trades"],
                "benchmark":    r["benchmark_return"],
                "alpha":        r["outperformance"],
            })
        except Exception as ex:
            results.append({"window": i + 1, "start": str(df.index[mid].date()),
                            "end": str(df.index[e - 1].date()), "error": str(ex)})
    return results
