"""
portfolio.py — Portfolio Optimizer (Phase 2)
Monte Carlo efficient frontier + Max Sharpe + Min Volatility + Risk Parity.
Uses scipy.optimize for precise optimal portfolio weights.
"""
import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import minimize

RF = 0.05  # risk-free rate 5%

def _safe(v, dec=4):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, dec)
    except Exception:
        return None


def _stats(w, mean_ret, cov):
    w  = np.array(w)
    r  = float(np.dot(w, mean_ret))
    v  = float(np.sqrt(np.clip(w @ cov @ w, 0, None)))
    sh = (r - RF) / (v + 1e-9)
    return r, v, sh


def optimize_portfolio(tickers: List[str], returns_df: pd.DataFrame,
                       n_portfolios: int = 3000) -> dict:
    returns_df = returns_df.dropna(how="all").fillna(0)
    if len(returns_df) < 30:
        raise ValueError("Need at least 30 trading days of return data.")
    if returns_df.shape[1] < 2:
        raise ValueError("Need at least 2 tickers for portfolio optimization.")

    mean_ret = returns_df.mean() * 252
    cov      = returns_df.cov() * 252
    n        = len(tickers)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bounds      = tuple((0.0, 1.0) for _ in range(n))
    init        = np.array([1 / n] * n)

    # ── Monte Carlo frontier ───────────────────────────────────
    frontier = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r, v, sh = _stats(w, mean_ret, cov)
        frontier.append({"return": _safe(r*100,2), "volatility": _safe(v*100,2),
                         "sharpe": _safe(sh,3),
                         "weights": {t: _safe(w[i],4) for i,t in enumerate(tickers)}})

    def _opt(obj_fn):
        res = minimize(obj_fn, init, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-9})
        return res.x

    # ── Max Sharpe ────────────────────────────────────────────
    w_sh   = _opt(lambda w: -_stats(w, mean_ret, cov)[2])
    r_sh, v_sh, sh_sh = _stats(w_sh, mean_ret, cov)

    # ── Min Volatility ────────────────────────────────────────
    w_mv   = _opt(lambda w: _stats(w, mean_ret, cov)[1])
    r_mv, v_mv, sh_mv = _stats(w_mv, mean_ret, cov)

    # ── Risk Parity (equal risk contribution) ─────────────────
    def rp_obj(w):
        sig = np.sqrt(np.clip(w @ cov @ w, 0, None))
        mrc = cov @ w / (sig + 1e-9)
        rc  = w * mrc / (sig + 1e-9)
        return float(np.sum((rc - 1/n) ** 2))

    w_rp   = _opt(rp_obj)
    w_rp   = np.abs(w_rp) / (np.abs(w_rp).sum() + 1e-9)
    r_rp, v_rp, sh_rp = _stats(w_rp, mean_ret, cov)

    # ── Equal Weight ─────────────────────────────────────────
    w_eq   = np.array([1/n]*n)
    r_eq, v_eq, sh_eq = _stats(w_eq, mean_ret, cov)

    def _fmt(label, w, r, v, sh):
        return {"label": label,
                "weights":    {t: _safe(w[i],4) for i,t in enumerate(tickers)},
                "return":     _safe(r*100,2),
                "volatility": _safe(v*100,2),
                "sharpe":     _safe(sh,3)}

    corr = returns_df.corr()
    individual = {
        t: {"annual_return":     _safe(float(mean_ret[t])*100, 2),
            "annual_volatility": _safe(float(np.sqrt(cov.loc[t,t]))*100, 2)}
        for t in tickers
    }

    return {
        "tickers":    tickers,
        "frontier":   frontier,
        "portfolios": {
            "max_sharpe":     _fmt("Max Sharpe",     w_sh, r_sh, v_sh, sh_sh),
            "min_volatility": _fmt("Min Volatility", w_mv, r_mv, v_mv, sh_mv),
            "risk_parity":    _fmt("Risk Parity",    w_rp, r_rp, v_rp, sh_rp),
            "equal_weight":   _fmt("Equal Weight",   w_eq, r_eq, v_eq, sh_eq),
        },
        "correlation": {
            "tickers": tickers,
            "matrix":  [[_safe(corr.iloc[i][j],4) for j in range(n)] for i in range(n)],
        },
        "individual": individual,
    }
