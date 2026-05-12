"""
earnings.py — Earnings & Fundamentals (Phase 1)
Uses our cookie session directly with quoteSummary — avoids yfinance 429.
"""
import math
import pandas as pd
from utils import _session, _get_crumb


def _safe(val, dec=2):
    if val is None: return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, dec)
    except Exception:
        return None


def _quotesummary(symbol: str, modules: list) -> dict:
    """Call Yahoo Finance quoteSummary via our authenticated session."""
    crumb = _get_crumb()
    url   = (
        f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        f"?modules={','.join(modules)}&crumb={crumb}&formatted=false"
    )
    r = _session.get(url, timeout=15)
    if not r.ok:
        raise RuntimeError(f"quoteSummary HTTP {r.status_code}")
    data = r.json().get("quoteSummary", {})
    if data.get("error"):
        raise RuntimeError(f"quoteSummary error: {data['error']}")
    result = data.get("result", [{}])
    return result[0] if result else {}


def get_earnings_data(symbol: str) -> dict:
    modules = [
        "earningsHistory",
        "earningsTrend",
        "financialData",
        "defaultKeyStatistics",
        "calendarEvents",
        "recommendationTrend",
    ]
    try:
        qs = _quotesummary(symbol, modules)
    except Exception as e:
        print(f"[earnings] quoteSummary failed: {e}")
        qs = {}

    # ── EPS History (actual vs estimate) ─────────────────────
    earnings_history = []
    try:
        for item in (qs.get("earningsHistory", {}).get("history") or []):
            actual   = _safe(item.get("epsActual",   {}).get("raw"))
            estimate = _safe(item.get("epsEstimate", {}).get("raw"))
            surprise = None
            if actual is not None and estimate and estimate != 0:
                surprise = _safe((actual - estimate) / abs(estimate) * 100)
            earnings_history.append({
                "quarter":  item.get("period", ""),
                "actual":   actual,
                "estimate": estimate,
                "surprise": surprise,
            })
    except Exception:
        pass

    # ── Next earnings date ─────────────────────────────────────
    next_earnings = None
    try:
        cal = qs.get("calendarEvents", {}).get("earnings", {})
        dates = cal.get("earningsDate", [])
        if dates:
            raw = dates[0].get("raw") or dates[0]
            next_earnings = pd.Timestamp(raw, unit="s").strftime("%Y-%m-%d")
    except Exception:
        pass

    # ── Key stats ──────────────────────────────────────────────
    ks  = qs.get("defaultKeyStatistics", {})
    fd  = qs.get("financialData",        {})

    def _r(obj, key, dec=2):
        v = obj.get(key)
        if isinstance(v, dict): v = v.get("raw")
        return _safe(v, dec)

    eps_ttm     = _r(ks, "trailingEps")
    eps_fwd     = _r(ks, "forwardEps")
    rev_ttm     = _r(fd, "totalRevenue", 0)
    gross_m     = _r(fd, "grossMargins", 4)
    profit_m    = _r(fd, "profitMargins", 4)

    # ── Analyst recommendations ────────────────────────────────
    analyst_summary = {}
    try:
        trend = qs.get("recommendationTrend", {}).get("trend", [])
        if trend:
            t = trend[0]
            analyst_summary = {
                "strong_buy":  t.get("strongBuy",  0),
                "buy":         t.get("buy",        0),
                "hold":        t.get("hold",       0),
                "sell":        t.get("sell",       0),
                "strong_sell": t.get("strongSell", 0),
            }
    except Exception:
        pass

    # ── Earnings trend (forward estimates) ────────────────────
    revenue_history = []
    try:
        for item in (qs.get("earningsTrend", {}).get("trend") or []):
            period = item.get("period", "")
            rev_est = item.get("revenueEstimate", {}).get("avg", {})
            rev_val = rev_est.get("raw") if isinstance(rev_est, dict) else rev_est
            if period and rev_val:
                revenue_history.append({
                    "quarter": period,
                    "revenue": int(rev_val),
                })
    except Exception:
        pass

    return {
        "ticker":           symbol,
        "next_earnings":    next_earnings,
        "eps_ttm":          eps_ttm,
        "eps_forward":      eps_fwd,
        "revenue_ttm":      int(rev_ttm) if rev_ttm else None,
        "gross_margin":     gross_m,
        "profit_margin":    profit_m,
        "earnings_history": earnings_history,
        "revenue_history":  revenue_history,
        "analyst_summary":  analyst_summary,
        "recent_recs":      [],
    }
