"""
utils.py — Multi-strategy Yahoo Finance fetcher with detailed logging.
S1: yfinance Ticker.history
S2: yf.download
S3: Yahoo v8 JSON chart API  (most reliable — returns JSON not CSV)
S4: Yahoo v7 CSV download    (fallback)
"""
import logging, warnings
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
import time, math, requests
import pandas as pd
import yfinance as yf
from io import StringIO

# ── Shared browser session ────────────────────────────────────
def _make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent":                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language":           "en-US,en;q=0.5",
        "Accept-Encoding":           "gzip, deflate, br",
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

import threading
_session = _make_session()
_crumb   = None
_crumb_lock = threading.Lock()

def _prewarm():
    """Pre-fetch Yahoo Finance cookies at startup so first request is fast."""
    try:
        global _crumb
        _session.get("https://finance.yahoo.com", timeout=8)
        import time as _t; _t.sleep(0.5)
        r = _session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=8)
        if r.ok and r.text and len(r.text) < 50 and "Too Many" not in r.text:
            _crumb = r.text.strip()
            print(f"[utils] ✓ Pre-warmed crumb: {_crumb[:8]}...")
    except Exception as e:
        print(f"[utils] Pre-warm skipped: {e}")

# Run pre-warm in background so server starts instantly
threading.Thread(target=_prewarm, daemon=True).start()


def _get_crumb(force: bool = False) -> str:
    global _crumb
    if _crumb and not force:
        return _crumb
    # Step 1: visit Yahoo Finance to get session cookies
    r0 = _session.get("https://finance.yahoo.com", timeout=10)
    print(f"[crumb] Yahoo homepage: HTTP {r0.status_code}, cookies: {list(_session.cookies.keys())}")
    time.sleep(1.0)
    # Step 2: fetch crumb
    r1 = _session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10)
    print(f"[crumb] Crumb endpoint: HTTP {r1.status_code}, text='{r1.text[:60]}'")
    if r1.ok and r1.text and len(r1.text) < 50 and "Too Many" not in r1.text:
        _crumb = r1.text.strip()
        print(f"[crumb] ✓ Valid crumb: {_crumb}")
        return _crumb
    # Try alternate crumb endpoint
    r2 = _session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=10)
    print(f"[crumb] Alt crumb: HTTP {r2.status_code}, text='{r2.text[:60]}'")
    if r2.ok and r2.text and len(r2.text) < 50 and "Too Many" not in r2.text:
        _crumb = r2.text.strip()
        print(f"[crumb] ✓ Valid crumb (alt): {_crumb}")
        return _crumb
    raise RuntimeError(f"Cannot get valid crumb. Got: '{r1.text[:80]}'")


def _v8_json(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Yahoo Finance v8 chart API — returns JSON, much more reliable than CSV."""
    crumb = _get_crumb()
    url   = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range={period}&crumb={crumb}"
    )
    r = _session.get(url, timeout=15)
    print(f"[v8] {symbol} HTTP {r.status_code} | first 120 chars: {r.text[:120]}")

    if not r.ok:
        raise RuntimeError(f"v8 HTTP {r.status_code}")

    data = r.json()
    err  = data.get("chart", {}).get("error")
    if err:
        raise RuntimeError(f"v8 error: {err}")

    result     = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote      = result["indicators"]["quote"][0]

    # Use adjclose if available, else close
    try:
        close = result["indicators"]["adjclose"][0]["adjclose"]
    except Exception:
        close = quote["close"]

    df = pd.DataFrame({
        "Open":   quote["open"],
        "High":   quote["high"],
        "Low":    quote["low"],
        "Close":  close,
        "Volume": quote["volume"],
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))
    df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df.dropna()


def _v7_csv(symbol: str, days: int = 185) -> pd.DataFrame:
    """Yahoo Finance v7 CSV download — fallback."""
    crumb = _get_crumb()
    end   = int(time.time())
    start = end - days * 86400
    url   = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
        f"?period1={start}&period2={end}&interval=1d&events=history&crumb={crumb}"
    )
    r = _session.get(url, timeout=15)
    print(f"[v7csv] {symbol} HTTP {r.status_code} | first 80 chars: {r.text[:80]}")
    if r.ok and r.text.strip().startswith("Date"):
        df = pd.read_csv(StringIO(r.text), parse_dates=["Date"])
        df = df.rename(columns={"Adj Close": "Close"}).dropna()
        return df.set_index("Date")
    raise RuntimeError(f"v7 CSV failed: HTTP {r.status_code} | {r.text[:150]}")


# ── Public helpers ────────────────────────────────────────────

def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _validate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    missing = {"Open", "High", "Low", "Close", "Volume"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {symbol}: {missing}")
    return df


def fetch_history(symbol: str, period: str = "6mo") -> pd.DataFrame:
    days_map = {"1mo":35,"3mo":95,"6mo":185,"1y":370,"2y":740,"5y":1850}
    days     = days_map.get(period, 185)
    errors   = []

    # S1 — yfinance Ticker.history
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if not df.empty:
            print(f"[utils] ✓ S1 Ticker.history ok for {symbol} ({len(df)} rows)")
            return _validate(flatten_df(df), symbol)
        errors.append("S1: empty")
    except Exception as e:
        errors.append(f"S1: {e}")

    # S2 — yf.download
    try:
        df = yf.download(symbol, period=period, auto_adjust=True,
                         progress=False, threads=False)
        if not df.empty:
            print(f"[utils] ✓ S2 yf.download ok for {symbol} ({len(df)} rows)")
            return _validate(flatten_df(df), symbol)
        errors.append("S2: empty")
    except Exception as e:
        errors.append(f"S2: {e}")

    # S3 — v8 JSON API
    try:
        global _crumb
        _crumb = None   # force fresh crumb
        df = _v8_json(symbol, period=period)
        if not df.empty:
            print(f"[utils] ✓ S3 v8 JSON ok for {symbol} ({len(df)} rows)")
            return _validate(df, symbol)
        errors.append("S3: empty")
    except Exception as e:
        errors.append(f"S3: {e}")

    # S4 — v7 CSV
    try:
        df = _v7_csv(symbol, days=days)
        if not df.empty:
            print(f"[utils] ✓ S4 v7 CSV ok for {symbol} ({len(df)} rows)")
            return _validate(df, symbol)
        errors.append("S4: empty")
    except Exception as e:
        errors.append(f"S4: {e}")

    raise ValueError(
        f"All 4 strategies failed for '{symbol}'. "
        f"Errors: {' | '.join(errors)}"
    )


def get_fast_info(symbol: str) -> dict:
    try:
        fi = yf.Ticker(symbol).fast_info
        def _g(*attrs, default=None):
            for a in attrs:
                try:
                    v = getattr(fi, a, None)
                    if v is not None:
                        f = float(v)
                        if not (math.isnan(f) or math.isinf(f)):
                            return f
                except Exception:
                    pass
            return default
        return {
            "last_price":     _g("last_price",     "lastPrice"),
            "previous_close": _g("previous_close", "previousClose"),
            "market_cap":     _g("market_cap",     "marketCap"),
            "52w_high":       _g("year_high",       "fiftyTwoWeekHigh"),
            "52w_low":        _g("year_low",        "fiftyTwoWeekLow"),
            "currency":       getattr(fi, "currency", None) or "USD",
            "exchange":       getattr(fi, "exchange", None) or "N/A",
        }
    except Exception:
        return {}


def get_slow_info(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


def get_info_v8(symbol: str) -> dict:
    """
    Get price + metadata entirely from our working v8 session.
    Avoids quoteSummary (always 429). Uses:
      - v8 chart API  → price, 52w range, currency, exchange
      - v1/search API → company name, quote type
    """
    crumb = _get_crumb()

    # ── Price + market data from v8 chart ─────────────────────
    url_chart = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range=5d&crumb={crumb}"
    )
    r = _session.get(url_chart, timeout=15)
    if not r.ok:
        raise ValueError(f"v8 chart failed HTTP {r.status_code}")

    meta = r.json()["chart"]["result"][0]["meta"]

    price = (meta.get("regularMarketPrice")
             or meta.get("chartPreviousClose")
             or meta.get("previousClose"))

    info = {
        "last_price":     price,
        "previous_close": meta.get("regularMarketPreviousClose") or meta.get("chartPreviousClose"),
        "52w_high":       meta.get("fiftyTwoWeekHigh"),
        "52w_low":        meta.get("fiftyTwoWeekLow"),
        "currency":       meta.get("currency", "USD"),
        "exchange":       meta.get("fullExchangeName") or meta.get("exchangeName", "N/A"),
        "quote_type":     meta.get("instrumentType", ""),
        "name":           symbol,   # fallback; search API will override
        "sector":         "N/A",
        "industry":       "N/A",
        "market_cap":     None,
        "description":    "",
    }

    # ── Company name from search API ──────────────────────────
    try:
        url_s = (
            f"https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={symbol}&quotesCount=1&lang=en-US&crumb={crumb}"
        )
        rs = _session.get(url_s, timeout=10)
        if rs.ok:
            quotes = rs.json().get("quotes", [])
            if quotes:
                q = quotes[0]
                info["name"]    = q.get("longname") or q.get("shortname") or symbol
                info["sector"]  = q.get("sector",   "N/A")
                info["industry"]= q.get("industry", "N/A")
    except Exception:
        pass

    return info
