"""
app.py — Streamlit Frontend (Phase 1)
Connects to FastAPI backend via BACKEND_URL env variable.
Tabs: Overview · Technical Analysis · News Sentiment · Earnings · Screener
"""
import os
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────
BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Full company names for display
TICKER_NAMES = {
    # US Tech
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"Nvidia","GOOGL":"Alphabet (Google)",
    "AMZN":"Amazon","META":"Meta Platforms","TSLA":"Tesla","AMD":"AMD",
    "INTC":"Intel","CRM":"Salesforce","ORCL":"Oracle","ADBE":"Adobe",
    "NFLX":"Netflix","QCOM":"Qualcomm","IBM":"IBM","CSCO":"Cisco",
    # US Finance
    "JPM":"JPMorgan Chase","BAC":"Bank of America","GS":"Goldman Sachs",
    "MS":"Morgan Stanley","WFC":"Wells Fargo","C":"Citigroup",
    "V":"Visa","MA":"Mastercard","BLK":"BlackRock","AXP":"American Express",
    # US Healthcare
    "JNJ":"Johnson & Johnson","PFE":"Pfizer","UNH":"UnitedHealth",
    "ABBV":"AbbVie","MRK":"Merck","TMO":"Thermo Fisher","ABT":"Abbott",
    "BMY":"Bristol-Myers","AMGN":"Amgen","GILD":"Gilead Sciences",
    # ETFs
    "SPY":"S&P 500 ETF","QQQ":"Nasdaq 100 ETF","VTI":"Total Market ETF",
    "IWM":"Russell 2000 ETF","GLD":"Gold ETF","SLV":"Silver ETF",
    "TLT":"Treasury Bond ETF","XLF":"Financial ETF","XLK":"Technology ETF",
    # India NSE
    "RELIANCE.NS":"Reliance Industries","TCS.NS":"Tata Consultancy",
    "INFY.NS":"Infosys","HDFCBANK.NS":"HDFC Bank","ICICIBANK.NS":"ICICI Bank",
    "WIPRO.NS":"Wipro","TATAMOTORS.NS":"Tata Motors","BAJFINANCE.NS":"Bajaj Finance",
    "MARUTI.NS":"Maruti Suzuki","SUNPHARMA.NS":"Sun Pharma",
    # Crypto
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","IBIT":"iShares Bitcoin ETF",
    "FBTC":"Fidelity Bitcoin ETF",
}

def ticker_label(t):
    name = TICKER_NAMES.get(t.upper(), "")
    return f"{name} ({t})" if name else t

WATCHLISTS = {
    "🇺🇸 US Tech":      ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "CRM"],
    "🇺🇸 US Finance":   ["JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA", "BLK", "AXP"],
    "🇺🇸 US Healthcare":["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "BMY", "AMGN", "GILD"],
    "🇮🇳 India NSE":    ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
                          "WIPRO.NS","TATAMOTORS.NS","BAJFINANCE.NS","MARUTI.NS","SUNPHARMA.NS"],
    "📊 ETFs":           ["SPY", "QQQ", "VTI", "IWM", "GLD", "SLV", "TLT", "XLF", "XLK"],
}

st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialization (must be before any widget) ─
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "AAPL"

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
.main-title  { font-size:2rem; font-weight:800; color:#58a6ff; letter-spacing:-0.5px; }
.sub-title   { color:#8b949e; font-size:0.85rem; margin-bottom:1.5rem; }
.badge-sbuy  { background:#1a7f37;color:#3fb950;border:1px solid #238636;padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.8rem; }
.badge-buy   { background:#0d3320;color:#00d084;border:1px solid #238636;padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.8rem; }
.badge-neu   { background:#21262d;color:#8b949e;border:1px solid #30363d;padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.8rem; }
.badge-sell  { background:#3d0000;color:#f85149;border:1px solid #6e1313;padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.8rem; }
.badge-ssell { background:#6e0000;color:#ff7b72;border:1px solid #8b1313;padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.8rem; }
.info-card   { background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;margin-bottom:8px; }
.disclaimer  { background:#161b22;border:1px solid #d29922;border-radius:6px;padding:10px 14px;font-size:0.78rem;color:#d29922;margin-top:1rem; }
div[data-testid="stMetric"] { background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def api_info(ticker):
    try:
        r = requests.get(f"{BACKEND}/info/{ticker}", timeout=20)
        data = r.json() if r.ok else None
        if not data or data.get("current_price") is None:
            raise Exception("incomplete data")
        return data
    except Exception:
        return None

@st.cache_data(ttl=60, show_spinner=False)
def api_indicators(ticker):
    try:
        r = requests.get(f"{BACKEND}/indicators/{ticker}", timeout=40)
        data = r.json() if r.ok else None
        if not data or not data.get("history"):
            raise Exception("incomplete data")
        return data
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def api_sentiment(ticker):
    try:
        r = requests.get(f"{BACKEND}/sentiment/{ticker}", timeout=25)
        return r.json() if r.ok else None
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def api_earnings(ticker):
    try:
        r = requests.get(f"{BACKEND}/earnings/{ticker}", timeout=25)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_screen(tickers, min_rsi=None, max_rsi=None, signal=None):
    try:
        body = {"tickers": tickers}
        if min_rsi is not None and min_rsi > 0:   body["min_rsi"] = min_rsi
        if max_rsi is not None and max_rsi < 100:  body["max_rsi"] = max_rsi
        if signal:                                 body["signal"]  = signal
        r = requests.post(f"{BACKEND}/screen", json=body, timeout=90)
        return r.json() if r.ok else None
    except Exception:
        return None

def badge(signal: str) -> str:
    s = signal.upper()
    cls = ("badge-sbuy"  if s == "STRONG BUY"  else
           "badge-buy"   if "BUY"  in s         else
           "badge-ssell" if s == "STRONG SELL"  else
           "badge-sell"  if "SELL" in s          else
           "badge-neu")
    return f'<span class="{cls}">{signal}</span>'

CHART_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", size=12),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

def _xaxis(): return dict(gridcolor="#21262d", showgrid=True, zeroline=False)
def _yaxis(): return dict(gridcolor="#21262d", showgrid=True, zeroline=False)


# ── Charts ────────────────────────────────────────────────────
def chart_price(hist: dict, ticker: str) -> go.Figure:
    d = hist["dates"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
    )
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=d, open=hist["open"], high=hist["high"], low=hist["low"], close=hist["close"],
        name="OHLC",
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="#3fb950",  decreasing_fillcolor="#f85149",
    ), row=1, col=1)
    # BB fill
    fig.add_trace(go.Scatter(
        x=d + d[::-1],
        y=hist["bb_upper"] + hist["bb_lower"][::-1],
        fill="toself", fillcolor="rgba(88,166,255,0.04)",
        line=dict(color="rgba(0,0,0,0)"), name="BB Zone", showlegend=False,
    ), row=1, col=1)
    # BB lines
    for key, col, nm in [("bb_upper","#58a6ff","BB Upper"),("bb_middle","#d29922","BB Mid"),("bb_lower","#58a6ff","BB Lower")]:
        fig.add_trace(go.Scatter(x=d, y=hist[key], name=nm,
            line=dict(color=col, width=1, dash="dot" if "Upper" in nm or "Lower" in nm else "solid"),
            opacity=0.6), row=1, col=1)
    # SMAs
    for key, col, nm in [("sma20","#ff7b72","SMA 20"),("sma50","#bc8cff","SMA 50"),("sma200","#ffa657","SMA 200")]:
        if any(v is not None for v in hist.get(key, [])):
            fig.add_trace(go.Scatter(x=d, y=hist[key], name=nm,
                line=dict(color=col, width=1.5), opacity=0.85), row=1, col=1)
    # Volume
    colors = ["#3fb950" if (hist["close"][i] or 0) >= (hist["open"][i] or 0) else "#f85149"
              for i in range(len(hist["close"]))]
    fig.add_trace(go.Bar(x=d, y=hist["volume"], name="Volume",
        marker_color=colors, opacity=0.65), row=2, col=1)

    fig.update_layout(title=f"{ticker} — Price & Bollinger Bands (90d)", height=500,
        xaxis_rangeslider_visible=False, **CHART_LAYOUT)
    fig.update_xaxes(**_xaxis())
    fig.update_yaxes(**_yaxis())
    return fig


def chart_rsi(hist: dict) -> go.Figure:
    d, rsi = hist["dates"], hist["rsi"]
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="#3d0000", opacity=0.15, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#0d3320", opacity=0.15, line_width=0)
    fig.add_hline(y=70, line=dict(color="#f85149", dash="dash", width=1))
    fig.add_hline(y=30, line=dict(color="#3fb950", dash="dash", width=1))
    fig.add_hline(y=50, line=dict(color="#6e7681", dash="dot",  width=1))
    fig.add_trace(go.Scatter(x=d, y=rsi, name="RSI (14)",
        line=dict(color="#bc8cff", width=2)))
    fig.update_layout(title="RSI (14)", height=240,
        yaxis=dict(range=[0, 100], **_yaxis()), xaxis=_xaxis(), **CHART_LAYOUT)
    return fig


def chart_macd(hist: dict) -> go.Figure:
    d = hist["dates"]
    h = hist["macd_histogram"]
    colors = ["#3fb950" if (v or 0) >= 0 else "#f85149" for v in h]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d, y=h, name="Histogram", marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=d, y=hist["macd_line"],   name="MACD",   line=dict(color="#58a6ff", width=2)))
    fig.add_trace(go.Scatter(x=d, y=hist["macd_signal"], name="Signal", line=dict(color="#d29922", width=1.5)))
    fig.update_layout(title="MACD (12, 26, 9)", height=240,
        yaxis=_yaxis(), xaxis=_xaxis(), **CHART_LAYOUT)
    return fig


def chart_sentiment_gauge(score: float) -> go.Figure:
    color = "#3fb950" if score > 0.05 else "#f85149" if score < -0.05 else "#8b949e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"color": "#e6edf3", "size": 36}},
        title={"text": "Sentiment Score", "font": {"color": "#8b949e", "size": 14}},
        gauge={
            "axis":  {"range": [-1, 1], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [-1.0, -0.3],  "color": "#3d0000"},
                {"range": [-0.3, -0.05], "color": "#5a1212"},
                {"range": [-0.05, 0.05], "color": "#1c2128"},
                {"range": [0.05,  0.3],  "color": "#0d3320"},
                {"range": [0.3,   1.0],  "color": "#1a7f37"},
            ],
        }
    ))
    fig.update_layout(height=220, paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"), margin=dict(l=20, r=20, t=40, b=10))
    return fig


def chart_earnings(history: list) -> go.Figure:
    if not history:
        return None
    q = [e["quarter"] for e in history]
    a = [e["actual"]   for e in history]
    s = [e["estimate"] for e in history]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=q, y=s, name="Estimate", marker_color="#58a6ff", opacity=0.55))
    fig.add_trace(go.Bar(x=q, y=a, name="Actual",   marker_color="#3fb950", opacity=0.9))
    fig.update_layout(title="Quarterly EPS — Actual vs Estimate", height=290,
        barmode="group", yaxis=_yaxis(), xaxis=_xaxis(), **CHART_LAYOUT)
    return fig


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Stock Analyzer")
    st.caption("Phase 1 — Indicators · Sentiment · Earnings")
    st.divider()

    ticker_input = st.text_input(
        "**Ticker Symbol**",
        key="ticker_widget",
        value=st.session_state["ticker"],
        placeholder="AAPL · TSLA · INFY.NS · BTC-USD",
        help="US: AAPL | India: RELIANCE.NS | Crypto: BTC-USD",
    ).upper().strip()

    analyze = st.button("🔍 Analyze", type="primary", width='stretch')

    st.divider()
    st.markdown("**Quick Tickers**")
    quick = ["AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","SPY","RELIANCE.NS","TCS.NS"]
    cols = st.columns(2)
    for i, t in enumerate(quick):
        name = TICKER_NAMES.get(t, t)
        if cols[i % 2].button(name, key=f"q_{t}", width='stretch', help=f"Ticker: {t}"):
            st.session_state["ticker"] = t
            st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:0.73rem;color:#6e7681;line-height:1.5;'>
    ⚠️ <b>Not financial advice.</b><br>
    All signals are informational only. 
    Past indicators ≠ future returns. 
    Consult a qualified advisor.
    </div>
    """, unsafe_allow_html=True)


# ── Resolve active ticker ─────────────────────────────────────
if analyze:
    st.session_state["ticker"] = ticker_input
active = st.session_state.get("ticker", "AAPL")


# ── Page header ───────────────────────────────────────────────
st.markdown('<div class="main-title">📈 Stock Market Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Technical Indicators · News Sentiment · Earnings & Fundamentals · Screener</div>', unsafe_allow_html=True)

tab_ov, tab_ta, tab_sent, tab_earn, tab_screen, tab_bt, tab_port, tab_ai, tab_de = st.tabs([
    "📊 Overview",
    "📉 Technical Analysis",
    "📰 News Sentiment",
    "💰 Earnings & Fundamentals",
    "🔍 Stock Screener",
    "⚡ Backtester",
    "💼 Portfolio",
    "🚀 AI Analysis",
    "🎯 Decision Engine",
])


# ═══════════════════════ OVERVIEW ════════════════════════════
with tab_ov:
    with st.spinner(f"Loading {active}..."):
        info = api_info(active)
        ind  = api_indicators(active)

    if not info:
        st.error(f"**{active}** not found. Check the ticker symbol (e.g. `AAPL`, `RELIANCE.NS`, `BTC-USD`).")
        st.stop()

    # ── Top bar: name + price + signal ───────────────────────
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        st.markdown(f"### {info.get('name', active)}")
        st.caption(f"{info.get('exchange','—')} · {info.get('sector','—')} · {info.get('industry','—')}")
    with c2:
        price = info.get("current_price") or (ind.get("price") if ind else None)
        prev  = info.get("previous_close")
        if price:
            st.metric("Current Price",
                      f"{info.get('currency','$')} {price:,.2f}",
                      delta=f"{price-prev:+.2f} ({(price-prev)/prev*100:+.2f}%)" if prev else None)
    with c3:
        if ind:
            sig = ind.get("overall_signal", "N/A")
            st.markdown(f"**Signal** &nbsp; {badge(sig)}", unsafe_allow_html=True)
            st.caption(f"▲ {ind.get('buy_count',0)} buy signals  ▼ {ind.get('sell_count',0)} sell signals")

    st.divider()

    # ── Key stats row ─────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    mc  = info.get("market_cap")
    c1.metric("Market Cap",   info.get("market_cap_fmt", "N/A"))
    c2.metric("P/E (TTM)",    f"{info.get('pe_ratio'):.1f}"  if info.get("pe_ratio") else "N/A")
    c3.metric("52W High",     f"${info.get('52w_high'):,.2f}" if info.get("52w_high") else "N/A")
    c4.metric("52W Low",      f"${info.get('52w_low'):,.2f}"  if info.get("52w_low")  else "N/A")
    c5.metric("Beta",         f"{info.get('beta'):.2f}"       if info.get("beta")     else "N/A")

    # ── Indicator snapshot ────────────────────────────────────
    if ind:
        st.subheader("Indicator Snapshot")
        iv = ind.get("indicators", {})
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("RSI",     f"{iv.get('rsi','—')}")
        c2.metric("SMA 20",  f"${iv.get('sma_20','—')}")
        c3.metric("SMA 50",  f"${iv.get('sma_50','—')}")
        c4.metric("SMA 200", f"${iv.get('sma_200','—')}")
        c5.metric("Stoch K", f"{iv.get('stoch_k','—')}")
        c6.metric("ATR",     f"{iv.get('atr','—')}")

        st.plotly_chart(chart_price(ind["history"], active), width='stretch', key="ov_price")

    # ── Company description ───────────────────────────────────
    if info.get("description"):
        with st.expander("About this company"):
            st.write(info["description"])


# ═══════════════════════ TECHNICAL ANALYSIS ══════════════════
with tab_ta:
    with st.spinner("Computing indicators..."):
        ind = api_indicators(active)

    if not ind:
        st.error("Could not compute indicators. Try again or check the ticker.")
    else:
        sig = ind.get("overall_signal", "NEUTRAL")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Overall Signal:** &nbsp; {badge(sig)}", unsafe_allow_html=True)
        c2.metric("Buy Signals",  ind.get("buy_count",  0))
        c3.metric("Sell Signals", ind.get("sell_count", 0))

        st.subheader("Signal Breakdown")
        rows = []
        for s in ind.get("signals", []):
            sig_txt = s["signal"]
            emoji   = "🟢" if "BUY" in sig_txt else ("🔴" if "SELL" in sig_txt else "⚪")
            rows.append({
                "Indicator": s["indicator"],
                "Signal":    f"{emoji} {sig_txt}",
                "Reason":    s["reason"],
                "Strength":  s["strength"],
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.subheader("Price Chart — 90 Days")
        st.plotly_chart(chart_price(ind["history"], active), width='stretch', key="ta_price")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_rsi(ind["history"]),  width='stretch', key="ta_rsi")
        with col2:
            st.plotly_chart(chart_macd(ind["history"]), width='stretch', key="ta_macd")

        with st.expander("📋 Raw Indicator Values"):
            iv = ind.get("indicators", {})
            df_raw = pd.DataFrame([
                {"Indicator": k.replace("_", " ").upper(), "Value": v}
                for k, v in iv.items() if v is not None
            ])
            st.dataframe(df_raw, width='stretch', hide_index=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ Technical signals are pattern-based heuristics — they identify tendencies, not certainties.
    No indicator has predictive accuracy above noise over the long run without proper risk management.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════ SENTIMENT ═══════════════════════════
with tab_sent:
    with st.spinner("Fetching news and scoring sentiment..."):
        sent = api_sentiment(active)

    if not sent:
        st.error("Could not fetch sentiment data.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(chart_sentiment_gauge(sent.get("overall_score", 0)), width='stretch', key="sent_gauge")
        with c2:
            label = sent.get("overall_label","N/A")
            emoji = sent.get("overall_emoji","")
            color = "#3fb950" if label=="POSITIVE" else "#f85149" if label=="NEGATIVE" else "#8b949e"
            st.markdown(f"<h2 style='color:{color}'>{emoji} {label}</h2>", unsafe_allow_html=True)
            st.caption(f"Analyzed {sent.get('article_count',0)} recent articles")
            st.markdown(f"""
| Sentiment  | Count |
|------------|-------|
| 🟢 Positive | {sent.get('positive_count',0)} |
| 🔴 Negative | {sent.get('negative_count',0)} |
| ⚪ Neutral  | {sent.get('neutral_count',0)} |
""")

        st.subheader("Recent News")
        if not sent.get("articles"):
            st.info("No recent news articles found for this ticker.")
        for art in sent.get("articles", []):
            lbl = art.get("label", "NEUTRAL")
            col = "#3fb950" if lbl=="POSITIVE" else "#f85149" if lbl=="NEGATIVE" else "#8b949e"
            score = art.get("compound", 0)
            with st.container():
                ca, cb = st.columns([7, 1])
                with ca:
                    url, title = art.get("url",""), art.get("title","")
                    st.markdown(f"**[{title}]({url})**" if url else f"**{title}**")
                    st.caption(f"{art.get('publisher','')} · {art.get('published','')}")
                    if art.get("summary"):
                        st.caption(art["summary"][:200])
                with cb:
                    st.markdown(f"<div style='color:{col};font-weight:700;font-size:0.8rem;text-align:right'>"
                                f"{art.get('emoji','')} {lbl}<br>{score:+.3f}</div>", unsafe_allow_html=True)
                st.divider()


# ═══════════════════════ EARNINGS ════════════════════════════
with tab_earn:
    with st.spinner("Loading earnings data..."):
        earn = api_earnings(active)

    if not earn:
        st.error("Could not load earnings data.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("EPS (TTM)",     f"${earn.get('eps_ttm')}"     if earn.get("eps_ttm")     else "N/A")
        c2.metric("EPS (Forward)", f"${earn.get('eps_forward')}" if earn.get("eps_forward") else "N/A")
        c3.metric("Next Earnings", earn.get("next_earnings", "N/A"))

        rev = earn.get("revenue_ttm")
        gm  = earn.get("gross_margin")
        pm  = earn.get("profit_margin")
        c1, c2, c3 = st.columns(3)
        c1.metric("Revenue (TTM)", f"${rev/1e9:.1f}B" if rev else "N/A")
        c2.metric("Gross Margin",  f"{gm*100:.1f}%"   if gm  else "N/A")
        c3.metric("Profit Margin", f"{pm*100:.1f}%"   if pm  else "N/A")

        st.divider()

        # EPS Chart
        eh = earn.get("earnings_history", [])
        if eh:
            fig = chart_earnings(eh)
            if fig:
                st.plotly_chart(fig, width='stretch', key="earn_eps")
        else:
            st.info("No quarterly EPS data available.")

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Analyst Consensus")
            asumm = earn.get("analyst_summary", {})
            if asumm:
                total = max(sum(asumm.values()), 1)
                for label, key, color in [
                    ("Strong Buy",  "strong_buy",  "#1a7f37"),
                    ("Buy",         "buy",         "#238636"),
                    ("Hold",        "hold",        "#6e7681"),
                    ("Sell",        "sell",        "#b62324"),
                    ("Strong Sell", "strong_sell", "#8b0000"),
                ]:
                    count = asumm.get(key, 0)
                    pct   = count / total
                    ca, cb = st.columns([4, 1])
                    ca.progress(pct, text=label)
                    cb.write(count)
            else:
                st.info("No analyst consensus data available.")

        with c2:
            st.subheader("Recent Analyst Actions")
            recs = earn.get("recent_recs", [])
            if recs:
                st.dataframe(pd.DataFrame(recs), width='stretch', hide_index=True)
            else:
                st.info("No recent analyst actions.")


# ═══════════════════════ SCREENER ════════════════════════════
with tab_screen:
    st.subheader("🔍 Stock Screener")
    st.caption("Scan a watchlist and filter by RSI range or signal type")

    with st.form("screen_form"):
        c1, c2 = st.columns(2)
        with c1:
            preset = st.selectbox("Preset Watchlist", ["Custom"] + list(WATCHLISTS.keys()))
            custom = st.text_input("Custom tickers (comma separated)", placeholder="AAPL, TSLA, NVDA")
        with c2:
            sig_filter = st.selectbox("Signal Filter", ["All", "BUY", "NEUTRAL", "SELL"])
            cr1, cr2   = st.columns(2)
            min_rsi = cr1.number_input("Min RSI", 0, 100, 0)
            max_rsi = cr2.number_input("Max RSI", 0, 100, 100)

        run = st.form_submit_button("🔍 Run Screener", type="primary", width='stretch')

    if run:
        tickers = (WATCHLISTS[preset] if preset != "Custom"
                   else [t.strip().upper() for t in custom.split(",") if t.strip()])

        if not tickers:
            st.warning("Enter tickers or select a preset watchlist.")
        else:
            sig_param = None if sig_filter == "All" else sig_filter
            with st.spinner(f"Scanning {len(tickers)} stocks... (may take ~{len(tickers)*2}s)"):
                result = api_screen(tickers, min_rsi, max_rsi, sig_param)

            if result and result.get("results"):
                rows = result["results"]
                st.success(f"✅ **{len(rows)} stocks** matched your criteria")

                for r in rows:
                    r["company"] = TICKER_NAMES.get(r["ticker"], r["ticker"])
                df_sc   = pd.DataFrame(rows)
                ordered = ["company"] + [c for c in df_sc.columns if c != "company"]
                df_sc   = df_sc[ordered]

                # Style
                def _sig_style(val):
                    if "BUY"  in str(val): return "background:#0d3320;color:#3fb950"
                    if "SELL" in str(val): return "background:#3d0000;color:#f85149"
                    return "background:#1c2128;color:#8b949e"

                def _rsi_style(val):
                    try:
                        v = float(val)
                        if v < 30: return "color:#3fb950;font-weight:bold"
                        if v > 70: return "color:#f85149;font-weight:bold"
                    except Exception:
                        pass
                    return ""

                styled = (df_sc.style
                    .applymap(_sig_style, subset=["signal"])
                    .applymap(_rsi_style, subset=["rsi"]))
                st.dataframe(styled, width='stretch', hide_index=True)

                # Drill-down
                st.divider()
                sel = st.selectbox("Drill down on a stock:", ["—"] + [r["ticker"] for r in rows])
                if sel != "—" and st.button(f"📊 Full Analysis for {sel}", type="primary"):
                    st.session_state["ticker"] = sel
                    st.rerun()
            else:
                st.warning("No stocks matched, or the screener timed out. Try fewer tickers.")

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Disclaimer:</b> All signals and screener results are for <b>educational and research purposes only</b>.
    This is <b>not financial advice</b>. Do your own research before making any investment decisions.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════ BACKTESTER ══════════════════════════
with tab_bt:
    st.subheader("⚡ Strategy Backtester")
    st.caption("Test trading strategies on historical data with walk-forward validation")

    with st.form("bt_form"):
        c1, c2, c3 = st.columns(3)
        bt_ticker   = c1.text_input("Ticker", value=active, placeholder="AAPL")
        bt_strategy = c2.selectbox("Strategy", [
            "RSI Mean Reversion", "MACD Momentum", "MA Cross (50/200)",
            "Bollinger Band", "ATR Momentum", "Multi-Factor Consensus",
        ])
        bt_period   = c3.selectbox("History Period", ["1y","2y","3y","5y"], index=1)

        c4, c5, c6 = st.columns(3)
        bt_capital  = c4.number_input("Initial Capital ($)", value=10000, step=1000, min_value=100)
        bt_comm     = c5.slider("Commission (%)", 0.0, 0.5, 0.1, 0.05) / 100
        bt_wf       = c6.checkbox("Walk-Forward Validation", value=True)

        # Strategy-specific params
        st.markdown("**Strategy Parameters**")
        p1, p2, p3 = st.columns(3)
        if bt_strategy == "RSI Mean Reversion":
            oversold   = p1.slider("Oversold",  10, 45, 30)
            overbought = p2.slider("Overbought", 55, 90, 70)
            bt_params  = {"oversold": oversold, "overbought": overbought}
        elif bt_strategy == "MACD Momentum":
            fast       = p1.slider("Fast EMA", 5, 20, 12)
            slow       = p2.slider("Slow EMA", 20, 50, 26)
            bt_params  = {"fast": fast, "slow": slow, "sig": 9}
        elif bt_strategy == "MA Cross (50/200)":
            fast       = p1.slider("Fast MA", 10, 100, 50)
            slow       = p2.slider("Slow MA", 100, 300, 200)
            bt_params  = {"fast": fast, "slow": slow}
        elif bt_strategy == "Bollinger Band":
            period     = p1.slider("Period", 10, 50, 20)
            width      = p2.slider("Std Dev", 1.0, 3.0, 2.0, 0.1)
            bt_params  = {"period": period, "width": width}
        elif bt_strategy == "ATR Momentum":
            ema_p      = p1.slider("EMA Period", 10, 50, 20)
            atr_m      = p2.slider("ATR Multiplier", 1.0, 4.0, 2.0, 0.1)
            bt_params  = {"ema_period": ema_p, "atr_mult": atr_m}
        else:  # Multi-Factor
            threshold  = p1.slider("Signal Threshold", 1.0, 4.0, 2.0, 0.5)
            bt_params  = {"threshold": threshold}

        run_bt = st.form_submit_button("⚡ Run Backtest", type="primary", use_container_width=True)

    if run_bt:
        with st.spinner(f"Backtesting {bt_strategy} on {bt_ticker.upper()} ({bt_period})..."):
            try:
                r = requests.post(f"{BACKEND}/backtest", json={
                    "ticker": bt_ticker.upper(), "strategy": bt_strategy,
                    "period": bt_period, "initial_capital": bt_capital,
                    "commission": bt_comm, "params": bt_params,
                    "walk_forward": bt_wf, "wf_windows": 5,
                }, timeout=60)
                bt_data = r.json() if r.ok else None
            except Exception:
                bt_data = None

        if not bt_data or "total_return" not in bt_data:
            st.error("Backtest failed. Try a longer period or different ticker.")
        else:
            # ── Metric cards ──────────────────────────────────
            st.divider()
            tr  = bt_data.get("total_return", 0) or 0
            bh  = bt_data.get("benchmark_return", 0) or 0
            out = bt_data.get("outperformance", 0) or 0

            m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
            m1.metric("Total Return",    f"{tr:+.1f}%",   delta=f"{out:+.1f}% vs B&H")
            m2.metric("CAGR",            f"{bt_data.get('cagr',0) or 0:.1f}%")
            m3.metric("Sharpe",          f"{bt_data.get('sharpe',0) or 0:.2f}")
            m4.metric("Sortino",         f"{bt_data.get('sortino',0) or 0:.2f}")
            m5.metric("Max Drawdown",    f"{bt_data.get('max_drawdown',0) or 0:.1f}%")
            m6.metric("Win Rate",        f"{bt_data.get('win_rate',0) or 0:.0f}%")
            m7.metric("Trades",          bt_data.get("total_trades", 0))

            c1, c2, c3 = st.columns(3)
            c1.metric("Profit Factor",   f"{bt_data.get('profit_factor',0) or 0:.2f}")
            c2.metric("Calmar Ratio",    f"{bt_data.get('calmar',0) or 0:.2f}")
            c3.metric("Buy & Hold",      f"{bh:+.1f}%")

            # ── Equity curve ──────────────────────────────────
            st.subheader("Equity Curve vs Buy & Hold")
            dates  = bt_data.get("dates", [])
            equity = bt_data.get("equity_curve", [])

            if dates and equity:
                bh_curve = [bt_capital * (1 + bh/100 * i/max(len(dates)-1,1))
                            for i in range(len(dates))]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=equity,   name="Strategy",
                    line=dict(color="#3fb950", width=2)))
                fig.add_trace(go.Scatter(x=dates, y=bh_curve, name="Buy & Hold",
                    line=dict(color="#58a6ff", width=1.5, dash="dash")))
                fig.update_layout(title="Portfolio Value", height=350,
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"),
                    legend=dict(bgcolor="#161b22"),
                    margin=dict(l=10,r=10,t=40,b=10),
                    yaxis=dict(gridcolor="#21262d"),
                    xaxis=dict(gridcolor="#21262d"))
                st.plotly_chart(fig, width="stretch", key="bt_equity")

            # ── Drawdown ──────────────────────────────────────
            dd = bt_data.get("drawdown_curve", [])
            if dd:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=dates, y=dd, fill="tozeroy",
                    fillcolor="rgba(248,81,73,0.2)", line=dict(color="#f85149", width=1),
                    name="Drawdown %"))
                fig2.update_layout(title="Drawdown (%)", height=220,
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"), margin=dict(l=10,r=10,t=40,b=10),
                    yaxis=dict(gridcolor="#21262d"), xaxis=dict(gridcolor="#21262d"))
                st.plotly_chart(fig2, width="stretch", key="bt_drawdown")

            # ── Walk-forward results ──────────────────────────
            wf = bt_data.get("walk_forward", [])
            if wf:
                st.subheader("Walk-Forward Validation (Out-of-Sample)")
                st.caption("Each window is tested on unseen data — this is your real strategy performance.")
                valid = [w for w in wf if "error" not in w]
                if valid:
                    wf_df = pd.DataFrame(valid)
                    def _color_ret(v):
                        try:
                            return "color:#3fb950" if float(v) >= 0 else "color:#f85149"
                        except: return ""
                    st.dataframe(
                        wf_df.style.applymap(_color_ret, subset=["total_return","alpha"]),
                        width="stretch", hide_index=True)
                    avg_ret = sum(w.get("total_return",0) or 0 for w in valid) / len(valid)
                    avg_sh  = sum(w.get("sharpe",0) or 0      for w in valid) / len(valid)
                    profitable = sum(1 for w in valid if (w.get("total_return",0) or 0) > 0)
                    ca, cb, cc = st.columns(3)
                    ca.metric("Avg OOS Return",      f"{avg_ret:+.1f}%")
                    cb.metric("Avg OOS Sharpe",      f"{avg_sh:.2f}")
                    cc.metric("Profitable Windows",  f"{profitable}/{len(valid)}")

            # ── Trade log ─────────────────────────────────────
            trades = [t for t in bt_data.get("trades",[]) if t.get("action") in ("SELL","CLOSE")]
            if trades:
                with st.expander(f"📋 Trade Log ({len(trades)} closed trades)"):
                    st.dataframe(pd.DataFrame(trades), width="stretch", hide_index=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ Backtesting uses historical data — past performance does not guarantee future results.
    Walk-forward results are more reliable than full-period results. Always account for slippage.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════ PORTFOLIO ════════════════════════════
with tab_port:
    st.subheader("💼 Portfolio Optimizer")
    st.caption("Efficient frontier · Max Sharpe · Min Volatility · Risk Parity")

    with st.form("port_form"):
        port_tickers = st.text_input(
            "Tickers (comma separated, min 2, max 15)",
            value="AAPL, MSFT, NVDA, GOOGL, JPM",
            placeholder="AAPL, MSFT, TSLA, BTC-USD",
        )
        c1, c2 = st.columns(2)
        port_period = c1.selectbox("History Period", ["1y","2y","3y"], index=1)
        n_port      = c2.slider("Monte Carlo Portfolios", 1000, 5000, 3000, 500)
        run_port    = st.form_submit_button("🚀 Optimize Portfolio", type="primary",
                                            use_container_width=True)

    if run_port:
        tickers_list = [t.strip().upper() for t in port_tickers.split(",") if t.strip()]
        if len(tickers_list) < 2:
            st.warning("Enter at least 2 tickers.")
        else:
            with st.spinner(f"Optimizing {len(tickers_list)}-asset portfolio..."):
                try:
                    r = requests.post(f"{BACKEND}/portfolio/optimize", json={
                        "tickers": tickers_list, "period": port_period,
                        "n_portfolios": n_port,
                    }, timeout=90)
                    pdata = r.json() if r.ok else None
                except Exception:
                    pdata = None

            if not pdata or "portfolios" not in pdata:
                st.error("Optimization failed. Check tickers or try fewer assets.")
            else:
                # ── Efficient Frontier ────────────────────────
                st.subheader("Efficient Frontier")
                frontier = pdata.get("frontier", [])
                if frontier:
                    vols    = [p["volatility"] for p in frontier]
                    rets    = [p["return"]     for p in frontier]
                    sharpes = [p["sharpe"]     for p in frontier]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=vols, y=rets, mode="markers",
                        marker=dict(color=sharpes, colorscale="Viridis", size=4,
                                    opacity=0.6, colorbar=dict(title="Sharpe")),
                        name="Portfolios", text=[f"Sharpe: {s:.2f}" for s in sharpes],
                        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<br>%{text}",
                    ))
                    # Highlight optimal portfolios
                    pf = pdata["portfolios"]
                    for key, color, sym in [
                        ("max_sharpe",     "#ffd700", "star"),
                        ("min_volatility", "#00d084", "diamond"),
                        ("risk_parity",    "#58a6ff", "circle"),
                        ("equal_weight",   "#f85149", "square"),
                    ]:
                        p = pf[key]
                        fig.add_trace(go.Scatter(
                            x=[p["volatility"]], y=[p["return"]],
                            mode="markers+text",
                            marker=dict(color=color, size=14, symbol=sym,
                                        line=dict(color="white", width=1.5)),
                            name=p["label"], text=[p["label"]],
                            textposition="top center",
                            textfont=dict(color=color, size=10),
                        ))
                    fig.update_layout(
                        title="Risk vs Return (Efficient Frontier)",
                        xaxis_title="Annual Volatility (%)",
                        yaxis_title="Annual Return (%)",
                        height=450, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font=dict(color="#e6edf3"),
                        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
                        margin=dict(l=10,r=10,t=40,b=10),
                        yaxis=dict(gridcolor="#21262d"),
                        xaxis=dict(gridcolor="#21262d"),
                    )
                    st.plotly_chart(fig, width="stretch", key="port_frontier")

                # ── Portfolio Cards ───────────────────────────
                st.subheader("Optimal Portfolios")
                pf   = pdata["portfolios"]
                cols = st.columns(4)
                colors_map = {
                    "max_sharpe":     ("#ffd700", "🏆"),
                    "min_volatility": ("#00d084", "🛡️"),
                    "risk_parity":    ("#58a6ff", "⚖️"),
                    "equal_weight":   ("#f85149", "📊"),
                }
                for col, (key, (color, icon)) in zip(cols, colors_map.items()):
                    p = pf[key]
                    with col:
                        st.markdown(f"<div style='border:1px solid {color};border-radius:8px;padding:12px;background:#161b22'>",
                                    unsafe_allow_html=True)
                        st.markdown(f"**{icon} {p['label']}**")
                        st.metric("Return",     f"{p['return']:.1f}%")
                        st.metric("Volatility", f"{p['volatility']:.1f}%")
                        st.metric("Sharpe",     f"{p['sharpe']:.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)

                # ── Weights Comparison ────────────────────────
                st.subheader("Weight Allocation")
                tickers_n = pdata["tickers"]
                weight_rows = []
                for key in ["max_sharpe","min_volatility","risk_parity","equal_weight"]:
                    p = pf[key]
                    row = {"Portfolio": p["label"]}
                    row.update({t: f"{(p['weights'].get(t,0) or 0)*100:.1f}%" for t in tickers_n})
                    weight_rows.append(row)
                st.dataframe(pd.DataFrame(weight_rows), width="stretch", hide_index=True)

                # Stacked bar chart of weights
                fig2 = go.Figure()
                for t in tickers_n:
                    vals = [(pf[k]["weights"].get(t,0) or 0)*100
                            for k in ["max_sharpe","min_volatility","risk_parity","equal_weight"]]
                    fig2.add_trace(go.Bar(
                        name=t,
                        x=[pf[k]["label"] for k in ["max_sharpe","min_volatility","risk_parity","equal_weight"]],
                        y=vals, text=[f"{v:.0f}%" for v in vals],
                        textposition="inside",
                    ))
                fig2.update_layout(
                    barmode="stack", title="Weight Allocation per Portfolio",
                    height=320, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"),
                    legend=dict(bgcolor="#161b22"),
                    margin=dict(l=10,r=10,t=40,b=10),
                    yaxis=dict(title="Weight (%)", gridcolor="#21262d"),
                    xaxis=dict(gridcolor="#21262d"),
                )
                st.plotly_chart(fig2, width="stretch", key="port_weights")

                # ── Correlation Heatmap ───────────────────────
                st.subheader("Asset Correlation Matrix")
                corr_data = pdata.get("correlation", {})
                if corr_data:
                    matrix = corr_data["matrix"]
                    fig3   = go.Figure(go.Heatmap(
                        z=matrix, x=tickers_n, y=tickers_n,
                        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                        text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in matrix],
                        texttemplate="%{text}",
                        colorbar=dict(title="Correlation"),
                    ))
                    fig3.update_layout(
                        title="Correlation Heatmap", height=350,
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font=dict(color="#e6edf3"),
                        margin=dict(l=10,r=10,t=40,b=10),
                    )
                    st.plotly_chart(fig3, width="stretch", key="port_corr")

                # ── Individual asset stats ────────────────────
                ind = pdata.get("individual", {})
                if ind:
                    with st.expander("Individual Asset Statistics"):
                        rows = [{"Ticker": t,
                                 "Annual Return (%)":     v["annual_return"],
                                 "Annual Volatility (%)": v["annual_volatility"]}
                                for t, v in ind.items()]
                        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ Portfolio optimization is based on historical returns — future correlations and volatilities
    will differ. This is a quantitative tool for research only, not investment advice.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════ AI ANALYSIS (Phase 3) ════════════════
with tab_ai:
    st.markdown('<div class="main-title" style="font-size:1.6rem">🚀 AI Analysis</div>', unsafe_allow_html=True)
    st.caption("Regime Detection · Probabilistic Forecast · Ensemble Signal Score")

    col_l, col_r = st.columns([3,1])
    ai_ticker = col_l.text_input("Ticker for AI Analysis", value=active,
                                  key="ai_ticker_input", placeholder="AAPL")
    ai_period = col_r.selectbox("Period", ["6mo","1y","2y"], index=1, key="ai_period")
    run_ai    = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True, key="run_ai")

    if run_ai or st.session_state.get("ai_ran"):
        if run_ai:
            st.session_state["ai_ran"] = True
            st.session_state["ai_ticker_val"] = ai_ticker.upper()

        ai_sym = st.session_state.get("ai_ticker_val", ai_ticker.upper())

        with st.spinner(f"Running AI Analysis on {ai_sym} — Regime · Forecast · Ensemble..."):
            try:
                resp    = requests.get(f"{BACKEND}/analysis/{ai_sym}",
                                       params={"period": ai_period, "forecast_days": 10, "n_sim": 500},
                                       timeout=90)
                ai_data = resp.json() if resp.ok else None
            except Exception:
                ai_data = None

        if not ai_data:
            st.error("AI Analysis failed. Check ticker or try again.")
            st.session_state["ai_ran"] = False
        else:
            regime   = ai_data.get("regime",   {})
            forecast = ai_data.get("forecast", {})
            ensemble = ai_data.get("ensemble", {})

            # ══ TOMORROW'S SIGNAL (top of page) ══════════════════
            tomorrow = ai_data.get("tomorrow", {})
            if tomorrow and "error" not in tomorrow:
                action     = tomorrow.get("action",     "HOLD")
                color      = tomorrow.get("color",      "#8b949e")
                arrow      = tomorrow.get("arrow",      "→")
                confidence = tomorrow.get("confidence", "LOW")
                conf_color = tomorrow.get("conf_color", "#8b949e")
                score      = tomorrow.get("score",      50)
                prob_gain  = tomorrow.get("prob_gain",  50)
                t_range    = tomorrow.get("tomorrow_range", {})
                summary    = tomorrow.get("summary",    "")
                reasons    = tomorrow.get("top_reasons",[])
                disclaimer = tomorrow.get("disclaimer", "")

                st.markdown(
                    f"""<div style='background:#161b22;border:2px solid {color};
                    border-radius:14px;padding:24px 28px;margin-bottom:20px'>
                    <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px'>
                        <div>
                            <div style='font-size:0.8rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px'>
                                Tomorrow's Signal for {ai_sym}</div>
                            <div style='font-size:3rem;font-weight:900;color:{color};line-height:1.1'>
                                {arrow} {action}</div>
                            <div style='font-size:0.9rem;color:#8b949e;margin-top:4px'>{summary}</div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:0.75rem;color:#8b949e'>Ensemble Score</div>
                            <div style='font-size:2rem;font-weight:700;color:{color}'>{score}</div>
                            <div style='font-size:0.8rem;color:{conf_color};font-weight:600'>
                                {confidence} CONFIDENCE</div>
                            <div style='font-size:0.8rem;color:#8b949e;margin-top:4px'>
                                {prob_gain}% probability of gain</div>
                        </div>
                    </div>
                    </div>""",
                    unsafe_allow_html=True
                )

                # Tomorrow price range row
                last  = t_range.get("last")
                t_p25 = t_range.get("p25")
                t_p50 = t_range.get("p50")
                t_p75 = t_range.get("p75")
                t_p5  = t_range.get("p5")
                t_p95 = t_range.get("p95")

                if last and t_p50:
                    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                    rc1.metric("Last Close",      f"${last:,.2f}")
                    rc2.metric("Likely Low (p25)", f"${t_p25:,.2f}" if t_p25 else "N/A",
                               delta=f"{(t_p25-last)/last*100:+.1f}%" if t_p25 and last else None)
                    rc3.metric("Median (p50)",    f"${t_p50:,.2f}" if t_p50 else "N/A",
                               delta=f"{(t_p50-last)/last*100:+.1f}%" if t_p50 and last else None)
                    rc4.metric("Likely High (p75)",f"${t_p75:,.2f}" if t_p75 else "N/A",
                               delta=f"{(t_p75-last)/last*100:+.1f}%" if t_p75 and last else None)
                    rc5.metric("Extreme High (p95)",f"${t_p95:,.2f}" if t_p95 else "N/A",
                               delta=f"{(t_p95-last)/last*100:+.1f}%" if t_p95 and last else None)

                # Top reasons
                if reasons:
                    st.markdown("**Key signals driving this recommendation:**")
                    for r in reasons:
                        parts = r.split(":")
                        ind   = parts[0].strip()
                        rest  = ":".join(parts[1:]).strip() if len(parts)>1 else r
                        bull  = "BUY" in rest.upper()
                        bear  = "SELL" in rest.upper()
                        dot   = "🟢" if bull else "🔴" if bear else "⚪"
                        st.markdown(
                            f"<div style='padding:6px 0;color:#e6edf3'>{dot} "
                            f"<b>{ind}</b>: {rest}</div>",
                            unsafe_allow_html=True)

                st.caption(f"⚠️ {disclaimer}")
                st.divider()

            # ══ SECTION 1: REGIME DETECTION ═══════════════════
            st.divider()
            st.subheader("🎯 Market Regime Detection")
            st.caption("GaussianMixture model classifies current market state from returns, volatility, RSI and volume features")

            if "error" not in regime:
                cur_regime = regime.get("current_regime","Unknown")
                cur_color  = regime.get("current_color","#8b949e")
                duration   = regime.get("duration_bars", 0)
                probs      = regime.get("probabilities", {})

                # Regime badge + stats
                r1, r2, r3, r4 = st.columns(4)
                r1.markdown(
                    f"<div style='background:#161b22;border:2px solid {cur_color};border-radius:10px;"
                    f"padding:16px;text-align:center'>"
                    f"<div style='font-size:0.8rem;color:#8b949e'>Current Regime</div>"
                    f"<div style='font-size:1.3rem;font-weight:800;color:{cur_color}'>{cur_regime}</div>"
                    f"<div style='font-size:0.75rem;color:#8b949e'>{duration} bars active</div></div>",
                    unsafe_allow_html=True)

                # Probability bars for each regime
                color_map = regime.get("color_map", {})
                for col, (reg, prob) in zip([r2,r3,r4], list(probs.items())[:3]):
                    rc = color_map.get(reg, "#8b949e")
                    col.markdown(
                        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
                        f"padding:12px;text-align:center'>"
                        f"<div style='font-size:0.75rem;color:#8b949e'>{reg}</div>"
                        f"<div style='font-size:1.2rem;font-weight:700;color:{rc}'>{prob*100:.0f}%</div>"
                        f"</div>", unsafe_allow_html=True)

                # Strategy recommendation
                strategy = regime.get("strategy","")
                if strategy:
                    st.markdown(
                        f"<div style='background:#161b22;border-left:4px solid {cur_color};"
                        f"padding:12px 16px;border-radius:0 8px 8px 0;margin:12px 0'>"
                        f"<b style='color:{cur_color}'>Strategy Recommendation:</b> "
                        f"<span style='color:#e6edf3'>{strategy}</span></div>",
                        unsafe_allow_html=True)

                # Regime history chart
                hist    = regime.get("history", [])
                dates_r = regime.get("dates",   [])
                if hist and dates_r:
                    regime_num = {"Bull Trend":3,"Consolidation":2,
                                  "High Volatility":1,"Bear Trend":0,"Unknown":2}
                    y_vals     = [regime_num.get(r,2) for r in hist]
                    bar_colors = [color_map.get(r,"#8b949e") for r in hist]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=dates_r, y=y_vals, marker_color=bar_colors,
                                         name="Regime", hovertext=hist,
                                         hovertemplate="%{hovertext}<extra></extra>"))
                    fig.update_layout(
                        title="Regime History (last 90 bars)", height=220,
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font=dict(color="#e6edf3"),
                        margin=dict(l=10,r=10,t=40,b=10),
                        yaxis=dict(tickvals=[0,1,2,3],
                                   ticktext=["Bear","High Vol","Consolidation","Bull"],
                                   gridcolor="#21262d"),
                        xaxis=dict(gridcolor="#21262d"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, width="stretch", key="ai_regime_hist")

                # Time in each regime
                regime_pct = regime.get("regime_pct",{})
                if regime_pct:
                    with st.expander("Time spent in each regime"):
                        rp_df = pd.DataFrame([
                            {"Regime": k, "% of Time": f"{v}%",
                             "Color":  color_map.get(k,"#8b949e")}
                            for k,v in regime_pct.items()
                        ])
                        st.dataframe(rp_df[["Regime","% of Time"]], width="stretch", hide_index=True)
            else:
                st.warning(f"Regime detection unavailable: {regime.get('error')}")

            # ══ SECTION 2: PROBABILISTIC FORECAST ═════════════
            st.divider()
            st.subheader("🔮 Probabilistic Price Forecast")
            st.caption("EWMA-GARCH volatility + ARIMA(1,1,1) drift + Bootstrap Monte Carlo (500 simulations)")

            if "error" not in forecast:
                last_p  = forecast.get("last_price", 0)
                p_gain  = forecast.get("prob_gain",  50)
                ann_vol = forecast.get("annual_vol_pct", 0)
                exp_r   = forecast.get("expected_range", {})

                f1,f2,f3,f4 = st.columns(4)
                f1.metric("Last Price",       f"${last_p:,.2f}" if last_p else "N/A")
                pg_color = "normal" if p_gain >= 50 else "inverse"
                f2.metric("Prob. of Gain",    f"{p_gain:.0f}%")
                f3.metric("Annual Volatility",f"{ann_vol:.1f}%")
                er_low  = exp_r.get("low_pct",0) or 0
                er_high = exp_r.get("high_pct",0) or 0
                f4.metric("80% Range (10d)",  f"{er_low:+.1f}% / {er_high:+.1f}%")

                # Fan chart
                bands   = forecast.get("bands",{})
                fdates  = forecast.get("forecast_dates",[])
                hdates  = forecast.get("hist_dates",[])
                hprices = forecast.get("hist_prices",[])

                if bands and fdates:
                    all_dates = hdates + fdates
                    fig = go.Figure()

                    # Historical prices
                    fig.add_trace(go.Scatter(
                        x=hdates, y=hprices, name="Historical",
                        line=dict(color="#58a6ff", width=2), mode="lines"))

                    # Confidence bands (p5–p95, p25–p75)
                    p5  = bands.get("p5",  [])
                    p25 = bands.get("p25", [])
                    p50 = bands.get("p50", [])
                    p75 = bands.get("p75", [])
                    p95 = bands.get("p95", [])

                    fig.add_trace(go.Scatter(
                        x=fdates+fdates[::-1], y=p95+p5[::-1],
                        fill="toself", fillcolor="rgba(63,185,80,0.08)",
                        line=dict(color="rgba(0,0,0,0)"), name="90% CI", showlegend=True))
                    fig.add_trace(go.Scatter(
                        x=fdates+fdates[::-1], y=p75+p25[::-1],
                        fill="toself", fillcolor="rgba(63,185,80,0.18)",
                        line=dict(color="rgba(0,0,0,0)"), name="50% CI", showlegend=True))
                    fig.add_trace(go.Scatter(
                        x=fdates, y=p50, name="Median (p50)",
                        line=dict(color="#3fb950", width=2.5, dash="dot")))
                    fig.add_trace(go.Scatter(
                        x=fdates, y=p5,  name="p5",
                        line=dict(color="#f85149", width=1, dash="dash")))
                    fig.add_trace(go.Scatter(
                        x=fdates, y=p95, name="p95",
                        line=dict(color="#3fb950", width=1, dash="dash")))

                    # Vertical line separating history from forecast
                    if hdates:
                        fig.add_vline(x=hdates[-1], line=dict(color="#d29922",
                                      dash="dash", width=1))

                    fig.update_layout(
                        title=f"{ai_sym} — 10-Day Probabilistic Forecast",
                        height=380, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font=dict(color="#e6edf3"),
                        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
                        margin=dict(l=10,r=10,t=40,b=10),
                        yaxis=dict(gridcolor="#21262d"),
                        xaxis=dict(gridcolor="#21262d"),
                    )
                    st.plotly_chart(fig, width="stretch", key="ai_forecast")

                    st.caption(f"📌 {forecast.get('methodology','')}")
            else:
                st.warning(f"Forecast unavailable: {forecast.get('error')}")

            # ══ SECTION 3: ENSEMBLE SCORE ══════════════════════
            st.divider()
            st.subheader("🤖 Ensemble Signal Score")
            st.caption("All Phase 1 indicators combined with regime-adaptive weighting → single 0–100 confidence score")

            if "error" not in ensemble:
                score = ensemble.get("score", 50)
                label = ensemble.get("label","NEUTRAL")
                color = ensemble.get("color","#8b949e")
                buy_c = ensemble.get("buy_components",0)
                sel_c = ensemble.get("sell_components",0)
                regime_used = ensemble.get("regime_used","default")

                # Big score display
                st.markdown(
                    f"<div style='background:#161b22;border:2px solid {color};border-radius:12px;"
                    f"padding:24px;text-align:center;margin:12px 0'>"
                    f"<div style='font-size:4rem;font-weight:900;color:{color}'>{score}</div>"
                    f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{label}</div>"
                    f"<div style='font-size:0.85rem;color:#8b949e;margin-top:6px'>"
                    f"Regime: {regime_used} · {buy_c} bullish · {sel_c} bearish components</div>"
                    f"</div>", unsafe_allow_html=True)

                # Score gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"font": {"color": color, "size": 40}},
                    gauge={
                        "axis":  {"range": [0,100], "tickcolor": "#8b949e"},
                        "bar":   {"color": color, "thickness": 0.25},
                        "bgcolor": "#161b22",
                        "bordercolor": "#30363d",
                        "steps": [
                            {"range":[0,20],  "color":"#3d0000"},
                            {"range":[20,35], "color":"#5a1212"},
                            {"range":[35,45], "color":"#2d2010"},
                            {"range":[45,55], "color":"#1c2128"},
                            {"range":[55,65], "color":"#0d3320"},
                            {"range":[65,80], "color":"#1a4030"},
                            {"range":[80,100],"color":"#1a7f37"},
                        ],
                        "threshold": {"line":{"color":"white","width":4},
                                      "thickness":0.75,"value":score},
                    }
                ))
                fig_g.update_layout(
                    height=250, paper_bgcolor="#0d1117",
                    font=dict(color="#e6edf3"),
                    margin=dict(l=30,r=30,t=20,b=10))
                st.plotly_chart(fig_g, width="stretch", key="ai_ensemble_gauge")

                # Component breakdown
                st.subheader("Component Breakdown")
                components = ensemble.get("components",{})
                weights    = ensemble.get("weights_used",{})
                wkey_map   = {"RSI":"rsi","MACD":"macd","MA Cross":"ma_cross",
                               "Bollinger":"bollinger","Stochastic":"stochastic",
                               "Volume":"volume","ATR Trend":"atr_trend"}
                comp_rows  = []
                for name, data in components.items():
                    wk    = wkey_map.get(name,"")
                    w_val = weights.get(wk, 0)
                    sc    = data.get("score",0.5)
                    sig   = data.get("signal","NEUTRAL")
                    sig_c = "#3fb950" if "BUY" in sig else "#f85149" if "SELL" in sig else "#8b949e"
                    comp_rows.append({
                        "Indicator": name,
                        "Value":     str(data.get("value","")),
                        "Score":     round(sc*100),
                        "Signal":    sig,
                        "Weight":    f"{w_val*100:.0f}%",
                        "Contribution": round(sc*w_val*100,1),
                    })

                comp_df = pd.DataFrame(comp_rows)
                def _color_sig(v):
                    if "BUY"  in str(v): return "color:#3fb950;font-weight:bold"
                    if "SELL" in str(v): return "color:#f85149;font-weight:bold"
                    return "color:#8b949e"
                st.dataframe(comp_df.style.applymap(_color_sig, subset=["Signal"]),
                             width="stretch", hide_index=True)

                # Radar / bar chart of component scores
                fig_bar = go.Figure()
                names   = [r["Indicator"] for r in comp_rows]
                scores  = [r["Score"]     for r in comp_rows]
                bar_c   = ["#3fb950" if s>=55 else "#f85149" if s<45 else "#8b949e"
                           for s in scores]
                fig_bar.add_trace(go.Bar(x=names, y=scores,
                    marker_color=bar_c, text=[f"{s}" for s in scores],
                    textposition="outside"))
                fig_bar.add_hline(y=55, line=dict(color="#3fb950",dash="dash",width=1))
                fig_bar.add_hline(y=45, line=dict(color="#f85149",dash="dash",width=1))
                fig_bar.update_layout(
                    title="Component Scores (>55 Bullish · <45 Bearish)",
                    height=280, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font=dict(color="#e6edf3"),
                    margin=dict(l=10,r=10,t=40,b=10),
                    yaxis=dict(range=[0,110], gridcolor="#21262d"),
                    xaxis=dict(gridcolor="#21262d"),
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, width="stretch", key="ai_components")

            else:
                st.warning(f"Ensemble unavailable: {ensemble.get('error')}")

    # ══ PLAIN LANGUAGE GUIDE ══════════════════════════════════
    st.divider()
    st.subheader("📖 What does this app tell you? (Plain English)")
    st.caption("For users new to stock markets — here's exactly what each section means")

    guide_items = [
        ("📊 Overview",            "Current price, market cap, 52-week range. Like checking the price tag and size of a company."),
        ("📉 Technical Analysis",  "Are traders currently buying or selling this stock? RSI, MACD, Bollinger Bands are like weather instruments — they measure market pressure, not guarantee direction."),
        ("📰 News Sentiment",      "Are recent news headlines positive or negative about this stock? Scored automatically — GREEN means mostly good news, RED means mostly bad news."),
        ("💰 Earnings & Fundamentals", "How much money is the company actually making? EPS = earnings per share. Higher actual vs estimate = good surprise. Analyst consensus = what Wall Street experts think."),
        ("🔍 Stock Screener",      "Scan many stocks at once and filter by signal or RSI. Like a search filter to find stocks worth investigating further."),
        ("⚡ Backtester",          "If you had used a trading strategy (e.g. buy when RSI<30, sell when RSI>70) in the past, how much money would you have made or lost? Walk-forward = tested on unseen data so results are honest."),
        ("💼 Portfolio Optimizer", "If you own multiple stocks, what % of your money should go into each to maximise return for minimum risk? Max Sharpe = best return per unit of risk. Risk Parity = equal risk from each asset."),
        ("🚀 AI Analysis",         "Combines everything: What market state are we in? (Regime) → What price range is realistic next 10 days? (Forecast) → Do all indicators agree on direction? (Ensemble Score 0-100)"),
    ]

    for icon_title, explanation in guide_items:
        with st.expander(icon_title):
            st.markdown(f"<p style='color:#e6edf3;font-size:0.95rem;line-height:1.6'>{explanation}</p>",
                        unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-top:12px'>
    <b style='color:#d29922'>⚠️ Most important thing to know:</b>
    <p style='color:#8b949e;margin:8px 0 0 0;font-size:0.88rem'>
    No tool — including this one — can reliably predict whether a stock will go up or down tomorrow.
    What this app gives you is <b style='color:#e6edf3'>evidence stacking</b>: when multiple independent
    indicators agree, you have a stronger (not guaranteed) signal. Professional traders use tools like
    this to improve their <i>odds</i>, manage <i>risk</i>, and avoid emotional decisions — not to
    predict the future with certainty.
    </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ AI Analysis uses statistical models on historical data. Regime detection, probabilistic forecasts,
    and ensemble scores are <b>research tools only</b> — not financial advice or trading signals.
    All models have limitations and can be wrong. Always apply independent judgment.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════ DECISION ENGINE ════════════════════
with tab_de:
    st.markdown('<div class="main-title" style="font-size:1.6rem">🎯 Decision Engine</div>',
                unsafe_allow_html=True)
    st.caption("Every signal combined automatically → one clear BUY / HOLD / SELL verdict")

    de_col1, de_col2 = st.columns([3,1])
    de_ticker = de_col1.text_input("Enter ticker", value=active, key="de_ticker",
                                    placeholder="AAPL, RELIANCE.NS, BTC-USD")
    de_period = de_col2.selectbox("Period", ["6mo","1y","2y"], index=1, key="de_period")
    run_de    = st.button("🎯 Analyze & Decide", type="primary",
                          use_container_width=True, key="run_de")

    if run_de:
        st.session_state["de_ran"] = True
        st.session_state["de_sym"] = de_ticker.upper()

    if st.session_state.get("de_ran"):
        de_sym = st.session_state.get("de_sym", de_ticker.upper())
        de_per = st.session_state.get("de_period", "1y")

        with st.spinner(f"Running full analysis on {de_sym} — Technical · Macro News · Earnings · Forecast..."):
            try:
                resp    = requests.get(f"{BACKEND}/decide/{de_sym}",
                                       params={"period": de_per}, timeout=120)
                de_data = resp.json() if resp.ok else None
            except Exception:
                de_data = None

        if not de_data or "decision" not in de_data:
            st.error("Analysis failed. Check ticker or try again.")
            st.session_state["de_ran"] = False
        else:
            dec    = de_data["decision"]
            macro  = de_data.get("macro",  {})
            fct    = de_data.get("forecast",{})
            sector = de_data.get("sector", "N/A")
            regime = de_data.get("regime", "Unknown")

            verdict    = dec.get("verdict",    "HOLD")
            color      = dec.get("color",      "#8b949e")
            arrow      = dec.get("arrow",      "→")
            conf       = dec.get("confidence", "LOW")
            conf_color = dec.get("conf_color", "#8b949e")
            score      = dec.get("final_score",50)
            plain      = dec.get("plain_english","")
            risks      = dec.get("risks",      [])
            levels     = dec.get("levels",     {})
            comp_sc    = dec.get("component_scores", {})
            last_p     = dec.get("last_price", 0)
            macro_sig  = dec.get("macro_signal","NEUTRAL")
            macro_risk = dec.get("macro_risk","UNKNOWN")

            # ── MAIN VERDICT CARD ─────────────────────────────
            st.markdown(f"""
            <div style='background:#161b22;border:3px solid {color};border-radius:16px;
            padding:28px 32px;text-align:center;margin:16px 0'>
                <div style='font-size:0.85rem;color:#8b949e;text-transform:uppercase;
                letter-spacing:2px;margin-bottom:8px'>Decision for {de_sym} · {sector}</div>
                <div style='font-size:4.5rem;font-weight:900;color:{color};
                line-height:1'>{arrow} {verdict}</div>
                <div style='display:flex;justify-content:center;gap:32px;margin-top:16px;flex-wrap:wrap'>
                    <div><span style='color:#8b949e;font-size:0.8rem'>SCORE</span>
                    <div style='font-size:1.8rem;font-weight:700;color:{color}'>{score}/100</div></div>
                    <div><span style='color:#8b949e;font-size:0.8rem'>CONFIDENCE</span>
                    <div style='font-size:1.2rem;font-weight:700;color:{conf_color}'>{conf}</div></div>
                    <div><span style='color:#8b949e;font-size:0.8rem'>REGIME</span>
                    <div style='font-size:1.2rem;font-weight:700;color:#e6edf3'>{regime}</div></div>
                    <div><span style='color:#8b949e;font-size:0.8rem'>MACRO</span>
                    <div style='font-size:1.2rem;font-weight:700;
                    color:{"#f85149" if "BEAR" in macro_sig else "#3fb950" if "BULL" in macro_sig else "#8b949e"}'>{macro_sig}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── PLAIN ENGLISH ─────────────────────────────────
            if plain:
                st.info(f"📖 **What this means:** {plain}")

            st.divider()

            # ── ENTRY / STOP / TARGET ─────────────────────────
            if levels and last_p:
                st.subheader("📍 Trade Levels")
                st.caption("Entry zone = forecast p25–p50 (buy on dip) · Stop = 1.5× ATR below entry · Target = forecast p75")

                l1,l2,l3,l4,l5 = st.columns(5)
                l1.metric("Last Price",    f"${last_p:,.2f}" if last_p else "N/A")
                el = levels.get("entry_low")
                eh = levels.get("entry_high")
                sl = levels.get("stop_loss")
                tp = levels.get("take_profit")
                rr = levels.get("rr_ratio")
                sp = levels.get("stop_pct")
                tp_p = levels.get("target_pct")
                l2.metric("Entry Zone",   f"${el:,.2f}–${eh:,.2f}" if el and eh else "N/A")
                l3.metric("Stop Loss",    f"${sl:,.2f}" if sl else "N/A",
                           delta=f"{sp:.1f}%" if sp else None, delta_color="inverse")
                l4.metric("Take Profit",  f"${tp:,.2f}" if tp else "N/A",
                           delta=f"+{tp_p:.1f}%" if tp_p else None)
                l5.metric("Risk/Reward",  f"1 : {rr:.1f}" if rr else "N/A")

                # Visual risk bar
                if sl and tp and last_p and el:
                    entry_mid = (el + (eh or el)) / 2
                    total_range = (tp - sl)
                    if total_range > 0:
                        loss_pct   = (entry_mid - sl) / total_range * 100
                        profit_pct = (tp - entry_mid) / total_range * 100
                        st.markdown(f"""
                        <div style='margin:12px 0'>
                        <div style='display:flex;height:16px;border-radius:8px;overflow:hidden'>
                            <div style='width:{loss_pct:.0f}%;background:#f85149;opacity:0.8'></div>
                            <div style='width:4px;background:#ffffff'></div>
                            <div style='width:{profit_pct:.0f}%;background:#3fb950;opacity:0.8'></div>
                        </div>
                        <div style='display:flex;justify-content:space-between;
                        font-size:0.75rem;color:#8b949e;margin-top:4px'>
                            <span>Stop ${sl:,.2f}</span>
                            <span>Entry ~${entry_mid:,.2f}</span>
                            <span>Target ${tp:,.2f}</span>
                        </div></div>""", unsafe_allow_html=True)

            st.divider()

            # ── SCORE BREAKDOWN ───────────────────────────────
            st.subheader("📊 Score Breakdown")
            st.caption("How each signal contributed to the final verdict")

            weights = dec.get("weights", {})
            score_rows = [
                ("🔧 Technical Indicators", "technical",       "All 7 indicators (RSI, MACD, BB, MA Cross, Stochastic, Volume, ATR)"),
                ("📰 Stock News",           "stock_sentiment", "Yahoo Finance news sentiment about this stock"),
                ("🌍 Global Macro News",    "macro_sentiment", "Reuters, BBC, AP, CNBC — global events relevant to this sector"),
                ("💰 Earnings Quality",     "earnings",        "Recent EPS beat/miss vs Wall Street estimates"),
                ("🔮 Forecast Direction",   "forecast",        "Monte Carlo probability of gain over 10 days"),
            ]

            for label, key, desc in score_rows:
                sc  = comp_sc.get(key, 50)
                w   = weights.get(key, 0)
                bar = sc / 100
                col = "#3fb950" if sc >= 55 else "#f85149" if sc < 45 else "#8b949e"
                st.markdown(f"""
                <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:12px 16px;margin:6px 0'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                        <div>
                            <span style='color:#e6edf3;font-weight:600'>{label}</span>
                            <span style='color:#6e7681;font-size:0.75rem;margin-left:8px'>
                            weight {int(w*100)}%</span>
                        </div>
                        <span style='color:{col};font-weight:700;font-size:1.1rem'>{sc:.0f}/100</span>
                    </div>
                    <div style='background:#21262d;border-radius:4px;height:6px;margin:8px 0'>
                        <div style='background:{col};width:{sc}%;height:6px;border-radius:4px'></div>
                    </div>
                    <div style='color:#6e7681;font-size:0.75rem'>{desc}</div>
                </div>""", unsafe_allow_html=True)

            st.divider()

            # ── GLOBAL MACRO NEWS ─────────────────────────────
            st.subheader("🌍 Global Macro Events")
            risk_color = {"HIGH":"#f85149","ELEVATED":"#d29922",
                          "MODERATE":"#58a6ff","LOW":"#3fb950"}.get(macro_risk,"#8b949e")
            st.markdown(
                f"Global Risk Level: <span style='color:{risk_color};font-weight:700'>"
                f"{macro_risk}</span> · {macro.get('summary','')}",
                unsafe_allow_html=True)

            # Only show MEDIUM/HIGH impact articles with relevance >= 3.5
            arts = [a for a in macro.get("articles",[])
                    if a.get("relevance",0) >= 3.5 and "LOW" not in a.get("impact","LOW")]
            if not arts:
                arts = sorted(macro.get("articles",[]),
                              key=lambda x: x.get("relevance",0), reverse=True)[:3]
            if arts:
                for art in arts[:8]:
                    impact = art.get("impact","🟢 LOW")
                    comp   = art.get("compound",0)
                    rel    = art.get("relevance",0)
                    sent_c = "#3fb950" if comp>=0.05 else "#f85149" if comp<=-0.05 else "#8b949e"
                    with st.container():
                        ca, cb = st.columns([7,1])
                        with ca:
                            url   = art.get("url","")
                            title = art.get("title","")
                            st.markdown(f"**{impact}** &nbsp; "
                                        f"[{title}]({url})" if url else f"**{impact}** {title}")
                            st.caption(f"{art.get('source','')} · Relevance: {rel}/10 · {art.get('pub','')}")
                        with cb:
                            st.markdown(f"<div style='color:{sent_c};font-weight:700;"
                                        f"text-align:right'>{comp:+.3f}</div>",
                                        unsafe_allow_html=True)
                        st.divider()
            else:
                st.info("No relevant global news found for this sector.")

            # ── RISK FACTORS ──────────────────────────────────
            if risks:
                st.subheader("⚠️ Risk Factors")
                for r in risks:
                    st.markdown(f"🔴 {r}")

        st.markdown("""
        <div class="disclaimer">
        ⚠️ Decision Engine combines multiple signals but <b>cannot guarantee tomorrow's direction</b>.
        Use entry/stop/target levels as guidelines — always apply your own judgment.
        This is a research tool, not financial advice.
        </div>""", unsafe_allow_html=True)
