"""
macro_sentiment.py — Global Macro News Sentiment
Sources: Reuters, BBC, AP, CNBC, MarketWatch (all free RSS, no API key)
Maps global events (wars, food shortage, rate hikes) to stock impact by sector.
"""
import xml.etree.ElementTree as ET
import re, time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import _session

_va = SentimentIntensityAnalyzer()

RSS_FEEDS = [
    ("Reuters Top News",  "https://feeds.reuters.com/reuters/topNews"),
    ("Reuters Business",  "https://feeds.reuters.com/reuters/businessNews"),
    ("AP News",           "https://feeds.apnews.com/rss/apf-topnews"),
    ("BBC World",         "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Business",      "http://feeds.bbci.co.uk/news/business/rss.xml"),
    ("CNBC Markets",      "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("MarketWatch",       "https://feeds.marketwatch.com/marketwatch/topstories/"),
]

# Sector → macro keywords that affect this sector
SECTOR_KEYWORDS = {
    "Technology":              ["semiconductor","chip","AI chip","artificial intelligence",
                                "China tech","chip export","tariff technology","data center",
                                "cloud computing","Taiwan strait","export control","tech regulation",
                                "cybersecurity breach","quantum computing","TSMC","Nvidia","Apple supply",
                                "iPhone","silicon","processor","GPU","tech earnings"],
    "Energy":                  ["oil","OPEC","crude","natural gas","pipeline","energy supply",
                                "war","sanction","refinery","fossil fuel","LNG","energy crisis"],
    "Financial Services":      ["interest rate","Fed","Federal Reserve","inflation","bank",
                                "credit","recession","ECB","central bank","bond yield","banking crisis"],
    "Healthcare":              ["FDA","drug approval","clinical trial","pandemic","vaccine",
                                "healthcare","pharma","biosimilar","Medicare","hospital"],
    "Consumer Cyclical":       ["consumer spending","retail sales","supply chain","inflation",
                                "housing","e-commerce","tariff","consumer confidence"],
    "Consumer Defensive":      ["food supply","food shortage","shortage","commodity","agriculture",
                                "drought","harvest","food price","grocery","wheat","corn"],
    "Industrials":             ["manufacturing","supply chain","trade war","infrastructure",
                                "construction","factory","freight","shipping"],
    "Automotive":              ["EV","electric vehicle","battery","lithium","charging",
                                "auto sales","car","emission","self-driving"],
    "Communication Services":  ["regulation","antitrust","media","streaming","social media",
                                "censorship","broadband","5G"],
    "Real Estate":             ["interest rate","mortgage","housing market","property",
                                "Fed","rent","commercial real estate"],
    "Utilities":               ["energy grid","electricity","water shortage","regulatory",
                                "power","nuclear","clean energy"],
    "Basic Materials":         ["commodity","metal","mining","iron ore","copper","gold",
                                "silver","supply shortage","rare earth"],
    "default":                 ["war","conflict","recession","inflation","Fed",
                                "interest rate","GDP","unemployment","trade war",
                                "sanction","geopolitical","central bank","dollar","crisis",
                                "food shortage","oil","pandemic","earthquake","flood"],
}

# High-impact events that affect ALL stocks regardless of sector
GLOBAL_SHOCK_KEYWORDS = [
    "nuclear attack","global pandemic","market crash","financial crisis",
    "emergency rate hike","oil embargo","global food crisis",
    "supply chain collapse","trade war escalation","world war",
    "stock market crash","economic collapse","global recession declared",
]

# Map ticker suffix / exchange to geography keywords
GEO_KEYWORDS = {
    ".NS": ["India","RBI","rupee","Sensex","Nifty","Modi","Indian economy"],
    ".BO": ["India","RBI","rupee","Sensex","BSE","Indian economy"],
    ".DE": ["Germany","EU","Euro","ECB","DAX","European"],
    ".L":  ["UK","Britain","BOE","FTSE","pound","Brexit"],
    "-USD":["crypto","bitcoin","blockchain","digital asset","SEC crypto"],
}


def _clean(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _fetch_rss(source_name: str, url: str) -> list:
    try:
        r = _session.get(url, timeout=8)
        if not r.ok:
            return []
        root  = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:15]:
            title = _clean(item.findtext("title", ""))
            desc  = _clean(item.findtext("description", ""))
            link  = item.findtext("link", "")
            pub   = item.findtext("pubDate", "")[:22] if item.findtext("pubDate") else ""
            if title:
                items.append({
                    "source": source_name,
                    "title":  title,
                    "desc":   desc[:300],
                    "url":    link,
                    "pub":    pub,
                })
        return items
    except Exception as e:
        print(f"[macro] RSS {source_name} failed: {e}")
        return []


def _relevance(text: str, sector: str, ticker: str) -> float:
    """Score 0–10 how relevant this article is to the stock."""
    tl = text.lower()
    score = 0.0

    # Sector keywords
    for kw in SECTOR_KEYWORDS.get(sector, SECTOR_KEYWORDS["default"]):
        if kw.lower() in tl:
            score += 1.0

    # Always-relevant macro keywords
    for kw in SECTOR_KEYWORDS["default"]:
        if kw.lower() in tl:
            score += 0.3

    # Geography bonus
    for suffix, geo_kws in GEO_KEYWORDS.items():
        if suffix in ticker.upper():
            for kw in geo_kws:
                if kw.lower() in tl:
                    score += 1.5

    # Global shock bonus
    for kw in GLOBAL_SHOCK_KEYWORDS:
        if kw.lower() in tl:
            score += 2.0

    # Direct ticker mention
    base = ticker.upper().replace(".NS","").replace(".BO","").replace(".DE","").replace(".L","").replace("-USD","")
    if base in text.upper():
        score += 3.0

    return round(min(score, 10.0), 2)


def _impact_label(rel: float, compound: float) -> str:
    if rel >= 5 and abs(compound) >= 0.3: return "🔴 HIGH"
    if rel >= 3 and abs(compound) >= 0.1: return "🟡 MEDIUM"
    return "🟢 LOW"


def get_macro_sentiment(ticker: str, sector: str = "default") -> dict:
    """
    Fetches global news from 7 RSS sources, scores each article for
    relevance to this stock's sector, applies VADER sentiment.
    Returns overall macro signal + top impactful articles.
    """
    all_articles = []
    for name, url in RSS_FEEDS:
        items = _fetch_rss(name, url)
        all_articles.extend(items)
        time.sleep(0.1)   # gentle rate limiting

    scored = []
    for art in all_articles:
        text     = f"{art['title']}. {art['desc']}"
        rel      = _relevance(text, sector, ticker)
        if rel < 3.0:           # strict threshold — must be genuinely relevant
            continue
        vs       = _va.polarity_scores(text)
        compound = vs["compound"]
        scored.append({
            "source":    art["source"],
            "title":     art["title"],
            "desc":      art["desc"],
            "url":       art["url"],
            "published": art["pub"],
            "relevance": rel,
            "compound":  round(compound, 4),
            "label":     "POSITIVE" if compound>=0.05 else "NEGATIVE" if compound<=-0.05 else "NEUTRAL",
            "emoji":     "🟢" if compound>=0.05 else "🔴" if compound<=-0.05 else "⚪",
            "impact":    _impact_label(rel, compound),
        })

    # Sort by relevance × abs(sentiment)
    scored.sort(key=lambda x: x["relevance"] * abs(x["compound"]), reverse=True)
    top = scored[:15]

    if not top:
        return {
            "ticker": ticker, "sector": sector,
            "overall_score": 0.0, "overall_label": "NO DATA",
            "macro_signal": "NEUTRAL", "articles": [],
            "risk_level": "UNKNOWN",
            "summary": "No relevant global news found.",
        }

    # Weighted average (weight by relevance)
    total_w = sum(a["relevance"] for a in top)
    avg     = sum(a["compound"] * a["relevance"] for a in top) / (total_w + 1e-9)
    pos_c   = sum(1 for a in top if a["compound"] >= 0.05)
    neg_c   = sum(1 for a in top if a["compound"] <= -0.05)
    high_c  = sum(1 for a in top if "HIGH" in a["impact"])

    # Risk level
    if high_c >= 3 or avg <= -0.3:   risk = "HIGH"
    elif high_c >= 1 or avg <= -0.1: risk = "ELEVATED"
    elif avg >= 0.2:                  risk = "LOW"
    else:                             risk = "MODERATE"

    # Macro signal
    if avg >= 0.15:   macro_sig = "BULLISH"
    elif avg >= 0.05: macro_sig = "WEAK BULLISH"
    elif avg <= -0.15:macro_sig = "BEARISH"
    elif avg <= -0.05:macro_sig = "WEAK BEARISH"
    else:             macro_sig = "NEUTRAL"

    # Plain English summary
    high_arts = [a for a in top if "HIGH" in a["impact"]]
    if high_arts:
        summary = f"{len(high_arts)} high-impact global event(s) affect this stock. "
        summary += f"Top: {high_arts[0]['title'][:80]}..."
    elif neg_c > pos_c:
        summary = f"Global news is mostly negative ({neg_c} bearish vs {pos_c} bullish articles relevant to this sector)."
    elif pos_c > neg_c:
        summary = f"Global news is mostly positive ({pos_c} bullish vs {neg_c} bearish articles relevant to this sector)."
    else:
        summary = "Global macro news is mixed with no dominant direction for this sector."

    return {
        "ticker":         ticker,
        "sector":         sector,
        "overall_score":  round(avg, 4),
        "overall_label":  "POSITIVE" if avg>=0.05 else "NEGATIVE" if avg<=-0.05 else "NEUTRAL",
        "macro_signal":   macro_sig,
        "risk_level":     risk,
        "positive_count": pos_c,
        "negative_count": neg_c,
        "high_impact_count": high_c,
        "article_count":  len(top),
        "summary":        summary,
        "articles":       top,
    }
