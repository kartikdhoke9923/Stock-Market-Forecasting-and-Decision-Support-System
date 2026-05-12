"""
sentiment.py — News Sentiment Analyzer (Phase 1)
Primary: Yahoo Finance RSS feed (free, no API key, uses cookie session)
Fallback: Yahoo Finance search API with crumb
Scoring: VADER sentiment analyzer
"""
import xml.etree.ElementTree as ET
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

_va = SentimentIntensityAnalyzer()

def _label(c): return "POSITIVE" if c >= 0.05 else "NEGATIVE" if c <= -0.05 else "NEUTRAL"
def _emoji(c): return ("🟢" if c >= 0.3 else "🟡" if c >= 0.05 else
                       "🔴" if c <= -0.3 else "🟠" if c <= -0.05 else "⚪")


def _rss_news(sym: str) -> list:
    """
    Yahoo Finance RSS feed — completely free, no crumb needed.
    Returns real headlines with descriptions for better VADER scoring.
    """
    from utils import _session
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
    try:
        r = _session.get(url, timeout=10)
        if not r.ok:
            return []
        root  = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item"):
            title = item.findtext("title",       "").strip()
            desc  = item.findtext("description", "").strip()
            link  = item.findtext("link",        "").strip()
            pub   = item.findtext("pubDate",     "").strip()
            src_el = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            publisher = "Yahoo Finance News"

            # Clean up HTML tags from description
            import re
            desc = re.sub(r"<[^>]+>", "", desc)[:300]

            if title:
                items.append({
                    "title":     title,
                    "summary":   desc,
                    "url":       link,
                    "publisher": publisher,
                    "published": pub[:22] if pub else "N/A",
                })
        return items
    except Exception as e:
        print(f"[sentiment] RSS failed: {e}")
        return []


def _search_api_news(sym: str) -> list:
    """
    Yahoo Finance search API news — uses cookie session + crumb.
    Fallback when RSS fails.
    """
    from utils import _session, _get_crumb
    try:
        crumb = _get_crumb()
        url   = (
            f"https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={sym}&quotesCount=0&newsCount=25&crumb={crumb}"
        )
        r    = _session.get(url, timeout=12)
        if not r.ok:
            return []
        news_items = r.json().get("news", [])
        result = []
        for item in news_items:
            title     = item.get("title", "").strip()
            link      = item.get("link",  "") or item.get("url", "")
            pub_time  = item.get("providerPublishTime")
            publisher = item.get("publisher", "Yahoo Finance")
            if not title:
                continue
            try:
                dt_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M") if pub_time else "N/A"
            except Exception:
                dt_str = "N/A"
            result.append({
                "title":     title,
                "summary":   "",
                "url":       link,
                "publisher": publisher,
                "published": dt_str,
            })
        return result
    except Exception as e:
        print(f"[sentiment] Search API failed: {e}")
        return []


def analyze_news_sentiment(sym: str) -> dict:
    # Try RSS first (best quality — has summaries)
    raw_news = _rss_news(sym)

    # Fallback to search API
    if not raw_news:
        raw_news = _search_api_news(sym)

    articles = []
    scores   = []

    for item in raw_news[:25]:
        title   = item.get("title",   "")
        summary = item.get("summary", "")
        if not title:
            continue

        text = f"{title}. {summary}" if summary else title
        vs   = _va.polarity_scores(text)
        c    = vs["compound"]
        scores.append(c)

        articles.append({
            "title":     title,
            "summary":   summary[:250] if summary else "",
            "url":       item.get("url",""),
            "publisher": item.get("publisher",""),
            "published": item.get("published",""),
            "compound":  round(c, 4),
            "positive":  round(vs["pos"], 4),
            "negative":  round(vs["neg"], 4),
            "neutral":   round(vs["neu"], 4),
            "label":     _label(c),
            "emoji":     _emoji(c),
        })

    avg = sum(scores) / len(scores) if scores else 0.0
    pos = sum(1 for s in scores if s >= 0.05)
    neg = sum(1 for s in scores if s <= -0.05)

    return {
        "ticker":         sym,
        "overall_score":  round(avg, 4),
        "overall_label":  _label(avg) if scores else "NO DATA",
        "overall_emoji":  _emoji(avg),
        "article_count":  len(articles),
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count":  len(scores) - pos - neg,
        "articles":       articles,
    }
