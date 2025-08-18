#
//  rss_summarizer.py
//  
//
//  Created by Angel Valchev on 18.08.25.
//


import argparse
import feedparser
import hashlib
import html
import json
import os
import sys
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from openai import OpenAI, APIConnectionError, APIError, RateLimitError
from pathlib import Path

# --- Secrets & config loading ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ Missing OPENAI_API_KEY environment variable. Set it in your shell.")
    sys.exit(1)

CONFIG_PATH = Path(__file__).with_name("config.json")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except FileNotFoundError:
    print(f"❌ Missing {CONFIG_PATH}. Create it with your feeds (see config.example.json).")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON in {CONFIG_PATH}: {e}")
    sys.exit(1)

RSS_FEEDS = cfg.get("rss_feeds", [])
if not RSS_FEEDS:
    print("❌ No 'rss_feeds' found in config.json.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# CONFIGURATION
# ----------------------------

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

RSS_FEEDS = config["rss_feeds"]
CATEGORIES = config["categories"]
EXCLUDED_TOPICS = config["excluded_topics"]

DEFAULT_HOURS = 24
OUTPUT_HTML = "rss_summary.html"

client = OpenAI(api_key=OPENAI_API_KEY)

# =======================
# Helpers
# =======================
def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    return BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)

def parse_date(entry):
    published = None
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        published = datetime(*entry.published_parsed[:6])
    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
        published = datetime(*entry.updated_parsed[:6])
    return published

def extract_summary(entry) -> str:
    summary = getattr(entry, "summary", None) or getattr(entry, "description", None)
    if not summary and hasattr(entry, "content"):
        content = entry.content
        if isinstance(content, list) and content:
            summary = content[0].get("value")
        elif isinstance(content, dict):
            summary = content.get("value")
    return clean_html(summary) if summary else ""

def extract_image(entry):
    # 1) media_content
    if hasattr(entry, "media_content"):
        for media in entry.media_content:
            url = media.get("url")
            if url and url.startswith("http"):
                return url
    # 2) media_thumbnail
    if hasattr(entry, "media_thumbnail"):
        for thumb in entry.media_thumbnail:
            url = thumb.get("url")
            if url and url.startswith("http"):
                return url
    # 3) first <img> in summary/description
    html_content = getattr(entry, "summary", None) or getattr(entry, "description", None) or ""
    soup = BeautifulSoup(html_content, "html.parser")
    img = soup.find("img")
    if img:
        src = img.get("src")
        if src and src.startswith("http"):
            return src
    return None

def article_matches_excluded(article: dict, excluded_topics) -> bool:
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    return any(keyword.lower() in text for keyword in excluded_topics)

# =======================
# Fetch & deduplicate
# =======================
def fetch_rss_articles(feeds, hours, excluded_topics):
    print("📡 Fetching RSS articles...")
    articles = []
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            published = parse_date(entry)
            if published and published < cutoff:
                continue  # keep undated items; skip only old-dated ones

            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = extract_summary(entry)
            image = extract_image(entry)

            article = {
                "title": title,
                "link": link,
                "summary": summary,
                "published": published.isoformat() if published else None,
                "image": image,
            }

            # Filter at collection stage
            if not article_matches_excluded(article, excluded_topics):
                articles.append(article)

    # Local dedup (by title + summary)
    seen = set()
    unique = []
    for a in articles:
        h = hashlib.sha256((a.get("title", "") + "||" + a.get("summary", "")).encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(a)

    print(f"✅ Fetched {len(unique)} articles after filtering & local deduplication")
    return unique

# =======================
# Summarize with OpenAI
# =======================
def summarize_articles_json(articles, categories, excluded_topics):
    print("🤖 Sending to ChatGPT for summarization...")

    if not articles:
        return {"categories": {}}

    system_prompt = (
        "You are an assistant that processes news RSS feeds.\n"
        "Tasks:\n"
        f"1) Categorize each article into exactly one of: {', '.join(categories)}. If none fits, use 'Uncategorized'.\n"
        f"2) Exclude articles related to these topics: {', '.join(excluded_topics)}.\n"
        "3) Deduplicate similar or repeated articles across feeds.\n"
        "4) For each article, provide a concise 1–2 sentence summary (context).\n"
        '5) Output valid JSON only with structure: {"categories": {"Category": [{"title","summary","link","image"}] } }.\n'
    )

    payload = {"articles": articles}

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},  # helps ensure parseable JSON
        )
    except APIConnectionError:
        print("❌ Connection error: Failed to reach OpenAI API. Please check your internet connection.")
        return {"categories": {}}
    except RateLimitError:
        print("❌ Rate limit reached: Too many requests to the OpenAI API. Try again later.")
        return {"categories": {}}
    except APIError as e:
        print(f"❌ API error: {e}")
        return {"categories": {}}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {"categories": {}}

    try:
        content = resp.choices[0].message.content.strip()
        return json.loads(content)
    except Exception:
        print("⚠️ Could not parse model response. Output may be malformed.")
        return {"categories": {}}

# =======================
# HTML writer
# =======================
def save_html(parsed_articles, output_file=OUTPUT_HTML):
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RSS News Summary</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; color:#333; }
h1 { background: #333; color: #fff; padding: 12px 16px; border-radius: 8px; }
h2 { color: #333; margin-top: 24px; background:#f0f0f0; padding:10px 12px; border-radius:8px; }
.article { background: #fff; border-radius: 8px; padding: 12px; margin: 12px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.08); }
.article a { text-decoration: none; color: #0066cc; font-weight: bold; }
.article a:hover { text-decoration: underline; }
.article p { margin: 6px 0 0; color: #444; line-height: 1.4; }
.article img { max-width: 100%; height: auto; margin: 6px 0; border-radius: 6px; display:block; }
.meta { font-size: 0.85em; color: #666; margin-bottom: 6px; }
</style>
</head>
<body>
<h1>📰 RSS News Summary</h1>
"""

    for category, items in parsed_articles.get("categories", {}).items():
        if not items:
            continue
        html_content += f"<h2>{html.escape(category)}</h2>\n"
        for item in items:
            title = html.escape(item.get("title", ""))
            link = html.escape(item.get("link", ""))
            summary = html.escape(item.get("summary", ""))
            img = item.get("image")

            html_content += '<div class="article">\n'
            html_content += f'  <a href="{link}" target="_blank">{title}</a>\n'
            if img:
                html_content += f'  <a href="{link}" target="_blank"><img src="{html.escape(img)}" alt="image"></a>\n'
            if summary:
                html_content += f'  <p>{summary}</p>\n'
            html_content += '</div>\n'

    html_content += "</body></html>"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML saved to {output_file}")

# =======================
# Main
# =======================
def main():
    parser = argparse.ArgumentParser(description="RSS Summarizer")
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS, help="Hours back to fetch (default=24)")
    args = parser.parse_args()

    articles = fetch_rss_articles(RSS_FEEDS, args.hours, EXCLUDED_TOPICS)
    if not articles:
        print("⚠️ No articles found.")
        return

    parsed_articles = summarize_articles_json(articles, CATEGORIES, EXCLUDED_TOPICS)
    if not parsed_articles.get("categories"):
        print("⚠️ No summaries generated.")
        return

    save_html(parsed_articles, OUTPUT_HTML)
    print("🎉 Done!")

if __name__ == "__main__":
    main()
