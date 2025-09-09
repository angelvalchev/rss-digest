import argparse
import feedparser
import hashlib
import html
import json
import os
import sys
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from openai import OpenAI, APIConnectionError, APIError, RateLimitError
from pathlib import Path
from urllib.parse import urlparse
from difflib import SequenceMatcher

# How many clusters to send to GPT per request (tune freely)
MAX_CLUSTERS_PER_BATCH = 50

# Replace your existing SYSTEM_FIX_PROMPT with this stricter version:
SYSTEM_FIX_PROMPT = """You are a JSON repair agent for a news digest.  
You MUST return a single JSON object and NOTHING ELSE.

REQUIRED OUTPUT SCHEMA (exactly):
{
  "categories": {
    "<CategoryName>": [
      {
        "title": "string (concise headline, ≤100 chars, must look like a real news headline)",
        "summary": "string (1–2 sentence factual digest, not generic filler, preserve substance of the story)",
        "full_story": "string",
        "image": "string (URL) or null",
        "primary_link": "string (URL)",
        "link": "string (duplicate of primary_link)",
        "sources": [ { "title": "string", "link": "string" } ],
        "inputClusterIds": [ "string", ... ]
      }
    ]
  },
  "metadata": {                     
    "represented_cluster_ids": [ "string", ... ],
    "omitted_cluster_ids": [ "string", ... ]   
  }
}

INPUT:
- raw_output: the previous model output (may be invalid or incomplete).
- missing_cluster_ids: array of cluster IDs that were not represented.
RULES (strict):
1. Preserve any valid stories and fields already present in raw_output.
2. For each id in missing_cluster_ids:
   a) FIRST try to MERGE that cluster into an existing story's inputClusterIds if they clearly describe the same event.
   b) ONLY if you can produce a **meaningful** merged or standalone story (title + summary + at least one real source link), add it.
   c) IF YOU CANNOT produce a meaningful story for a cluster, DO NOT create a placeholder or dummy story. Instead list that cluster id under metadata.omitted_cluster_ids.
3. Do NOT invent new cluster IDs or fabricate source links. Use exactly the provided IDs and provided source links. 
   - 🚫 Never use fake domains like "example.com" or "placeholder.com".
   - 🚫 Do not insert generic dummy titles like "New Developments in Cluster X".
4. Every story you include MUST have:
   - A concise, factual `title` (≤100 chars, must look like a real news headline, no cluster IDs, no "Placeholder..." text, no academic phrasing like "New Study Reveals...").
   - A non-empty `summary` (1–2 sentences, factual digest, preserve the original meaning, not vague filler like "details are emerging").
   - `sources`: must include at least one object with a non-empty `link` (must be a provided source link, never fabricated) and (preferably) a non-empty `title`.
   - `primary_link` must be exactly one of the `sources.link` values.
   - `inputClusterIds` must list the cluster IDs covered by that story (strings).
5. Do not create duplicates: each cluster_id must appear in exactly one story.
6. Preserve original facts — do not hallucinate critical details; you may summarize or omit uncertain details.
7. Categories:
   - Prefer the most fitting category from the allowed list provided by the caller 
     (e.g. Technology, World, Politics, Economy, Security, Bulgaria, EU, Culture, Health, Science).
   - Do NOT dump everything into "Uncategorized".
   - Use "Uncategorized" only if there is absolutely no reasonable category match.

8. Return ONLY valid JSON matching the schema above. 
   - No extra text
   - No explanation
   - No surrounding code blocks

EXIT/HELPFUL HINTS (for your program):
- metadata.represented_cluster_ids must list all cluster_ids present in the categories.
- metadata.omitted_cluster_ids must list all missing_cluster_ids not represented.
"""

# schema.py
BATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["categories"],
    "properties": {
        "categories": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "inputClusterIds"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1, "maxLength": 200},
                        "summary": {"type": "string"},
                        "inputClusterIds": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": ["integer", "string"]}  # allow int or str IDs
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}

# schema.py
from jsonschema import validate, ValidationError

def validate_json_schema(obj):
    try:
        validate(instance=obj, schema=BATCH_SCHEMA)
        return True, None
    except ValidationError as e:
        return False, f"{e.message} @ {'/'.join(map(str, e.path))}"

# --- Pre-clustering helpers (Quick Wins 1 & 2) ---
import re
import unicodedata
from difflib import SequenceMatcher

# minimal publisher hints to strip " | Reuters", " - BBC", etc.
_PUBLISHER_HINTS = {
    "reuters","bbc","ap","associated press","bloomberg","cnn","wsj","the verge",
    "techcrunch","nytimes","the new york times","guardian","financial times",
    "darik news","mediapool","bta","bloomberg line","forbes","axios","politico",
    "abc news","nbc news","cbs news","fox news","sky news","al jazeera","euronews"
}

# small stopword lists (EN + BG); keep numbers because they matter for events
_STOP_EN = {
    "the","a","an","to","of","in","on","for","and","or","but","is","are","was","were",
    "with","by","as","at","from","that","this","it","its","be","has","have","had",
    "says","say","said","will","would","over","after","into","about"
}
_STOP_BG = {
    "и","в","на","за","с","от","по","като","че","ще","са","е","беше","били","при",
    "след","над","под","между","или","но","срещу","без","до","към","защо","как"
}

def _squash_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _looks_like_publisher(t: str) -> bool:
    t_lc = t.lower().strip()
    if t_lc in _PUBLISHER_HINTS:
        return True
    if "news" in t_lc or ".com" in t_lc or ".bg" in t_lc or ".co" in t_lc or ".org" in t_lc:
        return True
    # very short tail that’s probably a brand
    if len(t_lc.split()) <= 3 and any(w in _PUBLISHER_HINTS for w in t_lc.split()):
        return True
    return False

def normalize_title(raw: str) -> str:
    """Lowercase, NFKC normalize, strip publisher tails, remove punctuation,
    drop simple stopwords, keep digits and Cyrillic letters."""
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw)
    s = _squash_spaces(s)

    # try removing final publisher tail after separators: " | ", " - ", " — ", " – "
    # split from the RIGHT once; drop tail if it looks like a publisher
    for sep in (" | ", " — ", " – ", " - "):
        if sep in s:
            left, right = s.rsplit(sep, 1)
            if _looks_like_publisher(right):
                s = left
                break

    s = _squash_spaces(s)

    # remove bracketed tags like [video], (opinion) if any
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"\([^)]+\)", " ", s)
    s = _squash_spaces(s)

    # lowercase
    s = s.lower()

    # keep unicode letters/digits/underscore/space; drop punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    s = _squash_spaces(s)

    # tokenize & drop stopwords
    tokens = [t for t in s.split() if t not in _STOP_EN and t not in _STOP_BG]
    return " ".join(tokens)

def _similarity(a: str, b: str) -> tuple[float, float]:
    """Return (sequence_ratio, jaccard) on normalized titles."""
    an = normalize_title(a)
    bn = normalize_title(b)
    if not an or not bn:
        return 0.0, 0.0
    seq = SequenceMatcher(None, an, bn).ratio()
    aset = set(an.split())
    bset = set(bn.split())
    jacc = (len(aset & bset) / len(aset | bset)) if aset and bset else 0.0
    return seq, jacc

def precluster_articles(
    articles: list,
    seq_weight: float = 0.4,
    jacc_weight: float = 0.6,
    threshold: float = 0.45
) -> list:
    """
    Greedy clustering by title similarity with hybrid scoring.
    - Normalizes titles (lowercase, stopwords removed, punctuation stripped).
    - Computes hybrid similarity: seq * weight + jacc * weight.
    - Clusters if similarity >= threshold.
    """

    clusters: list[dict] = []
    reps_norm: list[str] = []   # normalized representative title per cluster

    def normalize(text: str) -> str:
        if not text:
            return ""
        s = unicodedata.normalize("NFKC", text)
        s = re.sub(r"\s+", " ", s).strip().lower()
        s = re.sub(r"[^\w\s]", " ", s)  # remove punctuation
        tokens = [t for t in s.split() if t not in _STOP_EN and t not in _STOP_BG]
        return " ".join(tokens)

    def hybrid_similarity(a: str, b: str) -> float:
        a_norm, b_norm = normalize(a), normalize(b)
        if not a_norm or not b_norm:
            return 0.0
        seq = SequenceMatcher(None, a_norm, b_norm).ratio()
        aset, bset = set(a_norm.split()), set(b_norm.split())
        jacc = len(aset & bset) / len(aset | bset) if aset and bset else 0.0
        return seq_weight * seq + jacc_weight * jacc

    for art in articles:
        title = art.get("title", "") or ""
        best_i, best_score = -1, 0.0

        for i, rep in enumerate(reps_norm):
            score = hybrid_similarity(title, rep)
            if score > best_score:
                best_score, best_i = score, i

        if best_score >= threshold:
            clusters[best_i]["articles"].append(art)
        else:
            clusters.append({"cluster_name": "cluster", "articles": [art]})
            reps_norm.append(title)

    return clusters

def chunk_list(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def dedupe_sources_list(sources):
    seen = set()
    out = []
    for s in sources or []:
        url = (s.get("link") or "").strip()
        if url and url not in seen:
            seen.add(url)
            out.append({"title": s.get("title",""), "link": url})
    return out

def dedup_by_cluster_ids(merged: dict) -> dict:
    """
    Remove duplicates created by passthrough + GPT (or multi-batch overlaps).
    Prefers uniqueness by the set of inputClusterIds; falls back to (title, primary_link/link).
    """
    out = {"categories": {}}
    seen_keys = set()

    for cat, stories in merged.get("categories", {}).items():
        uniq = []
        for s in stories:
            ids = s.get("inputClusterIds") or []
            key = ("ids", tuple(sorted(str(x) for x in ids))) if ids else None
            if key is None:
                # fallback: title + best link
                best_link = s.get("primary_link") or s.get("link") or ""
                key = ("titlelink", s.get("title","").strip().lower(), best_link.strip().lower())

            if key not in seen_keys:
                seen_keys.add(key)
                uniq.append(s)
        out["categories"][cat] = uniq
    return out

def extract_batch_cluster_ids(batch):
    """
    Extract cluster_id values from a batch of clusters.
    Each cluster is expected to have a 'cluster_id' field.
    Returns a list of IDs (ints or strs).
    """
    ids = []
    for c in batch:
        cid = c.get("cluster_id")
        if cid is not None:
            ids.append(cid)
    return ids

def validate_batch_output(parsed_or_raw, required_ids):
    """
    Validate a GPT batch output against required cluster IDs.
    Accepts either a parsed dict or a raw JSON string.
    Returns a dict:
      {
        "ok": bool,
        "errors": list[str],
        "missing_ids": set,
        "covered_ids": set,
        "extra_ids": set
      }
    """
    report = {
        "ok": True,
        "errors": [],
        "missing_ids": set(),
        "covered_ids": set(),
        "extra_ids": set()
    }

    # Normalize into dict
    parsed = None
    if isinstance(parsed_or_raw, dict):
        parsed = parsed_or_raw
    elif isinstance(parsed_or_raw, str):
        try:
            parsed = json.loads(parsed_or_raw)
        except Exception as e:
            report["ok"] = False
            report["errors"].append(f"invalid JSON: {e}")
            report["missing_ids"] = set(required_ids)
            return report
    else:
        report["ok"] = False
        report["errors"].append(f"unexpected input type: {type(parsed_or_raw)}")
        report["missing_ids"] = set(required_ids)
        return report

    # Verify structure
    if not parsed or not isinstance(parsed, dict):
        report["ok"] = False
        report["errors"].append("parsed output missing or not dict")
        report["missing_ids"] = set(required_ids)
        return report

    # Collect all cluster IDs actually present in GPT stories
    covered = set()
    try:
        categories = parsed.get("categories", {})
        if isinstance(categories, dict):
            for stories in categories.values():
                if not isinstance(stories, list):
                    continue
                for story in stories:
                    if isinstance(story, dict):
                        for cid in story.get("inputClusterIds", []):
                            covered.add(cid)
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"exception while scanning stories: {e}")
        return report

    required = set(required_ids)
    missing = required - covered
    extra = covered - required

    if missing:
        report["ok"] = False
        report["errors"].append(f"missing {len(missing)} cluster_ids")

    report["missing_ids"] = missing
    report["covered_ids"] = covered
    report["extra_ids"] = extra

    return report

def merge_repair_results(original: dict, repaired: dict) -> dict:
    """
    Merge repaired batch output (dict with "categories") into the original.
    Ensures unique cluster_ids across all categories.
    """
    if not original or not isinstance(original, dict):
        original = {"categories": {}}
    if not repaired or not isinstance(repaired, dict):
        return original

    categories = original.setdefault("categories", {})

    # Collect all existing cluster_ids across all categories
    seen = set()
    for stories in categories.values():
        for story in stories:
            for cid in story.get("inputClusterIds", []):
                seen.add(cid)

    # Merge repaired categories
    for cat, stories in repaired.get("categories", {}).items():
        if not isinstance(stories, list):
            continue
        bucket = categories.setdefault(cat, [])
        for story in stories:
            cluster_ids = story.get("inputClusterIds", [])
            if not cluster_ids:
                continue
            if any(cid in seen for cid in cluster_ids):
                continue
            bucket.append(story)
            seen.update(cluster_ids)

    return original

def passthrough_batch(clusters):
    """
    Build a minimal valid batch output directly from raw clusters.
    Used for coverage repair when GPT missed some cluster_ids.
    """
    out = {"categories": {}}

    for cluster in clusters:
        # Just take the first article as representative
        art = cluster["articles"][0] if cluster["articles"] else {}
        cluster_id = cluster.get("cluster_id")

        story = {
            "title": art.get("title", "") or f"Untitled {cluster_id}",
            "summary": art.get("summary", "") or "",
            "full_story": art.get("summary", "") or "",
            "image": art.get("image", None),
            "primary_link": art.get("link", ""),
            "link": art.get("link", ""),
            "sources": [{"title": a.get("title", ""), "link": a.get("link", "")}
                        for a in cluster.get("articles", [])],
            "inputClusterIds": [cluster_id] if cluster_id is not None else []
        }

        out["categories"].setdefault("Uncategorized", []).append(story)

    return out

# --- New helpers: remove placeholder/invalid stories & merge near-duplicates ---

PLACEHOLDER_TITLE_RE = re.compile(r"\bplaceholder\b|\bcluster\s*\d+\b", flags=re.I)

def is_story_valid(story: dict) -> bool:
    """Return True if story looks real enough to keep."""
    if not isinstance(story, dict):
        return False
    title = (story.get("title") or "").strip()
    summary = (story.get("summary") or "").strip()
    input_ids = story.get("inputClusterIds") or []
    sources = story.get("sources") or []

    # title and summary required and not placeholder-like
    if not title or PLACEHOLDER_TITLE_RE.search(title):
        return False
    if not summary or "placeholder" in summary.lower():
        return False

    # must have at least one valid source link
    valid_sources = [s for s in sources if isinstance(s, dict) and (s.get("link") or "").strip()]
    if not valid_sources:
        return False

    # must have at least one cluster id
    if not input_ids:
        return False

    return True

def filter_out_invalid_and_placeholders(parsed: dict):
    """
    Remove placeholder-like or invalid stories from parsed.
    Returns (filtered_parsed, removed_cluster_ids_set)
    """
    removed = set()
    out = {"categories": {}}
    for cat, stories in parsed.get("categories", {}).items():
        kept = []
        for s in stories:
            if is_story_valid(s):
                # normalize inputClusterIds to strings
                s["inputClusterIds"] = [str(x) for x in (s.get("inputClusterIds") or [])]
                # dedupe sources and keep only non-empty links
                s["sources"] = dedupe_sources_list(s.get("sources") or [])
                kept.append(s)
            else:
                for cid in (s.get("inputClusterIds") or []):
                    removed.add(str(cid))
        if kept:
            out["categories"][cat] = kept
    return out, removed

def merge_near_duplicate_stories(parsed: dict, similarity_threshold: float = 0.86):
    """
    Coalesce near-duplicate stories across categories.
    Keeps the first canonical story and merges sources + inputClusterIds from duplicates.
    Uses existing normalize_title() + SequenceMatcher for similarity.
    """
    flat = []
    for cat, stories in parsed.get("categories", {}).items():
        for s in stories:
            flat.append((cat, s))

    canon = []  # list of (cat, story)
    for cat, s in flat:
        t = s.get("title","")
        matched = False
        for i, (ccat, cs) in enumerate(canon):
            # compare normalized titles
            seq = SequenceMatcher(None, normalize_title(t), normalize_title(cs.get("title",""))).ratio()
            if seq >= similarity_threshold:
                # merge into cs (prefer longer summary, more sources)
                cs["sources"] = dedupe_sources_list(cs.get("sources", []) + s.get("sources", []))
                cs["inputClusterIds"] = sorted(set(map(str, cs.get("inputClusterIds", [])) + list(map(str, s.get("inputClusterIds", [])))))
                # prefer the story with longer summary as canonical title/summary/full_story
                if len((s.get("summary") or "")) > len((cs.get("summary") or "")):
                    cs["summary"] = s.get("summary")
                    cs["full_story"] = s.get("full_story", cs.get("full_story",""))
                matched = True
                break
        if not matched:
            # normalize lists to strings
            s["inputClusterIds"] = [str(x) for x in (s.get("inputClusterIds") or [])]
            s["sources"] = dedupe_sources_list(s.get("sources") or [])
            canon.append((cat, s))

    # rebuild categories preserving canonical categories
    merged = {"categories": {}}
    for cat, s in canon:
        merged["categories"].setdefault(cat, []).append(s)
    return merged

def clean_and_validate_json(candidate: str):
    """
    Try to clean GPT's candidate JSON string and return a valid dict.
    - Strips markdown fences (```json … ```).
    - Removes HTML tags from text fields.
    - Fixes trailing commas.
    - Deduplicates inputClusterIds.
    - Ensures required fields exist.
    - Ensures primary_link is consistent with sources.
    - Logs any auto-heals applied.
    Returns (parsed_dict, error_message_or_None).
    """

    import logging
    logging.basicConfig(level=logging.INFO, format="🧹 [Cleaner] %(message)s")

    # --- Helper cleanup functions ---
    def strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text or "").strip()

    def fix_trailing_commas(text: str) -> str:
        # Remove ",}" and ",]" which break JSON
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text

    def strip_markdown_fences(text: str) -> str:
        # Remove markdown fences like ```json ... ```
        return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE)
    # --- Step 1: Normalize raw candidate ---
    candidate_fixed = candidate or ""
    candidate_fixed = strip_markdown_fences(candidate_fixed)
    candidate_fixed = fix_trailing_commas(candidate_fixed)
    candidate_fixed = candidate_fixed.strip("\ufeff")  # strip BOM if present

    # --- Step 2: Try parsing ---
    try:
        parsed = json.loads(candidate_fixed)
    except Exception as e:
        return None, f"JSON parse failed after cleanup: {e}"

    # --- Step 3: Validate schema & clean ---
    categories = parsed.get("categories", {})
    if not isinstance(categories, dict):
        return None, "Invalid categories format"

    fixes_applied = {
        "dedup_cluster_ids": 0,
        "added_sources": 0,
        "added_primary_to_sources": 0,
    }

    for cat_name, stories in categories.items():
        if not isinstance(stories, list):
            continue
        for story in stories:
            if not isinstance(story, dict):
                continue

            # Ensure required fields
            story.setdefault("title", "")
            story.setdefault("summary", "")
            story.setdefault("full_story", "")
            story.setdefault("image", None)
            story.setdefault("primary_link", "")
            story.setdefault("link", story.get("primary_link", ""))
            story.setdefault("sources", [])
            story.setdefault("inputClusterIds", [])

            # Clean text fields
            story["title"] = strip_html(story["title"])
            story["summary"] = strip_html(story["summary"])
            story["full_story"] = strip_html(story["full_story"])

            # Deduplicate cluster IDs
            cluster_ids = [str(cid) for cid in story.get("inputClusterIds", [])]
            if len(cluster_ids) != len(set(cluster_ids)):
                fixes_applied["dedup_cluster_ids"] += 1
            story["inputClusterIds"] = sorted(set(cluster_ids))

            # Auto-heal sources / primary_link consistency
            if story["primary_link"]:
                story["link"] = story["primary_link"]

                if not story["sources"]:
                    fixes_applied["added_sources"] += 1
                    story["sources"] = [{
                        "title": story["title"] or "Source",
                        "link": story["primary_link"]
                    }]

                links = [s.get("link") for s in story["sources"] if isinstance(s, dict)]
                if story["primary_link"] not in links:
                    fixes_applied["added_primary_to_sources"] += 1
                    story["sources"].append({
                        "title": story["title"] or "Source",
                        "link": story["primary_link"]
                    })

    # Log a summary of applied fixes (only if something was actually fixed)
    if any(fixes_applied.values()):
        logging.warning(
            f"Cleanup applied: "
            f"{fixes_applied['dedup_cluster_ids']} dedup cluster_id fixes, "
            f"{fixes_applied['added_sources']} missing sources added, "
            f"{fixes_applied['added_primary_to_sources']} primary_link source insertions."
        )

    return parsed, None

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
OUTPUT_HTML = "digest.html"

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

from urllib.parse import urlparse

def get_domain(url: str) -> str:
    """Extract the domain name from a URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return "source"

def article_matches_excluded(article: dict, excluded_topics) -> bool:
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    return any(keyword.lower() in text for keyword in excluded_topics)

def insert_placeholder_story(parsed: dict, cluster_id: str) -> dict:
    """
    Ensure parsed JSON has a placeholder story for the given cluster_id.
    Creates 'Uncategorized' category if needed.
    """
    if not parsed or not isinstance(parsed, dict):
        parsed = {"categories": {}}

    categories = parsed.setdefault("categories", {})
    bucket = categories.setdefault("Uncategorized", [])

    # Minimal valid schema
    placeholder = {
        "title": f"Cluster {cluster_id} (placeholder)",
        "summary": "No summary available due to repair failure.",
        "full_story": "This is a placeholder story because the batch repair could not recover details.",
        "image": None,
        "primary_link": "",
        "link": "",
        "sources": [],
        "inputClusterIds": [cluster_id],
    }

    bucket.append(placeholder)
    return parsed

def enforce_global_coverage(parsed_articles, clusters):
    """
    Ensure all input clusters are represented at least once in parsed_articles.
    Adds placeholder stories for any globally missing cluster IDs.
    """
    all_clusters = {str(c["cluster_id"]) for c in clusters}
    covered = set()

    for stories in parsed_articles.get("categories", {}).values():
        for story in stories:
            for cid in story.get("inputClusterIds", []):
                covered.add(str(cid))

    missing_globally = all_clusters - covered
    if missing_globally:
        print(f"⚠️ Final coverage fallback for {len(missing_globally)} clusters")
        for mid in missing_globally:
            parsed_articles = insert_placeholder_story(parsed_articles, mid)

    total = len(all_clusters)
    covered_count = total - len(missing_globally)
    print(f"✅ Final cluster coverage: {covered_count}/{total} = {(covered_count/total)*100:.1f}%")

    return parsed_articles

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

MAX_REPAIR_ATTEMPTS = 2

# =======================
# GPT Batch Helpers
# =======================

def build_system_prompt(batch_index: int, total_batches: int, categories: list[str], excluded_topics: list[str]) -> str:
    """
    Strong constraints: valid JSON only, full coverage of provided clusters,
    avoid 'Uncategorized' unless truly necessary, and emit the agreed schema.
    """
    cats = ", ".join(categories) if categories else "Uncategorized"
    banned = ", ".join(excluded_topics[:40]) if excluded_topics else ""

    return f"""
You are a news digest assistant. You MUST output **valid JSON only**.

This is batch {batch_index+1} of {total_batches}. Input = list of clusters, where each cluster contains articles.

GOALS:
1. Merge articles inside each cluster into one story object.
2. If two clusters are about the same event, merge them into one story and include all their cluster IDs in "inputClusterIds".
3. **Every cluster ID provided in the input must appear in at least one story’s inputClusterIds.**

RESTRICTIONS:
- Output JSON ONLY (no prose, no explanations).
- Use this exact schema:
  {{
    "categories": {{
      "CategoryName": [
        {{
          "title": str,
          "summary": str,
          "full_story": str,
          "image": str|null,
          "primary_link": str,
          "link": str,
          "sources": [{{ "title": str, "link": str }}],
          "inputClusterIds": [ str, ... ]
        }}
      ]
    }}
  }}
- Categories you may use: {cats}
- **You MUST assign each story to the best-fitting category. DO NOT use 'Uncategorized' unless absolutely no category fits.**

CATEGORY GUIDELINES:
- Politics → elections, government, diplomacy, policies, parties.
- World → international news not specific to Bulgaria.
- Bulgaria → news centered on Bulgarian events, politics, society, economy.
- Economy → business, markets, finance, companies, trade.
- Technology → AI, software, hardware, gadgets, startups, cybersecurity.
- Security → defense, crime, policing, military, cybersecurity threats.
- Health → medicine, public health, pandemics, hospitals.
- Science → research, discoveries, space, climate, environment.
- EU → EU policies, institutions, regulations.

OTHER RULES:
- Avoid or downplay topics with banned keywords: {banned}
- Titles concise; summaries factual; full_story longer narrative.
- sources = unique (title, link) pairs from merged articles.
- primary_link = most authoritative source if possible.
- image may be null if no suitable image.
- inputClusterIds must enumerate exactly the clusters covered.

Return JSON only. No text before/after.
"""

def call_gpt_batch(batch_index: int, batch, system_prompt: str, client):
    """
    Send one batch to GPT and try parsing the response as JSON.
    Returns (raw_text, parsed_json_or_None).
    """
    try:
        # Prepare user input (the articles)
        user_content = {"clusters": batch}

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
            ]
        )

        # ✅ New SDK: use .content, not ["content"]
        raw_text = resp.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = None

        # Save raw output for debugging
        with open(f"raw_batches/batch_{batch_index}.json", "w", encoding="utf-8") as f:
            f.write(raw_text)

        return raw_text, parsed

    except Exception as e:
        print(f"❌ GPT call failed for batch {batch_index}: {e}")
        return "", None

def repair_batch_output(batch_index, raw_text, missing_ids, system_fix_prompt, client, max_retries=2):
    """
    Try to repair or rebuild a batch output that failed JSON parsing
    or was missing cluster coverage.
    Returns (fixed_raw_text, parsed_json_or_None).
    """
    parsed = None
    fixed_text = raw_text or ""

    # Normalize missing_ids to strings (avoid int/str mismatches)
    missing_ids = [str(x) for x in (missing_ids or [])]
    if not missing_ids:
        return fixed_text, None

    for attempt in range(max_retries):
        try:
            # Construct a dynamic prompt
            prompt = system_fix_prompt + f"""

The following cluster IDs are missing and MUST be addressed: {missing_ids}

Rules reminder:
- Strict JSON only, no commentary.
- Schema must match exactly (categories → stories → fields).
- For each cluster:
  • If you can produce a meaningful story, include it with inputClusterIds.
  • If you cannot produce a valid story, DO NOT insert a placeholder —
    instead, list that ID under metadata.omitted_cluster_ids.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
            )
            candidate = response.choices[0].message.content.strip()

            # ✅ Save raw candidate for inspection
            debug_path = f"raw_batches/repair_batch_{batch_index}_attempt{attempt+1}.json"
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(candidate)
                print(f"📝 Saved repair attempt {attempt+1} output to {debug_path}")
            except Exception as e:
                print(f"⚠️ Could not save debug file for attempt {attempt+1}: {e}")
            # 🧹 Always try cleaner first
            cleaned, err = clean_and_validate_json(candidate)
            if not cleaned:
                print(f"⚠️ Cleaner failed: {err}")
                continue  # retry next attempt

            parsed = cleaned

            # ✅ Collect covered + omitted IDs
            categories = parsed.get("categories", {})
            metadata = parsed.get("metadata", {})

            covered_ids = set()
            for stories in categories.values():
                if not isinstance(stories, list):
                    continue
                for story in stories:
                    if not isinstance(story, dict):
                        continue
                    for cid in story.get("inputClusterIds", []):
                        covered_ids.add(str(cid))

            omitted_ids = set(str(x) for x in metadata.get("omitted_cluster_ids", []))

            still_missing = set(missing_ids) - covered_ids - omitted_ids
            if not still_missing:
                # ✅ Success
                return json.dumps(parsed, ensure_ascii=False, indent=2), parsed
            else:
                # Retry with reduced missing list
                missing_ids = list(still_missing)
                fixed_text = json.dumps(parsed, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ Repair attempt {attempt+1} failed: {e}")

    # If we end loop still missing, return last cleaned attempt (or raw_text)
    return fixed_text, parsed

def extract_batch_cluster_ids(batch):
    """
    Get the set of cluster_ids from one batch.
    """
    return {str(c.get("cluster_id")) for c in batch if c.get("cluster_id") is not None}

def passthrough_batch(missing_clusters):
    """
    Fallback: directly map missing clusters into minimal JSON stories.
    """
    out = {"categories": {"Uncategorized": []}}
    for c in missing_clusters:
        arts = c.get("articles", [])
        if not arts:
            continue
        a = arts[0]
        out["categories"]["Uncategorized"].append({
            "title": a.get("title", "Untitled"),
            "summary": a.get("summary", ""),
            "full_story": a.get("summary", ""),
            "image": a.get("image"),
            "primary_link": a.get("link"),
            "link": a.get("link"),
            "sources": [{"title": art.get("title", ""), "link": art.get("link", "")} for art in arts],
            "inputClusterIds": [str(c.get("cluster_id"))]
        })
    return out


def safe_process_batch(batch_index, batch, sys_prompt, client):
    required_ids = extract_batch_cluster_ids(batch)

    # 1. Ask GPT
    raw_text, parsed = call_gpt_batch(batch_index, batch, sys_prompt, client)

    # 2. First validation
    report = validate_batch_output(parsed if parsed else raw_text, required_ids)
    missing = report["missing_ids"]

    # 3. Try repair if missing
    attempts = 0
    while missing and attempts < MAX_REPAIR_ATTEMPTS:
        attempts += 1
        print(f"🔁 Repair attempt {attempts} for batch {batch_index}, missing {len(missing)}")

        fixed_text, recovered = repair_batch_output(
            batch_index,
            raw_text,
            missing,
            SYSTEM_FIX_PROMPT,
            client,
            max_retries=1
        )

        if recovered:
            parsed = merge_repair_results(parsed or {}, recovered)
            raw_text = fixed_text  # 🔑 update raw_text for next loop

        # Re-validate after merge
        report = validate_batch_output(parsed, required_ids)
        missing = report["missing_ids"]

    # 4. Still missing? → do NOT insert placeholders; record omitted IDs in metadata
    if missing:
        print(f"⚠️ Batch {batch_index}: still missing {len(missing)} → marking omitted (no placeholders will be created)")
        parsed = parsed or {"categories": {}}
        meta = parsed.setdefault("metadata", {})
        prev_omitted = set(meta.get("omitted_cluster_ids", []))
        meta["omitted_cluster_ids"] = sorted(prev_omitted.union({str(x) for x in missing}))

    return parsed

# =======================
# Summarize with OpenAI
# =======================
def summarize_articles_json(articles, categories, excluded_topics):
    print("🤖 Sending to ChatGPT for summarization & merging...")

    if not articles:
        return {"categories": {}}

    # --- Step 1: Pre-cluster articles ---
    clusters = precluster_articles(articles, threshold=0.40)
    for i, c in enumerate(clusters):
        c["cluster_id"] = i

    # --- Step 2: Batching ---
    batches = list(chunk_list(clusters, MAX_CLUSTERS_PER_BATCH))
    print(f"🧯 Batching: {len(clusters)} clusters → {len(batches)} batches of ≤ {MAX_CLUSTERS_PER_BATCH}")
    os.makedirs("raw_batches", exist_ok=True)

    # --- Step 3 & 4: Call GPT, validate, repair if needed ---
    validated_batches = []
    for bi, batch in enumerate(batches):
        print(f"🤖 Sending batch {bi+1}/{len(batches)} to GPT… (size={len(batch)})")

        sys_prompt = build_system_prompt(bi, len(batches), categories, excluded_topics)
        raw_text, parsed = call_gpt_batch(bi, batch, sys_prompt, client)

        required_ids = extract_batch_cluster_ids(batch)

        # If parse failed → repair
        if parsed is None:
            print(f"⚠️ Batch {bi}: no JSON parsed, attempting repair")
            _, parsed = repair_batch_output(bi, raw_text, required_ids, SYSTEM_FIX_PROMPT, client)

        # Validate coverage
        report = validate_batch_output(parsed, required_ids) if parsed else {
            "ok": False,
            "errors": ["no parsed JSON"],
            "missing_ids": required_ids,
            "covered_ids": set(),
            "extra_ids": set()
        }

        if not report["ok"]:
            print(f"🛠 Batch {bi}: invalid output → repair (missing {len(report['missing_ids'])})")
            _, parsed2 = repair_batch_output(bi, raw_text, report["missing_ids"], SYSTEM_FIX_PROMPT, client)
            report2 = validate_batch_output(parsed2, required_ids) if parsed2 else {"ok": False, "errors": ["repair failed"], "missing_ids": required_ids, "covered_ids": set(), "extra_ids": set()}
            if report2["ok"]:
                parsed = parsed2
                report = report2
            else:
                print(f"❌ Batch {bi}: still invalid after repair")

        cov = len(report["covered_ids"])
        need = len(required_ids)
        print(f"📋 Batch {bi}: coverage {cov}/{need} ({(cov/need*100 if need else 100):.1f}%), missing {len(report['missing_ids'])}")

        validated_batches.append((bi, parsed if isinstance(parsed, dict) else {"categories": {}}))

    # --- Step 5a: Merge validated batches ---
    merged = {"categories": {}}
    represented = set()

    for batch_id, parsed in validated_batches:
        cats = parsed.get("categories", {})
        for cat, stories in cats.items():
            if not isinstance(stories, list):
                continue
            for s in stories:
                s["sources"] = dedupe_sources_list(s.get("sources"))
            merged["categories"].setdefault(cat, []).extend(stories)

        meta = parsed.get("metadata") or {}
        for cid in meta.get("represented_cluster_ids", []):
            represented.add(cid)

    # --- Step 5b: Deduplicate across batches ---
    def dedup_stories(merged):
        deduped = {"categories": {}}
        seen_titles = set()
        for cat, stories in merged.get("categories", {}).items():
            out_stories = []
            for story in stories:
                title_key = story["title"].strip().lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                out_stories.append(story)
            deduped["categories"][cat] = out_stories
        return deduped

    # Keep your original title-based dedup first
    merged = dedup_stories(merged)

    # Then dedupe again by inputClusterIds to eliminate passthrough+GPT duplicates
    merged = dedup_by_cluster_ids(merged)

    # --- New: filter out placeholder/invalid stories (remove junk that came from repair) ---
    merged, removed_ids = filter_out_invalid_and_placeholders(merged)
    if removed_ids:
        print(f"🧹 Removed {len(removed_ids)} invalid/placeholder stories (cluster ids: {sorted(list(removed_ids))[:8]}{'...' if len(removed_ids)>8 else ''})")

    # --- New: merge near-duplicate stories across categories to reduce repetition ---
    merged = merge_near_duplicate_stories(merged, similarity_threshold=0.86)

    # --- Step 5c: Coverage repair (only for still-missing clusters) ---
    all_ids = {c.get("cluster_id") for c in clusters}

    # Recompute represented strictly from stories we currently have
    represented = set()
    for stories in merged.get("categories", {}).values():
        for s in stories:
            for cid in s.get("inputClusterIds", []):
                represented.add(cid)

    missing_ids = sorted(x for x in all_ids if x not in represented)
    if missing_ids:
        print(f"➕ Coverage repair: adding passthrough for {len(missing_ids)} missing clusters")
        missing = [c for c in clusters if c.get("cluster_id") in missing_ids]
        repair = passthrough_batch(missing)
        for cat, stories in repair["categories"].items():
            merged["categories"].setdefault(cat, []).extend(stories)

        # Recompute represented again after repair
        represented = set()
        for stories in merged.get("categories", {}).values():
            for s in stories:
                for cid in s.get("inputClusterIds", []):
                    represented.add(cid)

    # Final dedup pass after repair, just in case the repair reintroduced overlaps
    merged = dedup_by_cluster_ids(merged)

    # Clip represented to only valid input cluster IDs to avoid >100% anomalies
    represented = {cid for cid in represented if cid in all_ids}
    coverage_pct = (len(represented) / max(1, len(all_ids))) * 100.0
    print(f"🧮 Coverage (clusters): {len(represented)}/{len(all_ids)} = {coverage_pct:.1f}%")

    # --- Step 5d: Diagnostics ---
    total_stories = sum(len(stories) for stories in merged["categories"].values())
    print("📊 Final Digest Diagnostics:")
    print(f"  • Categories: {len(merged['categories'])}")
    print(f"  • Stories: {total_stories}")
    for cat, stories in merged["categories"].items():
        print(f"    - {cat}: {len(stories)} stories")

    return merged, clusters

# =======================
# HTML writer
# =======================
def save_html(parsed_articles, output_file=OUTPUT_HTML):
    if not parsed_articles or not isinstance(parsed_articles, dict):
        print("❌ No parsed articles to save")
        sys.exit(1)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RSS News Summary</title>
<style>
/* Default light theme */
body { 
  font-family: Arial, sans-serif; 
  margin: 20px; 
  background: #ffffff; 
  color: #1c1c1e; 
}

h1 { 
  background: #f0f0f5; 
  color: #1c1c1e; 
  padding: 12px 16px; 
  border-radius: 8px; 
}

h2 { 
  color: #1c1c1e; 
  margin-top: 24px; 
  background:#f0f0f5; 
  padding:10px 12px; 
  border-radius:8px; 
}

.articles {
  display: grid;
  grid-gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}

.article { 
  background: #f0f0f5; 
  border-radius: 8px; 
  padding: 12px; 
  box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
  transition: transform 0.2s;
}
.article:hover { transform: translateY(-3px); }

.article a { 
  text-decoration: none; 
  color: #0066cc; 
  font-weight: bold; 
}
.article a:hover { text-decoration: underline; }

.article p { 
  margin: 6px 0 0; 
  color: #333; 
  line-height: 1.4; 
}
.article .sources {
  font-size: 0.85em;   /* smaller text */
  color: #888;         /* light grey */
  margin-top: 4px;     /* little space above */
}

/* Limit preview summaries to 20 lines */
.summary {
  display: -webkit-box;
  -webkit-line-clamp: 20;   /* cap to 20 lines */
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  max-height: 30em;         /* 20 lines * 1.5em line-height = 30em fallback */
  line-height: 1.5;
}

/* Modal sources: same look as main page sources */
.modal-sources {
  font-size: 0.85em;
  color: #888;
  margin-top: 8px;
}
.modal-sources a { color: #888; text-decoration: none; }
.modal-sources a:hover { text-decoration: underline; }

/* Images adjustments */
.article img, .modal-img {
  max-width: 50%;        
  height: auto;
  margin: 6px 0;
  border-radius: 6px;
  display: block;
  filter: brightness(0.95) contrast(1.05);
}

/* Modal styles */
.modal {
  display: none; 
  position: fixed; 
  z-index: 9999; 
  inset: 0;
  background: rgba(0,0,0,0.5);
}

.modal-content {
  background: #f0f0f5; 
  margin: 5% auto; 
  padding: 20px; 
  border-radius: 12px;
  width: min(900px, 92vw); 
  max-height: 82vh; 
  overflow-y: auto; 
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  color: #1c1c1e;
}
.modal-header { 
  display:flex; 
  justify-content: space-between; 
  align-items: center; 
  gap: 12px; 
}

.modal-title { margin: 0; font-size: 1.25rem; }
.modal-close { cursor: pointer; font-size: 1.5rem; border:none; background:transparent; line-height:1; color: #1c1c1e; }

.modal-actions { margin-top: 12px; }

.btn { display:inline-block; padding: 10px 14px; border-radius:8px; text-decoration:none; font-weight:600; }
.btn-primary { background:#0066cc; color:#fff; }
.btn-primary:hover { background:#004999; }

/* --- Dark mode --- */
@media (prefers-color-scheme: dark) {
  body { background: #1c1c1e; color: #f0f0f5; }
  h1, h2, .article, .modal-content { background: #2c2c2e; color: #f0f0f5; }
  .article a { color: #0a84ff; }
  .article p { color: #d0d0d5; }
  .btn-primary { background: #0a84ff; color: #fff; }
  .modal-close { color: #f0f0f5; }
  .article img, .modal-img { filter: brightness(0.85) contrast(1.05); }
}
</style>
</head>
<body>
<h1>📰 RSS News Summary</h1>

<!-- Modal -->
<div id="articleModal" class="modal" role="dialog" aria-modal="true" aria-hidden="true">
  <div class="modal-content">
    <div class="modal-header">
      <h3 id="modalTitle" class="modal-title"></h3>
      <button class="modal-close" id="modalClose" aria-label="Close">&times;</button>
    </div>
    <img id="modalImg" class="modal-img" alt="" style="display:none;">
    <div id="modalBody"></div>
    <p id="modalSource" class="modal-sources"></p>
    <div class="modal-actions">
      <a id="modalLink" class="btn btn-primary" href="#" target="_blank" rel="noopener">Read Original ↗</a>
    </div>
  </div>
</div>
"""

    # --- Safety checks ---
    if not parsed_articles or not isinstance(parsed_articles, dict):
        print("❌ No parsed articles to save")
        sys.exit(1)

    # Insert categories + articles
    for category, items in parsed_articles.get("categories", {}).items():
        if not items:
            continue

        html_content += f"<h2>{html.escape(category)}</h2>\n"
        html_content += '<div class="articles">\n'  # wrapper for multi-column layout

        for item in items:
            title_raw = item.get("title", "") or ""
            link_raw = item.get("link", "") or ""
            source_raw = get_domain(link_raw)
            source_attr = html.escape(source_raw, quote=True)

            # SUMMARY: strip HTML tags, unescape HTML entities, then escape minimal set (<,>,&)
            summary_raw = item.get("summary", "") or ""
            # use your existing clean_html to strip tags
            summary_stripped = clean_html(summary_raw)
            # convert entities to characters then escape <>& but preserve quotes for nicer look
            summary_vis = html.escape(html.unescape(summary_stripped), quote=False)

            # Full story (kept as-is for modal). We escape for use in attribute safely.
            full_story_raw = item.get("full_story", "") or ""
            full_story_attr = html.escape(full_story_raw or "", quote=True).replace("'", "&#39;")

            img_raw = item.get("image", None)
            primary_link_raw = item.get("primary_link", link_raw)
            title_attr = html.escape(title_raw or "", quote=True).replace("'", "&#39;")
            link_attr = html.escape(link_raw or "", quote=True).replace("'", "&#39;")
            img_attr = html.escape(img_raw or "", quote=True).replace("'", "&#39;")
            primary_link_attr = html.escape(primary_link_raw or "", quote=True).replace("'", "&#39;")
            sources_list = item.get("sources", []) or []
            sources_json = html.escape(json.dumps(sources_list, ensure_ascii=False), quote=True)

            html_content += '<div class="article">\n'
            html_content += (
                f'  <a href="#" class="open-modal" '
                f'data-title="{title_attr}" '
                f'data-link="{link_attr}" '
                f'data-primary-link="{primary_link_attr}" '
                f'data-image="{img_attr}" '
                f'data-full-story="{full_story_attr}" '
                f'data-sources="{sources_json}" '
                f'data-source="{source_attr}">{html.escape(title_raw)}</a>\n'
            )

            if img_raw:
                html_content += (
                    f'  <a href="#" class="open-modal" '
                    f'data-title="{title_attr}" '
                    f'data-link="{link_attr}" '
                    f'data-primary-link="{primary_link_attr}" '
                    f'data-image="{img_attr}" '
                    f'data-full-story="{full_story_attr}" '
                    f'data-sources="{sources_json}" '
                    f'data-source="{source_attr}">'
                    f'<img src="{html.escape(img_raw)}" alt="image"></a>\n'
                )

            # Insert preview paragraph with the "summary" class so CSS clamp applies
            if summary_vis:
                html_content += f'  <p class="summary">{summary_vis}</p>\n'

            # Build sources line (light grey, small text)
            sources_list_vis = []
            for s in item.get("sources", []):
                try:
                    host = urlparse(s.get("link", "")).hostname or ""
                    host = host.replace("www.", "")
                    if host:
                        sources_list_vis.append(host)
                except:
                    pass

            if sources_list_vis:
                sources_html = f'  <p class="sources">{" • ".join(sources_list_vis)}</p>\n'
                html_content += sources_html

            html_content += '</div>\n'  # close .article

        html_content += '</div>\n'  # close .articles

    # Add script (modal + fetch + readability)
    html_content += """
<script src="https://unpkg.com/@mozilla/readability@0.4.4/Readability.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {
  const modal       = document.getElementById("articleModal");
  const modalBody   = document.getElementById("modalBody");
  const modalLink   = document.getElementById("modalLink");
  const modalClose  = document.getElementById("modalClose");
  const modalImg    = document.getElementById("modalImg");
  const modalSource = document.getElementById("modalSource");
  const modalTitle  = modal.querySelector(".modal-title");

  function clearModal() {
    modalImg.style.display = "none";
    modalImg.src = "";
    modalBody.innerHTML = "";
    modalSource.textContent = "";
    modalLink.href = "#";
    modalTitle.textContent = "";
  }

  document.querySelectorAll(".open-modal").forEach(el => {
    el.addEventListener("click", function (e) {
      e.preventDefault();
      clearModal();

      const url        = this.dataset.link || "";
      const title      = this.dataset.title || "";
      const img        = this.dataset.image || "";
      const fullStory  = this.dataset.fullStory || "";
      const sourcesRaw = this.dataset.sources || "[]";

      modalLink.href = url || this.dataset.primaryLink || "#";
      modalTitle.textContent = title;

      // Render sources as clickable chips
      try {
        const list  = JSON.parse(sourcesRaw);
        const chips = (Array.isArray(list) ? list : []).map(s => {
          const href = s && s.link ? s.link : "#";
          let host = "";
          try { host = new URL(href).hostname.replace(/^www\\./, ""); } catch {}
          return `<a href="${href}" target="_blank" rel="noopener">${host}</a>`;
        });
        modalSource.innerHTML = chips.join(" • ");
      } catch {
        modalSource.textContent = "";
      }

      // Image (optional)
      if (img) {
        modalImg.src = img;
        modalImg.style.display = "block";
      } else {
        modalImg.style.display = "none";
      }

      // Prefer merged full_story; fallback to Readability fetch
      if (fullStory && fullStory.trim()) {
        // The full_story was escaped for attribute safety and will be decoded by browser.
        // Insert as HTML so paragraphs / formatting are preserved.
        modalBody.innerHTML = fullStory;
        modal.style.display = "block";
        return;
      }

      // Otherwise attempt to fetch and extract (proxy)
      modalBody.innerHTML = "<p>Loading full article...</p>";
      const proxy  = "https://api.allorigins.win/get?url=";
      const target = encodeURIComponent(url);

      fetch(proxy + target)
        .then(r => r.json())
        .then(data => {
          const doc = new DOMParser().parseFromString(data.contents, "text/html");
          const article = new Readability(doc).parse();
          if (article) {
            // sanitize line breaks into paragraphs
            modalBody.innerHTML = "<p>" + article.textContent
              .replace(/\\r\\n/g, "<br>")
              .replace(/\\n/g, "<br>")
              .replace(/\\r/g, "<br>")
              .replace(/\\n{2,}/g, "</p><p>") + "</p>";
          } else {
            modalBody.innerHTML = "<p>⚠️ Could not extract article text.</p>";
          }
        })
        .catch(err => {
          console.error("Error loading article:", err);
          modalBody.innerHTML = "<p>⚠️ Could not load full article. Please open the original source.</p>";
        });

      modal.style.display = "block";
    });
  });

  modalClose.addEventListener("click", function () { modal.style.display = "none"; clearModal(); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") { modal.style.display = "none"; clearModal(); }});
  window.addEventListener("click", function (e) { if (e.target === modal) { modal.style.display = "none"; clearModal(); }});
});
</script>
</body></html>
"""

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

    parsed_articles, clusters = summarize_articles_json(articles, CATEGORIES, EXCLUDED_TOPICS)

    # If batch metadata was emitted, compute real cluster coverage from stories
    try:
        # Collect covered cluster IDs from stories
        covered = set()
        for items in parsed_articles.get("categories", {}).values():
            for s in items:
                for cid in s.get("source_cluster_ids", []):
                    covered.add(cid)
        # Load how many clusters we had
        if os.path.exists("raw_input_clusters.json"):
            with open("raw_input_clusters.json", "r", encoding="utf-8") as fin:
                rc = json.load(fin)
            total_clusters = len(rc.get("clusters", []))
            if total_clusters:
                print(f"🧮 True cluster coverage: {len(covered)}/{total_clusters} = {len(covered)/total_clusters*100:.1f}%")
    except Exception as e:
        print(f"⚠️ Coverage diagnostic failed: {e}")



    if not parsed_articles.get("categories"):
        print("⚠️ No summaries generated.")
        return

    # === Diagnostics ===
    print("📊 Digest diagnostics:")
    total_stories = 0
    multi_source_stories = 0
    all_sources_counts = []

    for cat, items in parsed_articles.get("categories", {}).items():
        count = len(items)
        print(f"  {cat}: {count} stories")
        for it in items:
            sc = len(it.get("sources", []))
            all_sources_counts.append(sc)
            if sc > 1:
                multi_source_stories += 1
            print(f"    - {it.get('title','')[:60]}... ({sc} sources)")
        total_stories += count

    print(f"➡️ Total: {total_stories} stories, {multi_source_stories} with multiple sources")

    if all_sources_counts:
        avg_sources = sum(all_sources_counts) / len(all_sources_counts)
        max_sources = max(all_sources_counts)
        print(f"📈 Avg sources per story: {avg_sources:.2f}")
        print(f"🔝 Largest merged cluster: {max_sources} sources")

    # Coverage check (read the saved files we wrote earlier)
    try:
        input_articles = 0
        input_clusters = 0
        if os.path.exists("raw_input_articles.json"):
            with open("raw_input_articles.json", "r", encoding="utf-8") as fin:
                raw_in = json.load(fin)
            input_articles = len(raw_in.get("articles", []))
        if os.path.exists("raw_input_clusters.json"):
            with open("raw_input_clusters.json", "r", encoding="utf-8") as fin:
                raw_clusters = json.load(fin)
            input_clusters = len(raw_clusters.get("clusters", []))

        covered = set()
        for items in parsed_articles.get("categories", {}).values():
            for s in items:
                for cid in s.get("inputClusterIds", []):
                    covered.add(cid)

        total_sources_output = sum(len(it.get("sources", []))
                                  for items in parsed_articles.get("categories", {}).values()
                                  for it in items)
        print(f"📥 Input had {input_articles} articles → {input_clusters} clusters")
        if input_articles:
            coverage_pct = (total_sources_output / input_articles) * 100
            print(f"📦 Coverage: {total_sources_output} sources represented in output")
            print(f"✅ Coverage ratio: {coverage_pct:.1f}%")
        else:
            print(f"📦 Coverage: {total_sources_output} sources represented in output")
    except Exception as e:
        print(f"⚠️ Could not compute coverage: {e}")


    parsed_articles = enforce_global_coverage(parsed_articles, clusters)



    save_html(parsed_articles, OUTPUT_HTML)
    print(f"✅ HTML saved to {OUTPUT_HTML}")
    print("🎉 Done!")


if __name__ == "__main__":
    main()
