import os
import re
import time
import html
import json
import math
import random
import hashlib
import difflib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests
import feedparser
from bs4 import BeautifulSoup
from openai import OpenAI


# ==================================================
# WND MAIN DIGEST
# Architecture:
# - Groq/Qwen only
# - Groq batch scoring per source
# - top 25% from EACH source
# - theme dedupe across sources
# - Groq batch Russian rewrite
# - no Gemini usage
# ==================================================


# =========================
# SETTINGS
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_FILE = "urls_base.txt"
SOURCES_FILE = "sources.txt"

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GROQ_REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-120b")

# Important:
# 25 means: from every RSS source, keep the best 25% of NEW items from that source.
TOP_PERCENT_PER_SOURCE = int(os.getenv("TOP_PERCENT_PER_SOURCE", "25"))
SOURCE_MIN_TAKE = int(os.getenv("SOURCE_MIN_TAKE", "1"))

# 0 means no global cap. This matches your request: publish all deduped top-25%-per-source stories.
# If later you want a hard cap, set MAX_NEWS_PER_RUN to e.g. 10.
MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "0"))

MAX_CANDIDATES_PER_SOURCE = int(os.getenv("MAX_CANDIDATES_PER_SOURCE", "20"))
ENTRIES_PER_SOURCE = int(os.getenv("ENTRIES_PER_SOURCE", "20"))

GROQ_BATCH_SIZE = int(os.getenv("GROQ_BATCH_SIZE", "20"))
GROQ_REWRITE_BATCH_SIZE = int(os.getenv("GROQ_REWRITE_BATCH_SIZE", "20"))

POST_DELAY_MIN_SECONDS = int(os.getenv("POST_DELAY_MIN_SECONDS", "60"))
POST_DELAY_MAX_SECONDS = int(os.getenv("POST_DELAY_MAX_SECONDS", "180"))

MAX_POSTS_PER_DOMAIN = int(os.getenv("MAX_POSTS_PER_DOMAIN", "3"))

DIGEST_CHANNEL_URL = os.getenv("DIGEST_CHANNEL_URL", "https://t.me/wnd_digest")

REQUEST_TIMEOUT = 30

groq_client = None


# =========================
# UTILS
# =========================

def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text):
    if not text:
        return ""
    soup = BeautifulSoup(str(text), "html.parser")
    value = soup.get_text(" ", strip=True)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def strip_markdown(text):
    if not text:
        return ""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "")
    text = text.replace("__", "")
    return text.strip()


def normalize_url(url):
    if not url:
        return ""

    url = url.strip()
    parsed = urlparse(url)

    blocked_prefixes = ("utm_",)
    blocked_keys = {
        "fbclid", "gclid", "mc_cid", "mc_eid", "cmpid",
        "ocid", "smid", "ref", "ref_src", "ito"
    }

    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_l = key.lower()
        if key_l in blocked_keys:
            continue
        if any(key_l.startswith(prefix) for prefix in blocked_prefixes):
            continue
        query.append((key, value))

    clean = parsed._replace(
        scheme=parsed.scheme.lower() or "https",
        netloc=parsed.netloc.lower().replace("www.", ""),
        path=parsed.path.rstrip("/"),
        query=urlencode(query, doseq=True),
        fragment=""
    )

    return urlunparse(clean)


def normalize_title(title):
    title = clean_text(title).lower()
    title = re.sub(r"[^\wа-яёіїєґ\s]", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def get_domain(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return "unknown"


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def extract_json_payload(text):
    if not text:
        return {}

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Prefer JSON object.
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        return json.loads(cleaned[obj_start:obj_end + 1])

    # Fallback JSON array.
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        return {"items": json.loads(cleaned[arr_start:arr_end + 1])}

    return json.loads(cleaned)


def has_cyrillic(text):
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


def is_valid_russian_rewrite(headline, post):
    if not headline or not post:
        return False
    joined = f"{headline} {post}"
    if not has_cyrillic(joined):
        return False
    # Avoid obviously untouched English fallback.
    latin_words = re.findall(r"\b[A-Za-z]{4,}\b", joined)
    cyr_words = re.findall(r"\b[А-Яа-яЁё]{3,}\b", joined)
    return len(cyr_words) >= max(3, len(latin_words))


# =========================
# DATABASE / DUPLICATES
# =========================

def load_db():
    url_hashes = set()
    title_hashes = set()
    raw_lines = set()

    if not os.path.exists(DB_FILE):
        return url_hashes, title_hashes, raw_lines

    with open(DB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue

            raw_lines.add(item)

            if item.startswith("urlhash:"):
                url_hashes.add(item.replace("urlhash:", "", 1))
            elif item.startswith("titlehash:"):
                title_hashes.add(item.replace("titlehash:", "", 1))
            elif item.startswith("seen_urlhash:"):
                url_hashes.add(item.replace("seen_urlhash:", "", 1))
            elif item.startswith("seen_titlehash:"):
                title_hashes.add(item.replace("seen_titlehash:", "", 1))
            elif item.startswith("http"):
                url_hashes.add(hash_text(normalize_url(item)))

    return url_hashes, title_hashes, raw_lines


def append_seen(item, reason):
    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write(f"seen_urlhash:{item['url_hash']}\n")
        f.write(f"seen_titlehash:{item['title_hash']}\n")
        f.write(f"seen_reason:{reason}\n")
        f.write(f"url:{item['link']}\n")
        f.write(f"title:{item['normalized_title']}\n")


def append_published(item):
    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write(f"urlhash:{item['url_hash']}\n")
        f.write(f"titlehash:{item['title_hash']}\n")
        f.write(f"published_url:{item['link']}\n")
        f.write(f"title:{item['normalized_title']}\n")


def is_similar_to_published_title(title, published_title_hashes, raw_db_lines):
    normalized = normalize_title(title)
    title_hash = hash_text(normalized)

    if title_hash in published_title_hashes:
        return True

    old_titles = [
        line.replace("title:", "", 1)
        for line in raw_db_lines
        if line.startswith("title:")
    ]

    for old in old_titles[-1000:]:
        ratio = difflib.SequenceMatcher(None, normalized, old).ratio()
        if ratio >= 0.86:
            return True

    return False


# =========================
# SOURCES
# =========================

def load_sources():
    if not os.path.exists(SOURCES_FILE):
        raise FileNotFoundError("sources.txt not found")

    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def collect_candidates_by_source(sources, published_url_hashes, published_title_hashes, raw_db_lines):
    groups = []
    total = 0

    for source_index, rss_url in enumerate(sources, start=1):
        print(f"Reading source: {rss_url}")

        try:
            feed = feedparser.parse(rss_url)
            source_items = []

            for entry_index, entry in enumerate(feed.entries[:ENTRIES_PER_SOURCE], start=1):
                link = normalize_url(entry.get("link", ""))
                title = clean_text(entry.get("title", ""))
                summary = clean_text(entry.get("summary", entry.get("description", "")))

                if not link or not title:
                    continue

                normalized_title = normalize_title(title)
                if len(normalized_title) < 12:
                    continue

                url_hash = hash_text(link)
                title_hash = hash_text(normalized_title)

                if url_hash in published_url_hashes:
                    continue

                if title_hash in published_title_hashes:
                    continue

                if is_similar_to_published_title(title, published_title_hashes, raw_db_lines):
                    continue

                item = {
                    "source_index": source_index,
                    "source_rss": rss_url,
                    "source_position": entry_index,
                    "domain": get_domain(link),
                    "entry": entry,
                    "link": link,
                    "url_hash": url_hash,
                    "title": title,
                    "summary": summary,
                    "normalized_title": normalized_title,
                    "title_hash": title_hash,
                }

                source_items.append(item)

            if len(source_items) > MAX_CANDIDATES_PER_SOURCE:
                source_items = source_items[:MAX_CANDIDATES_PER_SOURCE]

            print(f"New candidates from source: {len(source_items)}")
            total += len(source_items)

            groups.append({
                "source_url": rss_url,
                "items": source_items,
            })

        except Exception as e:
            print(f"Source error {rss_url}: {e}")

    print(f"Total new candidates across all sources: {total}")
    return groups


# =========================
# IMAGES
# =========================

def is_probably_good_image_url(url):
    if not url:
        return False

    lower = url.lower()

    bad_words = [
        "thumbnail", "thumb", "icon", "logo", "avatar",
        "sprite", "placeholder", "default"
    ]

    if any(word in lower for word in bad_words):
        return False

    return lower.startswith("http")


def extract_rss_image(entry):
    candidates = []

    if "media_content" in entry:
        for media in entry.media_content:
            url = media.get("url")
            width = int(media.get("width", 0) or 0)
            height = int(media.get("height", 0) or 0)
            candidates.append((url, width, height))

    if "media_thumbnail" in entry:
        for media in entry.media_thumbnail:
            url = media.get("url")
            width = int(media.get("width", 0) or 0)
            height = int(media.get("height", 0) or 0)
            candidates.append((url, width, height))

    if "links" in entry:
        for link in entry.links:
            href = link.get("href")
            link_type = link.get("type", "")
            rel = link.get("rel", "")
            if href and ("image" in link_type or rel == "enclosure"):
                candidates.append((href, 0, 0))

    good = []
    for url, width, height in candidates:
        if not is_probably_good_image_url(url):
            continue

        if width and height:
            if width >= 800 or height >= 450:
                good.append((url, width * height))
        else:
            good.append((url, 0))

    if not good:
        return None

    good.sort(key=lambda x: x[1], reverse=True)
    return good[0][0]


def extract_og_image(article_url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WNDNewsBot/1.0; +https://t.me/wnd_digest)"
        }

        response = requests.get(article_url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        selectors = [
            ("meta", {"property": "og:image:secure_url"}),
            ("meta", {"property": "og:image"}),
            ("meta", {"name": "twitter:image"}),
            ("meta", {"name": "twitter:image:src"}),
        ]

        for tag_name, attrs in selectors:
            tag = soup.find(tag_name, attrs=attrs)
            if tag and tag.get("content") and is_probably_good_image_url(tag.get("content")):
                return tag.get("content")

    except Exception as e:
        print(f"OpenGraph image error: {e}")

    return None


def get_best_image(entry, article_url):
    og = extract_og_image(article_url)
    if og:
        return og

    return extract_rss_image(entry)


# =========================
# GROQ SCORING PER SOURCE
# =========================

def ask_groq_score_batch(batch, source_url):
    payload = []

    for item in batch:
        payload.append({
            "id": item["batch_id"],
            "title": item["title"],
            "description": item["summary"][:700],
            "full_article_text": item.get("article_text", "")[:9000],
            "domain": item["domain"],
            "source": source_url,
        })

    prompt = f"""
You are the senior editor for WND | Main Digest, a serious international Telegram news digest.

You are scoring articles from ONE RSS source:
{source_url}

Task:
Score each item from 1 to 10 and identify its theme.

Channel profile:
- world news
- geopolitics
- wars
- international security
- diplomacy
- economy
- sanctions
- major global technology only if politically/economically important
- major crises and strategic events

Score scale:
10 = historic/global breaking news
9 = extremely important international story
8 = major publication-worthy story
7 = strong publication candidate
6 = interesting but secondary
5 and below = weak or routine

Important:
Even if the source has weak news, score honestly.
The code will take the top {TOP_PERCENT_PER_SOURCE}% from this source.

Theme key:
Create a short canonical English theme_key for deduplication across sources.
Example:
"us-russia-oil-sanctions-extension"
"belarus-russia-nuclear-drills"
"cuba-us-drone-claims"


КРИТИЧЕСКИ ВАЖНО ДЛЯ КАЧЕСТВА:
- не переписывай только первый абзац;
- используй full_article_text как главный источник фактов;
- title и description нужны только как подсказка;
- сжимай ВСЮ статью, а не RSS summary;
- сохраняй ключевые подробности: кто, что, где, когда, почему важно;
- сохраняй все числа, даты, суммы, проценты, мощности, названия стран, городов, организаций и должности;
- если в статье есть контекст, последствия, причины, детали или сравнения — выбери самые важные и включи в пост;
- не добавляй факты от себя;
- не используй пустые AI-фразы: "важный шаг", "новый этап", "меняет правила игры", "демонстрирует", "инновационный подход";
- текст должен выглядеть как работа живого редактора, а не машинный перевод.

СТРУКТУРА:
- headline: короткий сильный русский заголовок с главным фактом;
- post: 2–4 коротких абзаца;
- первый абзац: что произошло;
- второй/третий: ключевые детали и почему это важно;
- без ссылок и без упоминания источника.

Return STRICT JSON object only:
{{
  "items": [
    {{
      "id": 1,
      "score": 8,
      "category": "economy",
      "theme_key": "short-canonical-theme",
      "reason": "brief reason"
    }}
  ]
}}

No markdown. No text outside JSON.

Items:
{json.dumps(payload, ensure_ascii=False)}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown. No explanations. JSON must be valid. Escape all newlines inside strings."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def score_source_items(source_url, items):
    if not items:
        return []

    for idx, item in enumerate(items, start=1):
        item["batch_id"] = idx

    print(f"Scoring source with Groq: {source_url}")
    print(f"Source candidates: {len(items)} | batch size: {GROQ_BATCH_SIZE}")

    scored = []
    by_id = {item["batch_id"]: item for item in items}

    for batch_index, batch in enumerate(chunks(items, GROQ_BATCH_SIZE), start=1):
        try:
            print(f"Groq scoring batch {batch_index}: {len(batch)} articles")
            raw = ask_groq_score_batch(batch, source_url)
            payload = extract_json_payload(raw)
            results = payload.get("items", [])

        except Exception as e:
            print(f"Groq scoring failed for source batch: {e}")
            results = []

        for result in results:
            try:
                item_id = int(result.get("id"))
                score = int(result.get("score", 0))
            except Exception:
                continue

            item = by_id.get(item_id)
            if not item:
                continue

            item["score"] = max(1, min(10, score))
            item["category"] = str(result.get("category", "other"))
            item["theme_key"] = normalize_title(str(result.get("theme_key", ""))) or item["normalized_title"][:80]
            item["score_reason"] = str(result.get("reason", ""))

            scored.append(item)

            print(
                f"Score: {item['score']} | {item['category']} | "
                f"{item['domain']} | {item['title']}"
            )

    # If Groq failed to return some items, keep them with low score so top 25% logic still has candidates.
    scored_ids = {item["batch_id"] for item in scored}
    for item in items:
        if item["batch_id"] not in scored_ids:
            item["score"] = 1
            item["category"] = "other"
            item["theme_key"] = item["normalized_title"][:80]
            item["score_reason"] = "No Groq score returned"
            scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def take_top_percent_from_source(scored):
    if not scored:
        return []

    take_count = max(SOURCE_MIN_TAKE, math.ceil(len(scored) * TOP_PERCENT_PER_SOURCE / 100))
    take_count = min(take_count, len(scored))

    selected = scored[:take_count]

    print(
        f"Top {TOP_PERCENT_PER_SOURCE}% from source: "
        f"{len(selected)}/{len(scored)} selected"
    )

    return selected


def score_and_select_per_source(groups):
    selected_all = []
    processed_all = []

    print(f"Using Groq model: {GROQ_MODEL}")
    print(f"Top percent per source: {TOP_PERCENT_PER_SOURCE}%")

    for group in groups:
        source_url = group["source_url"]
        items = group["items"]

        if not items:
            continue

        processed_all.extend(items)

        scored = score_source_items(source_url, items)
        selected = take_top_percent_from_source(scored)
        selected_all.extend(selected)

    return selected_all, processed_all


# =========================
# THEME DEDUPE
# =========================

def theme_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def dedupe_themes(items):
    if not items:
        return []

    items = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    selected = []
    theme_keys = []
    title_keys = []
    domain_counts = {}

    for item in items:
        domain = item.get("domain", "unknown")
        if domain_counts.get(domain, 0) >= MAX_POSTS_PER_DOMAIN:
            continue

        theme_key = normalize_title(item.get("theme_key", "")) or item.get("normalized_title", "")
        title_key = item.get("normalized_title", normalize_title(item.get("title", "")))

        duplicate = False

        for old_theme in theme_keys:
            if theme_key and old_theme and theme_similarity(theme_key, old_theme) >= 0.72:
                duplicate = True
                break

        if not duplicate:
            for old_title in title_keys:
                if theme_similarity(title_key, old_title) >= 0.68:
                    duplicate = True
                    break

        if duplicate:
            print(f"Theme duplicate skipped: {item['title']}")
            continue

        selected.append(item)
        theme_keys.append(theme_key)
        title_keys.append(title_key)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    if MAX_NEWS_PER_RUN > 0:
        selected = selected[:MAX_NEWS_PER_RUN]

    print(f"After theme dedupe: {len(selected)} stories")
    return selected



# =========================
# ARTICLE TEXT EXTRACTION
# =========================

def fetch_article_text(article_url):
    """
    Fetch readable full article text only for already-selected stories.
    This improves rewrite quality without wasting requests on every RSS item.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WNDMainDigestBot/1.0; +https://t.me/wnd_digest)"
        }

        response = requests.get(article_url, headers=headers, timeout=25, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "iframe", "svg", "form", "nav", "footer", "header", "aside"]):
            tag.decompose()

        selectors = [
            "article",
            "main",
            "[itemprop='articleBody']",
            "[data-testid='article-body']",
            ".article-body",
            ".article__body",
            ".ArticleBody",
            ".story-body",
            ".entry-content",
            ".post-content",
            ".content",
        ]

        texts = []

        for selector in selectors:
            node = soup.select_one(selector)
            if node:
                paragraphs = [
                    clean_text(p.get_text(" ", strip=True))
                    for p in node.find_all(["p", "li", "h2", "h3"])
                ]
                paragraphs = [p for p in paragraphs if len(p) > 45]
                if paragraphs:
                    texts = paragraphs
                    break

        if not texts:
            paragraphs = [
                clean_text(p.get_text(" ", strip=True))
                for p in soup.find_all("p")
            ]
            texts = [p for p in paragraphs if len(p) > 45]

        junk_patterns = [
            "Sign up", "Subscribe", "All rights reserved", "Advertisement",
            "Read more", "Cookie", "cookies", "newsletter",
            "Follow us", "Share this", "Комментарии", "Читайте также",
        ]

        filtered = []
        for p in texts:
            if any(j.lower() in p.lower() for j in junk_patterns):
                continue
            filtered.append(p)

        full_text = "\n".join(filtered)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

        return full_text[:9000]

    except Exception as e:
        print(f"Article text fetch failed: {e}")
        return ""

# =========================
# GROQ RUSSIAN REWRITE
# =========================

def ask_groq_rewrite_batch(batch):
    payload = []

    for item in batch:
        payload.append({
            "id": item["global_id"],
            "score": item.get("score"),
            "category": item.get("category"),
            "title": item["title"],
            "description": item["summary"][:1000],
            "full_article_text": item.get("article_text", "")[:9000],
            "domain": item["domain"],
            "theme_key": item.get("theme_key", ""),
        })

    prompt = f"""
Ты редактор Telegram-канала WND | Мировой дайджест.

Твоя задача:
Перевести и переписать выбранные новости на русском языке для основного мирового дайджеста.

Критически важно:
- ВСЕ headline и post должны быть на русском языке.
- Нельзя оставлять английский оригинальный заголовок.
- Нельзя добавлять факты от себя.
- Нельзя терять важные факты из title/description.
- Нельзя использовать ИИ-фразы.
- Нельзя писать "это подчеркивает", "важный шаг", "на фоне", если этого нет в сути.
- Нельзя раздувать.
- Нельзя кликбейтить.
- Нужно писать коротко, ясно, человеческим новостным языком.

Стиль:
- серьезный мировой дайджест
- плотная новостная подача
- без эмодзи
- без Markdown
- без воды
- без канцелярита
- без ссылок внутри текста

HEADLINE:
- короткий
- сильный
- информативный
- только русский язык

POST:
- 2–4 коротких абзаца
- суть события
- почему это важно, только если это понятно из исходного текста
- без выдуманных деталей

Верни СТРОГО JSON object:
{{
  "items": [
    {{
      "id": 1,
      "headline": "Русский заголовок",
      "post": "Русский текст поста"
    }}
  ]
}}

No markdown. No text outside JSON.

Новости:
{json.dumps(payload, ensure_ascii=False)}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Return only valid JSON. All headline/post fields must be Russian."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def rewrite_selected_with_groq(items):
    if not items:
        return []

    for idx, item in enumerate(items, start=1):
        item["global_id"] = idx

    print(f"Groq Russian rewrite: {len(items)} articles in batches of {GROQ_REWRITE_BATCH_SIZE}")

    for item in items:
        item["article_text"] = fetch_article_text(item.get("link", ""))
        if item["article_text"]:
            print(f"Fetched full article text: {len(item['article_text'])} chars | {item['title'][:90]}")
        else:
            print(f"No full article text; using RSS summary only | {item['title'][:90]}")

    rewritten_by_id = {}

    for batch_index, batch in enumerate(chunks(items, GROQ_REWRITE_BATCH_SIZE), start=1):
        try:
            print(f"Groq rewrite batch {batch_index}: {len(batch)} articles")
            raw = ask_groq_rewrite_batch(batch)
            payload = extract_json_payload(raw)
            results = payload.get("items", [])

        except Exception as e:
            print(f"Groq rewrite batch failed: {e}")
            results = []

        for result in results:
            try:
                item_id = int(result.get("id"))
                headline = strip_markdown(str(result.get("headline", "")).strip())
                post = strip_markdown(str(result.get("post", "")).strip())
            except Exception:
                continue

            if is_valid_russian_rewrite(headline, post):
                rewritten_by_id[item_id] = {
                    "headline": headline,
                    "post_text": post,
                }

    final = []

    for item in items:
        rewrite = rewritten_by_id.get(item["global_id"])

        if rewrite:
            item["headline"] = rewrite["headline"]
            item["post_text"] = rewrite["post_text"]
            print(f"Russian rewrite OK: {item['headline'][:90]}")
            final.append(item)
        else:
            # No English fallback. If Groq failed to rewrite, skip the article.
            print(f"Skipped: no valid Russian rewrite for {item['title']}")

    return final


# =========================
# TELEGRAM
# =========================

def telegram_api(method, payload):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            print(f"Telegram {method} error:", response.status_code, response.text)

        return response.status_code == 200

    except Exception as e:
        print(f"Telegram request error: {e}")
        return False


def make_post_html(headline, body, source_url, score):
    clean_headline = strip_markdown(headline.strip())
    clean_body = strip_markdown(body.strip())

    safe_headline = html.escape(clean_headline)
    safe_body = html.escape(clean_body)

    if score >= 10:
        safe_headline = f"⚡ {safe_headline}"

    safe_source_url = html.escape(source_url.strip(), quote=True)
    safe_digest_url = html.escape(DIGEST_CHANNEL_URL.strip(), quote=True)

    return (
        f"<b>{safe_headline}</b>\n\n"
        f"{safe_body}\n\n"
        f'<a href="{safe_source_url}">Источник</a> | '
        f'<a href="{safe_digest_url}">WND Digest</a>'
    )


def send_photo_with_caption(image_url, caption_html):
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "photo": image_url,
        "caption": caption_html,
        "parse_mode": "HTML",
    }

    return telegram_api("sendPhoto", payload)


def send_text_with_preview(text_html):
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": text_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }

    return telegram_api("sendMessage", payload)


def send_text_without_preview(text_html):
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": text_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    return telegram_api("sendMessage", payload)


def publish_item(item):
    text_html = make_post_html(
        item["headline"],
        item["post_text"],
        item["link"],
        item["score"],
    )

    image_url = get_best_image(item["entry"], item["link"])

    if image_url and len(text_html) <= 1000:
        ok = send_photo_with_caption(image_url, text_html)
        if ok:
            return True

        print("Photo failed, fallback to text with link preview.")

    ok = send_text_with_preview(text_html)
    if ok:
        return True

    return send_text_without_preview(text_html)


def sleep_between_posts(current_index, total_count):
    if current_index >= total_count - 1:
        return

    delay = random.randint(POST_DELAY_MIN_SECONDS, POST_DELAY_MAX_SECONDS)
    print(f"Sleeping {delay} seconds before next post...")
    time.sleep(delay)


# =========================
# MAIN
# =========================

def validate_env():
    missing = []

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHANNEL_ID:
        missing.append("TELEGRAM_CHANNEL_ID")

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if missing:
        raise RuntimeError("Missing env variables: " + ", ".join(missing))


def init_clients():
    global groq_client

    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def mark_processed_as_seen(processed_items, published_url_hashes):
    published_hashes = {item["url_hash"] for item in published_url_hashes}

    for item in processed_items:
        if item["url_hash"] in published_hashes:
            continue

        try:
            append_seen(item, "processed_not_published")
        except Exception as e:
            print(f"append_seen failed: {e}")


def main():
    validate_env()
    init_clients()

    published_url_hashes, published_title_hashes, raw_db_lines = load_db()
    sources = load_sources()

    groups = collect_candidates_by_source(
        sources,
        published_url_hashes,
        published_title_hashes,
        raw_db_lines,
    )

    if not groups:
        print("No source groups.")
        return

    selected_by_source, processed_items = score_and_select_per_source(groups)

    print(f"Selected by top {TOP_PERCENT_PER_SOURCE}% per source: {len(selected_by_source)}")

    if not selected_by_source:
        print("No selected news after source selection.")
        mark_processed_as_seen(processed_items, [])
        return

    deduped = dedupe_themes(selected_by_source)

    if not deduped:
        print("No stories after theme dedupe.")
        mark_processed_as_seen(processed_items, [])
        return

    rewritten = rewrite_selected_with_groq(deduped)

    print(f"Selected for publishing: {len(rewritten)}")

    if not rewritten:
        print("No valid Russian rewrites to publish.")
        mark_processed_as_seen(processed_items, [])
        return

    published = []

    for index, item in enumerate(rewritten):
        print(
            f"Publishing {index + 1}/{len(rewritten)}: "
            f"score {item['score']} | {item['category']} | {item['domain']} | {item['headline']}"
        )

        try:
            if publish_item(item):
                append_published(item)
                published.append(item)
                print("Published.")
            else:
                append_seen(item, "publish_failed")
                print("Publish failed.")

        except Exception as e:
            append_seen(item, "publish_exception")
            print(f"Publish item error: {e}")

        sleep_between_posts(index, len(rewritten))

    mark_processed_as_seen(processed_items, published)

    print("Done.")


if __name__ == "__main__":
    main()
