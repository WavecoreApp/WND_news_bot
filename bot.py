
# ==================================================
# WND MAIN DIGEST — REFACTORED VERSION
# Architecture:
# - Groq Qwen3-32B for batch scoring
# - Gemini 2.5 Flash-Lite for rewrites
# - Optimized for hourly GitHub Actions runs
# ==================================================

import os

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

MAIN_DIGEST_MIN_SCORE = 8
TARGET_POSTS_MIN = 3
TARGET_POSTS_MAX = 10


import os
import re
import time
import html
import random
import json
import hashlib
import difflib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests
import feedparser
from bs4 import BeautifulSoup
from google import genai


# =========================
# SETTINGS
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DB_FILE = "urls_base.txt"
SOURCES_FILE = "sources.txt"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

MIN_SCORE_TO_POST = int(os.getenv("MIN_SCORE_TO_POST", "7"))
MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "5"))

POST_DELAY_MIN_SECONDS = int(os.getenv("POST_DELAY_MIN_SECONDS", "60"))
POST_DELAY_MAX_SECONDS = int(os.getenv("POST_DELAY_MAX_SECONDS", "180"))

ABOUT_POST_URL = os.getenv("ABOUT_POST_URL", "https://t.me/wnd_digest")
ABOUT_LINK_TEXT = os.getenv("ABOUT_LINK_TEXT", "ℹ️ О проекте WND")

ENTRIES_PER_SOURCE = int(os.getenv("ENTRIES_PER_SOURCE", "12"))
MAX_CANDIDATES_TO_ANALYZE = int(os.getenv("MAX_CANDIDATES_TO_ANALYZE", "20"))

# Важно: чтобы BBC/один сайт не забивал весь выпуск
MAX_POSTS_PER_DOMAIN = int(os.getenv("MAX_POSTS_PER_DOMAIN", "2"))

client = genai.Client()


# =========================
# UTILS
# =========================

def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text):
    if not text:
        return ""
    soup = BeautifulSoup(str(text), "html.parser")
    return soup.get_text(" ", strip=True)


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
    """
    Сохраняем не только опубликованные, но и обработанные/пропущенные.
    Так при ручных перезапусках бот не будет снова крутить те же BBC/Ebola новости.
    """
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

    for old in old_titles[-800:]:
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


def collect_candidates(sources, published_url_hashes, published_title_hashes, raw_db_lines):
    all_by_source = []

    for rss_url in sources:
        print(f"Reading source: {rss_url}")

        try:
            feed = feedparser.parse(rss_url)
            source_items = []

            for entry in feed.entries[:ENTRIES_PER_SOURCE]:
                link = normalize_url(entry.get("link", ""))

                title = clean_text(entry.get("title", ""))
                summary = clean_text(entry.get("summary", entry.get("description", "")))

                if not link or not title:
                    continue

                normalized_title = normalize_title(title)
                url_hash = hash_text(link)
                title_hash = hash_text(normalized_title)

                if url_hash in published_url_hashes:
                    continue

                if title_hash in published_title_hashes:
                    continue

                if is_similar_to_published_title(title, published_title_hashes, raw_db_lines):
                    continue

                source_items.append({
                    "source_rss": rss_url,
                    "domain": get_domain(link),
                    "entry": entry,
                    "link": link,
                    "url_hash": url_hash,
                    "title": title,
                    "summary": summary,
                    "normalized_title": normalized_title,
                    "title_hash": title_hash,
                })

            print(f"New candidates from source: {len(source_items)}")
            all_by_source.append(source_items)

        except Exception as e:
            print(f"Source error {rss_url}: {e}")

    # Round-robin по источникам, чтобы нижние источники тоже попадали в анализ
    balanced = []
    index = 0

    while True:
        added = False
        for source_items in all_by_source:
            if index < len(source_items):
                balanced.append(source_items[index])
                added = True
        if not added:
            break
        index += 1

    return balanced[:MAX_CANDIDATES_TO_ANALYZE]


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
# GEMINI
# =========================

def local_pre_score(item):
    """
    Cheap local ranking before Gemini.
    Gemini still makes the final score, but this prevents wasting quota on weak RSS items.
    """
    text_blob = f"{item.get('title', '')} {item.get('summary', '')} {item.get('domain', '')}".lower()

    high = [
        "war", "attack", "strike", "nuclear", "missile", "drone", "iran", "israel",
        "ukraine", "russia", "china", "taiwan", "trump", "xi", "putin", "g7",
        "sanction", "ceasefire", "opec", "oil", "gas", "market", "crisis",
        "ai", "artificial intelligence", "openai", "chatgpt", "google", "apple",
        "microsoft", "nvidia", "anthropic", "meta", "chip", "semiconductor",
        "cyber", "hack", "breach", "ransomware", "backdoor", "exploit",
        "ebola", "pandemic", "earthquake", "flood", "wildfire", "crash",
    ]

    medium = [
        "economy", "inflation", "election", "court", "president", "minister",
        "eu", "nato", "un ", "who ", "central bank", "rate", "deal",
        "сша", "иран", "израил", "украин", "росси", "китай", "тайван", "нато",
        "войн", "удар", "дрон", "ракета", "санкц", "эконом", "рын", "нефт",
        "ии", "нейросет", "apple", "google", "microsoft", "openai", "nvidia",
    ]

    low = [
        "review:", "what to watch", "what to read", "sport", "football", "celebrity",
        "recipe", "horoscope", "travel guide", "weekend", "movie review",
    ]

    score = 0
    for word in high:
        if word in text_blob:
            score += 4
    for word in medium:
        if word in text_blob:
            score += 2
    for word in low:
        if word in text_blob:
            score -= 5

    # Prefer real news domains a bit for Main Digest
    domain = item.get("domain", "")
    if any(d in domain for d in ["bbc", "guardian", "aljazeera", "nytimes", "npr", "dw.com", "meduza", "currenttime", "bloomberg", "cnbc", "arstechnica", "techcrunch"]):
        score += 1

    return score


def prepare_candidates_for_gemini(candidates):
    ranked = sorted(candidates, key=local_pre_score, reverse=True)
    selected = ranked[:MAX_CANDIDATES_TO_ANALYZE]

    for idx, item in enumerate(selected, start=1):
        item["batch_id"] = idx
        item["pre_score"] = local_pre_score(item)

    print(f"Using Gemini model: {GEMINI_MODEL}")
    print(f"Sending {len(selected)} candidates to Gemini in ONE batch request.")
    return selected


def ask_gemini_batch(candidates):
    news_lines = []

    for item in candidates:
        news_lines.append(
            {
                "id": item["batch_id"],
                "title": item["title"],
                "description": item["summary"],
                "link": item["link"],
                "domain": item["domain"],
                "pre_score": item.get("pre_score", 0),
            }
        )

    prompt = f"""
Ты редактор Telegram-канала WND | Мировой дайджест.

Задача:
Оцени список новостей и подготовь публикации на русском только для достойных новостей.

Формат канала:
• главные мировые события
• политика
• войны и безопасность
• дипломатия
• экономика и рынки
• технологии и AI, если событие реально значимое
• СНГ, Европа, США, Ближний Восток, Китай
• крупные катастрофы и глобальные риски

Что НЕ публиковать:
• мелкие локальные новости
• celebrity gossip
• спорт без мирового значения
• обычные обзоры гаджетов
• мелкие корпоративные пресс-релизы
• повторяющиеся routine updates

Оценка:
10 = крупнейшее мировое событие
9 = очень важная международная новость
8 = сильная новость
7 = хорошая публикабельная новость
6 и ниже = не публиковать

Правила текста:
1. Пиши на русском.
2. HEADLINE должен быть сильным, коротким и цепляющим, но без дешёвого кликбейта.
3. POST — короткая выжимка 2–5 коротких абзацев.
4. Без эмодзи.
5. Без Markdown.
6. Не выдумывай факты.
7. Не вставляй ссылки в текст.
8. Можно использовать списки с символом •.

Верни СТРОГО JSON-массив.
Без пояснений до или после JSON.

Формат каждого объекта:
{{
  "id": число,
  "score": число от 1 до 10,
  "category": "world/politics/war/economy/tech/ai/cis/disaster/other",
  "headline": "русский заголовок",
  "post": "короткий текст"
}}

Верни объекты только для новостей score >= {MIN_SCORE_TO_POST}.
Если достойных новостей нет — верни [].

Новости:
{json.dumps(news_lines, ensure_ascii=False)}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text.strip()


def parse_gemini_batch_response(text):
    try:
        cleaned = text.strip()

        # Gemini can sometimes wrap JSON into code fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        start = cleaned.find("[")
        end = cleaned.rfind("]")

        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

        data = json.loads(cleaned)

        if not isinstance(data, list):
            return []

        parsed = []
        for item in data:
            try:
                parsed.append({
                    "id": int(item.get("id")),
                    "score": int(item.get("score", 0)),
                    "category": str(item.get("category", "other")),
                    "headline": str(item.get("headline", "")).strip(),
                    "post": str(item.get("post", "")).strip(),
                })
            except Exception:
                continue

        return parsed

    except Exception as e:
        print(f"Gemini batch JSON parse error: {e}")
        print("Raw Gemini response:")
        print(text[:2000])
        return []


def analyze_candidates(candidates):
    prepared = prepare_candidates_for_gemini(candidates)

    if not prepared:
        return []

    try:
        gemini_text = ask_gemini_batch(prepared)
        results = parse_gemini_batch_response(gemini_text)

    except Exception as e:
        print(f"Gemini batch error: {e}")
        return []

    by_id = {item["batch_id"]: item for item in prepared}
    analyzed = []

    for result in results:
        item = by_id.get(result["id"])
        if not item:
            continue

        score = result["score"]

        if score < MIN_SCORE_TO_POST:
            continue

        item["score"] = score
        item["category"] = result["category"]
        item["headline"] = result["headline"] or item["title"]
        item["post_text"] = result["post"]

        print(f"Score: {score} | Category: {item['category']} | Domain: {item['domain']} | {item['headline']}")
        analyzed.append(item)

    analyzed.sort(key=lambda x: x["score"], reverse=True)
    return analyzed


def select_diverse_news(analyzed):
    selected = []
    domain_counts = {}

    for item in analyzed:
        domain = item["domain"]
        count = domain_counts.get(domain, 0)

        if count >= MAX_POSTS_PER_DOMAIN:
            continue

        selected.append(item)
        domain_counts[domain] = count + 1

        if len(selected) >= MAX_NEWS_PER_RUN:
            break

    return selected


# =========================
# TELEGRAM
# =========================

def telegram_api(method, payload):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

    try:
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code != 200:
            print(f"Telegram {method} error:", response.status_code, response.text)

        return response.status_code == 200

    except Exception as e:
        print(f"Telegram request error: {e}")
        return False


def make_post_html(headline, body, source_url, score):
    safe_headline = html.escape(headline.strip())
    safe_body = html.escape(body.strip())
    if score >= 10:
        safe_headline = f"⚡ {safe_headline}"
    safe_source_url = html.escape(source_url.strip(), quote=True)
    safe_about_url = html.escape(ABOUT_POST_URL.strip(), quote=True)
    safe_about_text = html.escape(ABOUT_LINK_TEXT.strip())

    return (
    f"<b>{safe_headline}</b>\n\n"
    f"{safe_body}\n\n"
    f'<a href="{safe_source_url}">Источник</a> | '
    f'<a href="https://t.me/wnd_digest/22">Information</a>'
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
    item["score"]
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

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if missing:
        raise RuntimeError("Missing secrets: " + ", ".join(missing))


def main():
    validate_env()

    published_url_hashes, published_title_hashes, raw_db_lines = load_db()
    sources = load_sources()

    candidates = collect_candidates(
        sources,
        published_url_hashes,
        published_title_hashes,
        raw_db_lines
    )

    print(f"Candidates collected: {len(candidates)}")

    if not candidates:
        print("No new candidates.")
        return

    analyzed = analyze_candidates(candidates)
    selected = select_diverse_news(analyzed)

    print(f"Selected for publishing: {len(selected)}")

    if not selected:
        print("No high-score diverse news found.")
        return

    for index, item in enumerate(selected):
        print(
            f"Publishing {index + 1}/{len(selected)}: "
            f"score {item['score']} | {item['category']} | {item['domain']} | {item['headline']}"
        )

        try:
            if publish_item(item):
                append_published(item)
                print("Published.")
            else:
                append_seen(item, "publish_failed")
                print("Publish failed.")

        except Exception as e:
            append_seen(item, f"publish_exception")
            print(f"Publish item error: {e}")

        sleep_between_posts(index, len(selected))

    print("Done.")


if __name__ == "__main__":
    main()


# Required GitHub Secrets:
# GROQ_API_KEY
# GEMINI_API_KEY
# TELEGRAM_BOT_TOKEN

# Required GitHub Variables:
# GROQ_MODEL=qwen/qwen3-32b
# GEMINI_MODEL=gemini-2.5-flash-lite
# TELEGRAM_CHANNEL_ID=@wnd_main_digest
