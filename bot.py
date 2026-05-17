import os
import re
import time
import random
import hashlib
from urllib.parse import urlparse, urlunparse

import requests
import feedparser
from bs4 import BeautifulSoup
from google import genai

# =========================
# SETTINGS
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

DB_FILE = "urls_base.txt"
SOURCES_FILE = "sources.txt"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "10"))
MIN_SCORE_TO_POST = int(os.getenv("MIN_SCORE_TO_POST", "7"))

POST_DELAY_MIN_SECONDS = int(os.getenv("POST_DELAY_MIN_SECONDS", "60"))
POST_DELAY_MAX_SECONDS = int(os.getenv("POST_DELAY_MAX_SECONDS", "180"))

# How many recent items from each RSS feed to inspect
ITEMS_PER_SOURCE = int(os.getenv("ITEMS_PER_SOURCE", "12"))

# Telegram caption limit is 1024. We keep margin.
TELEGRAM_CAPTION_LIMIT = 1000

CHANNEL_SIGNATURE = os.getenv("CHANNEL_SIGNATURE", "@wnd_digest")

client = genai.Client()


# =========================
# BASIC HELPERS
# =========================
def now_sleep_between_posts(index: int, total: int):
    if index >= total - 1:
        return
    delay = random.randint(POST_DELAY_MIN_SECONDS, POST_DELAY_MAX_SECONDS)
    print(f"Пауза перед следующим постом: {delay} сек")
    time.sleep(delay)


def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)


def normalize_url(url: str) -> str:
    """Remove tracking params and fragments to avoid duplicate URLs."""
    if not url:
        return ""

    parsed = urlparse(url.strip())

    # remove query and fragment completely; most RSS links use tracking params there
    clean = parsed._replace(query="", fragment="")
    normalized = urlunparse(clean)

    # remove trailing slash for stable comparison
    if normalized.endswith("/"):
        normalized = normalized[:-1]

    return normalized


def normalize_title(title: str) -> str:
    title = clean_html(title).lower()
    title = re.sub(r"[^a-zа-яё0-9\s]", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def fingerprint_url(url: str) -> str:
    return "url:" + hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


def fingerprint_title(title: str) -> str:
    return "title:" + hashlib.sha256(normalize_title(title).encode("utf-8")).hexdigest()


def title_words(title: str) -> set:
    words = normalize_title(title).split()
    # remove short/noisy words
    return {w for w in words if len(w) >= 4}


def jaccard_similarity(a: str, b: str) -> float:
    set_a = title_words(a)
    set_b = title_words(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# =========================
# DATABASE
# =========================
def load_database():
    """
    Supports old DB lines too:
    - raw URLs from early versions
    - url:<hash>
    - title:<hash>
    - title_text:<normalized title>
    """
    if not os.path.exists(DB_FILE):
        return {
            "raw": set(),
            "url_hashes": set(),
            "title_hashes": set(),
            "title_texts": [],
        }

    raw = set()
    url_hashes = set()
    title_hashes = set()
    title_texts = []

    with open(DB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue

            raw.add(value)

            if value.startswith("url:"):
                url_hashes.add(value)
            elif value.startswith("title:"):
                title_hashes.add(value)
            elif value.startswith("title_text:"):
                title_texts.append(value.replace("title_text:", "", 1).strip())
            elif value.startswith("http"):
                # migrate old raw URL lines in memory
                url_hashes.add(fingerprint_url(value))

    return {
        "raw": raw,
        "url_hashes": url_hashes,
        "title_hashes": title_hashes,
        "title_texts": title_texts[-500:],
    }


def is_duplicate(item, db) -> bool:
    url_fp = fingerprint_url(item["link"])
    title_fp = fingerprint_title(item["title"])
    normalized_link = normalize_url(item["link"])
    normalized_item_title = normalize_title(item["title"])

    if url_fp in db["url_hashes"]:
        return True

    if title_fp in db["title_hashes"]:
        return True

    # compatibility with old DB that saved raw URLs
    if item["link"] in db["raw"] or normalized_link in db["raw"]:
        return True

    # fuzzy title check for updated/repackaged stories
    for old_title in db["title_texts"][-300:]:
        if jaccard_similarity(normalized_item_title, old_title) >= 0.72:
            return True

    return False


def save_published_item(item):
    normalized_link = normalize_url(item["link"])
    normalized_item_title = normalize_title(item["title"])

    lines = [
        fingerprint_url(normalized_link),
        fingerprint_title(item["title"]),
        "title_text:" + normalized_item_title,
    ]

    with open(DB_FILE, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def save_skipped_item_as_seen(item):
    """Prevents weak/duplicate candidates from being reprocessed every run."""
    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write(fingerprint_url(item["link"]) + "\n")


# =========================
# SOURCES
# =========================
def load_sources():
    if not os.path.exists(SOURCES_FILE):
        raise FileNotFoundError("sources.txt not found")

    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def collect_candidates():
    sources = load_sources()
    candidates_by_source = []

    for rss_url in sources:
        source_items = []
        print(f"Читаю источник: {rss_url}")

        try:
            feed = feedparser.parse(rss_url)
            feed_title = clean_html(feed.feed.get("title", rss_url)) if hasattr(feed, "feed") else rss_url

            for entry in feed.entries[:ITEMS_PER_SOURCE]:
                link = normalize_url(entry.get("link", ""))
                title = clean_html(entry.get("title", ""))
                summary = clean_html(entry.get("summary", entry.get("description", "")))

                if not link or not title:
                    continue

                source_items.append({
                    "source": feed_title,
                    "rss_url": rss_url,
                    "entry": entry,
                    "link": link,
                    "title": title,
                    "summary": summary,
                })

        except Exception as e:
            print(f"Ошибка источника {rss_url}: {e}")

        candidates_by_source.append(source_items)

    # round-robin: BBC item, Guardian item, AP item, etc.
    balanced = []
    max_len = max((len(items) for items in candidates_by_source), default=0)
    for i in range(max_len):
        for items in candidates_by_source:
            if i < len(items):
                balanced.append(items[i])

    return balanced


# =========================
# IMAGE HANDLING
# =========================
def get_opengraph_image(url: str):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WND-NewsBot/1.0; +https://t.me/wnd_digest)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        res = requests.get(url, headers=headers, timeout=12)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        selectors = [
            ("meta", {"property": "og:image:secure_url"}, "content"),
            ("meta", {"property": "og:image"}, "content"),
            ("meta", {"name": "twitter:image"}, "content"),
            ("meta", {"name": "twitter:image:src"}, "content"),
        ]

        for tag, attrs, field in selectors:
            found = soup.find(tag, attrs=attrs)
            if found and found.get(field):
                return found.get(field).strip()

    except Exception as e:
        print(f"OpenGraph image error for {url}: {e}")

    return None


def get_rss_image(entry):
    try:
        if "media_content" in entry and entry.media_content:
            # Prefer biggest image if width exists
            sorted_images = sorted(
                entry.media_content,
                key=lambda x: int(x.get("width", 0) or 0),
                reverse=True,
            )
            for media in sorted_images:
                url = media.get("url")
                width = int(media.get("width", 0) or 0)
                if url and (width == 0 or width >= 600):
                    return url

        if "links" in entry:
            for link in entry.links:
                href = link.get("href", "")
                link_type = link.get("type", "")
                rel = link.get("rel", "")
                if href and ("image" in link_type or rel == "enclosure"):
                    return href

        if "media_thumbnail" in entry and entry.media_thumbnail:
            # Usually low quality; use only as last resort if URL doesn't look like a tiny thumb
            thumb = entry.media_thumbnail[0].get("url")
            if thumb and not re.search(r"(thumb|thumbnail|small|150x|120x|100x|80x)", thumb, re.I):
                return thumb
    except Exception as e:
        print(f"RSS image error: {e}")

    return None


def image_is_usable(image_url: str) -> bool:
    if not image_url:
        return False

    if re.search(r"(logo|avatar|icon|sprite|placeholder)", image_url, re.I):
        return False

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WND-NewsBot/1.0)",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
        res = requests.get(image_url, headers=headers, timeout=15, stream=True)
        if res.status_code != 200:
            return False

        content_type = res.headers.get("content-type", "").lower()
        if "image" not in content_type:
            return False

        content_length = int(res.headers.get("content-length", "0") or 0)
        # Low size often means thumbnail/logo. If header missing, don't reject.
        if content_length and content_length < 45_000:
            return False

        return True
    except Exception as e:
        print(f"Image validation error: {e}")
        return False


def get_best_image(item):
    # Prefer original article's high-quality OpenGraph image
    candidates = [
        get_opengraph_image(item["link"]),
        get_rss_image(item["entry"]),
    ]

    for image in candidates:
        if image and image_is_usable(image):
            return image

    return None


# =========================
# GEMINI
# =========================
def ask_gemini(item):
    prompt = f"""
Ты редактор русскоязычного Telegram-канала WND | Мировой дайджест.

Твоя задача:
1. Оцени важность новости от 1 до 10 для русскоязычной международной аудитории.
2. Приоритет: война, дипломатия, СНГ, экономика, санкции, технологии, AI, крупные катастрофы, выборы, протесты, решения правительств, мировые рынки.
3. Понижай оценку для локальных мелких новостей, криминала без международного значения, celebrity, спорта, повторов, слабых обновлений.
4. Сделай цепляющий, но не кликбейтный русский заголовок.
5. Переведи и сожми новость на русский.
6. Не выдумывай факты. Используй только заголовок и описание.
7. Стиль Telegram: коротко, ясно, живо. Можно использовать эмодзи и списки.
8. Не используй Markdown кроме жирного заголовка через HTML-тег <b>...</b>.
9. Не ставь ссылку на источник внутри текста — бот добавит её сам.

Верни ответ строго в таком формате:
SCORE: число от 1 до 10
HEADLINE: короткий русский заголовок
POST:
короткий текст новости, 2-5 строк или список

Источник: {item['source']}
Оригинальный заголовок: {item['title']}
Описание: {item['summary']}
Ссылка: {item['link']}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def parse_gemini_response(text):
    score = 0
    headline = ""
    post = text.strip()

    try:
        score_match = re.search(r"SCORE:\s*(\d+)", text, re.I)
        if score_match:
            score = int(score_match.group(1))

        headline_match = re.search(r"HEADLINE:\s*(.+)", text, re.I)
        if headline_match:
            headline = headline_match.group(1).strip()

        post_match = re.search(r"POST:\s*(.*)", text, re.I | re.S)
        if post_match:
            post = post_match.group(1).strip()

    except Exception as e:
        print(f"Gemini parse error: {e}")

    headline = re.sub(r"<\/?b>", "", headline).strip()
    post = re.sub(r"SCORE:.*", "", post).strip()
    post = re.sub(r"HEADLINE:.*", "", post).strip()

    return score, headline, post


# =========================
# TELEGRAM
# =========================
def truncate_for_caption(text: str) -> str:
    if len(text) <= TELEGRAM_CAPTION_LIMIT:
        return text
    return text[:TELEGRAM_CAPTION_LIMIT - 3].rstrip() + "..."


def build_post_text(headline, body, source_link):
    safe_headline = headline.strip() or "Главная новость"
    body = body.strip()

    return (
        f"<b>{safe_headline}</b>\n\n"
        f"{body}\n\n"
        f"Источник: {source_link}\n"
        f"Подписаться: {CHANNEL_SIGNATURE}"
    )


def send_photo_with_caption(image_url, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "photo": image_url,
        "caption": truncate_for_caption(text),
        "parse_mode": "HTML",
    }
    res = requests.post(url, json=payload, timeout=25)
    if res.status_code != 200:
        print("Telegram sendPhoto error:", res.status_code, res.text)
    return res.status_code == 200


def send_text(text, allow_preview=True):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": not allow_preview,
    }
    res = requests.post(url, json=payload, timeout=25)
    if res.status_code != 200:
        print("Telegram sendMessage error:", res.status_code, res.text)
    return res.status_code == 200


def publish(item, headline, body):
    text = build_post_text(headline, body, item["link"])
    image_url = get_best_image(item)

    if image_url:
        ok = send_photo_with_caption(image_url, text)
        if ok:
            return True

        print("Фото не отправилось, fallback на текст с preview")

    # If no good image, let Telegram generate preview from source URL.
    return send_text(text, allow_preview=True)


# =========================
# MAIN
# =========================
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ОШИБКА: TELEGRAM_BOT_TOKEN не найден")
        return
    if not TELEGRAM_CHANNEL_ID:
        print("ОШИБКА: TELEGRAM_CHANNEL_ID не найден")
        return

    db = load_database()
    candidates = collect_candidates()

    unique_candidates = []
    seen_this_run_url = set()
    seen_this_run_titles = []

    for item in candidates:
        url_fp = fingerprint_url(item["link"])
        norm_title = normalize_title(item["title"])

        if url_fp in seen_this_run_url:
            continue

        if is_duplicate(item, db):
            print(f"Дубль пропущен: {item['title']}")
            continue

        if any(jaccard_similarity(norm_title, old) >= 0.72 for old in seen_this_run_titles):
            print(f"Похожая новость в этом запуске пропущена: {item['title']}")
            continue

        seen_this_run_url.add(url_fp)
        seen_this_run_titles.append(norm_title)
        unique_candidates.append(item)

    print(f"Кандидатов после антидублей: {len(unique_candidates)}")

    scored_items = []

    for item in unique_candidates:
        try:
            print(f"Оценка Gemini: {item['title']}")
            gemini_text = ask_gemini(item)
            score, headline, post = parse_gemini_response(gemini_text)
            print(f"Score: {score} | {headline}")

            if score >= MIN_SCORE_TO_POST:
                scored_items.append({
                    "item": item,
                    "score": score,
                    "headline": headline,
                    "post": post,
                })
            else:
                # Save as seen so low-value news doesn't get evaluated forever
                save_skipped_item_as_seen(item)

            time.sleep(1)

        except Exception as e:
            print(f"Ошибка Gemini для новости: {e}")

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    to_publish = scored_items[:MAX_NEWS_PER_RUN]

    print(f"Готово к публикации: {len(to_publish)} из {len(scored_items)} хороших новостей")

    for index, pack in enumerate(to_publish):
        item = pack["item"]
        try:
            print(f"Публикация #{index + 1}: score {pack['score']} | {pack['headline']}")
            if publish(item, pack["headline"], pack["post"]):
                save_published_item(item)
                print("Опубликовано и сохранено в базу")
            else:
                print("Не удалось опубликовать")

            now_sleep_between_posts(index, len(to_publish))

        except Exception as e:
            print(f"Ошибка публикации: {e}")


if __name__ == "__main__":
    main()
