import os
import re
import time
import html
import random
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

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MIN_SCORE_TO_POST = int(os.getenv("MIN_SCORE_TO_POST", "7"))
MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "10"))

POST_DELAY_MIN_SECONDS = int(os.getenv("POST_DELAY_MIN_SECONDS", "60"))
POST_DELAY_MAX_SECONDS = int(os.getenv("POST_DELAY_MAX_SECONDS", "180"))

# Вставь сюда ссылку на твой info/about пост в канале.
# Пример: https://t.me/wnd_digest/15
ABOUT_POST_URL = os.getenv("ABOUT_POST_URL", "https://t.me/wnd_digest")

# Текст ссылки внизу каждого поста
ABOUT_LINK_TEXT = os.getenv("ABOUT_LINK_TEXT", "ℹ️ О проекте WND")

# Сколько новостей брать максимум из каждого RSS за один запуск
ENTRIES_PER_SOURCE = int(os.getenv("ENTRIES_PER_SOURCE", "10"))

# Чтобы не гонять Gemini по сотням новостей за один запуск
MAX_CANDIDATES_TO_ANALYZE = int(os.getenv("MAX_CANDIDATES_TO_ANALYZE", "60"))


client = genai.Client()


# =========================
# DATABASE / DUPLICATES
# =========================

def load_db():
    """
    Поддерживает старые и новые записи:
    - старые голые URL
    - новые строки urlhash:...
    - titlehash:...
    """
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
            else:
                # старый формат: URL или random строка
                raw_lines.add(item)
                if item.startswith("http"):
                    url_hashes.add(hash_text(normalize_url(item)))

    return url_hashes, title_hashes, raw_lines


def append_db(url_hash, title_hash, normalized_url, normalized_title):
    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write(f"urlhash:{url_hash}\n")
        f.write(f"titlehash:{title_hash}\n")
        f.write(f"url:{normalized_url}\n")
        f.write(f"title:{normalized_title}\n")


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_url(url):
    if not url:
        return ""

    url = url.strip()
    parsed = urlparse(url)

    # Убираем tracking-параметры
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

    path = parsed.path.rstrip("/")
    clean = parsed._replace(
        scheme=parsed.scheme.lower() or "https",
        netloc=parsed.netloc.lower(),
        path=path,
        query=urlencode(query, doseq=True),
        fragment=""
    )

    return urlunparse(clean)


def normalize_title(title):
    title = clean_text(title).lower()
    title = re.sub(r"[^\wа-яёіїєґ\s]", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def is_similar_to_published_title(title, published_title_hashes, raw_lines):
    """
    Быстрая проверка:
    1. title hash
    2. fuzzy similarity against old saved title:... lines
    """
    normalized = normalize_title(title)
    title_hash = hash_text(normalized)

    if title_hash in published_title_hashes:
        return True

    old_titles = [
        line.replace("title:", "", 1)
        for line in raw_lines
        if line.startswith("title:")
    ]

    for old in old_titles[-500:]:
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


def clean_text(text):
    if not text:
        return ""

    soup = BeautifulSoup(str(text), "html.parser")
    return soup.get_text(" ", strip=True)


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

                url_hash = hash_text(link)
                normalized_title = normalize_title(title)
                title_hash = hash_text(normalized_title)

                if url_hash in published_url_hashes:
                    continue

                if title_hash in published_title_hashes:
                    continue

                if is_similar_to_published_title(title, published_title_hashes, raw_db_lines):
                    continue

                source_items.append({
                    "source_rss": rss_url,
                    "entry": entry,
                    "link": link,
                    "url_hash": url_hash,
                    "title": title,
                    "summary": summary,
                    "normalized_title": normalized_title,
                    "title_hash": title_hash,
                })

            all_by_source.append(source_items)

        except Exception as e:
            print(f"Source error {rss_url}: {e}")

    # Балансировка: берём по одной новости из каждого источника по кругу
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

    # Сначала выбираем по размеру, если он есть
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
    # Сначала страница статьи: чаще всего там image лучше, чем RSS thumbnail
    og = extract_og_image(article_url)
    if og:
        return og

    return extract_rss_image(entry)


# =========================
# GEMINI
# =========================

def ask_gemini(title, description, source_link):
    prompt = f"""
Ты редактор Telegram-канала WND | Мировой дайджест.

Задача:
1. Оцени важность новости от 1 до 10.
2. Публиковать стоит только новости, интересные русскоязычной международной аудитории.
3. Приоритет: мировая политика, войны, дипломатия, экономика, технологии, AI, крупные катастрофы, СНГ, Европа, США, Ближний Восток, Китай.
4. Не публикуй мелкие локальные новости, celebrity gossip, спорт без большого мирового значения, мелкую партийную рутину.
5. Сделай цепляющий русский заголовок. Не кликбейт, но живой и сильный.
6. Сделай короткую выжимку на русском.
7. Не выдумывай факты. Используй только исходные данные.
8. Стиль Telegram: коротко, ясно, можно использовать 1–3 уместных эмодзи.
9. Можно использовать списки с символом •.
10. Не используй Markdown.
11. Не вставляй ссылки в текст.

Верни строго в таком формате:

SCORE: число от 1 до 10
HEADLINE: русский заголовок
POST:
короткий текст новости без заголовка

Исходная новость:
Title: {title}
Description: {description}
Link: {source_link}
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
        score_match = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
        headline_match = re.search(r"HEADLINE:\s*(.+)", text, re.IGNORECASE)

        if score_match:
            score = int(score_match.group(1))

        if headline_match:
            headline = headline_match.group(1).strip()

        if "POST:" in text:
            post = text.split("POST:", 1)[1].strip()

        # Убираем случайно оставшиеся служебные строки
        post = re.sub(r"^HEADLINE:.*$", "", post, flags=re.MULTILINE | re.IGNORECASE).strip()
        post = re.sub(r"^SCORE:.*$", "", post, flags=re.MULTILINE | re.IGNORECASE).strip()

    except Exception:
        pass

    return score, headline, post


def analyze_candidates(candidates):
    analyzed = []

    for item in candidates:
        print(f"Analyzing: {item['title']}")

        try:
            gemini_text = ask_gemini(
                item["title"],
                item["summary"],
                item["link"]
            )

            score, headline, post_text = parse_gemini_response(gemini_text)

            print(f"Score: {score} | {headline}")

            if score < MIN_SCORE_TO_POST:
                continue

            if not headline:
                headline = item["title"]

            item["score"] = score
            item["headline"] = headline
            item["post_text"] = post_text
            analyzed.append(item)

        except Exception as e:
            print(f"Gemini error: {e}")

    # Самые важные — первыми. При равном score сохраняем исходный balanced order.
    analyzed.sort(key=lambda x: x["score"], reverse=True)

    return analyzed[:MAX_NEWS_PER_RUN]


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


def make_post_html(headline, body, source_url):
    safe_headline = html.escape(headline.strip())
    safe_body = html.escape(body.strip())
    safe_source_url = html.escape(source_url.strip(), quote=True)
    safe_about_url = html.escape(ABOUT_POST_URL.strip(), quote=True)
    safe_about_text = html.escape(ABOUT_LINK_TEXT.strip())

    return (
        f"<b>{safe_headline}</b>\n\n"
        f"{safe_body}\n\n"
        f'<a href="{safe_source_url}">Источник</a>\n\n'
        f"—\n"
        f'<a href="{safe_about_url}">{safe_about_text}</a>'
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
        item["link"]
    )

    image_url = get_best_image(item["entry"], item["link"])

    # Telegram caption limit is 1024 chars.
    # Если текст длинный — отправляем текст без фото, но с preview ссылки.
    if image_url and len(text_html) <= 1000:
        ok = send_photo_with_caption(image_url, text_html)
        if ok:
            return True

        print("Photo failed, fallback to text with link preview.")

    # Если фото плохое/не отправилось/текст слишком длинный — пусть Telegram подтянет preview из ссылки "Источник".
    ok = send_text_with_preview(text_html)
    if ok:
        return True

    # Последний fallback — без preview.
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

    selected = analyze_candidates(candidates)

    print(f"Selected for publishing: {len(selected)}")

    if not selected:
        print("No high-score news found.")
        return

    for index, item in enumerate(selected):
        print(f"Publishing {index + 1}/{len(selected)}: score {item['score']} | {item['headline']}")

        try:
            if publish_item(item):
                append_db(
                    item["url_hash"],
                    item["title_hash"],
                    item["link"],
                    item["normalized_title"]
                )
                print("Published.")
            else:
                print("Publish failed.")

        except Exception as e:
            print(f"Publish item error: {e}")

        sleep_between_posts(index, len(selected))

    print("Done.")


if __name__ == "__main__":
    main()
