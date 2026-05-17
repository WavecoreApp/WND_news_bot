import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

import feedparser
import requests
from bs4 import BeautifulSoup
from google import genai

# =========================
# SETTINGS FROM GITHUB
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Stable model, good for summarization/translation.
# You can override it in GitHub: Settings -> Secrets and variables -> Actions -> Variables.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "3") or "3")
MIN_SCORE_TO_POST = int(os.getenv("MIN_SCORE_TO_POST", "8") or "8")

DB_FILE = "urls_base.txt"
SOURCES_FILE = "sources.txt"

REQUEST_TIMEOUT = 15
POST_DELAY_SECONDS = 4

client = genai.Client(api_key=GEMINI_API_KEY)


def load_published_ids() -> Set[str]:
    if not os.path.exists(DB_FILE):
        return set()
    with open(DB_FILE, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def save_published_id(news_id: str) -> None:
    with open(DB_FILE, "a", encoding="utf-8") as file:
        file.write(news_id + "\n")


def load_sources() -> List[str]:
    if not os.path.exists(SOURCES_FILE):
        raise FileNotFoundError(f"{SOURCES_FILE} not found")
    with open(SOURCES_FILE, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]


def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def normalize_url(url: str) -> str:
    if not url:
        return ""
    # Keep it simple: remove tracking query parameters that often create duplicate URLs.
    return url.split("?", 1)[0].strip()


def make_news_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def get_entry_image(entry: Any) -> Optional[str]:
    # RSS media extensions
    media_content = entry.get("media_content")
    if media_content and isinstance(media_content, list):
        url = media_content[0].get("url")
        if url:
            return url

    media_thumbnail = entry.get("media_thumbnail")
    if media_thumbnail and isinstance(media_thumbnail, list):
        url = media_thumbnail[0].get("url")
        if url:
            return url

    # Enclosure links
    links = entry.get("links", [])
    for link in links:
        href = link.get("href")
        link_type = link.get("type", "")
        rel = link.get("rel", "")
        if href and ("image" in link_type or rel == "enclosure"):
            return href

    return None


def get_opengraph_image(url: str) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 TelegramNewsBot/1.0"}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        selectors = [
            ("meta", {"property": "og:image"}),
            ("meta", {"name": "twitter:image"}),
            ("meta", {"property": "twitter:image"}),
        ]
        for tag_name, attrs in selectors:
            tag = soup.find(tag_name, attrs=attrs)
            if tag and tag.get("content"):
                return tag["content"].strip()
    except Exception as exc:
        print(f"OpenGraph image error for {url}: {exc}")
    return None


def get_best_image(entry: Any, url: str) -> Optional[str]:
    return get_entry_image(entry) or get_opengraph_image(url)


def collect_candidates(published_ids: Set[str]) -> List[Dict[str, Any]]:
    sources = load_sources()
    candidates: List[Dict[str, Any]] = []
    seen_this_run: Set[str] = set()

    for rss_url in sources:
        print(f"Reading source: {rss_url}")
        try:
            feed = feedparser.parse(rss_url)
            if getattr(feed, "bozo", 0):
                print(f"RSS warning for {rss_url}: {getattr(feed, 'bozo_exception', '')}")

            for entry in feed.entries[:12]:
                url = normalize_url(entry.get("link", ""))
                if not url:
                    continue

                news_id = make_news_id(url)
                if news_id in published_ids or news_id in seen_this_run:
                    continue

                title = clean_html(entry.get("title", ""))
                summary = clean_html(entry.get("summary", entry.get("description", "")))
                published = clean_html(entry.get("published", entry.get("updated", "")))

                if not title:
                    continue

                seen_this_run.add(news_id)
                candidates.append(
                    {
                        "id": news_id,
                        "url": url,
                        "title": title,
                        "summary": summary,
                        "published": published,
                        "entry": entry,
                    }
                )
        except Exception as exc:
            print(f"Source error {rss_url}: {exc}")

    return candidates


def ask_gemini(title: str, summary: str, published: str, url: str) -> Dict[str, Any]:
    prompt = f"""
Ты редактор Telegram-канала с главными мировыми новостями на русском языке.

Нужно оценить новость и подготовить короткий красивый Telegram-пост.

Правила:
- Используй только факты из входных данных. Ничего не додумывай.
- Пиши на русском.
- Сжимай новость: не длиннее 900 символов.
- Стиль: ясно, живо, без канцелярита и без кликбейта.
- Можно использовать 2–4 уместных эмодзи.
- Можно использовать короткий список, если это помогает.
- Не используй Markdown-разметку: никаких **, __, ###.
- Если новость не очень важная для международной аудитории, поставь низкий score.
- Важные темы: войны, дипломатия, политика, крупная экономика, технологии, общественные кризисы, катастрофы, международные решения.

Верни строго JSON без текста вокруг:
{{
  "score": 1-10,
  "headline_ru": "короткий заголовок на русском",
  "post_ru": "готовый текст поста для Telegram"
}}

Входные данные:
Title: {title}
Summary: {summary}
Published: {published}
URL: {url}
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.4,
        },
    )

    raw = (response.text or "").strip()
    print(f"Gemini raw response: {raw[:500]}")

    try:
        data = json.loads(raw)
        return {
            "score": int(data.get("score", 0)),
            "headline_ru": str(data.get("headline_ru", "")).strip(),
            "post_ru": str(data.get("post_ru", "")).strip(),
        }
    except Exception:
        # Fallback if model returns imperfect JSON.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "score": int(data.get("score", 0)),
                "headline_ru": str(data.get("headline_ru", "")).strip(),
                "post_ru": str(data.get("post_ru", "")).strip(),
            }
        raise ValueError(f"Could not parse Gemini JSON: {raw}")


def telegram_request(method: str, payload: Dict[str, Any]) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"Telegram {method} error: {response.status_code} {response.text}")
            return False
        return True
    except Exception as exc:
        print(f"Telegram {method} exception: {exc}")
        return False


def send_photo(image_url: str) -> bool:
    return telegram_request(
        "sendPhoto",
        {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "photo": image_url,
        },
    )


def send_message(text: str) -> bool:
    return telegram_request(
        "sendMessage",
        {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "text": text,
            "disable_web_page_preview": False,
        },
    )


def publish(post_ru: str, source_url: str, image_url: Optional[str]) -> bool:
    text = f"{post_ru}\n\nИсточник: {source_url}"

    # Best visual result: image first, then full text.
    # If Telegram cannot fetch the image, we still publish the text with link preview.
    if image_url:
        photo_ok = send_photo(image_url)
        print(f"Photo sent: {photo_ok}")
        time.sleep(1)

    return send_message(text)


def validate_settings() -> bool:
    ok = True
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN is missing")
        ok = False
    if not TELEGRAM_CHANNEL_ID:
        print("ERROR: TELEGRAM_CHANNEL_ID is missing")
        ok = False
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is missing")
        ok = False
    return ok


def main() -> None:
    if not validate_settings():
        return

    published_ids = load_published_ids()
    candidates = collect_candidates(published_ids)
    print(f"New candidates found: {len(candidates)}")

    posted_count = 0

    for item in candidates:
        if posted_count >= MAX_NEWS_PER_RUN:
            break

        print(f"Processing: {item['title']}")

        try:
            result = ask_gemini(
                title=item["title"],
                summary=item["summary"],
                published=item["published"],
                url=item["url"],
            )

            score = result["score"]
            post_ru = result["post_ru"]
            print(f"Score: {score}")

            # Mark low-quality/low-importance items as seen so the bot does not evaluate them forever.
            if score < MIN_SCORE_TO_POST:
                print("Skipped: low score")
                save_published_id(item["id"])
                continue

            if not post_ru:
                print("Skipped: empty Gemini post")
                save_published_id(item["id"])
                continue

            image_url = get_best_image(item["entry"], item["url"])
            success = publish(post_ru=post_ru, source_url=item["url"], image_url=image_url)

            if success:
                save_published_id(item["id"])
                posted_count += 1
                print("Published successfully")
                time.sleep(POST_DELAY_SECONDS)
            else:
                print("Publish failed; item was not marked as published")

        except Exception as exc:
            print(f"Processing error: {exc}")

    print(f"Done. Published: {posted_count}")


if __name__ == "__main__":
    main()
