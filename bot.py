import hashlib
import json
import os
import random
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

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

# Now the bot is quality-based, not "only 3 best".
# It checks many candidates and posts every strong one, up to this safety cap.
MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "20") or "20")
MIN_SCORE_TO_POST = int(os.getenv("MIN_SCORE_TO_POST", "8") or "8")
MAX_ITEMS_PER_SOURCE = int(os.getenv("MAX_ITEMS_PER_SOURCE", "8") or "8")
MAX_CANDIDATES_TO_CHECK = int(os.getenv("MAX_CANDIDATES_TO_CHECK", "60") or "60")

# Caption under photo has a Telegram limit. Keep the AI post compact.
MAX_CAPTION_LENGTH = 1024
MAX_AI_POST_LENGTH = 760

DB_FILE = "urls_base.txt"
SOURCES_FILE = "sources.txt"

REQUEST_TIMEOUT = 15
POST_DELAY_SECONDS = 3

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
        return [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]


def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(str(text), "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def normalize_url(url: str) -> str:
    if not url:
        return ""
    return url.split("?", 1)[0].strip()


def make_news_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def get_entry_image(entry: Any) -> Optional[str]:
    media_content = entry.get("media_content")
    if media_content and isinstance(media_content, list):
        for item in media_content:
            url = item.get("url")
            if url:
                return url

    media_thumbnail = entry.get("media_thumbnail")
    if media_thumbnail and isinstance(media_thumbnail, list):
        for item in media_thumbnail:
            url = item.get("url")
            if url:
                return url

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
    by_source: List[List[Dict[str, Any]]] = []
    seen_this_run: Set[str] = set()

    for rss_url in sources:
        print(f"Reading source: {rss_url}")
        source_items: List[Dict[str, Any]] = []

        try:
            feed = feedparser.parse(rss_url)
            if getattr(feed, "bozo", 0):
                print(f"RSS warning for {rss_url}: {getattr(feed, 'bozo_exception', '')}")

            source_name = clean_html(feed.feed.get("title", rss_url))

            for entry in feed.entries[:MAX_ITEMS_PER_SOURCE]:
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
                source_items.append(
                    {
                        "id": news_id,
                        "url": url,
                        "title": title,
                        "summary": summary,
                        "published": published,
                        "source_name": source_name,
                        "entry": entry,
                    }
                )
        except Exception as exc:
            print(f"Source error {rss_url}: {exc}")

        if source_items:
            by_source.append(source_items)

    # Round-robin mixing: BBC cannot consume the whole run before other sources are checked.
    candidates: List[Dict[str, Any]] = []
    index = 0
    while len(candidates) < MAX_CANDIDATES_TO_CHECK:
        added = False
        for source_items in by_source:
            if index < len(source_items):
                candidates.append(source_items[index])
                added = True
                if len(candidates) >= MAX_CANDIDATES_TO_CHECK:
                    break
        if not added:
            break
        index += 1

    return candidates


def ask_gemini(title: str, summary: str, published: str, source_name: str, url: str) -> Dict[str, Any]:
    prompt = f"""
Ты редактор Telegram-канала «WND | Мировой дайджест». Канал публикует главные мировые новости на русском.

Твоя задача: оценить новость и сделать компактный Telegram-пост ПОД ФОТО.

Правила оценки:
- 9-10: очень важная международная новость, война, дипломатия, крупная экономика, санкции, кризис, катастрофа, AI/Big Tech с мировым эффектом, важные события СНГ/Европы/США/Ближнего Востока.
- 8: хорошая важная новость для широкой аудитории.
- 6-7: нормальная, но не обязательная.
- 1-5: локальная, мелкая, повторяющаяся, celebrity, спорт без мирового эффекта, слабый инфоповод.

Правила текста:
- Используй только факты из входных данных. Ничего не додумывай.
- Пиши на русском.
- Сжимай: максимум {MAX_AI_POST_LENGTH} символов.
- Стиль Telegram: живо, ясно, без кликбейта и без канцелярита.
- Можно использовать 2-4 уместных эмодзи.
- Можно использовать короткие пункты через •, если это реально помогает.
- Не используй Markdown: никаких **, __, ###.
- Не добавляй ссылку на источник внутри post_ru — бот добавит её сам.
- Не добавляй «подписывайтесь» внутри post_ru — бот добавит это сам.

Верни строго JSON без текста вокруг:
{{
  "score": 1-10,
  "headline_ru": "короткий заголовок на русском",
  "post_ru": "готовый текст поста"
}}

Входные данные:
Source: {source_name}
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
            "temperature": 0.35,
        },
    )

    raw = (response.text or "").strip()
    print(f"Gemini raw response: {raw[:500]}")

    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse Gemini JSON: {raw}")
        data = json.loads(match.group(0))

    return {
        "score": int(data.get("score", 0)),
        "headline_ru": str(data.get("headline_ru", "")).strip(),
        "post_ru": str(data.get("post_ru", "")).strip(),
    }


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


def build_caption(post_ru: str, source_url: str) -> str:
    footer = f"\n\nИсточник: {source_url}\n\nПодписаться: {TELEGRAM_CHANNEL_ID}"
    available = MAX_CAPTION_LENGTH - len(footer)

    post_ru = post_ru.strip()
    if len(post_ru) > available:
        post_ru = post_ru[: max(0, available - 1)].rstrip() + "…"

    return post_ru + footer


def send_photo_with_caption(image_url: str, caption: str) -> bool:
    return telegram_request(
        "sendPhoto",
        {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "photo": image_url,
            "caption": caption,
        },
    )


def send_message_no_preview(text: str) -> bool:
    # If there is no photo, we disable link preview so Telegram does not add a random duplicate picture below.
    return telegram_request(
        "sendMessage",
        {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
    )


def publish(post_ru: str, source_url: str, image_url: Optional[str]) -> bool:
    caption = build_caption(post_ru, source_url)

    if image_url:
        ok = send_photo_with_caption(image_url, caption)
        if ok:
            return True
        print("Photo with caption failed. Sending text without preview instead.")

    return send_message_no_preview(caption)


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
    checked_count = 0

    for item in candidates:
        if posted_count >= MAX_NEWS_PER_RUN:
            print(f"Safety cap reached: MAX_NEWS_PER_RUN={MAX_NEWS_PER_RUN}")
            break

        checked_count += 1
        print(f"Processing: {item['title']}")

        try:
            result = ask_gemini(
                title=item["title"],
                summary=item["summary"],
                published=item["published"],
                source_name=item["source_name"],
                url=item["url"],
            )

            score = result["score"]
            post_ru = result["post_ru"]
            print(f"Score: {score}")

            # Mark checked low-quality items as seen so the bot does not waste future runs on them.
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

    print(f"Done. Checked: {checked_count}. Published: {posted_count}")


if __name__ == "__main__":
    main()
