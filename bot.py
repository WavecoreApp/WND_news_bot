import hashlib
import html
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

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

# Main posting logic:
# - evaluate candidates from all RSS sources
# - keep only news with score >= MIN_SCORE_TO_POST
# - sort by score from highest to lowest
# - publish up to MAX_NEWS_PER_RUN
MAX_NEWS_PER_RUN = int(os.getenv("MAX_NEWS_PER_RUN", "10") or "10")
MIN_SCORE_TO_POST = int(os.getenv("MIN_SCORE_TO_POST", "7") or "7")
MAX_ITEMS_PER_SOURCE = int(os.getenv("MAX_ITEMS_PER_SOURCE", "8") or "8")
MAX_CANDIDATES_TO_CHECK = int(os.getenv("MAX_CANDIDATES_TO_CHECK", "60") or "60")

# Random delay between posts, so the channel does not look like spam.
# Default: 1-3 minutes.
POST_DELAY_MIN_SECONDS = int(os.getenv("POST_DELAY_MIN_SECONDS", "60") or "60")
POST_DELAY_MAX_SECONDS = int(os.getenv("POST_DELAY_MAX_SECONDS", "180") or "180")

# Caption under photo has a Telegram limit. Keep the AI post compact.
MAX_CAPTION_LENGTH = 1024
MAX_AI_POST_LENGTH = 660
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "700") or "700")
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "350") or "350")
MIN_IMAGE_CONTENT_LENGTH = int(os.getenv("MIN_IMAGE_CONTENT_LENGTH", "90000") or "90000")

DB_FILE = "urls_base.txt"
SOURCES_FILE = "sources.txt"

REQUEST_TIMEOUT = 15

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


def parse_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(str(value).strip())
    except Exception:
        return None


def image_url_looks_small(url: str) -> bool:
    lowered = url.lower()
    bad_markers = [
        "thumbnail", "thumb", "small", "60x", "80x", "100x", "120x",
        "150x", "180x", "200x", "240x", "300x", "320x"
    ]
    return any(marker in lowered for marker in bad_markers)


def image_has_enough_size(url: str, width: Optional[int] = None, height: Optional[int] = None) -> bool:
    # If the page gives real OpenGraph dimensions, use them.
    if width and height:
        return width >= MIN_IMAGE_WIDTH and height >= MIN_IMAGE_HEIGHT

    # Obvious thumbnails are rejected.
    if image_url_looks_small(url):
        return False

    # If dimensions are missing, use content-length as a rough quality check.
    # This avoids many tiny RSS thumbnails without downloading the whole image.
    try:
        headers = {"User-Agent": "Mozilla/5.0 TelegramNewsBot/1.0"}
        response = requests.head(url, allow_redirects=True, headers=headers, timeout=REQUEST_TIMEOUT)
        content_type = response.headers.get("content-type", "").lower()
        content_length = parse_int(response.headers.get("content-length"))

        if content_type and "image" not in content_type:
            return False

        if content_length is not None:
            return content_length >= MIN_IMAGE_CONTENT_LENGTH

        # If server does not provide content-length, allow it unless URL looks like a thumbnail.
        return True
    except Exception as exc:
        print(f"Image HEAD check failed for {url}: {exc}")
        return False


def get_entry_images(entry: Any) -> List[Tuple[str, Optional[int], Optional[int]]]:
    images: List[Tuple[str, Optional[int], Optional[int]]] = []

    media_content = entry.get("media_content")
    if media_content and isinstance(media_content, list):
        for item in media_content:
            url = item.get("url")
            if url:
                images.append((url.strip(), parse_int(item.get("width")), parse_int(item.get("height"))))

    media_thumbnail = entry.get("media_thumbnail")
    if media_thumbnail and isinstance(media_thumbnail, list):
        for item in media_thumbnail:
            url = item.get("url")
            if url:
                images.append((url.strip(), parse_int(item.get("width")), parse_int(item.get("height"))))

    links = entry.get("links", [])
    for link in links:
        href = link.get("href")
        link_type = link.get("type", "")
        rel = link.get("rel", "")
        if href and ("image" in link_type or rel == "enclosure"):
            images.append((href.strip(), parse_int(link.get("width")), parse_int(link.get("height"))))

    return images


def get_opengraph_image(url: str) -> Optional[Tuple[str, Optional[int], Optional[int]]]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 TelegramNewsBot/1.0"}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        image_url = None
        for tag_name, attrs in [
            ("meta", {"property": "og:image:secure_url"}),
            ("meta", {"property": "og:image"}),
            ("meta", {"name": "twitter:image"}),
            ("meta", {"property": "twitter:image"}),
        ]:
            tag = soup.find(tag_name, attrs=attrs)
            if tag and tag.get("content"):
                image_url = urljoin(url, tag["content"].strip())
                break

        if not image_url:
            return None

        width = None
        height = None
        width_tag = soup.find("meta", property="og:image:width")
        height_tag = soup.find("meta", property="og:image:height")
        if width_tag and width_tag.get("content"):
            width = parse_int(width_tag["content"])
        if height_tag and height_tag.get("content"):
            height = parse_int(height_tag["content"])

        return image_url, width, height
    except Exception as exc:
        print(f"OpenGraph image error for {url}: {exc}")
    return None


def get_best_image(entry: Any, url: str) -> Optional[str]:
    # Prefer the article page OpenGraph image. RSS images are often small thumbnails.
    og_image = get_opengraph_image(url)
    if og_image:
        image_url, width, height = og_image
        if image_has_enough_size(image_url, width, height):
            print(f"Using OpenGraph image: {image_url}")
            return image_url
        print(f"Rejected low-quality OpenGraph image: {image_url} ({width}x{height})")

    # Fallback to RSS images only if they look good enough.
    for image_url, width, height in get_entry_images(entry):
        image_url = urljoin(url, image_url)
        if image_has_enough_size(image_url, width, height):
            print(f"Using RSS image: {image_url}")
            return image_url
        print(f"Rejected low-quality RSS image: {image_url} ({width}x{height})")

    return None


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

    # Round-robin mixing: one source cannot consume the whole evaluation list.
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

Твоя задача: оценить новость, придумать сильный русский заголовок и сделать компактный Telegram-пост ПОД ФОТО.

Правила оценки:
- 10: топовая мировая новость дня. Очень важное событие с международными последствиями.
- 9: очень важная международная новость: война, дипломатия, крупная экономика, санкции, кризис, катастрофа, AI/Big Tech с мировым эффектом, важные события СНГ/Европы/США/Ближнего Востока.
- 8: важная новость для широкой аудитории.
- 7: интересная и полезная новость, которую можно публиковать, если она не локальная мелочь.
- 1-6: локальная, мелкая, повторяющаяся, celebrity, спорт без мирового эффекта, слабый инфоповод.

Правила текста:
- Используй только факты из входных данных. Ничего не додумывай.
- Пиши на русском.
- Сжимай: максимум {MAX_AI_POST_LENGTH} символов.
- Стиль Telegram: живо, ясно, без кликбейта и без канцелярита.
- Можно использовать 2-4 уместных эмодзи.
- Можно использовать короткие пункты через •, если это реально помогает.
- Заголовок headline_ru должен быть цепляющим, но без кликбейта. Опирайся на оригинальный заголовок статьи.
- Не используй Markdown: никаких **, __, ###.
- Не добавляй ссылку на источник внутри post_ru — бот добавит её сам.
- Не добавляй «подписывайтесь» внутри post_ru — бот добавит это сам.

Верни строго JSON без текста вокруг:
{{
  "score": 1-10,
  "headline_ru": "короткий цепляющий заголовок на русском",
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

    score = int(data.get("score", 0))
    score = max(0, min(10, score))

    return {
        "score": score,
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


def build_post_html(headline_ru: str, post_ru: str, source_url: str, for_caption: bool) -> str:
    safe_headline = html.escape((headline_ru or "Главное к этому часу").strip())
    safe_post = html.escape((post_ru or "").strip())

    # Raw source URL is intentionally visible.
    # In text-only mode Telegram can use it for the link preview.
    footer = f"\n\nИсточник: {source_url}\nПодписаться: {TELEGRAM_CHANNEL_ID}"

    if safe_post:
        body = f"<b>{safe_headline}</b>\n\n{safe_post}"
    else:
        body = f"<b>{safe_headline}</b>"

    if for_caption:
        available = MAX_CAPTION_LENGTH - len(footer)
        if len(body) > available:
            body = body[: max(0, available - 1)].rstrip() + "…"

    return body + footer


def send_photo_with_caption(image_url: str, caption_html: str) -> bool:
    return telegram_request(
        "sendPhoto",
        {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "photo": image_url,
            "caption": caption_html,
            "parse_mode": "HTML",
        },
    )


def send_message_with_preview(text_html: str) -> bool:
    return telegram_request(
        "sendMessage",
        {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "text": text_html,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        },
    )


def publish(headline_ru: str, post_ru: str, source_url: str, image_url: Optional[str]) -> bool:
    if image_url:
        caption = build_post_html(headline_ru, post_ru, source_url, for_caption=True)
        ok = send_photo_with_caption(image_url, caption)
        if ok:
            return True
        print("Photo with caption failed. Sending text message with source preview instead.")

    # No good-quality image, or Telegram rejected the image.
    # Send text with preview enabled so Telegram can pull the original article preview.
    message = build_post_html(headline_ru, post_ru, source_url, for_caption=False)
    return send_message_with_preview(message)


def random_delay_after_post(current_index: int, total_to_publish: int) -> None:
    # No need to wait after the final post.
    if current_index >= total_to_publish:
        return

    low = min(POST_DELAY_MIN_SECONDS, POST_DELAY_MAX_SECONDS)
    high = max(POST_DELAY_MIN_SECONDS, POST_DELAY_MAX_SECONDS)
    delay = random.randint(low, high)
    print(f"Waiting {delay} seconds before next post...")
    time.sleep(delay)


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

    evaluated: List[Dict[str, Any]] = []
    checked_count = 0

    # First pass: evaluate all candidates, do NOT publish yet.
    # This lets the bot publish the strongest stories first.
    for item in candidates:
        checked_count += 1
        print(f"Evaluating: {item['title']}")

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

            if score < MIN_SCORE_TO_POST:
                print("Skipped: low score")
                # Mark low-quality checked items as seen, so future runs don't waste time on them.
                save_published_id(item["id"])
                continue

            if not post_ru:
                print("Skipped: empty Gemini post")
                save_published_id(item["id"])
                continue

            item["score"] = score
            item["post_ru"] = post_ru
            item["headline_ru"] = result["headline_ru"]
            evaluated.append(item)

        except Exception as exc:
            print(f"Evaluation error: {exc}")

    print(f"Strong news found: {len(evaluated)}")

    # Sort by quality: 10 first, then 9, then 8, then 7.
    # If scores are equal, keep the earlier round-robin order.
    evaluated.sort(key=lambda x: x.get("score", 0), reverse=True)
    to_publish = evaluated[:MAX_NEWS_PER_RUN]

    if len(evaluated) > MAX_NEWS_PER_RUN:
        print(
            f"Publish cap: found {len(evaluated)} strong news, "
            f"publishing top {MAX_NEWS_PER_RUN} by score."
        )

    posted_count = 0

    for index, item in enumerate(to_publish, start=1):
        print(f"Publishing #{index}/{len(to_publish)} | Score {item['score']} | {item['title']}")

        try:
            image_url = get_best_image(item["entry"], item["url"])
            success = publish(headline_ru=item.get("headline_ru", ""), post_ru=item["post_ru"], source_url=item["url"], image_url=image_url)

            if success:
                save_published_id(item["id"])
                posted_count += 1
                print("Published successfully")
                random_delay_after_post(index, len(to_publish))
            else:
                print("Publish failed; item was not marked as published")

        except Exception as exc:
            print(f"Publish error: {exc}")

    print(f"Done. Checked: {checked_count}. Strong: {len(evaluated)}. Published: {posted_count}")


if __name__ == "__main__":
    main()
