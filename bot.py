import os
import re
import io
import json
import html
import time
import base64
import random
import hashlib
import calendar
import tempfile
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin, quote

import requests
import feedparser
from bs4 import BeautifulSoup
from PIL import Image
from google import genai
from google.genai import types as genai_types


# ==================================================
# WND MAIN DIGEST
# Pipeline (one run per hour):
# 1. collect fresh RSS entries (last 24 h) from sources.txt
# 2. score all new candidates with Gemini in batches (importance / freshness / reach)
# 3. keep score >= MIN_SCORE_TO_POST
# 4. dedupe: exact hashes -> token Jaccard + event_key -> one LLM event check
# 5. for each selected story: fetch article, Russian rewrite (one Gemini call per article)
# 6. publish with the article's own image/video and two URL buttons,
#    random 30-120 s pauses between posts; overflow goes to the queue
# State lives in state.json (small, bounded, written atomically).
# ==================================================


def env_str(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def env_int(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        print(f"Invalid integer env {name}={value!r}; using default {default}")
        return default


def env_float(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        print(f"Invalid number env {name}={value!r}; using default {default}")
        return default


# =========================
# SETTINGS
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GEMINI_API_KEYS = [
    key.strip()
    for key in os.getenv("GEMINI_API_KEYS", "").split(",")
    if key.strip()
]

GEMINI_SCORE_MODEL = env_str("GEMINI_SCORE_MODEL", "gemini-3.1-flash-lite")
GEMINI_REWRITE_MODEL = env_str("GEMINI_REWRITE_MODEL", "gemini-3.8-flash")
GROQ_FALLBACK_MODEL = env_str("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")

MIN_SCORE_TO_POST = env_float("MIN_SCORE_TO_POST", 7.0)
MAX_POSTS_PER_RUN = max(1, env_int("MAX_POSTS_PER_RUN", 12))
MAX_POSTS_PER_DOMAIN = max(1, env_int("MAX_POSTS_PER_DOMAIN", 6))
ENTRIES_PER_SOURCE = max(1, env_int("ENTRIES_PER_SOURCE", 25))

POST_DELAY_MIN_SECONDS = max(0, env_int("POST_DELAY_MIN_SECONDS", 30))
POST_DELAY_MAX_SECONDS = max(0, env_int("POST_DELAY_MAX_SECONDS", 120))
if POST_DELAY_MAX_SECONDS < POST_DELAY_MIN_SECONDS:
    POST_DELAY_MIN_SECONDS, POST_DELAY_MAX_SECONDS = POST_DELAY_MAX_SECONDS, POST_DELAY_MIN_SECONDS

DIGEST_CHANNEL_URL = env_str("DIGEST_CHANNEL_URL", "https://t.me/wnd_digest")

STATE_FILE = "state.json"
LEGACY_DB_FILE = "urls_base.txt"
SOURCES_FILE = "sources.txt"

MAX_ENTRY_AGE_HOURS = 24
SCORE_CHUNK_SIZE = 40
SCORE_DESCRIPTION_CHARS = 300

PUBLISHED_RETENTION_DAYS = 7
SEEN_LIMIT = 5000
QUEUE_MAX_AGE_HOURS = 12
QUEUE_MAX_RETRIES = 2

DEDUP_WINDOW_HOURS = 72
JACCARD_THRESHOLD = 0.6
JACCARD_THRESHOLD_STRICT = 0.5
LLM_DEDUP_RECENT_EVENTS = 40
MAX_ALTERNATES = 3

ARTICLE_TEXT_LIMIT = 9000
MIN_FULL_TEXT_CHARS = 300
MIN_DESCRIPTION_CHARS = 200

HEADLINE_MAX_CHARS = 140
POST_TARGET_CHARS = 550
POST_MAX_CHARS = 650
POST_MIN_CHARS = 80
BOLD_SPAN_MAX_CHARS = 80
CAPTION_LIMIT = 1024
TEXT_MESSAGE_LIMIT = 4096

IMAGE_MIN_BYTES = 15 * 1024
IMAGE_MAX_BYTES = 10 * 1024 * 1024
IMAGE_MIN_WIDTH = 600
IMAGE_MIN_HEIGHT = 315
VIDEO_MAX_BYTES = 50 * 1024 * 1024
PAGE_MAX_BYTES = 4 * 1024 * 1024

GEMINI_MAX_ATTEMPTS = 3
GEMINI_RATE_LIMIT_BACKOFF = [10, 30]
GEMINI_TIMEOUT_MS = 120000

REQUEST_TIMEOUT = 25
TELEGRAM_TIMEOUT = 60

# Stop starting new posts after this many seconds so the run ends well before
# the 30-minute workflow timeout and the next hourly run.
RUN_TIME_BUDGET_SECONDS = 22 * 60

RUN_STARTED_AT = time.time()

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

HTTP = requests.Session()
HTTP.headers.update(BROWSER_HEADERS)

gemini_clients = []
gemini_key_index = 0


# =========================
# LOGGING (secrets are always redacted)
# =========================

def _secret_values():
    values = [TELEGRAM_BOT_TOKEN, GROQ_API_KEY] + GEMINI_API_KEYS
    return [v for v in values if v and len(v) >= 6]


def redact(text):
    text = str(text)
    for secret in _secret_values():
        text = text.replace(secret, "***")
    text = re.sub(r"bot\d{5,}:[A-Za-z0-9_-]{20,}", "bot***", text)
    text = re.sub(r"([?&]key=)[A-Za-z0-9_-]{16,}", r"\1***", text)
    text = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "***", text)
    text = re.sub(r"gsk_[0-9A-Za-z]{20,}", "***", text)
    return text


def log(*parts):
    print(redact(" ".join(str(p) for p in parts)), flush=True)


def short_error(error, limit=300):
    return redact(str(error)).replace("\n", " ")[:limit]


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
    """Remove markdown leftovers from plain text (used for headlines and fallbacks)."""
    if not text:
        return ""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").replace("`", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\((?:https?://)?[^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("*", "")
    return text.strip()


EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF️‍]"
)


def strip_emoji(text):
    return EMOJI_RE.sub("", text or "")


def normalize_url(url):
    if not url:
        return ""

    url = url.strip()
    parsed = urlparse(url)

    blocked_prefixes = ("utm_",)
    blocked_keys = {
        "fbclid", "gclid", "mc_cid", "mc_eid", "cmpid",
        "ocid", "smid", "ref", "ref_src", "ito",
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
        fragment="",
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


def slugify(text, max_len=80):
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    value = re.sub(r"-{2,}", "-", value)
    return value[:max_len].strip("-")


def to_number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_score(value):
    number = to_number(value, 0.0)
    if number <= 0:
        return 0.0
    return max(1.0, min(10.0, number))


def parse_json_tolerant(text):
    """Parse model output into a dict. Accepts fenced code, extra text, or a bare list."""
    if text is None:
        raise ValueError("empty model output")

    cleaned = str(text).strip()
    if not cleaned:
        raise ValueError("empty model output")

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    candidates = [cleaned]

    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(cleaned[obj_start:obj_end + 1])

    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(cleaned[arr_start:arr_end + 1])

    last_error = None
    for candidate in candidates:
        for variant in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                value = json.loads(variant)
            except Exception as e:
                last_error = e
                continue
            if isinstance(value, dict):
                return value
            if isinstance(value, list):
                return {"items": value}

    raise ValueError(f"could not parse JSON from model output: {last_error}")


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


def utf16_len(text):
    return len((text or "").encode("utf-16-le")) // 2


def visible_html_length(text_html):
    plain = re.sub(r"<[^>]+>", "", text_html or "")
    return utf16_len(html.unescape(plain))


STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "but", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "after", "over", "into", "amid", "says", "say",
    "said", "new", "will", "would", "could", "can", "has", "have", "had", "not", "no",
    "up", "out", "about", "than", "more", "his", "her", "their", "they", "he", "she",
    "we", "you", "what", "who", "how", "why", "when", "s",
}


def title_tokens(title):
    words = normalize_title(title).split()
    return frozenset(w for w in words if len(w) > 1 and w not in STOPWORDS)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def is_http_url(url):
    try:
        parsed = urlparse(url or "")
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def clean_url_for_use(url):
    """Drop tracking parameters and the fragment but keep the original host (www. etc.)."""
    try:
        parsed = urlparse((url or "").strip())
        query = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid", "mc_cid", "mc_eid")
        ]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True), fragment=""))
    except Exception:
        return (url or "").strip()


def url_without_query(url):
    try:
        parsed = urlparse(url or "")
        return urlunparse(parsed._replace(query="", fragment="")).lower()
    except Exception:
        return (url or "").lower()


# =========================
# HARD CONTENT FILTERS
# =========================

BLOCKED_CATEGORIES = {
    "coupon", "coupons", "shopping", "deals", "promotions", "affiliate",
    "casino", "betting", "gambling",
    "sport", "sports", "celebrity", "celebrities", "gossip", "entertainment",
}

BLOCKED_KEYWORDS = [

    # obvious sports
    "champions league", "premier league", "world cup", "fifa world cup",
    "women's champions league", "uefa champions league", "caf champions league",
    "nba finals", "super bowl", "grand prix",
    "ipl match", "cricket match", "football match",
    "soccer match", "tennis match", "basketball game", "baseball game",
    "hockey game", "match preview", "match report", "starting lineup",
    "transfer window", "transfer rumours", "fantasy football",
    "live score", "league table", "playoffs", "quarter-final",
    "semi-final", "final score", "season opener",

    # obvious entertainment/culture spam
    "movie trailer", "box office", "celebrity gossip", "red carpet",
    "album review", "concert tour", "netflix trailer",
    "tv series", "reality show", "streaming platform",

    # obvious shopping/affiliate/promos
    "coupon code", "promo code", "discount code", "voucher code",
    "black friday", "cyber monday", "limited-time deal", "limited time deal",
    "shop now", "buy now", "special offer", "flash sale",
    "affiliate link", "sponsored post", "cashback", "trip.com", "AnimeJapan",

    # gambling / betting
    "betting odds", "sports betting", "online casino",
    "jackpot", "slot machine", "poker tournament",

    # low-value clickbait
    "you won't believe", "viral video", "top 10",
    "best moments", "must watch", "internet reacts",
]


def is_blocked_news_item(title, summary, category=""):
    category_l = str(category or "").lower().strip()
    if category_l in BLOCKED_CATEGORIES:
        return True, f"category:{category}"

    blob = f"{title} {summary}".lower()
    blob = re.sub(r"\s+", " ", blob)

    for keyword in BLOCKED_KEYWORDS:
        keyword_l = keyword.lower().strip()
        if not keyword_l:
            continue

        # For Latin spam/sports phrases, require word boundaries.
        if re.search(r"[a-z]", keyword_l):
            pattern = r"(?<![a-z0-9])" + re.escape(keyword_l) + r"(?![a-z0-9])"
            if re.search(pattern, blob):
                return True, f"keyword:{keyword}"
        else:
            # For Cyrillic phrases, only match full phrase, not tiny fragments.
            if keyword_l in blob:
                return True, f"keyword:{keyword}"

    return False, ""


# Source priority when several outlets report the same event (best first).
SOURCE_PRIORITY = [
    ("reuters.com",),
    ("apnews.com",),
    ("bbc.co.uk", "bbc.com"),
    ("dw.com",),
    ("wsj.com",),
    ("france24.com",),
    ("aljazeera.com",),
    ("politico.eu",),
]

AGGREGATOR_DOMAINS = ("news.google.com", "flipboard.com")


def _domain_matches(domain, name):
    return domain == name or domain.endswith("." + name)


def source_priority(domain):
    d = (domain or "").lower()
    for index, names in enumerate(SOURCE_PRIORITY):
        if any(_domain_matches(d, name) for name in names):
            return index
    return len(SOURCE_PRIORITY)


def canonical_source(domain):
    d = (domain or "").lower()
    for names in SOURCE_PRIORITY:
        if any(_domain_matches(d, name) for name in names):
            return names[0]
    return d or "unknown"


def is_aggregator_url(url):
    d = get_domain(url)
    return any(_domain_matches(d, name) for name in AGGREGATOR_DOMAINS)


def rank_key(item):
    return (source_priority(item.get("domain", "")), -to_number(item.get("score"), 0.0))


# =========================
# STATE (state.json)
# =========================

QUEUE_FIELDS = [
    "link", "url", "url_hash", "title", "title_hash", "normalized_title", "summary",
    "domain", "aggregator", "alt_links", "rss_images", "rss_videos", "source_rss",
    "published_ts", "importance", "freshness", "reach", "score", "category",
    "event_key", "reason", "queued_at", "attempts",
]


def public_fields(item):
    return {key: item[key] for key in QUEUE_FIELDS if key in item}


def read_last_lines(path, count, block_size=65536):
    """Read only the last `count` lines of a (possibly huge) text file."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        position = f.tell()
        buffer = b""
        while position > 0 and buffer.count(b"\n") <= count:
            read_size = min(block_size, position)
            position -= read_size
            f.seek(position)
            buffer = f.read(read_size) + buffer
    lines = buffer.decode("utf-8", errors="replace").splitlines()
    return lines[-count:]


class BotState:
    def __init__(self):
        self.published = []
        self.seen_url_hashes = []
        self.seen_title_hashes = []
        self.queue = []
        self._seen_url_set = set()
        self._seen_title_set = set()

    # ---------- load / save ----------

    @classmethod
    def load(cls):
        state = cls()
        data = None
        need_migration = False

        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("state.json root is not an object")
            except Exception as e:
                log(f"state.json is unreadable ({short_error(e)}); starting with an empty state")
                data = None
                need_migration = True
        else:
            need_migration = True

        if data:
            if isinstance(data.get("published"), list):
                state.published = [p for p in data["published"] if isinstance(p, dict)]
            if isinstance(data.get("seen_url_hashes"), list):
                state.seen_url_hashes = [str(h) for h in data["seen_url_hashes"] if h]
            if isinstance(data.get("seen_title_hashes"), list):
                state.seen_title_hashes = [str(h) for h in data["seen_title_hashes"] if h]
            if isinstance(data.get("queue"), list):
                state.queue = [q for q in data["queue"] if isinstance(q, dict) and q.get("url_hash")]

        if need_migration and os.path.exists(LEGACY_DB_FILE):
            state.migrate_legacy_db()

        state.prune()
        return state

    def migrate_legacy_db(self):
        """One-time import of the last SEEN_LIMIT lines of urls_base.txt (never the whole file)."""
        try:
            lines = read_last_lines(LEGACY_DB_FILE, SEEN_LIMIT)
        except Exception as e:
            log(f"Legacy DB migration failed: {short_error(e)}")
            return

        url_hashes = []
        title_hashes = []
        for line in lines:
            item = line.strip()
            if item.startswith("urlhash:"):
                url_hashes.append(item.split(":", 1)[1])
            elif item.startswith("seen_urlhash:"):
                url_hashes.append(item.split(":", 1)[1])
            elif item.startswith("titlehash:"):
                title_hashes.append(item.split(":", 1)[1])
            elif item.startswith("seen_titlehash:"):
                title_hashes.append(item.split(":", 1)[1])

        for h in url_hashes:
            self.add_seen(url_hash=h)
        for h in title_hashes:
            self.add_seen(title_hash=h)

        log(
            f"Migrated legacy {LEGACY_DB_FILE}: {len(set(url_hashes))} url hashes, "
            f"{len(set(title_hashes))} title hashes (last {len(lines)} lines only)"
        )

    def prune(self):
        now = time.time()

        published_cutoff = now - PUBLISHED_RETENTION_DAYS * 86400
        self.published = [
            p for p in self.published
            if to_number(p.get("timestamp"), 0) >= published_cutoff
        ]

        queue_cutoff = now - QUEUE_MAX_AGE_HOURS * 3600
        kept_queue = []
        seen_queue = set()
        for q in self.queue:
            if to_number(q.get("queued_at"), 0) < queue_cutoff:
                log(f"Queue entry expired: {q.get('title', '')[:90]}")
                continue
            if q.get("url_hash") in seen_queue:
                continue
            seen_queue.add(q.get("url_hash"))
            kept_queue.append(q)
        self.queue = kept_queue

        self.seen_url_hashes = self._dedupe_tail(self.seen_url_hashes, SEEN_LIMIT)
        self.seen_title_hashes = self._dedupe_tail(self.seen_title_hashes, SEEN_LIMIT)
        self._seen_url_set = set(self.seen_url_hashes)
        self._seen_title_set = set(self.seen_title_hashes)

    @staticmethod
    def _dedupe_tail(values, limit):
        # Keep the most recent occurrence of each hash, preserve order, keep last `limit`.
        result = []
        seen = set()
        for value in reversed(values):
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
            if len(result) >= limit:
                break
        result.reverse()
        return result

    def save(self):
        self.seen_url_hashes = self.seen_url_hashes[-SEEN_LIMIT:]
        self.seen_title_hashes = self.seen_title_hashes[-SEEN_LIMIT:]

        data = {
            "version": 2,
            "updated_at": int(time.time()),
            "published": self.published,
            "queue": self.queue,
            "seen_url_hashes": self.seen_url_hashes,
            "seen_title_hashes": self.seen_title_hashes,
        }

        directory = os.path.dirname(os.path.abspath(STATE_FILE))
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=".state-", suffix=".json.tmp", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, STATE_FILE)
            tmp_path = None
        except Exception as e:
            log(f"Saving state failed: {short_error(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ---------- seen ----------

    def add_seen(self, url_hash=None, title_hash=None):
        if url_hash:
            self.seen_url_hashes.append(url_hash)
            self._seen_url_set.add(url_hash)
        if title_hash:
            self.seen_title_hashes.append(title_hash)
            self._seen_title_set.add(title_hash)

    def is_seen_url(self, url_hash):
        return url_hash in self._seen_url_set

    def is_seen_title(self, title_hash):
        return title_hash in self._seen_title_set

    # ---------- published ----------

    def published_url_hashes(self):
        result = set()
        for p in self.published:
            for key in ("url_hash", "resolved_url_hash"):
                if p.get(key):
                    result.add(p[key])
        return result

    def published_title_hashes(self):
        return {p.get("title_hash") for p in self.published if p.get("title_hash")}

    def recent_published(self, hours):
        cutoff = time.time() - hours * 3600
        return [p for p in self.published if to_number(p.get("timestamp"), 0) >= cutoff]

    def recent_image_keys(self):
        return {url_without_query(p.get("image_url")) for p in self.published if p.get("image_url")}

    def record_published(self, entry):
        self.published.append(entry)
        self.add_seen(url_hash=entry.get("url_hash"), title_hash=entry.get("title_hash"))
        if entry.get("resolved_url_hash"):
            self.add_seen(url_hash=entry["resolved_url_hash"])

    # ---------- queue ----------

    def queue_hashes(self):
        return {q.get("url_hash") for q in self.queue}

    def remove_from_queue(self, url_hash):
        self.queue = [q for q in self.queue if q.get("url_hash") != url_hash]

    def upsert_queue(self, item):
        entry = public_fields(item)
        entry["alternates"] = [public_fields(a) for a in item.get("alternates", [])][:MAX_ALTERNATES]
        entry.setdefault("queued_at", int(time.time()))
        entry.setdefault("attempts", 0)
        for index, existing in enumerate(self.queue):
            if existing.get("url_hash") == entry["url_hash"]:
                self.queue[index] = entry
                return
        self.queue.append(entry)


# =========================
# SOURCES / COLLECTION
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


def fetch_feed(rss_url):
    """Download a feed with a timeout (feedparser itself has none) and parse it."""
    response = HTTP.get(rss_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    if getattr(feed, "bozo", 0) and not feed.entries:
        raise ValueError(f"unparseable feed: {getattr(feed, 'bozo_exception', '')}")
    return feed


def entry_timestamp(entry):
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            try:
                return int(calendar.timegm(value))
            except Exception:
                continue
    return None


def _int_attr(media, key):
    try:
        return int(media.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def extract_rss_images(entry):
    candidates = []

    for media in entry.get("media_content", []) or []:
        url = media.get("url", "")
        media_type = str(media.get("type", "")).lower()
        medium = str(media.get("medium", "")).lower()
        if not url:
            continue
        if "video" in media_type or medium == "video":
            continue
        if media_type.startswith("image") or medium == "image" or re.search(r"\.(jpe?g|png|webp)(\?|$)", url.lower()):
            candidates.append((url, _int_attr(media, "width") * _int_attr(media, "height")))

    for media in entry.get("media_thumbnail", []) or []:
        url = media.get("url", "")
        if url:
            candidates.append((url, _int_attr(media, "width") * _int_attr(media, "height")))

    for link in entry.get("links", []) or []:
        href = link.get("href", "")
        link_type = str(link.get("type", "")).lower()
        if href and link_type.startswith("image"):
            candidates.append((href, 0))

    candidates.sort(key=lambda x: x[1], reverse=True)
    result = []
    for url, _ in candidates:
        if is_http_url(url) and url not in result:
            result.append(url)
    return result[:5]


def extract_rss_videos(entry):
    result = []

    for media in entry.get("media_content", []) or []:
        url = media.get("url", "")
        media_type = str(media.get("type", "")).lower()
        medium = str(media.get("medium", "")).lower()
        if url and ("video" in media_type or medium == "video" or is_probably_good_video_url(url)):
            result.append(url)

    for link in entry.get("links", []) or []:
        href = link.get("href", "")
        link_type = str(link.get("type", "")).lower()
        if href and ("video" in link_type or is_probably_good_video_url(href)):
            result.append(href)

    return [u for u in dict.fromkeys(result) if is_probably_good_video_url(u)][:3]


def extract_alt_links(entry):
    """Possible direct publisher links inside an aggregator entry (Google News / Flipboard)."""
    candidates = []
    for key in ("feedburner_origlink", "origlink", "id", "guid"):
        value = entry.get(key)
        if isinstance(value, str):
            candidates.append(value)
    for link in entry.get("links", []) or []:
        href = link.get("href", "")
        link_type = str(link.get("type", "")).lower()
        if href and not link_type.startswith(("image", "video", "audio")):
            candidates.append(href)

    result = []
    for url in candidates:
        if not is_http_url(url) or is_aggregator_url(url):
            continue
        if len(urlparse(url).path.strip("/")) < 5:
            continue  # a homepage, not an article
        usable = clean_url_for_use(url)
        if usable not in result:
            result.append(usable)
    return result


def entry_source_info(entry):
    source = entry.get("source") or {}
    try:
        href = source.get("href", "") or source.get("url", "")
        title = clean_text(source.get("title", ""))
    except AttributeError:
        href, title = "", ""
    return href, title


def collect_candidates(sources, state):
    published_urls = state.published_url_hashes()
    published_titles = state.published_title_hashes()
    queued = state.queue_hashes()

    run_url_hashes = set()
    run_title_hashes = set()
    candidates = []
    now = time.time()
    max_age = MAX_ENTRY_AGE_HOURS * 3600

    for rss_url in sources:
        try:
            feed = fetch_feed(rss_url)
        except Exception as e:
            log(f"Source error {rss_url}: {short_error(e)}")
            continue

        try:
            entries = list(feed.entries[:ENTRIES_PER_SOURCE])
        except Exception:
            entries = []

        stats = {"entries": len(entries), "new": 0, "old": 0, "known": 0, "blocked": 0}

        for entry in entries:
            try:
                raw_link = entry.get("link", "")
                if not is_http_url(raw_link):
                    continue
                link = normalize_url(raw_link)
                title = clean_text(entry.get("title", ""))
                summary = clean_text(entry.get("summary", entry.get("description", "")))
                if not link or not title:
                    continue

                ts = entry_timestamp(entry)
                if ts is not None and now - ts > max_age:
                    stats["old"] += 1
                    continue

                aggregator = is_aggregator_url(link)
                source_href, source_title = entry_source_info(entry)

                # Google News titles look like "Headline - Reuters".
                if source_title:
                    for sep in (" - ", " | ", " — "):
                        suffix = sep + source_title
                        if title.endswith(suffix):
                            title = title[: -len(suffix)].strip()
                            break

                # Google News descriptions only repeat the headline.
                norm_summary = normalize_title(summary)
                norm_title = normalize_title(title)
                if summary and norm_title and norm_summary.startswith(norm_title[:40]) and len(summary) < len(title) + 60:
                    summary = ""

                blocked, blocked_reason = is_blocked_news_item(title, summary)
                if blocked:
                    stats["blocked"] += 1
                    log(f"Blocked at collection [{blocked_reason}]: {title[:100]}")
                    continue

                normalized_title = normalize_title(title)
                if len(normalized_title) < 12:
                    continue

                url_hash = hash_text(link)
                title_hash = hash_text(normalized_title)

                if (
                    url_hash in published_urls
                    or url_hash in queued
                    or state.is_seen_url(url_hash)
                    or title_hash in published_titles
                    or state.is_seen_title(title_hash)
                    or url_hash in run_url_hashes
                    or title_hash in run_title_hashes
                ):
                    stats["known"] += 1
                    continue

                if aggregator:
                    domain = get_domain(source_href) if is_http_url(source_href) else get_domain(link)
                    alt_links = extract_alt_links(entry)
                else:
                    domain = get_domain(link)
                    alt_links = []

                run_url_hashes.add(url_hash)
                run_title_hashes.add(title_hash)
                stats["new"] += 1

                candidates.append({
                    "link": link,                       # normalized: used for hashing only
                    "url": clean_url_for_use(raw_link),  # original host kept: used for fetching and the button
                    "url_hash": url_hash,
                    "title": title,
                    "title_hash": title_hash,
                    "normalized_title": normalized_title,
                    "summary": summary,
                    "domain": domain,
                    "aggregator": aggregator,
                    "alt_links": alt_links,
                    "rss_images": extract_rss_images(entry),
                    "rss_videos": extract_rss_videos(entry),
                    "source_rss": rss_url,
                    "published_ts": ts,
                })
            except Exception as e:
                log(f"Entry error in {rss_url}: {short_error(e)}")

        log(
            f"Source {rss_url[:90]} | entries {stats['entries']} | new {stats['new']} | "
            f"known {stats['known']} | older than {MAX_ENTRY_AGE_HOURS}h {stats['old']} | blocked {stats['blocked']}"
        )

    return candidates


# =========================
# GEMINI / GROQ HELPERS
# =========================

def init_gemini_clients():
    global gemini_clients
    gemini_clients = []
    for key in GEMINI_API_KEYS:
        try:
            try:
                client = genai.Client(
                    api_key=key,
                    http_options=genai_types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
                )
            except Exception:
                # Older SDKs without HttpOptions(timeout=...).
                client = genai.Client(api_key=key)
            gemini_clients.append(client)
        except Exception as e:
            log(f"Gemini client init failed for key #{len(gemini_clients) + 1}: {short_error(e)}")
    log(f"Gemini keys available: {len(gemini_clients)}")


def next_gemini_client():
    global gemini_key_index
    if not gemini_clients:
        raise RuntimeError("No Gemini API keys configured")
    index = gemini_key_index % len(gemini_clients)
    gemini_key_index += 1
    return gemini_clients[index], index + 1


def is_rate_limit_error(error):
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    message = str(error)
    return code == 429 or "RESOURCE_EXHAUSTED" in message or re.search(r"\b429\b", message) is not None


def is_fatal_request_error(error):
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    return code in (400, 401, 403, 404)


def gemini_generate_json(prompt, model, temperature, label):
    """The single entry point for every Gemini call: key rotation, retries, backoff, JSON parsing."""
    last_error = None

    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        client, key_no = next_gemini_client()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            text = getattr(response, "text", None) or ""
            return parse_json_tolerant(text)

        except Exception as e:
            last_error = e
            rate_limited = is_rate_limit_error(e)
            kind = "rate limit" if rate_limited else type(e).__name__
            log(
                f"Gemini {label} failed | model {model} | key #{key_no}/{len(gemini_clients)} | "
                f"attempt {attempt}/{GEMINI_MAX_ATTEMPTS} | {kind}: {short_error(e, 200)}"
            )

            if attempt >= GEMINI_MAX_ATTEMPTS:
                break
            if not rate_limited and is_fatal_request_error(e):
                break  # wrong model name / bad key: retrying will not help

            if rate_limited:
                delay = GEMINI_RATE_LIMIT_BACKOFF[min(attempt - 1, len(GEMINI_RATE_LIMIT_BACKOFF) - 1)]
            else:
                delay = 3
            log(f"Retrying Gemini {label} in {delay}s with the next key")
            time.sleep(delay)

    raise RuntimeError(f"Gemini {label} failed: {short_error(last_error, 200)}")


def groq_generate_json(prompt, temperature, label):
    """Optional last-resort fallback (rewrite step only, only when GROQ_API_KEY is set)."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_FALLBACK_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "Return only one valid JSON object. No markdown, no text outside JSON."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    for attempt in range(2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
        except Exception as e:
            raise RuntimeError(f"Groq {label} request error: {short_error(e, 200)}")

        if response.status_code == 400 and attempt == 0 and "response_format" in payload:
            payload.pop("response_format", None)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"Groq {label} HTTP {response.status_code}: {short_error(response.text, 200)}")

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Groq {label} bad response: {short_error(e, 200)}")
        return parse_json_tolerant(content)

    raise RuntimeError(f"Groq {label} failed")


# =========================
# SCORING
# =========================

SCORING_PROMPT = """
You are the senior editor of "WND | Main Digest", a serious international Telegram news digest.
Score every item below. Be strict: the scale is NOT meant to be generous. Most items in a normal
news feed are routine and must get 4 or below. Only a few items per hour deserve 7+.

Channel profile: world affairs - geopolitics, wars, international security, diplomacy, economy,
sanctions, energy, major crises and disasters; global technology only when it is politically or
economically important. Not wanted: sports, celebrities, entertainment, lifestyle, shopping,
gambling, local crime, human-interest features.

Criteria (integers 1-10):

importance - how significant the event is for world affairs.
  3 = routine or local: "City council in Lyon approves new tram line".
  6 = notable international development with limited consequences:
      "EU adds 12 officials to its sanctions list over election fraud".
  9 = major event with global consequences: "Russia and Ukraine sign a ceasefire agreement".

freshness - is this genuinely new news.
  3 = explainer, analysis, opinion, "what we know", recap or a follow-up with nothing new:
      "What is behind the trade war? A guide".
  6 = follow-up with some new details: "Death toll from Turkey quake rises to 45".
  9 = first report of a new development: "US announces new tariffs on Chinese chips, effective Monday".

reach - how likely a broad audience is to care about and forward this.
  3 = interests only specialists: "Bond spreads in Portugal narrow slightly".
  6 = interests people who follow international news: "Japan's prime minister visits Australia".
  9 = everybody talks about it: "Oil jumps 15% after attack on Saudi export terminal".

Also return:
  category  - one of: geopolitics, war, security, diplomacy, politics, economy, sanctions, energy,
              technology, disaster, health, climate, science, society, crime, sports, celebrity,
              entertainment, lifestyle, shopping, gambling, other
  event_key - short canonical English slug of the underlying EVENT (not of the headline wording),
              lowercase words joined by hyphens, e.g. "us-russia-oil-sanctions-extension".
              Two articles about the same event must get the same event_key.
  reason    - at most 12 words.

Return one JSON object only, exactly in this shape, with one entry for every input id:
{"items": [{"id": 1, "importance": 5, "freshness": 6, "reach": 4, "category": "economy",
            "event_key": "example-event-key", "reason": "brief reason"}]}

Items:
__ITEMS__
"""


def compute_score(importance, freshness, reach):
    return round(0.5 * importance + 0.2 * freshness + 0.3 * reach, 1)


def score_candidates(candidates):
    """Returns (scored, unscored). Unscored items are not marked as seen so the next run retries them."""
    scored = []
    unscored = []

    for chunk_index, chunk in enumerate(chunks(candidates, SCORE_CHUNK_SIZE), start=1):
        by_id = {}
        payload = []
        for local_id, item in enumerate(chunk, start=1):
            by_id[local_id] = item
            payload.append({
                "id": local_id,
                "title": item["title"],
                "description": (item.get("summary") or "")[:SCORE_DESCRIPTION_CHARS],
                "domain": item["domain"],
            })

        prompt = SCORING_PROMPT.replace("__ITEMS__", json.dumps(payload, ensure_ascii=False))

        try:
            result = gemini_generate_json(prompt, GEMINI_SCORE_MODEL, 0, f"scoring chunk {chunk_index}")
            rows = result.get("items", [])
            if not isinstance(rows, list):
                raise ValueError("'items' is not a list")
        except Exception as e:
            log(f"Scoring chunk {chunk_index} failed, {len(chunk)} items left for the next run: {short_error(e, 200)}")
            unscored.extend(chunk)
            continue

        done_ids = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                local_id = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            item = by_id.get(local_id)
            if item is None or local_id in done_ids:
                continue

            importance = clamp_score(row.get("importance"))
            freshness = clamp_score(row.get("freshness"))
            reach = clamp_score(row.get("reach"))
            if not importance or not freshness or not reach:
                continue  # incomplete row: treat as unscored

            item["importance"] = importance
            item["freshness"] = freshness
            item["reach"] = reach
            item["score"] = compute_score(importance, freshness, reach)
            item["category"] = str(row.get("category") or "other").strip().lower()[:40]
            item["event_key"] = slugify(row.get("event_key")) or slugify(" ".join(item["normalized_title"].split()[:8]))
            item["reason"] = str(row.get("reason") or "")[:120]
            done_ids.add(local_id)
            scored.append(item)

            log(
                f"Score {item['score']:.1f} (i{importance:g} f{freshness:g} r{reach:g}) | "
                f"{item['category']} | {item['domain']} | {item['title'][:100]}"
            )

        missing = [item for local_id, item in by_id.items() if local_id not in done_ids]
        if missing:
            log(f"Scoring chunk {chunk_index}: {len(missing)} items without a score, left for the next run")
            unscored.extend(missing)

    return scored, unscored


# =========================
# DEDUPLICATION
# =========================

def build_reference_events(state):
    """Everything already published in the last 72 h plus everything waiting in the queue."""
    references = []
    for p in state.recent_published(DEDUP_WINDOW_HOURS):
        references.append({
            "headline": p.get("headline_ru") or p.get("normalized_title", ""),
            "event_key": p.get("event_key", ""),
            "tokens": title_tokens(p.get("normalized_title", "")),
            "timestamp": to_number(p.get("timestamp"), 0),
        })
    for q in state.queue:
        references.append({
            "headline": q.get("title", ""),
            "event_key": q.get("event_key", ""),
            "tokens": title_tokens(q.get("normalized_title") or q.get("title", "")),
            "timestamp": to_number(q.get("queued_at"), 0),
        })
    return references


def is_duplicate_of_reference(item, references, threshold):
    tokens = item.get("_tokens") or title_tokens(item["title"])
    key = item.get("event_key", "")
    for ref in references:
        if key and ref["event_key"] and key == ref["event_key"]:
            return True, f"same event_key '{key}'"
        similarity = jaccard(tokens, ref["tokens"])
        if similarity >= threshold:
            return True, f"jaccard {similarity:.2f} with '{ref['headline'][:70]}'"
    return False, ""


def attach_alternates(winner, members):
    """Keep the other reports of the same event as fallbacks (used when the winner has no usable text)."""
    pool = []
    for member in members:
        if member is not winner:
            pool.append(public_fields(member))
        pool.extend(member.get("alternates") or [])

    result = []
    used = {winner.get("url_hash")}
    for alt in sorted(pool, key=rank_key):
        if alt.get("url_hash") in used:
            continue
        used.add(alt.get("url_hash"))
        result.append(alt)

    winner["alternates"] = result[:MAX_ALTERNATES]
    return winner


def group_cheap(items, threshold):
    """Group candidates of this run by event_key / Jaccard; the best-priority item wins."""
    groups = []
    for item in sorted(items, key=rank_key):
        target = None
        for group in groups:
            leader = group[0]
            same_key = item.get("event_key") and item.get("event_key") == leader.get("event_key")
            if same_key or jaccard(item["_tokens"], leader["_tokens"]) >= threshold:
                target = group
                break
        if target is None:
            groups.append([item])
        else:
            target.append(item)
    return groups


DEDUP_PROMPT = """
You are the deduplication editor of a world-news Telegram channel.

CANDIDATES are stories that may be published now. PUBLISHED are events the channel already
covered in the last days (Russian headline + event_key) or has already queued for publishing.

For every candidate decide:
1. duplicate_of_published: true if the candidate reports the same event as one of PUBLISHED and
   adds no materially new facts.
2. group: candidates that describe the same event must share the same group value; a candidate
   about a different event gets its own group value.

Important: different developments of the same story are NOT duplicates when they add materially
new facts - for example a new vote, a new casualty number, a new official reaction, a new
decision, a new attack. Only mark duplicates when the core news is the same.

Return one JSON object only:
{"items": [{"id": 1, "duplicate_of_published": false, "group": "g1"}]}
Include every candidate id exactly once.

CANDIDATES:
__CANDIDATES__

PUBLISHED:
__PUBLISHED__
"""


def llm_dedup(items, state):
    """One LLM call per run. Returns list of winners or raises on failure."""
    recent = sorted(state.published, key=lambda p: to_number(p.get("timestamp"), 0))
    published_payload = [
        {"headline": p.get("headline_ru") or p.get("normalized_title", ""), "event_key": p.get("event_key", "")}
        for p in recent[-LLM_DEDUP_RECENT_EVENTS:]
    ]
    for q in state.queue:
        published_payload.append({"headline": q.get("title", ""), "event_key": q.get("event_key", "")})

    by_id = {}
    candidates_payload = []
    for local_id, item in enumerate(items, start=1):
        by_id[local_id] = item
        candidates_payload.append({
            "id": local_id,
            "title": item["title"],
            "event_key": item.get("event_key", ""),
            "domain": item["domain"],
        })

    prompt = (
        DEDUP_PROMPT
        .replace("__CANDIDATES__", json.dumps(candidates_payload, ensure_ascii=False))
        .replace("__PUBLISHED__", json.dumps(published_payload, ensure_ascii=False))
    )

    result = gemini_generate_json(prompt, GEMINI_SCORE_MODEL, 0, "dedup")
    rows = result.get("items", [])
    if not isinstance(rows, list):
        raise ValueError("'items' is not a list")

    decisions = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            local_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if local_id not in by_id:
            continue
        dup = row.get("duplicate_of_published")
        if isinstance(dup, str):
            dup = dup.strip().lower() in ("true", "yes", "1")
        group = row.get("group")
        group = str(group).strip() if group not in (None, "") else f"solo-{local_id}"
        decisions[local_id] = (bool(dup), group)

    if not decisions:
        raise ValueError("dedup output has no usable rows")

    groups = {}
    duplicates = []
    for local_id, item in by_id.items():
        dup, group = decisions.get(local_id, (False, f"solo-{local_id}"))
        if dup:
            duplicates.append(item)
            log(f"LLM dedup: duplicate of published | {item['title'][:100]}")
            continue
        groups.setdefault(group, []).append(item)

    winners = []
    grouped_out = []
    for members in groups.values():
        members.sort(key=rank_key)
        winner = attach_alternates(members[0], members)
        winners.append(winner)
        for loser in members[1:]:
            grouped_out.append(loser)
            log(f"LLM dedup: same event as '{winner['title'][:60]}' | dropped {loser['domain']} | {loser['title'][:80]}")

    return winners, duplicates + grouped_out


def deduplicate(items, state):
    """Returns (winners, rejected). Only runs on items that already passed the score threshold."""
    if not items:
        return [], []

    for item in items:
        item["_tokens"] = title_tokens(item["title"])

    rejected = []

    # Layer 1: exact hashes (cheap re-check against everything published / queued).
    published_urls = state.published_url_hashes()
    published_titles = state.published_title_hashes()
    queued = state.queue_hashes()
    layer1 = []
    for item in items:
        if item["url_hash"] in published_urls or item["url_hash"] in queued or item["title_hash"] in published_titles:
            log(f"Exact duplicate: {item['title'][:100]}")
            rejected.append(item)
        else:
            layer1.append(item)

    # Layer 2: token Jaccard / event_key against the last 72 h (token sets precomputed once).
    references = build_reference_events(state)
    layer2 = []
    for item in layer1:
        dup, why = is_duplicate_of_reference(item, references, JACCARD_THRESHOLD)
        if dup:
            log(f"Fuzzy duplicate ({why}): {item['title'][:100]}")
            rejected.append(item)
        else:
            layer2.append(item)

    # Same event_key / near-identical titles inside this run: merge before the LLM sees them.
    layer2_winners = []
    for group in group_cheap(layer2, JACCARD_THRESHOLD):
        winner = attach_alternates(group[0], group)
        layer2_winners.append(winner)
        for loser in group[1:]:
            log(f"In-run duplicate of '{winner['title'][:60]}': {loser['domain']} | {loser['title'][:80]}")
            rejected.append(loser)

    if not layer2_winners:
        return [], rejected

    # Layer 3: one LLM event check.
    try:
        winners, llm_rejected = llm_dedup(layer2_winners, state)
        rejected.extend(llm_rejected)
        return winners, rejected
    except Exception as e:
        log(f"LLM dedup failed, falling back to strict Jaccard {JACCARD_THRESHOLD_STRICT}: {short_error(e, 200)}")

    # Layer 4 (fallback): stricter cheap check, never publish unfiltered.
    strict = []
    for item in layer2_winners:
        dup, why = is_duplicate_of_reference(item, references, JACCARD_THRESHOLD_STRICT)
        if dup:
            log(f"Strict fuzzy duplicate ({why}): {item['title'][:100]}")
            rejected.append(item)
        else:
            strict.append(item)

    winners = []
    for group in group_cheap(strict, JACCARD_THRESHOLD_STRICT):
        winner = attach_alternates(group[0], group)
        winners.append(winner)
        for loser in group[1:]:
            log(f"Strict in-run duplicate of '{winner['title'][:60]}': {loser['title'][:80]}")
            rejected.append(loser)
    return winners, rejected


# =========================
# ARTICLE PAGES / GOOGLE NEWS RESOLVER
# =========================

def fetch_page(url, timeout=REQUEST_TIMEOUT):
    """GET a page with size and time limits. Returns (final_url, html_text) or (final_url, "")."""
    deadline = time.time() + timeout * 2
    try:
        with HTTP.get(url, timeout=timeout, allow_redirects=True, stream=True) as response:
            final_url = response.url or url
            if response.status_code != 200:
                log(f"Page fetch HTTP {response.status_code}: {get_domain(final_url)}")
                return final_url, ""
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "html" not in content_type and "xml" not in content_type:
                return final_url, ""
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > PAGE_MAX_BYTES or time.time() > deadline:
                    break
            # requests assumes ISO-8859-1 when no charset is sent; most news sites are UTF-8.
            encoding = response.encoding if "charset=" in content_type else "utf-8"
            try:
                text = bytes(data).decode(encoding, errors="replace")
            except LookupError:
                text = bytes(data).decode("utf-8", errors="replace")
            return final_url, text
    except Exception as e:
        log(f"Page fetch failed ({get_domain(url)}): {short_error(e, 200)}")
        return url, ""


def _looks_like_article_url(url):
    return is_http_url(url) and not is_aggregator_url(url) and len(urlparse(url).path.strip("/")) >= 5


def _decode_google_news_id(url):
    """Old-style Google News ids are base64 protobuf that contains the article URL."""
    try:
        path = urlparse(url).path
        match = re.search(r"/articles/([A-Za-z0-9_-]+)", path)
        if not match:
            return None, None
        article_id = match.group(1)
        padded = article_id + "=" * (-len(article_id) % 4)
        raw = base64.urlsafe_b64decode(padded)
        found = re.search(rb"https?://[\x21-\x7e]+", raw)
        if found:
            candidate = found.group(0).decode("ascii", errors="ignore")
            if _looks_like_article_url(candidate):
                return article_id, candidate
        return article_id, None
    except Exception:
        return None, None


def _google_news_batchexecute(article_id, page_html):
    """Newer Google News ids need a signed batchexecute request. Best effort only."""
    if not page_html:
        return None
    soup = BeautifulSoup(page_html, "html.parser")
    node = soup.find(attrs={"data-n-a-sg": True, "data-n-a-ts": True})
    if not node:
        return None
    signature = node.get("data-n-a-sg")
    timestamp = node.get("data-n-a-ts")
    if not signature or not timestamp:
        return None

    inner = (
        '["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],'
        '"X","X",1,[1,1,1],1,1,null,0,0,null,0],"' + article_id + '",' + str(timestamp) + ',"' + signature + '"]'
    )
    body = "f.req=" + quote(json.dumps([[["Fbv4je", inner, None, "generic"]]]))
    try:
        response = HTTP.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return None
        text = response.text.replace('\\"', '"').replace("\\/", "/")
        match = re.search(r'"garturlres","(https?://[^"]+)"', text)
        if not match:
            return None
        candidate = match.group(1)
        try:
            candidate = json.loads('"' + candidate + '"')
        except Exception:
            pass
        return candidate if _looks_like_article_url(candidate) else None
    except Exception as e:
        log(f"Google News batchexecute failed: {short_error(e, 150)}")
        return None


def resolve_publisher_url(item):
    """Best-effort resolution of aggregator links (Google News / Flipboard) to the publisher URL."""
    link = item.get("url") or item["link"]
    if not is_aggregator_url(link):
        return link

    # 1. Follow HTTP redirects.
    final_url, page_html = fetch_page(link)
    if _looks_like_article_url(final_url):
        return clean_url_for_use(final_url)

    # 2. Hints in the aggregator page (canonical / og:url / data-n-au / meta refresh).
    if page_html:
        try:
            soup = BeautifulSoup(page_html, "html.parser")
            hints = []
            for tag in soup.find_all(attrs={"data-n-au": True}):
                hints.append(tag.get("data-n-au"))
            for rel in ("canonical",):
                tag = soup.find("link", rel=rel)
                if tag and tag.get("href"):
                    hints.append(tag.get("href"))
            tag = soup.find("meta", attrs={"property": "og:url"})
            if tag and tag.get("content"):
                hints.append(tag.get("content"))
            tag = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
            if tag and tag.get("content"):
                match = re.search(r"url=(.+)$", tag.get("content"), flags=re.I)
                if match:
                    hints.append(match.group(1).strip("'\" "))
            for hint in hints:
                if _looks_like_article_url(hint):
                    return clean_url_for_use(hint)
        except Exception as e:
            log(f"Aggregator page parse failed: {short_error(e, 150)}")

    # 3. Direct links inside the feed entry.
    for alt in item.get("alt_links") or []:
        if _looks_like_article_url(alt):
            return clean_url_for_use(alt)

    # 4. Google News article id (old base64 format, then signed batchexecute).
    if get_domain(link).endswith("news.google.com"):
        article_id, decoded = _decode_google_news_id(link)
        if decoded:
            return clean_url_for_use(decoded)
        if article_id:
            decoded = _google_news_batchexecute(article_id, page_html)
            if decoded:
                return clean_url_for_use(decoded)

    return None


def fetch_article_text(article_url, page_html=None):
    """
    Fetch readable full article text only for already-selected stories.
    Pass page_html to reuse an already downloaded page.
    """
    try:
        if page_html is None:
            _, page_html = fetch_page(article_url)
        if not page_html:
            return ""

        soup = BeautifulSoup(page_html, "html.parser")

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

        return full_text[:ARTICLE_TEXT_LIMIT]

    except Exception as e:
        log(f"Article text extraction failed: {short_error(e, 200)}")
        return ""


# =========================
# IMAGES / VIDEO
# =========================

BAD_IMAGE_WORDS = [
    "logo", "icon", "favicon", "avatar", "sprite", "placeholder", "default",
    "logos", "icons", "avatars", "sprites", "placeholders",
    "blank", "spacer", "pixel", "1x1", "fallback", "no-image", "noimage", "badge",
]


def is_probably_good_image_url(url):
    if not is_http_url(url):
        return False
    lower = url.lower()
    if lower.endswith(".svg") or ".svg?" in lower or lower.endswith(".gif"):
        return False
    path = urlparse(lower).path
    for word in BAD_IMAGE_WORDS:
        if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", path):
            return False
    return True


def page_meta_values(soup, names):
    values = []
    for name in names:
        for attr in ("property", "name", "itemprop"):
            for tag in soup.find_all("meta", attrs={attr: name}):
                content = (tag.get("content") or "").strip()
                if content and content not in values:
                    values.append(content)
    return values


def extract_page_media(page_url, page_html):
    """Image and video candidates from the article's own page, in priority order."""
    if not page_html:
        return [], []
    try:
        soup = BeautifulSoup(page_html, "html.parser")
    except Exception:
        return [], []

    images = page_meta_values(soup, ["og:image:secure_url", "og:image", "og:image:url", "twitter:image", "twitter:image:src"])
    videos = page_meta_values(soup, ["og:video:secure_url", "og:video:url", "og:video", "twitter:player:stream"])

    images = [urljoin(page_url, u) for u in images]
    videos = [urljoin(page_url, u) for u in videos]
    return images, videos


def download_limited(url, max_bytes, timeout=REQUEST_TIMEOUT):
    """Returns (bytes, content_type) or (None, reason)."""
    deadline = time.time() + timeout * 2
    try:
        with HTTP.get(url, timeout=timeout, allow_redirects=True, stream=True) as response:
            if response.status_code != 200:
                return None, f"HTTP {response.status_code}"
            content_type = response.headers.get("Content-Type", "").lower()
            length = response.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > max_bytes:
                return None, f"too large ({length} bytes)"
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > max_bytes:
                    return None, "too large"
                if time.time() > deadline:
                    return None, "download timeout"
            return bytes(data), content_type
    except Exception as e:
        return None, short_error(e, 150)


def validate_image(url):
    """Download and check an image. Returns dict(url, data, content_type, width, height) or None."""
    data, info = download_limited(url, IMAGE_MAX_BYTES)
    if data is None:
        log(f"Image rejected ({info}): {url[:120]}")
        return None
    content_type = info
    if not content_type.startswith("image/"):
        log(f"Image rejected (content-type {content_type or 'missing'}): {url[:120]}")
        return None
    if len(data) < IMAGE_MIN_BYTES:
        log(f"Image rejected (only {len(data)} bytes): {url[:120]}")
        return None
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            image_format = (img.format or "").upper()
    except Exception as e:
        log(f"Image rejected (unreadable: {short_error(e, 80)}): {url[:120]}")
        return None
    if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
        log(f"Image rejected ({width}x{height} too small): {url[:120]}")
        return None
    ratio = width / float(height)
    if ratio > 3.5 or ratio < 0.4:
        log(f"Image rejected (aspect ratio {ratio:.2f}, banner/sprite): {url[:120]}")
        return None
    return {
        "url": url,
        "data": data,
        "content_type": content_type,
        "format": image_format,
        "width": width,
        "height": height,
    }


def pick_image(candidates, blocked_image_keys):
    tried = set()
    for url in candidates:
        if not url or url in tried:
            continue
        tried.add(url)
        if not is_probably_good_image_url(url):
            log(f"Image URL looks like a logo/icon/placeholder, skipped: {url[:120]}")
            continue
        if url_without_query(url) in blocked_image_keys:
            log(f"Image already used by another post (site default?), skipped: {url[:120]}")
            continue
        image = validate_image(url)
        if image:
            return image
    return None


def is_probably_good_video_url(url):
    if not is_http_url(url):
        return False

    lower = url.lower()
    blocked = [
        "youtube.com", "youtu.be", "twitter.com", "//x.com", ".x.com",
        "tiktok.com", ".m3u8", "/embed",
    ]
    if any(x in lower for x in blocked):
        return False

    path = urlparse(lower).path
    return path.endswith((".mp4", ".webm", ".mov"))


def check_video(url):
    """Only direct video files up to VIDEO_MAX_BYTES with a known Content-Length."""
    if not is_probably_good_video_url(url):
        return None
    try:
        response = HTTP.head(url, allow_redirects=True, timeout=15)
        if response.status_code != 200:
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        length = response.headers.get("Content-Length", "")
        if not length.isdigit() or int(length) <= 0 or int(length) > VIDEO_MAX_BYTES:
            log(f"Video rejected (size {length or 'unknown'}): {url[:120]}")
            return None
        if content_type and not (content_type.startswith("video/") or content_type == "application/octet-stream"):
            return None
        final_url = response.url or url
        return final_url if is_probably_good_video_url(final_url) else url
    except Exception as e:
        log(f"Video check failed: {short_error(e, 120)}")
        return None


def pick_video(candidates):
    for url in dict.fromkeys(candidates):
        checked = check_video(url)
        if checked:
            return checked
    return None


# =========================
# RUSSIAN REWRITE
# =========================

BANNED_PHRASES = [
    "важный шаг", "знаковое событие", "демонстрирует", "подчеркивает",
    "меняет правила игры", "на фоне напряженности", "новый этап",
    "инновационный подход", "стоит отметить", "следует отметить", "важно отметить",
    "в заключение", "не секрет, что", "открывает новые возможности",
    "в современном мире", "играет ключевую роль", "беспрецедентный шаг",
]

REWRITE_PROMPT = """
Ты редактор Telegram-канала «WND | Мировой дайджест». Напиши по статье ниже короткий пост на русском языке.

ЗАГОЛОВОК (headline):
- короткий (до 110 символов), передаёт главный факт;
- без кликбейта, без вопросов, без кавычек-ёлочек вокруг всего заголовка.

ПОСТ (post):
- 2-4 коротких предложения или два коротких абзаца (абзацы разделяй пустой строкой);
- всего не больше __POST_CHARS__ символов;
- плотно и по фактам: что произошло, ключевые детали, почему это важно; без воды;
- выдели 1-3 самых важных факта (число, имя, место, решение) двойными звёздочками: **так**;
  выделяй только короткий фрагмент, никогда не целое предложение;
- сохраняй все числа, даты, суммы, имена и должности точно как в источнике;
- не добавляй фактов, которых нет в источнике; никаких мнений и оценок;
- без эмодзи, без ссылок, без упоминания источника или издания;
- запрещены фразы: __BANNED__.

Если полного текста нет, опирайся только на заголовок и описание и пиши 2-3 предложения.

Верни строго один JSON-объект:
{"headline": "Заголовок", "post": "Текст поста с **ключевым фактом**"}

ИСТОЧНИК (домен, только для контекста): __DOMAIN__

ЗАГОЛОВОК СТАТЬИ:
__TITLE__

ОПИСАНИЕ:
__DESCRIPTION__

ПОЛНЫЙ ТЕКСТ:
__TEXT__
"""

BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")


def normalize_for_ban(text):
    return (text or "").lower().replace("ё", "е")


def validate_rewrite(headline, post):
    if not headline or not post:
        return False, "empty headline or post"
    if not is_valid_russian_rewrite(headline, post):
        return False, "not Russian"
    if len(headline) > HEADLINE_MAX_CHARS:
        return False, f"headline too long ({len(headline)})"
    plain_post = post.replace("**", "")
    if len(plain_post) > POST_MAX_CHARS:
        return False, f"post too long ({len(plain_post)})"
    if len(plain_post) < POST_MIN_CHARS:
        return False, f"post too short ({len(plain_post)})"
    spans = BOLD_RE.findall(post)
    if len(spans) < 1:
        return False, "no bold facts"
    if len(spans) > 3:
        return False, f"too many bold spans ({len(spans)})"
    if any(len(span) > BOLD_SPAN_MAX_CHARS for span in spans):
        return False, "a whole sentence is bold"
    blob = normalize_for_ban(f"{headline} {post}")
    for phrase in BANNED_PHRASES:
        if normalize_for_ban(phrase) in blob:
            return False, f"banned phrase '{phrase}'"
    if re.search(r"https?://|www\.|t\.me/", blob):
        return False, "contains a link"
    return True, ""


def rewrite_article(item, article_text):
    description = (item.get("summary") or "")[:1500]
    prompt = (
        REWRITE_PROMPT
        .replace("__POST_CHARS__", str(POST_TARGET_CHARS))
        .replace("__BANNED__", ", ".join(f"«{p}»" for p in BANNED_PHRASES))
        .replace("__DOMAIN__", item.get("domain", ""))
        .replace("__TITLE__", item.get("title", ""))
        .replace("__DESCRIPTION__", description or "(нет)")
        .replace("__TEXT__", (article_text or "")[:ARTICLE_TEXT_LIMIT] or "(полный текст недоступен)")
    )

    attempts = [("gemini", GEMINI_REWRITE_MODEL)]
    if GEMINI_SCORE_MODEL != GEMINI_REWRITE_MODEL:
        attempts.append(("gemini", GEMINI_SCORE_MODEL))
    if GROQ_API_KEY:
        attempts.append(("groq", GROQ_FALLBACK_MODEL))

    for provider, model in attempts:
        try:
            if provider == "gemini":
                result = gemini_generate_json(prompt, model, 0.3, "rewrite")
            else:
                result = groq_generate_json(prompt, 0.3, "rewrite")
        except Exception as e:
            log(f"Rewrite with {model} failed: {short_error(e, 200)}")
            continue

        headline = strip_markdown(strip_emoji(str(result.get("headline") or ""))).strip()
        post = strip_emoji(str(result.get("post") or "")).strip()
        post = re.sub(r"\r\n?", "\n", post)
        post = re.sub(r"\n{3,}", "\n\n", post)

        ok, reason = validate_rewrite(headline, post)
        if ok:
            log(f"Rewrite OK with {model}: {headline[:90]}")
            return {"headline": headline, "post": post, "model": model}
        log(f"Rewrite from {model} rejected ({reason}): {headline[:90]}")

    return None


# =========================
# TELEGRAM RENDERING
# =========================

def render_body_html(post):
    """Escape FIRST, then turn **x** into <b>x</b>, then drop stray markdown."""
    text = html.escape(post or "", quote=False)
    text = BOLD_RE.sub(lambda m: "<b>" + m.group(1).strip() + "</b>", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("*", "").replace("__", "").replace("`", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"<b>\s*</b>", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def render_post_html(headline, post, score):
    safe_headline = html.escape(strip_markdown(headline), quote=False)
    if to_number(score, 0) >= 9:
        safe_headline = "⚡ " + safe_headline
    return f"<b>{safe_headline}</b>\n\n{render_body_html(post)}"


def split_post_sentences(post):
    """List of (paragraph_index, sentence) so sentences can be dropped from the end."""
    result = []
    for p_index, paragraph in enumerate(re.split(r"\n\s*\n", post.strip())):
        for sentence in re.split(r"(?<=[.!?…])\s+", paragraph.strip()):
            if sentence.strip():
                result.append((p_index, sentence.strip()))
    return result


def join_post_sentences(parts):
    paragraphs = []
    current_index = None
    for p_index, sentence in parts:
        if p_index != current_index:
            paragraphs.append([])
            current_index = p_index
        paragraphs[-1].append(sentence)
    return "\n\n".join(" ".join(p) for p in paragraphs)


def build_caption(headline, post, score, limit=CAPTION_LIMIT):
    """Render the post; if the visible text is too long, drop whole sentences from the end."""
    parts = split_post_sentences(post)
    while True:
        text_html = render_post_html(headline, join_post_sentences(parts), score)
        if visible_html_length(text_html) <= limit or len(parts) <= 1:
            break
        parts = parts[:-1]
    if visible_html_length(text_html) > limit:
        # Still too long with a single sentence: send the headline only (never cut inside a tag).
        text_html = render_post_html(headline, "", score).strip()
    return text_html


def build_keyboard(article_url):
    row = []
    if is_http_url(article_url):
        row.append({"text": "📰 Оригинал", "url": article_url})
    if is_http_url(DIGEST_CHANNEL_URL):
        row.append({"text": "✨ WND Digest", "url": DIGEST_CHANNEL_URL})
    return {"inline_keyboard": [row]} if row else None


# =========================
# TELEGRAM SENDING
# =========================

def telegram_call(method, payload, files=None):
    """Returns (ok, status_code, description). Retries once after a 429 using retry_after."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

    for attempt in range(2):
        try:
            if files:
                response = requests.post(url, data=payload, files=files, timeout=TELEGRAM_TIMEOUT)
            else:
                response = requests.post(url, json=payload, timeout=TELEGRAM_TIMEOUT)
        except Exception as e:
            log(f"Telegram {method} request error: {short_error(e, 200)}")
            return False, None, "request error"

        try:
            body = response.json()
        except Exception:
            body = {}

        if response.status_code == 200 and body.get("ok"):
            return True, 200, ""

        description = str(body.get("description") or response.text[:300])

        if response.status_code == 429 and attempt == 0:
            parameters = body.get("parameters") or {}
            retry_after = int(to_number(parameters.get("retry_after"), 5))
            wait = max(1, min(retry_after, 300)) + 1
            log(f"Telegram {method} rate limited, waiting {wait}s before one retry")
            time.sleep(wait)
            continue

        log(f"Telegram {method} error {response.status_code}: {short_error(description, 200)}")
        return False, response.status_code, description

    return False, 429, "rate limited"


def upload_bytes_for_photo(image):
    """Telegram uploads accept JPEG/PNG reliably; convert anything else to JPEG."""
    data = image["data"]
    image_format = image.get("format", "")
    if image_format in ("JPEG", "PNG"):
        ext = "jpg" if image_format == "JPEG" else "png"
        return data, f"image.{ext}", "image/jpeg" if ext == "jpg" else "image/png"
    try:
        with Image.open(io.BytesIO(data)) as img:
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=90)
            return buffer.getvalue(), "image.jpg", "image/jpeg"
    except Exception:
        return data, "image.jpg", image.get("content_type") or "image/jpeg"


def send_text_message(text_html, keyboard):
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": text_html,
        "parse_mode": "HTML",
        "link_preview_options": {"is_disabled": True},
    }
    if keyboard:
        payload["reply_markup"] = keyboard
    ok, status, description = telegram_call("sendMessage", payload)
    if ok:
        return True
    if keyboard and status == 400 and re.search(r"button|reply markup|url", description or "", flags=re.I):
        log("Retrying text message without buttons (Telegram rejected a button URL)")
        payload.pop("reply_markup", None)
        ok, _, _ = telegram_call("sendMessage", payload)
        return ok
    return False


def send_post(caption_html, text_html, keyboard, video_url, image):
    """Video -> photo by URL -> photo upload -> plain text. Returns (ok, kind)."""
    markup_json = json.dumps(keyboard, ensure_ascii=False) if keyboard else None

    if video_url:
        payload = {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "video": video_url,
            "caption": caption_html,
            "parse_mode": "HTML",
            "supports_streaming": True,
        }
        if keyboard:
            payload["reply_markup"] = keyboard
        ok, _, _ = telegram_call("sendVideo", payload)
        if ok:
            return True, "video"
        log("sendVideo failed, falling back to the image")

    if image:
        payload = {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "photo": image["url"],
            "caption": caption_html,
            "parse_mode": "HTML",
        }
        if keyboard:
            payload["reply_markup"] = keyboard
        ok, _, _ = telegram_call("sendPhoto", payload)
        if ok:
            return True, "photo"

        log("sendPhoto by URL failed, retrying as an upload")
        data, filename, content_type = upload_bytes_for_photo(image)
        form = {"chat_id": TELEGRAM_CHANNEL_ID, "caption": caption_html, "parse_mode": "HTML"}
        if markup_json:
            form["reply_markup"] = markup_json
        ok, _, _ = telegram_call("sendPhoto", form, files={"photo": (filename, data, content_type)})
        if ok:
            return True, "photo-upload"
        log("Photo upload failed, falling back to a text message")

    if send_text_message(text_html, keyboard):
        return True, "text"
    return False, "failed"


# =========================
# PUBLISHING
# =========================

def prepare_story(item, blocked_image_keys):
    """
    Resolve the link, fetch the article, rewrite it and pick media.
    Returns (status, data): status is "ok", "no_content" (permanent skip) or "error" (retry later).
    """
    article_url = item.get("url") or item["link"]
    resolved_url = ""

    if item.get("aggregator"):
        resolved = resolve_publisher_url(item)
        if resolved:
            resolved_url = resolved
            article_url = resolved
            log(f"Resolved aggregator link -> {get_domain(resolved)}")
        else:
            log(f"Aggregator link not resolved, headline-only: {item['title'][:90]}")

    page_url, page_html = ("", "")
    if not is_aggregator_url(article_url):
        page_url, page_html = fetch_page(article_url)
        # Never use text/images of a different page (homepage, paywall or login redirect).
        if page_html and (is_aggregator_url(page_url) or len(urlparse(page_url).path.strip("/")) < 5):
            log(f"Article page redirected away ({get_domain(page_url)}), ignoring page content")
            page_html = ""

    article_text = fetch_article_text(page_url or article_url, page_html) if page_html else ""
    summary = item.get("summary") or ""

    if len(article_text) >= MIN_FULL_TEXT_CHARS:
        log(f"Full text: {len(article_text)} chars | {item['title'][:90]}")
    elif len(summary) >= MIN_DESCRIPTION_CHARS:
        log(f"No full text, using RSS description ({len(summary)} chars) | {item['title'][:90]}")
        article_text = ""
    else:
        log(
            f"Not enough text to publish (text {len(article_text)}, description {len(summary)}) | "
            f"{item['domain']} | {item['title'][:90]}"
        )
        return "no_content", None

    rewrite = rewrite_article(item, article_text)
    if not rewrite:
        return "error", None

    page_images, page_videos = extract_page_media(page_url, page_html)
    video_url = pick_video(page_videos + list(item.get("rss_videos") or []))
    # The image is picked even when a video exists: it is the fallback if sendVideo fails.
    image = pick_image(page_images + list(item.get("rss_images") or []), blocked_image_keys)

    return "ok", {
        "article_url": article_url,
        "resolved_url": resolved_url,
        "headline": rewrite["headline"],
        "post": rewrite["post"],
        "model": rewrite["model"],
        "video_url": video_url,
        "image": image,
    }


def time_left():
    return RUN_TIME_BUDGET_SECONDS - (time.time() - RUN_STARTED_AT)


def publish_all(state, new_items):
    """Queue first, then new stories, each ordered by score; pauses of 30-120 s between posts."""
    now = int(time.time())

    queue_items = sorted(state.queue, key=lambda q: -to_number(q.get("score"), 0))
    new_items = sorted(new_items, key=lambda q: -to_number(q.get("score"), 0))

    # Everything that qualified goes into the queue first, so a crash never loses it.
    for item in new_items:
        item["queued_at"] = now
        item["attempts"] = 0
        state.upsert_queue(item)
        state.add_seen(url_hash=item["url_hash"])
    state.save()

    to_publish = [dict(q) for q in queue_items] + [state_entry_for(state, item) for item in new_items]
    to_publish = [item for item in to_publish if item]

    stats = {"published": 0, "failed": 0, "skipped": 0, "deferred": 0}
    send_attempts = 0
    domain_counts = {}
    last_post_at = None
    blocked_image_keys = set(state.recent_image_keys())
    published_urls = state.published_url_hashes()
    published_titles = state.published_title_hashes()

    for item in to_publish:
        title = item.get("title", "")[:90]
        domain_key = canonical_source(item.get("domain", ""))

        if item.get("url_hash") in published_urls or item.get("title_hash") in published_titles:
            log(f"Already published, removing from queue: {title}")
            state.remove_from_queue(item["url_hash"])
            continue

        if send_attempts >= MAX_POSTS_PER_RUN:
            stats["deferred"] += 1
            continue
        if domain_counts.get(domain_key, 0) >= MAX_POSTS_PER_DOMAIN:
            log(f"Domain cap reached for {domain_key}, deferred to next run: {title}")
            stats["deferred"] += 1
            continue
        if time_left() <= 0:
            log(f"Run time budget used up, deferred to next run: {title}")
            stats["deferred"] += 1
            continue

        # Try the winner first, then other reports of the same event if the winner has no usable text.
        status, data, used = "no_content", None, item
        for option in [item] + list(item.get("alternates") or []):
            try:
                status, data = prepare_story(option, blocked_image_keys)
            except Exception as e:
                log(f"Preparing story failed: {short_error(e, 200)}")
                status, data = "error", None
            used = option
            if status != "no_content":
                break
            if option is not item:
                state.add_seen(url_hash=option.get("url_hash"))

        if status == "no_content":
            log(f"Skipped (no usable text in any report): {title}")
            state.remove_from_queue(item["url_hash"])
            stats["skipped"] += 1
            state.save()
            continue

        if status == "ok":
            caption_html = build_caption(data["headline"], data["post"], item.get("score"))
            text_html = build_caption(data["headline"], data["post"], item.get("score"), TEXT_MESSAGE_LIMIT)
            keyboard = build_keyboard(data["article_url"])

            if last_post_at is not None:
                delay = random.uniform(POST_DELAY_MIN_SECONDS, POST_DELAY_MAX_SECONDS)
                wait = delay - (time.time() - last_post_at)
                if wait > 0 and wait > time_left():
                    log(f"Pause would exceed the run time budget, deferred to next run: {title}")
                    stats["deferred"] += 1
                    continue
                if wait > 0:
                    log(f"Pause before next post: {wait:.0f}s (drawn {delay:.0f}s since the previous post)")
                    time.sleep(wait)

            send_attempts += 1
            log(
                f"Publishing: score {to_number(item.get('score'), 0):.1f} | {used.get('domain', '')} | "
                f"model {data['model']} | {data['headline'][:90]}"
            )
            ok, kind = send_post(caption_html, text_html, keyboard, data["video_url"], data["image"])
        else:
            ok, kind = False, "prepare-error"

        if ok:
            last_post_at = time.time()
            image_url = ""
            if kind in ("photo", "photo-upload") and data.get("image"):
                image_url = data["image"]["url"]
                blocked_image_keys.add(url_without_query(image_url))
            entry = {
                "url_hash": item["url_hash"],
                "resolved_url_hash": hash_text(normalize_url(data["resolved_url"])) if data.get("resolved_url") else "",
                "title_hash": item["title_hash"],
                "normalized_title": item.get("normalized_title", ""),
                "event_key": item.get("event_key", ""),
                "headline_ru": data["headline"],
                "domain": used.get("domain", ""),
                "url": data["article_url"],
                "image_url": image_url,
                "media": kind,
                "score": to_number(item.get("score"), 0),
                "timestamp": int(time.time()),
            }
            state.record_published(entry)
            if used is not item:
                state.add_seen(url_hash=used.get("url_hash"), title_hash=used.get("title_hash"))
            published_urls.add(item["url_hash"])
            published_titles.add(item["title_hash"])
            domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
            state.remove_from_queue(item["url_hash"])
            stats["published"] += 1
            log(f"Published as {kind}.")
        else:
            stats["failed"] += 1
            attempts = int(to_number(item.get("attempts"), 0)) + 1
            if attempts > QUEUE_MAX_RETRIES:
                log(f"Publish failed {attempts} times, dropping: {title}")
                state.remove_from_queue(item["url_hash"])
            else:
                log(f"Publish failed ({kind}), will retry next run ({attempts}/{QUEUE_MAX_RETRIES}): {title}")
                for q in state.queue:
                    if q.get("url_hash") == item["url_hash"]:
                        q["attempts"] = attempts

        state.save()

    return stats


def state_entry_for(state, item):
    for q in state.queue:
        if q.get("url_hash") == item["url_hash"]:
            return dict(q)
    return None


# =========================
# MAIN
# =========================

def validate_env():
    missing = []

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHANNEL_ID:
        missing.append("TELEGRAM_CHANNEL_ID")

    if not GEMINI_API_KEYS:
        missing.append("GEMINI_API_KEYS")

    if missing:
        raise RuntimeError("Missing env variables: " + ", ".join(missing))


def log_settings():
    log(
        f"Settings | score model {GEMINI_SCORE_MODEL} | rewrite model {GEMINI_REWRITE_MODEL} | "
        f"groq fallback {'on (' + GROQ_FALLBACK_MODEL + ')' if GROQ_API_KEY else 'off'} | "
        f"min score {MIN_SCORE_TO_POST} | max posts/run {MAX_POSTS_PER_RUN} | "
        f"max posts/domain {MAX_POSTS_PER_DOMAIN} | pause {POST_DELAY_MIN_SECONDS}-{POST_DELAY_MAX_SECONDS}s | "
        f"entries/source {ENTRIES_PER_SOURCE}"
    )


def main():
    validate_env()
    log_settings()
    init_gemini_clients()

    state = BotState.load()
    log(
        f"State | published (7d) {len(state.published)} | seen urls {len(state.seen_url_hashes)} | "
        f"queue {len(state.queue)}"
    )

    try:
        sources = load_sources()
        candidates = collect_candidates(sources, state)
        log(f"SUMMARY collected: {len(candidates)} new candidates from {len(sources)} sources")

        passed = []
        if candidates:
            scored, unscored = score_candidates(candidates)
            log(f"SUMMARY scored: {len(scored)} | unscored (retry next run): {len(unscored)}")

            for item in scored:
                blocked, reason = is_blocked_news_item(item["title"], item.get("summary", ""), item.get("category", ""))
                if blocked:
                    log(f"Blocked after scoring [{reason}]: {item['title'][:100]}")
                    state.add_seen(url_hash=item["url_hash"], title_hash=item["title_hash"])
                    continue
                if item["score"] < MIN_SCORE_TO_POST:
                    state.add_seen(url_hash=item["url_hash"], title_hash=item["title_hash"])
                    continue
                passed.append(item)
            log(f"SUMMARY passed threshold {MIN_SCORE_TO_POST}: {len(passed)}")
            state.save()

        winners, rejected = deduplicate(passed, state)
        for item in rejected:
            state.add_seen(url_hash=item["url_hash"], title_hash=item["title_hash"])
        log(f"SUMMARY after dedup: {len(winners)} new stories | duplicates dropped: {len(rejected)} | queued from earlier runs: {len(state.queue)}")
        state.save()

        if not winners and not state.queue:
            log("Nothing qualifies for publishing this run.")
            return

        stats = publish_all(state, winners)
        log(
            f"SUMMARY published: {stats['published']} | failed: {stats['failed']} | "
            f"skipped (no text): {stats['skipped']} | deferred: {stats['deferred']} | "
            f"queued for next run: {len(state.queue)}"
        )
    finally:
        state.prune()
        state.save()
        log("Done.")


if __name__ == "__main__":
    main()
