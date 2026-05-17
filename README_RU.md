# News Telegram Bot — готовые файлы

Этот архив содержит готовую систему для GitHub Actions:

- `bot.py` — основная логика
- `sources.txt` — RSS-источники
- `urls_base.txt` — база уже обработанных новостей
- `requirements.txt` — Python-библиотеки
- `.github/workflows/main.yml` — автоматический запуск каждые 30 минут

## Какие секреты нужны в GitHub

В репозитории откройте:

Settings -> Secrets and variables -> Actions -> New repository secret

Добавьте:

1. `TELEGRAM_BOT_TOKEN` — токен от @BotFather
2. `TELEGRAM_CHANNEL_ID` — username канала, например `@my_world_news_2026`
3. `GEMINI_API_KEY` — ключ из Google AI Studio

## Опциональные переменные

Settings -> Secrets and variables -> Actions -> Variables -> New repository variable

Можно добавить:

- ABOUT_POST_URL — link to your channel info/about post, e.g. https://t.me/wnd_digest/15
- ABOUT_LINK_TEXT — default: ℹ️ О проекте WND
- MIN_SCORE_TO_POST — default: 7
- MAX_NEWS_PER_RUN — default: 10
- POST_DELAY_MIN_SECONDS — default: 60
- POST_DELAY_MAX_SECONDS — default: 180
- MAX_CANDIDATES_TO_ANALYZE — default: 60
- ENTRIES_PER_SOURCE — default: 10

