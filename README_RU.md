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

- `MAX_NEWS_PER_RUN` = `3`
- `MIN_SCORE_TO_POST` = `8`
- `GEMINI_MODEL` = `gemini-2.5-flash`

Если не добавлять, эти значения уже стоят по умолчанию в коде.
