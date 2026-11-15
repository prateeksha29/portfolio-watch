# Portfolio Watch

A self-hosted news brief for the stocks and funds you own. Three times a day it collects market news, SEC filings
and macro data for your holdings, merges duplicate stories, summarizes the important ones with a local LLM, and writes
you a short digest, largest positions first. A phone web app adds research pages, positions, performance vs the
S&P 500, new stock ideas and Ask (questions answered from your positions and the news).

Your brokerage files stay on your machine; only ticker symbols go out in news searches. Not financial advice.

## Features

- **Digest**: earnings, company announcements, insider trades, analyst moves, news per holding, themes, macro
- **Stock pages**: chart, analyst targets, valuation, your position, news
- **Positions & Performance**: today's value and gains; what you earned over 1M–1Y vs the S&P 500
- **Buys vs S&P**: each buy compared with the same dollars in the S&P 500
- **Ideas**: notable insider buying in stocks you don't own
- **Ask**: "Should I add to NVDA?", answered from your positions, market data and cited articles
- **Multi-user**: each person signs in and sees only their own portfolio
- **Installable phone app** that works offline

Reads Fidelity exports (positions and activity CSVs).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.toml config.toml           # sources, models, thresholds
cp .env.example .env && chmod 600 .env        # WEB_SECRET (required), SEC_USER_AGENT, optional API keys
.venv/bin/python -m fin_news adduser alice --admin
```

Then sign in to the web app and upload your Fidelity exports on **Your files**.

## Models

You need a chat model (summaries, Ask) and a small embedding model (grouping, relevance, search).

| Option | Settings in `config.toml` |
|---|---|
| **Your own GPU** (default) | Any OpenAI-compatible server (llama-swap, llama.cpp, vLLM, Ollama): `[llm] base_url`, `model`. Defaults: Qwen3.8-27B (~24 GB GPU) and Qwen3-Embedding-0.6B (runs on a CPU too). |
| **Claude** | `pip install anthropic`; `provider = "anthropic"`, `model = "claude-opus-5-5"`, `ANTHROPIC_API_KEY` in `.env` |
| **OpenAI** (untested) | `base_url = "https://api.openai.com/v1"`, `local = false`, `api_key_env = "OPENAI_API_KEY"` |

With Claude or OpenAI, keep the embedding model on a local server (`[embedding] base_url`). A hosted API sees the
articles it summarizes and your Ask questions (with your positions); nothing else leaves your machine. Changing the
embedding model needs a new database.

## Running

```
0 6,14,18 * * * /path/to/portfolio-watch/run.sh   # collect, summarize, write digests
*/5 * * * *     /path/to/portfolio-watch/web.sh   # keeps the web app up (port 8050)
```

For phone install you need https (e.g. an nginx reverse proxy). For access away from home, use a tunnel with its own
sign-in (e.g. Cloudflare Tunnel + Access) instead of opening a router port. Logs are in `logs/`.

## Commands

```bash
python -m fin_news run [--no-llm]          # a run now; digests in digests/<name>/latest.md
python -m fin_news ask "..." [--user NAME] # ask from the command line
python -m fin_news search "Micron HBM"     # search stored news
python -m fin_news market [AMD ...]        # prices, P/E, analyst targets
python -m fin_news adduser NAME | passwd NAME
python -m fin_news files check             # an export's structure, without any values
python -m fin_news files scramble          # a copy with fake accounts and amounts, for debugging
```

`files check` and `files scramble` let someone help fix a parsing problem without seeing your numbers.

More detail on each step, data sources and tuning: [docs/how-it-works.md](docs/how-it-works.md).

## License

MIT (see `LICENSE`). Each news source and data API has its own terms; you are responsible for following them.
