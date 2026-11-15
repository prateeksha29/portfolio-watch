# Holdings Brief

Local news pipeline and phone web app for Fidelity portfolios. Three times a day it collects US market news, SEC
filings and macro data for everyone's stocks and fund themes, groups duplicate stories with an embedding model,
summarizes the important ones with a local LLM, and writes each person a Markdown digest ordered by their own position
sizes. Nothing is sent to a cloud model; portfolio files never leave the machine (only ticker symbols appear in feed
URLs).

Personal finance tool, not financial advice. Check each data source's terms before turning it on.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.toml config.toml        # shared settings: sources, models, thresholds
cp .env.example .env && chmod 600 .env     # WEB_SECRET (required), SEC_USER_AGENT, optional API keys
.venv/bin/python -m fin_news adduser alice --admin   # asks for a password; repeat (without --admin) per person
```

**Models.** Two kinds: a chat model writes story and filing summaries and answers Ask (`[llm]`), and a small embedding
model groups stories, scores relevance and powers search (`[embedding]`). The defaults are Qwen3.8-27B (Gemma 4
26B-A4B as fallback) and Qwen3-Embedding-0.6B (1024 dims). Changing the embedding model or its dimensions needs a new
database (and the `[filter]` thresholds were tuned for this one), so keep it when you switch chat models.

- **Your own GPU.** Run an OpenAI-compatible server: llama-swap (loads models on demand), llama.cpp's `llama-server`,
  vLLM or Ollama. Set `[llm] base_url` (e.g. `http://localhost:8080/v1`) and `model` / `ask_model` to the names its
  `/v1/models` lists. The same server (or `[embedding] base_url`) must serve the embedding model at `/v1/embeddings`;
  at 0.6B it also runs on a CPU. A 27B chat model needs a ~24 GB GPU at 4-bit; on a smaller card pick a smaller model.
  Thinking mode is turned off per request so the model answers in JSON directly.
- **OpenAI** (untested): `base_url = "https://api.openai.com/v1"`, `local = false` (no llama.cpp-only settings, room
  for reasoning tokens), `api_key_env = "OPENAI_API_KEY"` with the key in `.env`, and an OpenAI model name. Point
  `[embedding] base_url` at a local server for the embedding model.
- **Claude**: `pip install anthropic`, then `provider = "anthropic"`, `model = "claude-opus-5-5"` (or a cheaper
  one such as `claude-haiku-5-5`) and `ANTHROPIC_API_KEY=...` in `.env`; `base_url` is not used. Summaries run at
  `effort = "low"` and Ask at `ask_effort = "high"`; Claude returns the same structured JSON, and a request it
  declines is retried on another Claude model. Keep a local embedding server (`[embedding] base_url`).

A hosted API is sent the article text it summarizes, Ask questions and, for Ask, your positions: everything else
stays on your machine.

**People.** Each person has `users/<name>/`: `user.toml` (display name, extra tickers and topics, the accounts their
returns count), a hashed `password`, and `fidelity/` with their exports, which they upload on the app's Files page
(or copy there). Admins also see source problems in the digest and the Benchmarks page. A run is shared: news and
prices are collected once for everyone's stocks, each story is summarized once, and the stories that get summaries
interleave everyone's digest order, so no one's largest holdings are crowded out. Each person sees only their own
digest, positions, returns, ideas and Ask answers; other people's stocks never appear in their digest.

## Commands

```bash
.venv/bin/python -m fin_news run                           # full run for everyone -> digests/<name>/latest.md
.venv/bin/python -m fin_news run --no-llm --no-fulltext    # fast run
.venv/bin/python -m fin_news holdings [--user NAME]        # tickers and fund themes read from a person's export
.venv/bin/python -m fin_news search "Micron HBM pricing" --since 2026-10-01
.venv/bin/python -m fin_news ask "Based on how much NVDA I hold, should I buy more?" [--user NAME]
.venv/bin/python -m fin_news market [AMD ...]              # prices, P/E and analyst targets now
.venv/bin/python -m fin_news funds [--force]               # fund holdings from SEC filings, and real exposure per stock
.venv/bin/python -m fin_news prune                         # apply retention now
.venv/bin/python -m fin_news adduser NAME [--display "Name"] [--admin]
.venv/bin/python -m fin_news passwd NAME
.venv/bin/python -m fin_news files check [--user NAME]     # the exports' structure only: columns, kinds of values,
                                                           # accounts as A/B/C, rows the app can't read (no values)
.venv/bin/python -m fin_news files scramble [--user NAME]  # a copy in data/scrambled/NAME/ with fake accounts and
                                                           # amounts, for reproducing a parsing bug without real data
```

`files check` and `files scramble` exist so problems with an export can be fixed by someone (or an AI assistant)
who must not see your numbers. The scrambled copy keeps the format, symbols, prices and dates; account names and
numbers are replaced, every quantity and dollar amount is multiplied by a random factor per stock, and transfer,
bank and card lines keep only their first words. It refuses to write a copy in which an original amount or account
still appears. `scripts/e2e_test.py` runs on such copies.

## Schedule and phone app

```
0 6,14,18 * * * /path/to/fin-news/run.sh   # collect, summarize, write digests, apply retention
*/5 * * * *     /path/to/fin-news/web.sh   # keeps the web app running (exits at once if it already is)
@reboot         /path/to/fin-news/web.sh
```

The web app (`python -m fin_news.web`, FastAPI, port 8050) has the digest, holdings with per-stock pages (chart,
analyst targets and ratings, valuation, position, news), positions, performance vs the S&P 500, ideas, search and
Ask; More -> Your files (upload exports), Account (password, sign out), Status, How it works and Benchmarks. Each
person signs in with their name and password; changing a password signs out their other devices. It is an
installable PWA that keeps pages for offline use (cleared when someone signs out); installing needs https, e.g. an
nginx reverse proxy (`[web] proxy`, then `direct_access = false`). For access away from home, put it behind a tunnel
with its own sign-in (e.g. Cloudflare Tunnel + Access) rather than opening a router port; let `/ping` through it. Logs: `logs/run.log`, `logs/cron.err` (last run), `logs/web.log`.

## Pipeline

| Step | What happens |
|---|---|
| Holdings | Each person's newest Fidelity positions export in `users/<name>/fidelity/` (by the date in its name; a file without positions is skipped with a warning) -> stocks (by position size) and funds (mapped to themes). A changed portfolio is noted at the top of that person's next digest. The CSV never leaves the machine; only ticker symbols appear in feed URLs. |
| Collect | Yahoo, Nasdaq, Seeking Alpha, Google News per ticker; Google News per topic; Fed, BLS, CNBC, MarketWatch, Investing.com, Benzinga, SA Market Currents, PR Newswire, Motley Fool RSS (Business Wire comes in via Yahoo/Finnhub/Nasdaq); the fool.com news sitemap (every free article and earnings call transcript of the last 2 days, with the tickers Fool tagged it with; those tags count like a ticker in the headline); Reddit RSS; SEC EDGAR; Finnhub news + earnings calendar; FRED. Parallel across sites, ~1 request/second per site, conditional GETs. |
| Discovery | `fin_news/discover.py`, `[discovery]`; once a day (first run after 06:00). New stock ideas the owner doesn't hold: insider buying from Finviz's insider page (robots.txt allows `/insidertrading`; its screener is disallowed; one fetch a day, about a week of Form 4 purchases, kept in `insider_buys`). Insider rules: officers and directors only (not 10% owners, funds, trustees), companies $1B+ and $5+; 2+ insiders buying $100k+ within 30 days, or a CEO/CFO/president/chair buying $250k+. Each idea (`ideas` table) is shown once and gets a Yahoo snapshot that day (price when flagged) and 30 days later; watched ideas daily. **Ideas tab**: New (14 days, with price vs range and 200-day average, Street view and 30-day moves, earnings, fund overlap; Watch/Dismiss), Watching (return since flagged vs the S&P, and each source's 30-day scorecard). The digest lists the day's new ideas. Zacks was considered and left out: its terms prohibit automated collection. |
| Filter | Drop items older than 3 days and URLs already stored. |
| Tag | Company names, aliases and cashtags -> tickers. Words like MU, ON or META only count as $MU, (MU) or NASDAQ:MU. |
| Embed | Qwen3-Embedding-0.6B served by llama-swap (`[embedding] served_model`, `/v1/embeddings`), 1024 dims, with each model's text formats and instructions added by us (`embed.py` FORMATS). llama-swap swaps it in and out of the shared GPU; a run waits up to `wait_seconds` (15 min) for it, then leaves its articles for the next run with a ⚠️ in the digest. Chosen over EmbeddingGemma 2 (used until 2026-11-02) on the benchmark: duplicates AUC 0.79 vs 0.70, search nDCG 0.86 vs 0.82. Changing the model or dims needs a re-embedded database. |
| Classify | News vs analysis/opinion vs press release vs transcript: the feed's own label, then title and wire-service source rules, then nearest example. |
| Score | Similarity to each holding and topic decides relevance. |
| Group | Mean-centered headline similarity >= 0.65, the same with holdings' names masked out >= 0.30 (so one company's unrelated headlines don't merge), and published within 36 h -> one story; never merged: "X vs. Y" headlines, headlines naming different holdings, analyst actions by different brokerages; the main link is chosen by source type and priority. |
| Full text | Article pages for each relevant story's main link, only where robots.txt allows and not on paywalled sites. |
| LLM | Top ~80 stories: summary of up to 3 sentences, sentiment for the holding, key numbers (each checked against the article text), why it matters. Qwen3.8-27B via llama-swap (falls back to Gemma 4 26B-A4B); model and prompt chosen on the benchmark (`fin_news/bench`, `reports/bench-*.md`): facts covered 0.85 vs 0.58 for Gemma on the 44 held-out stories. Step timings are kept per run (`runs.stats.step_seconds`). |
| SEC contents | 8-K/6-K: the attached press release (EX-99.x) is read and summarized by the LLM (headline, results vs guidance, key numbers checked against the filing). Form 4: insider name, role, open-market buys/sells, price, value and 10b5-1 plan flag. |
| Market data | Twice a day, in the first run after 06:00 and after 14:00 (`[market] refresh_hours`; the 2pm fetch has that day's close): the Yahoo Finance analysis page per holding (robots.txt allows `/quote/`), giving price, returns vs the S&P 500, P/E, forward P/E, price/sales, 52-week range, analyst consensus and counts, price targets (average, median, range), each firm's latest rating/target action, next earnings. Daily closes build the chart; the first month is backfilled once per stock from its Google Finance page (robots.txt allows `/finance/`). Nasdaq's and Yahoo's JSON APIs are not used (robots.txt disallows them). Other tickers are fetched when looked up in the web app. Kept 120 days. |
| Holdings views | Holdings has four views. **Research**: the news/analyst table (unchanged). **Positions**: Fidelity's positions view for the `[returns]` accounts at the latest close: value, the day's change, gain vs cost basis (the export's basis, moved by later trades; matches Fidelity's Total gain/loss), cash, and a concentration line (`[returns] groups`). **Performance**: for 1M/3M/6M/YTD/1Y, invested and earned (end value - start value - money invested + dividends), a time-weighted % vs the S&P 500, a chart of value vs money put in vs the same money in the S&P, and month by month. **Buys vs S&P**: pick a window of buy dates (presets or custom); every buy, ESPP purchase and vest in it valued today, vs the same dollars in the S&P 500 on the same days, by month bought and by stock; buys under 30 days old are shown but not judged.
| Returns data | From each person's Fidelity activity exports (`users/<name>/fidelity/Accounts_History*.csv`, 90 days each, merged; duplicates across overlapping files counted once) for the accounts in their `[returns] accounts` (all accounts if not set; money-market funds are cash). For 1M/3M/6M/YTD/1Y: earned = end value - start value - money invested + dividends and interest, split into holdings already owned / new investments / dividends, with a time-weighted (Modified Dietz) % vs the S&P 500; each new investment (buy, ESPP purchase, vest) is compared with the same dollars put into the S&P 500 on the same day, grouped per stock. Shares on any date come from the newest positions export and the transactions since (checked: an older export is rebuilt exactly); sales take the oldest shares first. Daily closes for every symbol involved and ^GSPC come from Yahoo Finance's history page (a year once, then the last week, daily; robots.txt allows `/quote/`); funds Yahoo doesn't list use your own trade prices (shown as approximate). Ask gets the same summary. |
| Fund holdings | SEC Form N-PORT per fund (data.sec.gov submissions -> filing header naming the fund's series -> `primary_doc.xml`): every position with its share of the fund, as of the last quarter end (public ~60 days later; checked every 30 days). Pooled 401(k) funds use the public fund with the same strategy (`[funds] proxies`). Holdings are matched to tickers by company name; with them come your real exposure per stock (direct + your slice of each fund), shown on Holdings, stock pages, fund pages and in Ask. |
| Digest | Per person, `digests/<name>/YYYY-MM-DD_HHMM.md` and `latest.md`: heads-up (earnings, big stories, and for admins: a source that broke since the last run), company announcements, insider trades, other SEC filings, analyst moves (rating changes and moved targets, 3 days), holdings by size with price and average target, themes and funds, macro table, analysis, Reddit. Each source's numbers are on the Status page. |
| Retention | After every run: delete articles, stories and their vectors older than 14 days, and digest files older than 3 days (`[retention]`). |

## Tuning

Embedding models give every text a large shared component (with EmbeddingGemma 2 an unrelated cake recipe scored
~0.46 against "Nvidia"), so relevance uses **mean-centered** cosine: the average article vector is subtracted first,
which puts unrelated articles at ~0 and headline-named ones at ~0.35. The thresholds were retuned for Qwen3-Embedding
on the benchmark (2026-11-02). A holding counts only when the article names it; similarity
alone could not attribute unnamed articles to the right holding. Adjust in `config.toml` after reading
`digests/<name>/latest.md`: `dedup_threshold` (stories wrongly merged -> raise; duplicates still shown -> lower),
`ticker_text_min`, `topic_min`, `topic_min_z` (noise -> raise; missing stories -> lower).

Junk dropped at ingest: Yahoo option-contract quote pages and Motley Fool author pages.
Finnhub links are resolved to the publisher URL when the article text is fetched.

## License

MIT (see `LICENSE`). Not financial advice. Each news source, data API and subscription has its own terms; you are responsible for following them.
