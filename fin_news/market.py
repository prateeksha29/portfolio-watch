"""Prices, valuation and analyst views per stock, from the Yahoo Finance analysis page (one page per ticker; its
robots.txt allows /quote/ pages). Daily closes come from those snapshots; the first time a stock is seen, its last
month of closes is backfilled from its Google Finance page (robots.txt allows /finance/). Nasdaq's and Yahoo's JSON
APIs are not used: their robots.txt disallow all."""
import bisect
import datetime as dt
import json
import logging
import re
import threading
import time
from zoneinfo import ZoneInfo

from lxml import html as lxml_html

from .collect import Collector, iso, redact, run_by_host, utcnow
from .returns import BENCHMARK, months_back

log = logging.getLogger(__name__)

YAHOO = "https://finance.yahoo.com/quote/{}/analysis/"
YAHOO_QUOTE = "https://finance.yahoo.com/quote/{}/"  # fallback: Yahoo sometimes renders the analysis page empty;
                                                    # the quote page has no EPS forecasts
YAHOO_HISTORY = "https://finance.yahoo.com/quote/{}/history/?period1={}&period2={}"  # daily rows as an HTML table
GOOGLE = "https://www.google.com/finance/quote/{}:{}"
# Yahoo exchange codes -> Google Finance exchange names; unknown ones are tried as NASDAQ, then NYSE
GOOGLE_EXCHANGE = {"NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ", "NYQ": "NYSE",
                   "NYS": "NYSE", "ASE": "NYSEAMERICAN", "PCX": "NYSEARCA", "BTS": "BATS"}
DAILY_ROW = re.compile(r'\[[\d.]+,([\d.]+),[\d.]+,[\d.]+,"(\d{4}-\d{2}-\d{2})T16:00:00')  # [open, close, high, low, date
EXCHANGE_TZ = ZoneInfo("America/New_York")
TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.-][A-Z]{1,2})?$")
GRADE_ACTIONS = {"up": "upgrades", "down": "downgrades", "init": "starts at", "reit": "reiterates", "main": "keeps"}

NAME_SUFFIX = re.compile(r"[,\s]+(inc|corp|corporation|co|company|ltd|limited|plc|n\.?v|s\.?a|se|ag|holdings?|"
                         r"group|/[a-z]+)\.?$", re.I)
# Descriptive last words people leave out: 'Palantir' for Palantir Technologies, 'Meta' for Meta Platforms.
GENERIC_END = re.compile(r" (technologies|technology|platforms|systems|international|industries|enterprises|brands|"
                         r"solutions|communications|networks|semiconductor|semiconductors|software|entertainment|"
                         r"financial|motors|interactive|labs|therapeutics|pharmaceuticals|biosciences|"
                         r"markets|computer|computers|devices|global|worldwide|incorporated|the)$")
# Capitalised abbreviations that are also ticker symbols; in a question they mean the word, not the stock.
NOT_TICKERS = {"AI", "CEO", "CFO", "EPS", "ETF", "IPO", "USA", "US", "GDP", "CPI", "FED", "SEC", "IRA", "YTD", "ATH",
               "EV", "PE", "Q1", "Q2", "Q3", "Q4", "OK", "IT", "ALL", "NOW", "ON", "ARE", "BE", "SO", "ANY", "ONE"}

_live_lock = threading.Lock()
_no_history: set[str] = set()  # tickers Google had no history for, not retried until restart
_index: dict = {}


def _norm_name(name: str) -> str:
    n = name.lower().replace("&", " and ")
    while NAME_SUFFIX.search(n):
        n = NAME_SUFFIX.sub("", n)
    return re.sub(r"[^a-z0-9]+", " ", n).strip()


def company_index(cfg: dict) -> tuple[dict[str, tuple[str, str]], set[str]]:
    """(normalized company name -> (ticker, name), all tickers) from the SEC ticker list the pipeline downloads
    weekly, plus the configured aliases ('Google' -> GOOGL). The SEC list is ordered by size, so for a name with
    several share classes the first (main) one wins."""
    path = cfg["paths"]["data"] / "sec_company_tickers.json"
    mtime = path.stat().st_mtime if path.exists() else 0
    if _index.get("mtime") != mtime:
        names, tickers = {}, set()
        for t, aliases in cfg.get("aliases", {}).items():
            for a in aliases:
                names.setdefault(_norm_name(a), (t, aliases[0]))
        rows = json.loads(path.read_text()).values() if path.exists() else []
        for row in rows:
            t = row["ticker"].upper()
            if valid_ticker(t):
                tickers.add(t)
                names.setdefault(_norm_name(row["title"]), (t, row["title"]))
        for row in rows:  # second pass, so a full name always wins over a shortened one
            t = row["ticker"].upper()
            short = GENERIC_END.sub("", _norm_name(row["title"]))
            if valid_ticker(t) and short and short != _norm_name(row["title"]):
                names.setdefault(short, (t, row["title"]))
        _index.update(mtime=mtime, names=names, tickers=tickers)
    return _index["names"], _index["tickers"]


def find_companies(cfg: dict, text: str, limit: int = 5) -> tuple[tuple[str, str] | None, list[tuple[str, str]]]:
    """A search box entry -> (exact name match or None, other companies whose name starts with it)."""
    names, _ = company_index(cfg)
    key = _norm_name(text)
    exact = names.get(key)
    partial = []
    if len(key) >= 4:
        seen = {exact[0]} if exact else set()
        for n, (t, title) in names.items():
            if n.startswith(key + " ") and t not in seen:
                seen.add(t)
                partial.append((t, title))
                if len(partial) == limit:
                    break
    return exact, partial


def companies_in(cfg: dict, question: str) -> list[str]:
    """Stocks named in a question: tickers in capitals ('ORCL', '$orcl') and company names written with capitals
    ('Oracle', 'Advanced Micro Devices'), longest name first."""
    names, tickers = company_index(cfg)
    found = [m.upper() for m in re.findall(r"\$([A-Za-z]{1,5})\b", question)]
    found += [t for t in re.findall(r"\b([A-Z]{2,5}(?:\.[A-Z])?)\b", question) if t in tickers and t not in NOT_TICKERS]
    for run in re.findall(r"\b[A-Z][\w&'.-]*(?:\s+[A-Z][\w&'.-]*){0,3}", question):
        words = run.split()
        i = 0
        while i < len(words):
            for j in range(len(words), i, -1):
                hit = names.get(_norm_name(" ".join(words[i:j])))
                if hit and " ".join(words[i:j]).upper() not in NOT_TICKERS:
                    found.append(hit[0])
                    i = j - 1
                    break
            i += 1
    return list(dict.fromkeys(found))


def valid_ticker(t: str) -> bool:
    return bool(TICKER.match(t or ""))


def _raw(v):
    return v.get("raw") if isinstance(v, dict) else v


def parse_analysis(content: str | bytes, ticker: str) -> dict:
    """The quoteSummary JSON that Yahoo embeds in the analysis page -> a compact snapshot. Raises LookupError when
    the page has no data for the ticker (unknown symbol)."""
    doc = lxml_html.fromstring(content)
    res = None
    for s in doc.xpath("//script[@data-url]"):
        if f"/quoteSummary/{yahoo_symbol(ticker)}" not in s.get("data-url", ""):
            continue
        try:
            body = json.loads(json.loads(s.text)["body"])
            res = (body.get("quoteSummary", {}).get("result") or [None])[0]
        except (ValueError, KeyError, TypeError):
            continue
        if res:
            break
    if not res or "price" not in res:
        raise LookupError(f"No Yahoo Finance data for {ticker}")
    price, sd, fd = res["price"], res.get("summaryDetail", {}), res.get("financialData", {})
    perf = res.get("quoteUnadjustedPerformanceOverview", {})
    ret = perf.get("performanceOverview", {})
    earnings = res.get("calendarEvents", {}).get("earnings", {})
    actions = []
    history = (res.get("upgradeDowngradeHistory") or res.get("upgradeDowngradeHistoryLatest") or {}).get("history", [])
    for a in history[:20]:
        actions.append({"date": dt.datetime.fromtimestamp(a["epochGradeDate"], dt.timezone.utc).date().isoformat(),
                        "firm": a.get("firm"), "to": a.get("toGrade"), "from": a.get("fromGrade"),
                        "action": a.get("action"), "target": a.get("currentPriceTarget") or None,
                        "prior": a.get("priorPriceTarget") or None})
    eps = None
    trend = next((t for t in res.get("earningsTrend", {}).get("trend", []) if t.get("period") == "0q"), None)
    if trend and _raw(trend.get("earningsEstimate", {}).get("avg")) is not None:
        est, et, rev = trend["earningsEstimate"], trend.get("epsTrend", {}), trend.get("epsRevisions", {})
        eps = {"quarter_end": trend.get("endDate"), "avg": _raw(est.get("avg")), "low": _raw(est.get("low")),
               "high": _raw(est.get("high")), "analysts": _raw(est.get("numberOfAnalysts")),
               "year_ago": _raw(est.get("yearAgoEps")),
               "days_ago": {d: _raw(et.get(f"{d}daysAgo")) for d in (7, 30, 90)},
               # Yahoo spells these inconsistently (downLast7Days, downLast30days)
               "up_7": _raw(rev.get("upLast7days")), "up_30": _raw(rev.get("upLast30days")),
               "down_7": _raw(rev.get("downLast7Days", rev.get("downLast7days"))),
               "down_30": _raw(rev.get("downLast30days", rev.get("downLast30Days")))}
    history = [{"quarter": (h.get("quarter") or {}).get("fmt"), "actual": _raw(h.get("epsActual")),
                "estimate": _raw(h.get("epsEstimate")), "surprise": _raw(h.get("surprisePercent"))}
               for h in res.get("earningsHistory", {}).get("history", []) if _raw(h.get("epsActual")) is not None]
    periods = ("fiveDaysReturn", "oneMonthReturn", "threeMonthReturn", "ytdReturnPct", "oneYearTotalReturn")
    return {
        "ticker": ticker, "name": price.get("longName") or price.get("shortName") or ticker,
        "type": price.get("quoteType"), "currency": price.get("currency") or "USD", "exchange": price.get("exchange"),
        "price": _raw(price.get("regularMarketPrice")), "change_pct": _raw(price.get("regularMarketChangePercent")),
        "market_time": price.get("regularMarketTime"), "market_state": price.get("marketState"),
        "market_cap": _raw(price.get("marketCap")),
        "pe": _raw(sd.get("trailingPE")), "forward_pe": _raw(sd.get("forwardPE")),
        "price_to_sales": _raw(sd.get("priceToSalesTrailing12Months")), "beta": _raw(sd.get("beta")),
        "dividend_yield": _raw(sd.get("dividendYield")), "high_52w": _raw(sd.get("fiftyTwoWeekHigh")),
        "low_52w": _raw(sd.get("fiftyTwoWeekLow")), "avg_50d": _raw(sd.get("fiftyDayAverage")),
        "avg_200d": _raw(sd.get("twoHundredDayAverage")),
        "revenue_growth": _raw(fd.get("revenueGrowth")), "earnings_growth": _raw(fd.get("earningsGrowth")),
        "profit_margin": _raw(fd.get("profitMargins")),
        "target_mean": _raw(fd.get("targetMeanPrice")), "target_median": _raw(fd.get("targetMedianPrice")),
        "target_high": _raw(fd.get("targetHighPrice")), "target_low": _raw(fd.get("targetLowPrice")),
        "analysts": _raw(fd.get("numberOfAnalystOpinions")), "rating": fd.get("recommendationKey"),
        "rating_mean": _raw(fd.get("recommendationMean")),
        "trend": res.get("recommendationTrend", {}).get("trend", [])[:4],
        "actions": actions,
        "returns": {p: _raw(ret.get(p)) for p in periods},
        "earnings_date": next((d.get("fmt") for d in earnings.get("earningsDate", [])), None),
        "eps_estimate": _raw(earnings.get("earningsAverage")),
        "eps": eps, "eps_history": history[-4:],
    }


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


class Market:
    """Fetches snapshots and closes and stores them in the quotes and prices tables."""

    def __init__(self, cfg: dict, store, collector: Collector | None = None):
        self.cfg, self.store = cfg, store
        self.collector = collector or Collector(cfg)

    # ---------- fetching ----------

    def _yahoo(self, job) -> dict:
        for page in (YAHOO, YAHOO, YAHOO_QUOTE):  # the analysis page (with EPS forecasts) twice, then the quote page
            url = page.format(yahoo_symbol(job["ticker"]))
            if not self.collector.robots.allowed(url):
                return {"error": "robots.txt disallows"}
            r = self.collector.http.get(url)
            if r.status_code == 404:
                break
            r.raise_for_status()
            try:
                return {"quote": parse_analysis(r.content, job["ticker"])}
            except LookupError:
                time.sleep(self.collector.delay_for("finance.yahoo.com"))
        raise LookupError(f"No Yahoo Finance data for {job['ticker']}")

    def _google(self, job) -> dict:
        """About 21 daily closes for the last month, from the data Google Finance embeds in its quote page. Today's
        row is skipped: until the close it is the live price, and the Yahoo snapshot covers today."""
        known = GOOGLE_EXCHANGE.get(job.get("exchange") or "")
        today = dt.datetime.now(EXCHANGE_TZ).date().isoformat()
        for exchange in ([known] if known else ["NASDAQ", "NYSE"]):
            url = GOOGLE.format(job["ticker"], exchange)
            if not self.collector.robots.allowed(url):
                return {"error": "robots.txt disallows"}
            r = self.collector.http.get(url)
            r.raise_for_status()
            head = f'[[[["{job["ticker"]}","{exchange}"]'
            best = []
            for block in re.findall(r"AF_initDataCallback\(\{key: 'ds:\d+'.*?data:(.*?), sideChannel", r.text, re.S):
                if block.startswith(head):
                    rows = [(d, float(c)) for c, d in DAILY_ROW.findall(block) if d < today]
                    best = max(best, rows, key=len)
            if len(best) >= 5:
                return {"closes": best}
            time.sleep(self.collector.delay_for("www.google.com"))
        raise LookupError(f"No Google Finance price history for {job['ticker']}")

    def _backfill_jobs(self, tickers) -> list[dict]:
        """Stocks with fewer than 15 closes in the last 30 days: new holdings and first lookups."""
        return [{"ticker": t, "host": "www.google.com", "exchange": (self.latest(t) or {}).get("exchange")}
                for t in tickers if len(self.closes(t)) < 15]

    def refresh(self, tickers: list[str]) -> dict:
        """Snapshot every ticker (one Yahoo page each, paced), store today's close, and backfill a month of closes
        from Google Finance for tickers with fewer than 15. Returns source health in the pipeline's format."""
        health = {}

        def tally(name, results):
            h = health.setdefault(name, {"requests": 0, "errors": 0, "not_modified": 0, "items": 0, "last_error": ""})
            for r in results:
                h["requests"] += 1
                if r.get("error"):
                    h["errors"] += 1
                    h["last_error"] = r["error"][:200]
                else:
                    h["items"] += 1

        jobs = [{"ticker": t, "host": "finance.yahoo.com"} for t in tickers]
        results = run_by_host(jobs, self._yahoo, self.collector.delay_for, 1)
        for r in results:
            if r.get("quote"):
                self.save(r["quote"])
            elif r.get("error"):
                log.warning("Market data for %s: %s", r["job"]["ticker"], r["error"])
        tally("Yahoo Finance (prices, targets)", results)
        results = run_by_host(self._backfill_jobs(tickers), self._google, self.collector.delay_for, 1)
        for r in results:
            self.save_closes(r["job"]["ticker"], r.get("closes", []))
        if results:
            tally("Google Finance (price history)", results)
        self.store.commit()
        return health

    def live(self, ticker: str) -> dict:
        """Fetch one ticker now (web app refresh, or a ticker not looked up today). Raises on failure."""
        with _live_lock:
            out = self._yahoo({"ticker": ticker})
            if "quote" not in out:
                raise RuntimeError(out.get("error", "fetch failed"))
            self.save(out["quote"])
            self.backfill(ticker)
            self.store.commit()
            return out["quote"]

    def live_price(self, ticker: str) -> dict:
        """Like live(), but price only: one fetch of the light quote page. It lacks the EPS forecasts and carries
        only 4 analyst actions, so those come from the stored snapshot. Falls back to live() if it won't parse."""
        url = YAHOO_QUOTE.format(yahoo_symbol(ticker))
        if not self.collector.robots.allowed(url):
            raise RuntimeError("robots.txt disallows")
        r = self.collector.http.get(url)
        r.raise_for_status()
        try:
            fresh = parse_analysis(r.content, ticker)
        except LookupError:
            return self.live(ticker)
        with _live_lock:
            stored = self.latest(ticker) or {}
            stored.pop("fetched", None)
            merged = {**stored, **{k: v for k, v in fresh.items() if v not in (None, [], {}, "")}}
            seen = {(a["date"], a["firm"]) for a in fresh["actions"]}  # the quote page's 4 + the older stored ones
            merged["actions"] = (fresh["actions"] + [a for a in stored.get("actions", [])
                                                     if (a["date"], a["firm"]) not in seen])[:20]
            self.save(merged)
            self.backfill(ticker)
            self.store.commit()
            return merged

    def _history(self, job) -> dict:
        """Daily closes (split-adjusted, not dividend-adjusted) from Yahoo's history page, for any symbol Yahoo
        lists: stocks, mutual funds, the S&P 500 index (^GSPC)."""
        start = dt.datetime.combine(job["since"], dt.time(), dt.timezone.utc)
        url = YAHOO_HISTORY.format(yahoo_symbol(job["ticker"]).replace("^", "%5E"), int(start.timestamp()),
                                   int(time.time()) + 86400)
        if not self.collector.robots.allowed(url):
            return {"error": "robots.txt disallows"}
        r = self.collector.http.get(url)
        r.raise_for_status()
        closes = []
        for tr in lxml_html.fromstring(r.content).iter("tr"):
            cells = [c.text_content().strip() for c in tr.findall("td")]
            if len(cells) == 7:  # dividend and split rows have two cells
                try:
                    closes.append((dt.datetime.strptime(cells[0], "%b %d, %Y").date().isoformat(),
                                   float(cells[4].replace(",", ""))))
                except ValueError:
                    continue
        if not closes:
            raise LookupError(f"No price history for {job['ticker']}")
        return {"closes": closes}

    def history(self, tickers: list[str], since: dt.date) -> dict:
        """Store daily closes since `since` for each ticker (one page each, paced). Returns source health."""
        jobs = [{"ticker": t, "since": since, "host": "finance.yahoo.com"} for t in tickers]
        h = {"requests": 0, "errors": 0, "not_modified": 0, "items": 0, "last_error": ""}
        for r in run_by_host(jobs, self._history, self.collector.delay_for, 1):
            h["requests"] += 1
            if r.get("closes"):
                self.save_closes(r["job"]["ticker"], r["closes"])
                h["items"] += 1
            else:
                h["errors"] += 1
                h["last_error"] = (r.get("error") or "no rows")[:200]
                log.warning("Price history for %s: %s", r["job"]["ticker"], h["last_error"])
        self.store.commit()
        return h

    def backfill(self, ticker: str) -> None:
        """A month of closes from Google Finance when the stock has fewer than 15 (once per stock)."""
        if ticker in _no_history:
            return
        for job in self._backfill_jobs([ticker]):
            try:
                self.save_closes(ticker, self._google(job)["closes"])
                self.store.commit()
            except Exception as e:  # the snapshot is still useful without the history
                _no_history.add(ticker)
                log.warning("Price history for %s: %s", ticker, redact(str(e)))

    # ---------- storage ----------

    def save(self, q: dict) -> None:
        now = utcnow()
        self.store.db.execute("INSERT OR REPLACE INTO quotes(ticker, date, fetched, data) VALUES (?, ?, ?, ?)",
                              (q["ticker"], now.date().isoformat(), iso(now), json.dumps(q)))
        if q.get("price") and q.get("market_time"):
            # Before the open this is the previous session's close; a later fetch on the same day replaces it.
            day = dt.datetime.fromtimestamp(q["market_time"], EXCHANGE_TZ).date().isoformat()
            self.save_closes(q["ticker"], [(day, q["price"])])

    def save_closes(self, ticker: str, closes) -> None:
        self.store.db.executemany("INSERT OR REPLACE INTO prices(ticker, date, close) VALUES (?, ?, ?)",
                                  [(ticker, d, c) for d, c in closes])

    def latest(self, ticker: str) -> dict | None:
        row = self.store.db.execute("SELECT fetched, data FROM quotes WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                                    (ticker,)).fetchone()
        if not row:
            return None
        q = {**json.loads(row["data"]), "fetched": row["fetched"]}
        day = (dt.datetime.fromtimestamp(q["market_time"], EXCHANGE_TZ).date().isoformat()
               if q.get("market_time") else None)
        q["sp500_returns"] = self.sp500_returns(day)
        return q

    def sp500_returns(self, day: str | None = None) -> dict:
        """The S&P 500's price returns to its close on or before `day` (default: the latest stored), from the daily
        closes the Returns step stores. Not Yahoo's benchmark row: for a foreign company that is its home market's
        index (TSM: Taiwan's), not the S&P 500. A period the stored closes don't reach is None."""
        rows = self.store.db.execute("SELECT date, close FROM prices WHERE ticker = ? AND date <= ? ORDER BY date",
                                     (BENCHMARK, day or "9999-12-31")).fetchall()
        out = dict.fromkeys(("fiveDaysReturn", "oneMonthReturn", "threeMonthReturn", "ytdReturnPct",
                             "oneYearTotalReturn"))
        if len(rows) < 2:
            return out
        days, end = [r[0] for r in rows], rows[-1][1]
        last = dt.date.fromisoformat(rows[-1][0])

        def close_on(d: dt.date) -> float | None:  # the last close on or before d, within a week
            i = bisect.bisect_right(days, d.isoformat())
            return rows[i - 1][1] if i and (d - dt.date.fromisoformat(days[i - 1])).days <= 6 else None

        if len(rows) >= 6:
            out["fiveDaysReturn"] = end / rows[-6][1] - 1
        for key, start in (("oneMonthReturn", months_back(last, 1)), ("threeMonthReturn", months_back(last, 3)),
                           ("ytdReturnPct", dt.date(last.year - 1, 12, 31)),
                           ("oneYearTotalReturn", months_back(last, 12))):
            c = close_on(start)
            out[key] = end / c - 1 if c else None
        return out

    def closes(self, ticker: str, days: int = 30) -> list[tuple[str, float]]:
        since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        return [(r[0], r[1]) for r in self.store.db.execute(
            "SELECT date, close FROM prices WHERE ticker = ? AND date >= ? ORDER BY date", (ticker, since))]


def market_open(now: dt.datetime | None = None) -> bool:
    """Weekdays 9:30-16:15 New York time (15 min of slack for the closing print). No holiday calendar."""
    t = (now or dt.datetime.now(EXCHANGE_TZ)).astimezone(EXCHANGE_TZ)
    return t.weekday() < 5 and dt.time(9, 30) <= t.time() <= dt.time(16, 15)


def refresh_slot(cfg: dict) -> str | None:
    """The current market-data slot: today's date and the latest [market] refresh_hours hour already reached
    (6 = before the open, for analyst actions; 14 = after the close, for that day's prices). None before the first."""
    m = cfg.get("market", {})
    hours = m.get("refresh_hours") or [m.get("refresh_hour", 6)]
    now = dt.datetime.now()
    reached = [h for h in hours if now.hour >= h]
    return f"{now.date().isoformat()}@{max(reached)}" if reached else None


def refresh_due(cfg: dict, store) -> bool:
    """In the first pipeline run of each slot (see refresh_slot)."""
    if not cfg.get("market", {}).get("enabled", True):
        return False
    slot = refresh_slot(cfg)
    return slot is not None and store.get_meta("market_refreshed") != slot


def stale(q: dict | None, hours: float = 20) -> bool:
    if not q or "eps_history" not in q:  # missing, or saved before EPS forecasts were collected
        return True
    fetched = dt.datetime.strptime(q["fetched"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return (utcnow() - fetched).total_seconds() > hours * 3600


# ---------- presentation ----------

def money(v: float | None) -> str:
    if v is None:
        return "n/a"
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    return f"${v:,.2f}"


def pct(v: float | None, signed: bool = True) -> str:
    return "n/a" if v is None else (f"{v:+.1%}" if signed else f"{v:.1%}")


def upside(q: dict) -> float | None:
    return q["target_mean"] / q["price"] - 1 if q.get("target_mean") and q.get("price") else None


def counts(q: dict, period: str = "0m") -> dict | None:
    return next((t for t in q.get("trend", []) if t.get("period") == period), None)


def action_line(a: dict) -> str:
    """'Cantor Fitzgerald keeps Overweight, target $350' / 'BMO downgrades to Market Perform, target $140 → $125'"""
    verb = GRADE_ACTIONS.get(a.get("action") or "", "rates")
    if a.get("action") in ("up", "down"):
        text = f"{a['firm']} {verb} to {a['to']}" + (f" (from {a['from']})" if a.get("from") else "")
    else:
        text = f"{a['firm']} {verb} {a['to']}"
    if a.get("target"):
        prior = a.get("prior")
        text += (f", target ${prior:,.0f} → ${a['target']:,.0f}" if prior and prior != a["target"]
                 else f", target ${a['target']:,.0f}")
    return text


def notable(a: dict) -> bool:
    """A rating change, a new rating, or a moved target; not 'reiterates, same target'."""
    return a.get("action") in ("up", "down", "init") or bool(a.get("prior") and a.get("target")
                                                             and a["prior"] != a["target"])


def eps_change(e: dict, days: int) -> float | None:
    """Change in the quarter's average EPS forecast over the last `days` (7, 30 or 90)."""
    before = (e.get("days_ago") or {}).get(str(days)) or (e.get("days_ago") or {}).get(days)
    return e["avg"] / before - 1 if before and e.get("avg") else None


def eps_lines(q: dict) -> list[str]:
    """'EPS forecast for the quarter ending 2026-10-31: $2.47 (44 analysts) ...' for Ask and the stock page."""
    e, out = q.get("eps"), []
    if e:
        line = f"EPS forecast for the quarter ending {e['quarter_end']}: ${e['avg']:.2f} ({e['analysts']} analysts"
        if e.get("year_ago"):
            line += f"; ${e['year_ago']:.2f} a year earlier, {e['avg'] / e['year_ago'] - 1:+.0%}"
        line += ")"
        moves = [f"{label} {pct(eps_change(e, d))}" for d, label in ((30, "30 days"), (90, "90 days"))
                 if eps_change(e, d) is not None]
        if moves:
            line += "; change in the forecast: " + ", ".join(moves)
        line += (f"; analysts raising/cutting it in the last 30 days: {e.get('up_30') or 0} up, "
                 f"{e.get('down_30') or 0} down")
        out.append(line)
    hist = q.get("eps_history") or []
    if hist:
        out.append("Last quarters, EPS actual vs forecast: " + "; ".join(
            f"{h['quarter']} ${h['actual']:.2f} vs ${h['estimate']:.2f} ({pct(h['surprise'])})"
            for h in hist if h.get("estimate") is not None and h.get("surprise") is not None))
    return out


def revision_flag(q: dict) -> str | None:
    """A digest line when the quarter's average EPS forecast moved 2% or more in the last 7 days. Based on the
    average, not Yahoo's counts of analysts revising, which can exceed the number of analysts."""
    e = q.get("eps")
    change = eps_change(e, 7) if e else None
    if change is None or abs(change) < 0.02:
        return None
    before = e["days_ago"].get("7") or e["days_ago"].get(7)
    return (f"{'📈 Analysts raised' if change > 0 else '📉 Analysts cut'} this quarter's EPS forecast "
            f"{abs(change):.0%} in the last 7 days (${before:.2f} → ${e['avg']:.2f})")


def price_when(q: dict, closes: list[tuple[str, float]]) -> str:
    """Which trading day the quoted price is from: the close it matches, else the (local) day it was fetched. The
    fetch date alone misled the model: a Sunday fetch of Friday's close read as a drop on Sunday."""
    from .askfmt import day
    d = next((d for d, c in reversed(closes) if abs(c - q["price"]) < 0.006), None)
    if d:
        return f"close on {day(d)}"
    try:
        local = dt.datetime.fromisoformat(q["fetched"].replace("Z", "+00:00")).astimezone().date().isoformat()
    except (KeyError, ValueError):
        local = (q.get("fetched") or "")[:10]
    return f"latest trade, fetched {day(local)}"


def context_lines(q: dict, closes: list[tuple[str, float]]) -> list[str]:
    """Market data for the Ask prompt."""
    lines = [f"Market data (Yahoo Finance): price ${q['price']:,.2f} ({price_when(q, closes)})"
             + (f", {q['change_pct']:+.1%} that day" if q.get("change_pct") is not None else "")]
    r, b = q.get("returns", {}), q.get("sp500_returns", {})
    if r.get("oneMonthReturn") is not None:
        lines.append(f"Returns: 5 days {pct(r.get('fiveDaysReturn'))}, 1 month {pct(r.get('oneMonthReturn'))}, "
                     f"3 months {pct(r.get('threeMonthReturn'))}, YTD {pct(r.get('ytdReturnPct'))}, 1 year "
                     f"{pct(r.get('oneYearTotalReturn'))} (S&P 500: 1 month {pct(b.get('oneMonthReturn'))}, "
                     f"YTD {pct(b.get('ytdReturnPct'))}, 1 year {pct(b.get('oneYearTotalReturn'))})")
    if len(closes) >= 2:
        lines.append(f"Closes over the last {len(closes)} trading days: low ${min(c for _, c in closes):,.2f}, "
                     f"high ${max(c for _, c in closes):,.2f}")
    val = [f"{k} {v:.1f}" for k, v in (("trailing P/E", q.get("pe")), ("forward P/E", q.get("forward_pe")),
                                       ("price/sales", q.get("price_to_sales")), ("beta", q.get("beta"))) if v]
    val.append(f"market cap {money(q.get('market_cap'))}")
    if q.get("low_52w") and q.get("high_52w"):
        val.append(f"52-week range ${q['low_52w']:,.2f}-${q['high_52w']:,.2f}")
    if q.get("revenue_growth") is not None:
        val.append(f"revenue growth {pct(q['revenue_growth'])} year over year")
    lines.append("Valuation: " + ", ".join(val))
    if q.get("target_mean"):
        c = counts(q)
        split = (f" ({c['strongBuy']} strong buy, {c['buy']} buy, {c['hold']} hold, {c['sell']} sell, "
                 f"{c['strongSell']} strong sell)" if c else "")
        lines.append(f"Analysts: {q.get('analysts')} analysts, consensus {(q.get('rating') or 'n/a').replace('_', ' ')}"
                     f"{split}; price target average ${q['target_mean']:,.2f} ({pct(upside(q))} vs price), median "
                     f"${q['target_median']:,.2f}, range ${q['target_low']:,.2f}-${q['target_high']:,.2f}")
    lines += eps_lines(q)
    recent = [f"{a['date']} {action_line(a)}" for a in q.get("actions", [])[:6]]
    if recent:
        lines.append("Latest analyst actions: " + "; ".join(recent))
    return lines


def chart_svg(closes: list[tuple[str, float]], q: dict | None) -> str:
    """30-day closing prices with the analyst target range (band) and average target (dashed). Inline SVG coloured
    by the page's CSS variables; the y-axis covers the prices and the average target, and targets outside it are
    labelled at the edge."""
    if len(closes) < 2:
        return ""
    w, h, left, right, top, bottom = 600, 220, 8, 92, 12, 22
    prices = [c for _, c in closes]
    mean = (q or {}).get("target_mean")
    lo, hi = min(prices + ([mean] if mean else [])), max(prices + ([mean] if mean else []))
    pad = (hi - lo) * 0.08 or hi * 0.02
    lo, hi = lo - pad, hi + pad

    def x(i):
        return left + (w - left - right) * i / (len(closes) - 1)

    def y(v):
        return top + (h - top - bottom) * (hi - min(max(v, lo), hi)) / (hi - lo)

    parts, labels = [], []  # labels: (y, text, class) on the right edge
    t_low, t_high = (q or {}).get("target_low"), (q or {}).get("target_high")
    if t_low and t_high:
        parts.append(f'<rect x="{left}" y="{y(t_high):.1f}" width="{w - left - right}" '
                     f'height="{max(y(t_low) - y(t_high), 0):.1f}" fill="var(--pos)" opacity="0.10"/>')
        labels.append((y(t_high) + 4, f"high ${t_high:,.0f}{' ↑' if t_high > hi else ''}", "lbl"))
        labels.append((y(t_low) + 4, f"low ${t_low:,.0f}{' ↓' if t_low < lo else ''}", "lbl"))
    if mean:
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(mean):.1f}" y2="{y(mean):.1f}" stroke="var(--pos)" '
                     f'stroke-dasharray="5 4" stroke-width="1.5"/>')
        labels.append((y(mean) + 4, f"avg ${mean:,.0f}", "lbl tgt"))
    labels.sort()
    for i in range(1, len(labels)):  # keep 15px between labels
        if labels[i][0] - labels[i - 1][0] < 15:
            labels[i] = (labels[i - 1][0] + 15, *labels[i][1:])
    for ly, text, cls in labels:
        parts.append(f'<text x="{w - right + 6}" y="{min(ly, h - bottom + 4):.1f}" class="{cls}">{text}</text>')
    points = " ".join(f"{x(i):.1f},{y(c):.1f}" for i, c in enumerate(prices))
    parts.append(f'<polyline points="{points}" fill="none" stroke="var(--accent)" stroke-width="2.5" '
                 f'stroke-linejoin="round"/>')
    parts.append(f'<circle cx="{x(len(prices) - 1):.1f}" cy="{y(prices[-1]):.1f}" r="4" fill="var(--accent)"/>')
    for i, anchor in ((0, "start"), (len(closes) - 1, "end")):
        label = dt.date.fromisoformat(closes[i][0]).strftime("%b %d")
        parts.append(f'<text x="{x(i):.1f}" y="{h - 4}" text-anchor="{anchor}" class="lbl">{label}</text>')
    hi_i, lo_i = prices.index(max(prices)), prices.index(min(prices))
    for i, dy in ((hi_i, -7), (lo_i, 15)):
        anchor, dx = ("start", 0) if i < len(prices) / 2 else ("end", 0)
        if mean and abs(y(prices[i]) + dy - 4 - y(mean)) < 12:  # would sit on the target line: beside the point
            dy, dx = 4, (8 if anchor == "start" else -8)
        parts.append(f'<text x="{x(i) + dx:.1f}" y="{y(prices[i]) + dy:.1f}" text-anchor="{anchor}" '
                     f'class="lbl">${prices[i]:,.2f}</text>')
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Closing prices, last '
            f'{len(closes)} trading days">{"".join(parts)}</svg>')
