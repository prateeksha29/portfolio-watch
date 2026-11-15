"""Discovery: new stock ideas, once a day.

- A private add-on (fin_news/advisor.py, when present): a newsletter's recent recommendations of stocks the owner
  doesn't hold, from the picks it already read. A one-sentence reason comes from its write-up (local LLM).
- Insider buying: Finviz's insider page (robots.txt allows /insidertrading; its screener pages are disallowed) lists
  about a week of Form 4 open-market purchases. Read once a day and kept, so buys add up across days. Only officers
  and directors of companies worth $1B+ count (not funds, 10% owners or fund trustees). A stock qualifies when 2+
  insiders bought $100k+ in total within 30 days, or a CEO, CFO, president or chair bought $250k+.

Each new idea gets one Yahoo snapshot (context, and its price the day it was flagged) and is shown once; the owner can
watch or dismiss it. Watched ideas are priced daily, and every idea once more 30 days after it was flagged, so the
Ideas page can show how each source's ideas did against the S&P 500."""
import datetime as dt
import json
import logging
import re

from lxml import html as lxml_html

from .addons import advisor
from .market import EXCHANGE_TZ, Market, valid_ticker

log = logging.getLogger(__name__)

INSIDERS = "https://finviz.com/insidertrading?tc=1"  # tc=1: purchases only
GROUP = "Finviz (insider buying)"
SCHEMA = """
CREATE TABLE IF NOT EXISTS insider_buys(filing TEXT, ticker TEXT, owner TEXT, role TEXT, date TEXT, price REAL,
  shares REAL, value REAL, company TEXT, industry TEXT, market_cap REAL, PRIMARY KEY(filing, ticker, date, shares, price));
CREATE TABLE IF NOT EXISTS ideas(key TEXT PRIMARY KEY, ticker TEXT, source TEXT, flagged TEXT, data TEXT,
  status TEXT DEFAULT 'new', status_date TEXT, price REAL, price_date TEXT, sp500 REAL, price_30d REAL,
  sp500_30d REAL);
-- Each person's Watch / Dismiss of an idea (ideas.status is no longer used; ideas are shared, choices aren't)
CREATE TABLE IF NOT EXISTS idea_status(user TEXT, key TEXT, status TEXT, status_date TEXT, PRIMARY KEY(user, key));
"""
IDEA_COLS = "i.key, i.ticker, i.source, i.flagged, i.data, i.price, i.price_date, i.sp500, i.price_30d, i.sp500_30d"
WATCHED = "key IN (SELECT key FROM idea_status WHERE status = 'watch')"  # by anyone: priced daily
SOURCES = {**({advisor.IDEA_SOURCE[0]: advisor.IDEA_SOURCE[1]} if advisor else {}), "insider": "Insider buying"}
ICONS = {**({advisor.IDEA_SOURCE[0]: advisor.IDEA_SOURCE[2]} if advisor else {}), "insider": "👔"}
INSIDER_ROLE = re.compile(r"\b(CEO|CFO|COO|Chief|President|Chair|Director|Officer|EVP|SVP|VP|Treasurer|Counsel|"
                          r"Secretary)", re.I)
NOT_INSIDER = re.compile(r"10%|Trustee|Control Person|See Remarks", re.I)  # funds and holding companies
SENIOR = re.compile(r"\b(CEO|CFO|COO|Chief (Executive|Financial|Operating)|(?<!Vice )President|Chair)", re.I)
FUNDS = re.compile(r"Fund", re.I)  # closed-end funds and ETFs: their trustees' buys say little
SP500 = "^GSPC"


def usd(v: float | None) -> str:
    """$2.1M / $250k / $8,400"""
    if v is None:
        return "n/a"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e4:
        return f"${v / 1e3:.0f}k"
    return f"${v:,.0f}"


def _cap(s: str | None) -> float | None:
    m = re.match(r"([\d.]+)([KMBT])$", (s or "").strip())
    return float(m[1]) * {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m[2]] if m else None


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def parse_insiders(page: str) -> list[dict]:
    """Purchases on Finviz's insider page, one per Form 4 row."""
    out = []
    for tr in lxml_html.fromstring(page).xpath('//tr[contains(@class, "fv-insider-row")]'):
        td = tr.xpath("./td")
        if len(td) < 10 or td[4].text_content().strip() != "Buy":  # 1-cell rows are ad banners
            continue
        link, filing = td[0].xpath(".//a"), td[9].xpath(".//a/@href")
        try:
            day = dt.datetime.strptime(td[3].text_content().strip(), "%b %d '%y").date()
        except ValueError:
            continue
        if not link or not filing:
            continue
        out.append({"ticker": link[-1].text_content().strip().replace("-", "."), "owner": td[1].text_content().strip(),
                    "role": td[2].text_content().strip(), "date": day.isoformat(),
                    "price": _num(td[5].text_content()), "shares": _num(td[6].text_content()),
                    "value": _num(td[7].text_content()), "filing": filing[0],
                    "company": td[0].get("data-boxover-company"), "industry": td[0].get("data-boxover-industry"),
                    "market_cap": _cap(td[0].get("data-boxover-value"))})
    return out


def _create(store) -> None:
    store.db.executescript(SCHEMA)
    if "price_date" not in {r[1] for r in store.db.execute("PRAGMA table_info(ideas)")}:
        store.db.execute("ALTER TABLE ideas ADD COLUMN price_date TEXT")  # ideas recorded on 2026-10-31


def _owner(name: str) -> str:
    return name.title() if name.isupper() else name


class Discovery:
    def __init__(self, cfg: dict, store, collector, market: Market | None = None):
        self.cfg, self.store, self.collector = cfg, store, collector
        self.c = cfg.get("discovery", {})
        self.market = market or Market(cfg, store, collector)
        _create(store)

    @property
    def enabled(self) -> bool:
        return self.c.get("enabled", True)

    def due(self) -> bool:
        """Once a day, in the first run at or after refresh_hour (after an add-on has read its picks)."""
        now = dt.datetime.now()
        return (self.enabled and now.hour >= self.c.get("refresh_hour", 6)
                and self.store.get_meta("discovery_ran") != now.date().isoformat())

    # ---------- sources ----------

    def read_insiders(self) -> dict:
        health = {"requests": 1, "errors": 0, "not_modified": 0, "items": 0, "last_error": ""}
        try:
            if not self.collector.robots.allowed(INSIDERS):
                raise PermissionError(f"robots.txt disallows {INSIDERS}")
            r = self.collector.http.get(INSIDERS)
            if not all(self.collector.robots.allowed(u) for u in [h.url for h in r.history] + [r.url]):
                raise PermissionError(f"Finviz now redirects to {r.url}, which robots.txt disallows")  # as its screener does
            r.raise_for_status()
            rows = parse_insiders(r.text)
            if not rows:
                raise LookupError("no purchases found on the Finviz insider page (layout changed?)")
            self.store.db.executemany(
                "INSERT OR REPLACE INTO insider_buys(filing, ticker, owner, role, date, price, shares, value, company, "
                "industry, market_cap) VALUES (:filing, :ticker, :owner, :role, :date, :price, :shares, :value, "
                ":company, :industry, :market_cap)", rows)
            keep = (dt.date.today() - dt.timedelta(days=self.c.get("keep_insider_days", 120))).isoformat()
            self.store.db.execute("DELETE FROM insider_buys WHERE date < ?", (keep,))
            self.store.commit()
            health["items"] = len(rows)
        except Exception as e:
            health["errors"] = 1
            health["last_error"] = str(e)[:200]
            log.warning("Finviz insider page: %s", e)
        return health

    def insider_candidates(self, held: set[str]) -> list[dict]:
        days = self.c.get("insider_days", 30)
        since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        by: dict[str, list] = {}
        for r in self.store.db.execute("SELECT * FROM insider_buys WHERE date >= ? ORDER BY date", (since,)):
            if (r["ticker"] in held or not valid_ticker(r["ticker"]) or NOT_INSIDER.search(r["role"])
                    or not INSIDER_ROLE.search(r["role"]) or FUNDS.search(r["industry"] or "")
                    or (r["market_cap"] or 0) < self.c.get("min_market_cap", 1e9)
                    or (r["price"] or 0) < self.c.get("min_price", 5) or not r["value"]):
                continue
            by.setdefault(r["ticker"], []).append(dict(r))
        out = []
        for t, rows in by.items():
            people: dict[str, dict] = {}
            for r in rows:
                p = people.setdefault(r["owner"], {"owner": _owner(r["owner"]), "role": r["role"], "value": 0.0,
                                                   "last": r["date"]})
                p["value"] += r["value"]
                p["last"] = max(p["last"], r["date"])
            total = sum(p["value"] for p in people.values())
            senior = sum(p["value"] for p in people.values() if SENIOR.search(p["role"]))
            if not ((len(people) >= 2 and total >= self.c.get("cluster_min_value", 1e5))
                    or senior >= self.c.get("senior_min_value", 2.5e5)):
                continue
            people_list = sorted(people.values(), key=lambda p: -p["value"])
            first, last = rows[0]["date"], max(r["date"] for r in rows)
            who = (f"{len(people)} insiders" if len(people) > 1 else people_list[0]["role"])
            out.append({"ticker": t, "source": "insider", "name": rows[-1]["company"], "sector": rows[-1]["industry"],
                        "market_cap": rows[-1]["market_cap"], "value": total,
                        "headline": f"{who} bought {usd(total)} ({_span(first, last)})",
                        "people": [{**p, "value": round(p["value"])} for p in people_list], "last_buy": last})
        return out

    # ---------- the daily run ----------

    def run(self, held: set[str], use_llm: bool = True) -> dict:
        """Read the sources, record new ideas, price ideas that need it. Returns {"health", "new": [idea summaries]}."""
        health = {GROUP: self.read_insiders()}
        today = dt.date.today().isoformat()
        known = {r["key"]: r for r in self.store.db.execute("SELECT key, ticker, source, flagged, data FROM ideas")}
        recent_insider = {r["ticker"]: r["key"] for r in known.values() if r["source"] == "insider"
                          and r["flagged"] >= (dt.date.today() - dt.timedelta(days=90)).isoformat()}
        new, late = [], []  # late: add-on ideas whose write-up came after they were flagged (login was expired)
        for c in advisor.idea_candidates(self.store, self.c, held) if advisor else []:
            key = f"{c['source']}:{c['ticker']}:{c['rec_date']}"
            if key not in known:
                new.append((key, c))
            elif c.get("thesis") and not json.loads(known[key]["data"]).get("why"):
                late.append((key, c))
        for c in self.insider_candidates(held):
            if c["ticker"] in recent_insider:  # flagged in the last 90 days: add the newer buys, don't flag again
                key = recent_insider[c["ticker"]]
                self.store.db.execute("UPDATE ideas SET data = ? WHERE key = ?",
                                      (json.dumps({**json.loads(known[key]["data"]), **c}), key))
            else:
                new.append((f"insider:{c['ticker']}:{today}", c))

        reasons = self._reasons([c for _, c in new + late if c.get("thesis")]) if use_llm else {}
        for key, c in late:
            if reasons.get(c["ticker"]):
                self.store.db.execute("UPDATE ideas SET data = ? WHERE key = ?", (json.dumps(
                    {**json.loads(known[key]["data"]), "why": reasons[c["ticker"]]}), key))
        for key, c in new:
            from_writeup = "thesis" in c
            c = {k: v for k, v in c.items() if k != "thesis"}
            if from_writeup and reasons.get(c["ticker"]):
                c["why"] = reasons[c["ticker"]]
            self.store.db.execute("INSERT OR IGNORE INTO ideas(key, ticker, source, flagged, data) VALUES (?, ?, ?, ?, ?)",
                                  (key, c["ticker"], c["source"], today, json.dumps(c)))
        self.store.commit()

        health.update(self._price(today))
        self.store.set_meta("discovery_ran", today)
        log.info("Discovery: %d new ideas (%s)", len(new), ", ".join(c["ticker"] for _, c in new) or "none")
        return {"health": health, "new": [{"ticker": c["ticker"], "name": c.get("name"), "source": c["source"],
                                           "headline": c["headline"], "why": c.get("why"),
                                           "both": both_sources(self.store, c["ticker"], c["source"])}
                                          for _, c in new]}

    def _reasons(self, cands: list[dict]) -> dict[str, str]:
        """One sentence per new add-on idea on why the newsletter recommends it, from its write-up."""
        if not cands or not self.cfg["llm"].get("enabled", True):
            return {}
        from .llm import LLM
        llm = LLM(self.cfg)
        if not llm.available:
            return {}
        out = {}
        for c in cands:
            try:
                out[c["ticker"]] = llm.pick_reason(c["name"] or c["ticker"], c["thesis"])
            except Exception as e:
                log.warning("Reason for %s: %s", c["ticker"], e)
        return {t: r for t, r in out.items() if r}

    def _price(self, today: str) -> dict:
        """Yahoo snapshots for: ideas without a price yet (flagged in the last 3 days; normally just today's), watched
        ideas, and ideas 30-45 days old without their 30-day price. The stock and the S&P 500 are always compared on
        the same trading day: the S&P close is filled in once that day's close is stored."""
        watched = {r[0] for r in self.store.db.execute(f"SELECT key FROM ideas WHERE {WATCHED}")}
        rows = self.store.db.execute(
            f"SELECT * FROM ideas WHERE (price IS NULL AND flagged >= ?) OR {WATCHED} OR "
            "(price_30d IS NULL AND flagged <= ? AND flagged >= ?) OR (sp500 IS NULL AND flagged >= ?) "
            "ORDER BY price IS NOT NULL, flagged DESC", (_ago(3), _ago(30), _ago(45), _ago(7))).fetchall()
        need = [r["ticker"] for r in rows if r["price"] is None or r["key"] in watched
                or (r["price_30d"] is None and r["flagged"] <= _ago(30))]
        tickers = list(dict.fromkeys(need))[: self.c.get("max_priced", 40)]
        health = self.market.refresh(tickers) if tickers else {}
        fetched = set(tickers)
        for r in rows:
            price, day = r["price"], r["price_date"]
            if price is None and r["ticker"] in fetched:  # only a snapshot taken now is the price "when flagged"
                q = self.market.latest(r["ticker"])
                if q and q.get("price") and q.get("market_time") and q["fetched"][:10] >= _ago(1):
                    price = q["price"]
                    day = dt.datetime.fromtimestamp(q["market_time"], EXCHANGE_TZ).date().isoformat()
                    self.store.db.execute("UPDATE ideas SET price = ?, price_date = ? WHERE key = ?",
                                          (price, day, r["key"]))
            if price is not None and r["sp500"] is None and day:
                self.store.db.execute("UPDATE ideas SET sp500 = ? WHERE key = ?", (self._close_on(SP500, day), r["key"]))
            if price is not None and r["price_30d"] is None and day and r["flagged"] <= _ago(30):
                target = (dt.date.fromisoformat(day) + dt.timedelta(days=30)).isoformat()
                row = self.store.db.execute("SELECT date, close FROM prices WHERE ticker = ? AND date >= ? "
                                            "ORDER BY date LIMIT 1", (r["ticker"], target)).fetchone()
                sp = self._close_on(SP500, row["date"]) if row else None
                if row and (sp or r["flagged"] <= _ago(44)):  # wait for the S&P close of the same day, up to 2 weeks
                    self.store.db.execute("UPDATE ideas SET price_30d = ?, sp500_30d = ? WHERE key = ?",
                                          (row["close"], sp, r["key"]))
        self.store.commit()
        return {f"{k} — ideas": v for k, v in health.items()}

    def _close_on(self, ticker: str, day: str) -> float | None:
        row = self.store.db.execute("SELECT close FROM prices WHERE ticker = ? AND date = ?", (ticker, day)).fetchone()
        return row[0] if row else None


def both_sources(store, ticker: str, source: str, user: str = "") -> bool:
    """The other source flagged the same stock within 30 days (and the person hasn't dismissed it)."""
    return store.db.execute("SELECT 1 FROM ideas WHERE ticker = ? AND source != ? AND flagged >= ? AND key NOT IN "
                            "(SELECT key FROM idea_status WHERE user = ? AND status = 'dismissed')",
                            (ticker, source, _ago(30), user)).fetchone() is not None


def _ago(days: int) -> str:
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()


def _span(first: str, last: str) -> str:
    a, b = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    return f"{a:%b %-d}" if a == b else f"{a:%b %-d}–{b:%-d}" if a.month == b.month else f"{a:%b %-d}–{b:%b %-d}"


# ---------- reading (web page, digest) ----------

def ideas(store, where: str = "1", params=(), user: str = "") -> list[dict]:
    """Ideas matching `where`, with `status` (new / watch / dismissed) as this person set it."""
    _create(store)
    rows = store.db.execute(
        f"SELECT * FROM (SELECT {IDEA_COLS}, coalesce(u.status, 'new') AS status, u.status_date FROM ideas i "
        f"LEFT JOIN idea_status u ON u.key = i.key AND u.user = ?) WHERE {where} ORDER BY flagged DESC",
        (user, *params)).fetchall()
    return [{**dict(r), **json.loads(r["data"])} for r in rows]


def set_status(store, key: str, status: str, user: str) -> None:
    _create(store)
    store.db.execute("INSERT INTO idea_status(user, key, status, status_date) VALUES (?, ?, ?, ?) "
                     "ON CONFLICT(user, key) DO UPDATE SET status = excluded.status, status_date = excluded.status_date",
                     (user, key, status, dt.date.today().isoformat()))
    store.commit()


def scorecard(store) -> list[dict]:
    """Per source: ideas with a 30-day price, their average return over those 30 days, and the S&P 500's."""
    out = []
    for source, label in SOURCES.items():
        done = [i for i in ideas(store, "source = ? AND price_30d IS NOT NULL AND price > 0", (source,))]
        rets = [i["price_30d"] / i["price"] - 1 for i in done]
        sp = [i["sp500_30d"] / i["sp500"] - 1 for i in done if i.get("sp500") and i.get("sp500_30d")]
        beat = [i for i in done if i.get("sp500") and i.get("sp500_30d")
                and i["price_30d"] / i["price"] > i["sp500_30d"] / i["sp500"]]
        out.append({"source": label, "n": len(done), "avg": sum(rets) / len(rets) if rets else None,
                    "sp500": sum(sp) / len(sp) if sp else None, "beat": len(beat) / len(sp) if sp else None,
                    "pending": len(ideas(store, "source = ? AND price_30d IS NULL", (source,)))})
    return out


def sp500_close(store, day: str | None = None) -> float | None:
    q = ("SELECT close FROM prices WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1" if day else
         "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1")
    row = store.db.execute(q, (SP500, day) if day else (SP500,)).fetchone()
    return row[0] if row else None
