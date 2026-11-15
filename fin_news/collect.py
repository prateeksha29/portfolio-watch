"""Collect items from RSS feeds and APIs, politely and in parallel across hosts."""
import calendar
import datetime as dt
import html
import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qsl, quote_plus, urlencode, urljoin, urlparse, urlunparse

import feedparser
import requests
import trafilatura
from lxml import etree
from lxml import html as lxml_html

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (compatible; fin-news/0.1; personal feed reader)"
ROBOTS_AGENT = "fin-news"
TRACKING_PARAMS = {"source", "ncid", "guccounter", "guce_referrer", "guce_referrer_sig", "tsrc", ".tsrc", "ref",
                   "cmpid", "mod", "siteid", "yptr", "fbclid", "gclid", "mc_cid", "mc_eid"}
GOOGLE_NEWS = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="
FOOL_SITEMAP = "https://www.fool.com/news-sitemap.xml"
SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9", "n": "http://www.google.com/schemas/sitemap-news/0.9"}
REDIRECT_HOSTS = ("finnhub.io",)  # news links that 302 to the publisher; resolved before fetching

SEC_FORMS = {"8-K": "Current report", "10-Q": "Quarterly report", "10-K": "Annual report",
             "4": "Insider transaction", "6-K": "Foreign issuer report",
             "20-F": "Annual report (foreign issuer)", "40-F": "Annual report (Canadian issuer)"}
SEC_8K_ITEMS = {"1.01": "material agreement", "1.02": "agreement terminated", "2.01": "acquisition or disposal completed",
                "2.02": "earnings results", "2.03": "new debt or obligation", "2.05": "restructuring costs",
                "2.06": "impairment", "3.01": "listing notice", "5.02": "director/officer change",
                "5.07": "shareholder vote results", "7.01": "Reg FD disclosure", "8.01": "other events"}

FILING_FORM = re.compile(r" filed ([\w/-]+):")
# Form 4 transaction codes worth naming; open-market buys (P) and sales (S) are the ones that carry a signal.
FORM4_CODES = {"P": "bought", "S": "sold", "M": "exercised options", "A": "received award", "F": "withheld for tax",
               "G": "gifted", "C": "converted", "X": "exercised options"}

_secrets: list[str] = []


def set_secrets(values) -> None:
    _secrets[:] = [v for v in values if v]


def redact(text: str) -> str:
    for s in _secrets:
        text = text.replace(s, "***")
    return text


def filing_text(content: bytes) -> str:
    """Plain text of an SEC HTML document, tables included (they hold the numbers)."""
    doc = lxml_html.fromstring(content)
    for bad in doc.xpath("//script|//style"):
        bad.drop_tree()
    for cell in doc.xpath("//td|//th|//p|//div|//br|//tr"):
        cell.tail = (cell.tail or "") + (" | " if cell.tag in ("td", "th") else "\n")
    lines = (re.sub(r"[ \t\xa0|]*\|[ \t\xa0|]*", " | ", ln).strip(" |\t\xa0") for ln in doc.text_content().splitlines())
    return "\n".join(ln for ln in lines if ln)


def parse_form4(content: bytes) -> dict:
    """Who, what role, and each non-derivative transaction (code, shares, price) from a Form 4 XML."""
    root = etree.fromstring(content)

    def val(node, path):
        found = node.find(path)
        return (found.text or "").strip() if found is not None and found.text else ""

    rel = root.find("reportingOwner/reportingOwnerRelationship")
    roles = []
    if rel is not None:
        if val(rel, "officerTitle"):
            title = " ".join(val(rel, "officerTitle").split())
            roles.append(title.title() if title.isupper() and len(title) > 4 else title)
        elif val(rel, "isOfficer") in ("1", "true"):
            roles.append("Officer")
        if val(rel, "isDirector") in ("1", "true"):
            roles.append("Director")
        if val(rel, "isTenPercentOwner") in ("1", "true"):
            roles.append("10% owner")
    trades = []
    for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        try:
            shares = float(val(t, "transactionAmounts/transactionShares/value") or 0)
            price = float(val(t, "transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        trades.append({"code": val(t, "transactionCoding/transactionCode"), "shares": shares, "price": price,
                       "date": val(t, "transactionDate/value"),
                       "owned_after": val(t, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")})
    name = val(root, "reportingOwner/reportingOwnerId/rptOwnerName")
    return {"owner": " ".join(reversed(name.split(" ", 1))).title() if " " in name else name.title(),
            "role": ", ".join(roles), "plan_10b5_1": val(root, "aff10b5One") in ("1", "true"), "trades": trades}


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(s: str | None) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def canonical_url(url: str) -> str:
    p = urlparse(url.strip())
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS]
    return urlunparse((p.scheme.lower() or "https", p.netloc.lower(), p.path, "", urlencode(query), ""))


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def host_matches(url: str, hosts) -> bool:
    h = host_of(url)
    return any(h == s or h.endswith("." + s) for s in hosts)


def _entry_time(entry) -> str:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            return iso(dt.datetime.fromtimestamp(calendar.timegm(st), tz=dt.timezone.utc))
    return iso(utcnow())


class Http:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self.local, "session", None)
        if s is None:
            s = self.local.session = requests.Session()
            s.headers["User-Agent"] = UA
        return s

    def get(self, url: str, headers: dict | None = None, allow_redirects: bool = True) -> requests.Response:
        for attempt in range(3):
            try:
                r = self._session().get(url, headers=headers or {}, timeout=(10, self.timeout),
                                        allow_redirects=allow_redirects)
            except requests.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(redact(f"{type(e).__name__}: {e}")) from None
                time.sleep(3 * (attempt + 1))
                continue
            if (r.status_code == 429 or r.status_code >= 500) and attempt < 2:
                retry_after = r.headers.get("Retry-After", "")
                time.sleep(min(int(retry_after), 30) if retry_after.isdigit() else 5)
                continue
            return r
        raise AssertionError("unreachable")


class RobotRules:
    """robots.txt as RFC 9309 and the major crawlers read it: the groups naming our agent (else the '*' groups),
    the longest matching rule wins, Allow wins a tie, '*' and a trailing '$' are wildcards. Python's
    urllib.robotparser differs in ways that matter: it takes the first matching rule, ignores wildcards, and turns
    'Disallow: /?' (the home page with a query) into 'Disallow: /' (the whole site, as on google.com)."""

    def __init__(self, text: str, agent: str):
        groups, agents, rules, in_rules = [], [], [], False
        for line in text.splitlines():
            key, _, value = line.split("#", 1)[0].partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                if in_rules:
                    groups.append((agents, rules))
                    agents, rules, in_rules = [], [], False
                agents.append(value.lower())
            elif key in ("allow", "disallow") and agents:
                in_rules = True
                if value:  # an empty Disallow allows everything
                    rules.append((key == "allow", value))
        if agents:
            groups.append((agents, rules))
        agent = agent.lower()
        mine = [r for a, rs in groups if any(x != "*" and x in agent for x in a) for r in rs]
        chosen = mine if any(any(x != "*" and x in agent for x in a) for a, _ in groups) else \
            [r for a, rs in groups if "*" in a for r in rs]
        self.rules = [(allow, len(path), self._pattern(path)) for allow, path in chosen]

    @staticmethod
    def _pattern(path: str) -> re.Pattern:
        end = path.endswith("$")
        body = ".*".join(re.escape(part) for part in (path[:-1] if end else path).split("*"))
        return re.compile(body + ("$" if end else ""))

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        target = (p.path or "/") + (f"?{p.query}" if p.query else "")
        best = None  # (length, allow)
        for allow, length, pattern in self.rules:
            if pattern.match(target) and (best is None or length > best[0] or (length == best[0] and allow)):
                best = (length, allow)
        return best is None or best[1]


class Robots:
    def __init__(self, http: Http):
        self.http = http
        self.cache: dict[str, RobotRules | bool] = {}
        self.lock = threading.Lock()

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        with self.lock:
            rules = self.cache.get(base)
        if rules is None:
            try:
                r = self.http.get(base + "/robots.txt")
                if r.status_code >= 500:
                    rules = False  # server trouble: assume disallowed for now
                elif r.status_code >= 400:
                    rules = True  # no robots.txt: everything allowed
                else:
                    rules = RobotRules(r.text, ROBOTS_AGENT)
            except RuntimeError:
                rules = False
            with self.lock:
                self.cache[base] = rules
        return rules if isinstance(rules, bool) else rules.allowed(url)


def run_by_host(jobs: list[dict], fn, delay_for, max_workers: int) -> list[dict]:
    """Run jobs in parallel across hosts, sequentially (with a delay) within each host."""
    by_host = defaultdict(list)
    for job in jobs:
        by_host[job["host"]].append(job)
    results, lock = [], threading.Lock()

    def worker(host, host_jobs):
        delay = delay_for(host)
        for i, job in enumerate(host_jobs):
            if i:
                time.sleep(delay)
            try:
                out = fn(job)
            except Exception as e:  # one bad source must not stop the run
                out = {"job": job, "error": redact(f"{type(e).__name__}: {e}")}
            out.setdefault("job", job)
            with lock:
                results.append(out)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f in [ex.submit(worker, h, js) for h, js in by_host.items()]:
            f.result()
    return results


class Collector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        f = cfg["fetch"]
        self.http = Http(f["timeout_seconds"])
        self.robots = Robots(self.http)
        self.default_delay = f["delay_seconds"]
        self.host_delays = f.get("host_delays", {})
        self.workers = f.get("max_parallel_hosts", 8)
        self.skip_hosts = f.get("fulltext_skip_hosts", [])
        s = cfg["secrets"]
        self.fred_key, self.finnhub_key = s["fred"], s["finnhub"]
        self.sec_headers = ({"User-Agent": s["sec_user_agent"], "Accept-Encoding": "gzip, deflate"}
                            if s["sec_user_agent"] else None)
        set_secrets([self.fred_key, self.finnhub_key, *(v for k, v in s.items() if k != "sec_user_agent")])

    def delay_for(self, host: str) -> float:
        return self.host_delays.get(host, self.default_delay)

    def fetchable(self, url: str) -> bool:
        if not url.startswith(("http://", "https://")):
            return False
        return host_matches(url, REDIRECT_HOSTS) or not host_matches(url, self.skip_hosts)

    # ---------- SEC ticker map ----------

    def sec_ticker_map(self, data_dir: Path) -> dict[str, tuple[int, str]]:
        if not self.sec_headers:
            log.warning("SEC_USER_AGENT not set; skipping SEC filings")
            return {}
        path = data_dir / "sec_company_tickers.json"
        fresh = path.exists() and time.time() - path.stat().st_mtime < 7 * 86400
        if not fresh:
            try:
                r = self.http.get("https://www.sec.gov/files/company_tickers.json", self.sec_headers)
                r.raise_for_status()
                path.write_bytes(r.content)
            except Exception as e:
                log.warning("Could not refresh SEC ticker list: %s", redact(str(e)))
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {row["ticker"].upper(): (int(row["cik_str"]), row["title"]) for row in data.values()}

    # ---------- jobs ----------

    def build_jobs(self, tickers, names, topics, sec_map) -> list[dict]:
        src, jobs = self.cfg["sources"], []

        def add(name, group, url, kind, hints=(), fn="feed", **extra):
            jobs.append({"name": name, "group": group, "url": url, "host": host_of(url), "kind": kind,
                         "hints": list(hints), "fn": fn, **extra})

        for t in tickers:
            name = names.get(t, t)
            if src.get("yahoo"):
                add("Yahoo Finance", "Yahoo Finance",
                    f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US", "news", [t])
            if src.get("nasdaq"):
                add("Nasdaq", "Nasdaq", f"https://www.nasdaq.com/feed/rssoutbound?symbol={t}", "news", [t])
            if src.get("seeking_alpha"):
                add("Seeking Alpha", "Seeking Alpha", f"https://seekingalpha.com/api/sa/combined/{t}.xml", "news", [t])
            if src.get("google_news"):
                add("Google News", "Google News", GOOGLE_NEWS + quote_plus(f'"{name}" OR {t} stock when:2d'),
                    "news", [t], google_news=True)
            if src.get("google_news_fool"):
                add("Motley Fool", "Google News (fool.com)", GOOGLE_NEWS + quote_plus(f"site:fool.com {name} when:2d"),
                    "analysis", [t], google_news=True)
            if src.get("finnhub") and self.finnhub_key:
                add("Finnhub", "Finnhub", f"https://finnhub.io/api/v1/company-news?symbol={t}", "news", [t],
                    fn="finnhub_news", ticker=t)
            if self.sec_headers and t in sec_map:
                cik = sec_map[t][0]
                add("SEC EDGAR", "SEC EDGAR", f"https://data.sec.gov/submissions/CIK{cik:010d}.json", "filing", [t],
                    fn="sec", ticker=t, cik=cik)
        for topic in topics:
            add("Google News", "Google News (topics)", GOOGLE_NEWS + quote_plus(f"{topic} when:2d"), "news",
                google_news=True)
        if src.get("fool_sitemap"):
            add("Motley Fool", "Motley Fool (news sitemap)", FOOL_SITEMAP, "analysis", fn="news_sitemap")
        for feed in src.get("feeds", []):
            add(feed["name"], feed["name"], feed["url"], feed.get("kind", "news"))
        subs = src.get("reddit_subreddits", [])
        if subs:  # one combined feed: without an account Reddit allows about one feed request a minute
            add("Reddit", "Reddit", f"https://www.reddit.com/r/{'+'.join(subs)}/top/.rss?t=day&limit=100", "social",
                reddit=True)
        if self.finnhub_key:
            add("Finnhub earnings", "Finnhub earnings", "https://finnhub.io/api/v1/calendar/earnings", "event",
                fn="finnhub_earnings", tickers=list(tickers))
        if self.fred_key:
            for s in self.cfg.get("fred", {}).get("series", []):
                add("FRED", "FRED", f"https://api.stlouisfed.org/fred/series/observations?series_id={s['id']}",
                    "macro", fn="fred", series=s["id"])
        return jobs

    def collect(self, tickers, names, topics, sec_map, feed_cache) -> list[dict]:
        jobs = self.build_jobs(tickers, names, topics, sec_map)
        log.info("Fetching %d sources across %d sites", len(jobs), len({j["host"] for j in jobs}))
        handlers = {"feed": lambda j: self._feed(j, feed_cache), "sec": self._sec, "fred": self._fred,
                    "finnhub_news": self._finnhub_news, "finnhub_earnings": self._finnhub_earnings,
                    "news_sitemap": self._news_sitemap}
        return run_by_host(jobs, lambda j: handlers[j["fn"]](j), self.delay_for, self.workers)

    # ---------- handlers ----------

    def _feed(self, job, feed_cache) -> dict:
        headers = {}
        etag, modified = feed_cache.get(job["url"], (None, None))
        if etag:
            headers["If-None-Match"] = etag
        if modified:
            headers["If-Modified-Since"] = modified
        r = self.http.get(job["url"], headers)
        out = {"status": r.status_code, "items": [], "etag": r.headers.get("ETag"),
               "modified": r.headers.get("Last-Modified")}
        if r.status_code == 304:
            return out
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}"
            return out
        for e in feedparser.parse(r.content).entries:
            link, title = e.get("link"), clean_text(e.get("title"))
            if not link or not title:
                continue
            source, summary = job["name"], clean_text(e.get("summary"))[:2000]
            if job.get("google_news"):
                publisher = (e.get("source") or {}).get("title")
                if publisher:
                    source = f"Google News · {publisher}"
                    title = re.sub(rf"\s+-\s+{re.escape(publisher)}$", "", title)
                summary = ""  # Google News summaries are just the headline again
            elif job.get("reddit"):
                m = re.search(r"/r/([^/]+)/", link)
                source = f"Reddit r/{m.group(1)}" if m else source
            out["items"].append({"url": canonical_url(link), "title": title, "summary": summary,
                                 "published": _entry_time(e), "source": source, "kind": job["kind"],
                                 "hints": job["hints"]})
        return out

    def _news_sitemap(self, job) -> dict:
        """A Google News sitemap: the publisher's last two days of articles, each with the stock tickers it is about
        (passed on as hints; for these the publisher's tags count like a ticker in the headline)."""
        if not self.robots.allowed(job["url"]):
            return {"error": "robots.txt disallows"}
        r = self.http.get(job["url"])
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        out = {"items": []}
        for u in etree.fromstring(r.content).iterfind("s:url", SITEMAP_NS):
            url, title = u.findtext("s:loc", namespaces=SITEMAP_NS), clean_text(u.findtext("n:news/n:title",
                                                                                          namespaces=SITEMAP_NS))
            when = u.findtext("n:news/n:publication_date", namespaces=SITEMAP_NS)
            if not url or not title or not when:
                continue
            tickers = [t.split(":")[-1].strip().upper() for t in
                       (u.findtext("n:news/n:stock_tickers", namespaces=SITEMAP_NS) or "").split(",") if t.strip()]
            out["items"].append({"url": canonical_url(url), "title": title, "summary": "",
                                 "published": iso(dt.datetime.fromisoformat(when)),
                                 "source": job["name"], "hints": tickers,
                                 "kind": "transcript" if "/call-transcripts/" in url else job["kind"]})
        return out

    def _finnhub_news(self, job) -> dict:
        today = utcnow().date()
        params = {"symbol": job["ticker"], "from": (today - dt.timedelta(days=2)).isoformat(),
                  "to": today.isoformat(), "token": self.finnhub_key}
        r = self.http.get("https://finnhub.io/api/v1/company-news?" + urlencode(params))
        if r.status_code != 200:
            return {"status": r.status_code, "error": f"HTTP {r.status_code}"}
        items = []
        for d in r.json():
            if not d.get("url") or not d.get("headline"):
                continue
            items.append({"url": canonical_url(d["url"]), "title": clean_text(d["headline"]),
                          "summary": clean_text(d.get("summary"))[:2000],
                          "published": iso(dt.datetime.fromtimestamp(d.get("datetime") or time.time(), tz=dt.timezone.utc)),
                          "source": f"Finnhub · {d.get('source') or 'unknown'}", "kind": "news", "hints": job["hints"]})
        return {"status": 200, "items": items}

    def _finnhub_earnings(self, job) -> dict:
        today = utcnow().date()
        params = {"from": today.isoformat(), "to": (today + dt.timedelta(days=14)).isoformat(),
                  "token": self.finnhub_key}
        r = self.http.get("https://finnhub.io/api/v1/calendar/earnings?" + urlencode(params))
        if r.status_code != 200:
            return {"status": r.status_code, "error": f"HTTP {r.status_code}"}
        wanted = set(job["tickers"])
        events = [(e["symbol"], e["date"], "earnings",
                   json.dumps({"hour": e.get("hour"), "eps_estimate": e.get("epsEstimate"),
                               "revenue_estimate": e.get("revenueEstimate")}))
                  for e in r.json().get("earningsCalendar", []) if e.get("symbol") in wanted]
        return {"status": 200, "events": events}

    def _fred(self, job) -> dict:
        params = {"series_id": job["series"], "api_key": self.fred_key, "file_type": "json",
                  "sort_order": "desc", "limit": 15}
        r = self.http.get("https://api.stlouisfed.org/fred/series/observations?" + urlencode(params))
        if r.status_code != 200:
            return {"status": r.status_code, "error": f"HTTP {r.status_code}"}
        macro = [(job["series"], o["date"], float(o["value"]))
                 for o in r.json().get("observations", []) if o.get("value") not in (".", "", None)]
        return {"status": 200, "macro": macro}

    def _sec(self, job) -> dict:
        r = self.http.get(job["url"], self.sec_headers)
        if r.status_code != 200:
            return {"status": r.status_code, "error": f"HTTP {r.status_code}"}
        recent = r.json()["filings"]["recent"]
        cutoff = (utcnow() - dt.timedelta(days=self.cfg["sec"]["lookback_days"])).date().isoformat()
        forms, t, items = set(self.cfg["sec"]["forms"]), job["ticker"], []
        n = len(recent["form"])
        codes = recent.get("items") or [""] * n
        for i in range(n):
            form = recent["form"][i]
            if recent["filingDate"][i] < cutoff or form not in forms:
                continue
            acc, doc = recent["accessionNumber"][i], recent["primaryDocument"][i]
            if form == "8-K" and codes[i]:
                what = ", ".join(SEC_8K_ITEMS[c] for c in codes[i].split(",") if c in SEC_8K_ITEMS) or SEC_FORMS[form]
            else:
                what = SEC_FORMS.get(form) or recent["primaryDocDescription"][i] or form
            accepted = recent.get("acceptanceDateTime", [""] * n)[i]
            published = accepted[:19] + "Z" if accepted else recent["filingDate"][i] + "T00:00:00Z"
            items.append({"url": f"https://www.sec.gov/Archives/edgar/data/{job['cik']}/{acc.replace('-', '')}/{doc}",
                          "title": f"{t} filed {form}: {what}", "summary": f"Form {form} — {what}",
                          "published": published, "source": "SEC EDGAR", "kind": "filing", "hints": [t],
                          "form": form})
        return {"status": 200, "items": items}

    # ---------- SEC filing contents ----------

    def sec_details(self, filings: list[tuple[int, str, str]]) -> dict[int, str]:
        """(article id, url, title) -> body. 8-K/6-K: text of the attached press release (EX-99.x), else the main
        document. Form 4: the insider transactions as JSON. Other forms: '' (nothing to fetch; marks them done)."""
        jobs = [{"id": aid, "url": url, "form": (FILING_FORM.search(title) or [None, ""])[1], "host": host_of(url)}
                for aid, url, title in filings]
        results = run_by_host(jobs, self._sec_detail, self.delay_for, self.workers)
        out = {r["job"]["id"]: r.get("body", "") for r in results if "body" in r}
        log.info("SEC filing contents: %d of %d read", sum(1 for b in out.values() if b), len(jobs))
        return out

    def _sec_detail(self, job) -> dict:
        form, url = job["form"], job["url"]
        if form == "4":
            r = self.http.get(url.replace("/xslF345X06/", "/"), self.sec_headers)  # raw XML behind the rendered page
            return {"body": json.dumps(parse_form4(r.content))} if r.status_code == 200 else {"error": r.status_code}
        if form not in ("8-K", "6-K"):
            return {"body": ""}
        folder = url.rsplit("/", 1)[0]
        acc = folder.rsplit("/", 1)[1]
        r = self.http.get(f"{folder}/{acc[:10]}-{acc[10:12]}-{acc[12:]}-index.htm", self.sec_headers)
        if r.status_code != 200:
            return {"error": r.status_code}
        exhibits = []
        for tr in lxml_html.fromstring(r.content).xpath("//table[@class='tableFile']//tr")[1:]:
            cells, href = [td.text_content().strip() for td in tr.xpath("td")], tr.xpath(".//a/@href")
            if len(cells) > 3 and cells[3].startswith("EX-99") and href and href[0].endswith((".htm", ".html", ".txt")):
                exhibits.append(urljoin(url, href[0]))
        texts = []
        for doc in exhibits[:2] or [url]:  # the press release, plus a second exhibit if there is one
            d = self.http.get(doc, self.sec_headers)
            if d.status_code == 200:
                texts.append(filing_text(d.content))
        return {"body": "\n\n".join(t for t in texts if t)}

    # ---------- full text ----------

    def fetch_full_texts(self, articles: list[tuple[int, str]]) -> tuple[dict[int, str], dict[int, str]]:
        """Returns (article id -> body, article id -> resolved publisher URL)."""
        jobs = [{"id": aid, "url": url, "host": host_of(url)} for aid, url in articles if self.fetchable(url)]
        log.info("Fetching full text for %d articles", len(jobs))
        results = run_by_host(jobs, self._fulltext, self.delay_for, self.workers)
        bodies = {r["job"]["id"]: r["body"] for r in results if r.get("body")}
        resolved = {r["job"]["id"]: r["final_url"] for r in results if r.get("final_url")}
        outcomes = Counter("ok" if r.get("body") else str(r.get("status") or r.get("error", "error")[:40])
                           for r in results)
        log.info("Full text outcomes: %s", dict(outcomes.most_common()))
        return bodies, resolved

    def _fulltext(self, job) -> dict:
        url, out = job["url"], {}
        if host_matches(url, REDIRECT_HOSTS):
            r = self.http.get(url, allow_redirects=False)
            location = r.headers.get("Location")
            if not location:
                return {"status": r.status_code}
            url = out["final_url"] = canonical_url(urljoin(url, location))
            if host_matches(url, self.skip_hosts):
                return {**out, "status": "skipped"}
        if not self.robots.allowed(url):
            return {**out, "status": "disallowed"}
        r = self.http.get(url)
        if r.status_code != 200 or "html" not in r.headers.get("Content-Type", ""):
            return {**out, "status": r.status_code}
        body = trafilatura.extract(r.text, include_comments=False, include_tables=False, favor_precision=True)
        return {**out, "status": 200, "body": (body or "")[:50000] or None}
