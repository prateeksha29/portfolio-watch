"""What the owner's funds hold, from SEC Form N-PORT: each registered fund's full list of positions with their share
of the fund, filed quarterly and public about 60 days after the quarter ends. Pooled funds (collective trusts in a
401(k)) don't file, so config maps them to the public fund with the same strategy or index. From the holdings comes
the owner's real exposure to each stock: shares held directly plus their slice of every fund that holds it.

Found through data.sec.gov submissions and the filing archive (robots.txt allows both; EDGAR's cgi-bin browse pages
are disallowed)."""
import datetime as dt
import json
import logging
import re
import time
from collections import Counter, defaultdict

from lxml import etree

from .collect import Collector, iso, redact, utcnow

log = logging.getLogger(__name__)

MF_TICKERS = "https://www.sec.gov/files/company_tickers_mf.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{}/{}/{}"
SHARE_CLASS = re.compile(r"(?:\s+|\s*[-,/]\s*)(CL(ASS)?\s*[A-C]|SPONSORED ADR|ADR|SPON ADR|NY REG(ISTRY)?( SH(ARES|S)?)?|ORD(INARY)?( SHARES)?|"
                         r"COM(MON)?( STOCK)?|REIT|UNIT|SHS)\b.*$", re.I)
# Fund filings abbreviate company names; the SEC ticker list mostly doesn't. Both sides are reduced to one key.
ABBREVIATIONS = {"mfg": "manufacturing", "hldgs": "holdings", "hldg": "holding", "cos": "companies",
                 "assocs": "associates", "intl": "international", "grp": "group", "svcs": "services", "sys": "systems",
                 "tech": "technologies", "techs": "technologies", "pharma": "pharmaceuticals", "bk": "bank",
                 "natl": "national", "amer": "america", "fin": "financial", "corpn": "corporation", "&": "and"}
DROP_WORDS = {"inc", "corp", "corporation", "co", "company", "ltd", "limited", "nv", "sa", "se", "ag", "the",
              "del", "new", "holding", "holdings", "group", "lp", "llc", "incorporated"}
_fund_index: dict = {}
ASSET_NAMES = {"EC": "stocks", "EP": "preferred stock", "DBT": "bonds", "STIV": "cash & short-term", "RE": "real estate",
               "DE": "derivatives", "DIR": "derivatives", "DFE": "derivatives", "DCO": "derivatives",
               "ABS-MBS": "mortgage bonds", "LON": "loans", "OTHER": "other"}


def match_key(name: str) -> str:
    """'Taiwan Semiconductor Mfg Co Ltd', 'TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD' -> 'taiwan semiconductor
    manufacturing'; 'Western Digital Corp/De' -> 'western digital'; 'Coca-Cola Co/The' -> 'coca cola'."""
    n = SHARE_CLASS.sub("", name.replace(".", " ").replace("&", " and ")).lower()
    n = re.sub(r"/[a-z]{2,3}\b|\(the\)", " ", n)  # state of incorporation (/de, /md), '(The)'
    n = re.sub(r"[^a-z0-9&]+", " ", n)
    n = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", n)  # initials: 's a' -> 'sa', 'd r horton' -> 'dr horton'
    words = [ABBREVIATIONS.get(w, w) for w in n.split()]
    while words and words[-1] in DROP_WORDS | {"and", "ohio", "de", "md", "nj", "ny"}:
        words.pop()
    return " ".join(w for w in words if w not in ("the", "of"))


def fund_name_index(cfg: dict) -> dict[str, str]:
    """match_key(company name) -> ticker, and the same key without spaces ('exxonmobil'), from [funds.names], the
    configured aliases and the SEC ticker list (largest company first)."""
    path = cfg["paths"]["data"] / "sec_company_tickers.json"
    mtime = path.stat().st_mtime if path.exists() else 0
    if _fund_index.get("mtime") != mtime:
        names = [(n, t) for n, t in cfg.get("funds", {}).get("names", {}).items()]
        names += [(a, t) for t, aliases in cfg.get("aliases", {}).items() for a in aliases]
        names += [(row["title"], row["ticker"].replace("-", ".")) for row in
                  (json.loads(path.read_text()).values() if path.exists() else [])]
        index = {}
        for name, t in names:
            if re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", t):
                key = match_key(name)
                index.setdefault(key, t)
                index.setdefault("~" + key.replace(" ", ""), t)
        index.pop("", None)
        index.pop("~", None)
        _fund_index.update(mtime=mtime, index=index)
    return _fund_index["index"]


def lookup(index: dict[str, str], name: str) -> str | None:
    key = match_key(name)
    return index.get(key) or index.get("~" + key.replace(" ", ""))


class FundData:
    def __init__(self, cfg: dict, store, collector: Collector | None = None):
        self.cfg, self.store = cfg, store
        self.collector = collector or Collector(cfg)
        self.headers = self.collector.sec_headers
        f = cfg.get("funds", {})
        self.proxies = f.get("proxies", {})
        self.refresh_days = f.get("refresh_days", 30)

    def source_of(self, symbol: str) -> str:
        """The fund whose filings stand in for this holding: itself, or its public twin."""
        return self.proxies.get(symbol, symbol)

    # ---------- fetching ----------

    def _get(self, url: str):
        r = self.collector.http.get(url, self.headers)
        r.raise_for_status()
        time.sleep(0.12)  # SEC asks for at most 10 requests a second
        return r

    def _series(self) -> dict[str, dict]:
        path = self.cfg["paths"]["data"] / "sec_company_tickers_mf.json"
        if not path.exists() or time.time() - path.stat().st_mtime > 7 * 86400:
            path.write_bytes(self._get(MF_TICKERS).content)
        data = json.loads(path.read_text())
        f = data["fields"]
        return {r[f.index("symbol")]: dict(zip(f, r)) for r in data["data"]}

    def _latest_filing(self, cik: int, series: str) -> tuple[str, str] | None:
        """(accession, filing date) of the newest N-PORT for the series. A registrant files one per series, so
        the small header files are checked newest first until one names the series."""
        recent = self._get(SUBMISSIONS.format(cik)).json()["filings"]["recent"]
        cands = sorted(((recent["filingDate"][i], recent["accessionNumber"][i]) for i in range(len(recent["form"]))
                        if recent["form"][i] == "NPORT-P"), reverse=True)
        for filed, acc in cands[:200]:
            hdr = self._get(ARCHIVE.format(cik, acc.replace("-", ""), f"{acc}.hdr.sgml")).text
            if f"<SERIES-ID>{series}" in hdr:
                return acc, filed
        return None

    def refresh(self, symbols: list[str], force: bool = False) -> dict:
        """Fetch holdings for funds never fetched or not checked in refresh_days (a new filing appears quarterly).
        Returns source health in the pipeline's format."""
        h = {"requests": 0, "errors": 0, "not_modified": 0, "items": 0, "last_error": ""}
        cutoff = iso(utcnow() - dt.timedelta(days=self.refresh_days))
        sources = sorted({self.source_of(s) for s in symbols})
        due = [s for s in sources if force or not (row := self.store.db.execute(
            "SELECT checked FROM fund_filings WHERE fund = ?", (s,)).fetchone()) or row[0] < cutoff]
        if not due or not self.headers:
            return h
        series = self._series()
        for src in due:
            h["requests"] += 1
            info = series.get(src)
            if not info:
                h["not_modified"] += 1  # not an SEC-registered fund (a pooled fund without a proxy)
                continue
            try:
                found = self._latest_filing(int(info["cik"]), info["seriesId"])
                if not found:
                    raise LookupError(f"no N-PORT filing found for {src}")
                acc, filed = found
                known = self.store.db.execute("SELECT accession FROM fund_filings WHERE fund = ?", (src,)).fetchone()
                if known and known[0] == acc:
                    h["not_modified"] += 1
                else:
                    doc = self._get(ARCHIVE.format(int(info["cik"]), acc.replace("-", ""), "primary_doc.xml")).content
                    self._save(src, acc, filed, doc)
                    h["items"] += 1
                self.store.db.execute("UPDATE fund_filings SET checked = ? WHERE fund = ?", (iso(utcnow()), src))
                self.store.commit()
            except Exception as e:  # one fund must not stop the rest
                h["errors"] += 1
                h["last_error"] = redact(f"{src}: {type(e).__name__}: {e}")[:200]
                log.warning("Fund holdings for %s: %s", src, h["last_error"])
        return h

    def _save(self, src: str, acc: str, filed: str, doc: bytes) -> None:
        root = etree.fromstring(doc)
        index = fund_name_index(self.cfg)
        rows = []
        for inv in root.iter("{*}invstOrSec"):
            name = (inv.findtext("{*}name") or inv.findtext("{*}title") or "").strip()
            pct = float(inv.findtext("{*}pctVal") or 0)
            if not name or pct <= 0:  # short positions and derivatives with negative value
                continue
            ticker = lookup(index, name) if inv.findtext("{*}assetCat") == "EC" else None
            rows.append((src, name.title() if name.isupper() else name, ticker,
                         inv.findtext("{*}cusip"), pct, inv.findtext("{*}assetCat") or "OTHER",
                         inv.findtext("{*}invCountry") or ""))
        db = self.store.db
        db.execute("DELETE FROM fund_holdings WHERE fund = ?", (src,))
        db.executemany("INSERT INTO fund_holdings(fund, name, ticker, cusip, pct, asset, country) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        db.execute("INSERT OR REPLACE INTO fund_filings(fund, accession, period, filed, net_assets, name, checked) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (src, acc, root.findtext(".//{*}repPdDate"), filed, float(root.findtext(".//{*}netAssets") or 0),
                    root.findtext(".//{*}seriesName"), iso(utcnow())))
        self.store.commit()
        log.info("Fund holdings for %s: %d positions as of %s", src, len(rows), root.findtext(".//{*}repPdDate"))

    def remap(self) -> int:
        """Re-match stored holdings to tickers (after the matching rules change); returns how many are matched."""
        index = fund_name_index(self.cfg)
        rows = self.store.db.execute("SELECT rowid, name FROM fund_holdings WHERE asset = 'EC'").fetchall()
        found = [(lookup(index, r[1]), r[0]) for r in rows]
        self.store.db.executemany("UPDATE fund_holdings SET ticker = ? WHERE rowid = ?", found)
        self.store.commit()
        return sum(1 for t, _ in found if t)

    # ---------- reading ----------

    def filing(self, symbol: str) -> dict | None:
        row = self.store.db.execute("SELECT * FROM fund_filings WHERE fund = ?", (self.source_of(symbol),)).fetchone()
        return dict(row) if row else None

    def holdings(self, symbol: str) -> list[dict]:
        return [dict(r) for r in self.store.db.execute(
            "SELECT name, ticker, pct, asset, country FROM fund_holdings WHERE fund = ? ORDER BY pct DESC",
            (self.source_of(symbol),))]

    def mix(self, symbol: str) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """(asset types, countries) as % of the fund, largest first."""
        assets, countries = Counter(), Counter()
        for r in self.holdings(symbol):
            assets[ASSET_NAMES.get(r["asset"], "other")] += r["pct"]
            if r["country"]:
                countries[r["country"]] += r["pct"]
        return assets.most_common(), countries.most_common(8)

    def exposures(self, holdings) -> dict:
        """Real exposure per stock: {ticker: {direct, via: {fund: $}, total}}, plus the funds without holdings data
        and the portfolio total. A fund's slice of a stock is the owner's fund value times the stock's share of the
        fund in its latest filing."""
        out = defaultdict(lambda: {"direct": 0.0, "via": {}, "total": 0.0})
        for t, e in holdings.stocks.items():
            out[t]["direct"] = e["value"]
        missing = []
        for sym, f in holdings.funds.items():
            rows = self.holdings(sym)
            if not rows:
                missing.append(sym)
                continue
            for r in rows:
                if r["ticker"]:
                    v = f["value"] * r["pct"] / 100
                    out[r["ticker"]]["via"][sym] = out[r["ticker"]]["via"].get(sym, 0) + v
        for d in out.values():
            d["total"] = d["direct"] + sum(d["via"].values())
        total = sum(e["value"] for e in (*holdings.stocks.values(), *holdings.funds.values())) or 1
        return {"by_ticker": dict(out), "missing": missing, "portfolio": total,
                "missing_value": sum(holdings.funds[s]["value"] for s in missing)}


def exposure_line(t: str, d: dict, total: float) -> str:
    """'NVDA: $10,000 direct + $5,000 through funds (VGT $3,000, FXAIX $2,000, ...) = $15,000, 3.0% of portfolio'"""
    via = sorted(d["via"].items(), key=lambda kv: -kv[1])
    parts = ", ".join(f"{s} ${v:,.0f}" for s, v in via[:5]) + (", ..." if len(via) > 5 else "")
    return (f"{t}: ${d['direct']:,.0f} direct + ${d['total'] - d['direct']:,.0f} through funds"
            + (f" ({parts})" if parts else "") + f" = ${d['total']:,.0f}, {d['total'] / total:.1%} of portfolio")


FUND_WORDS = re.compile(r"\b(funds?|etfs?|index|pool|401\(?k\)?|portfolio|contrafund|target[- ]date)\b", re.I)
GENERIC = {"FIDELITY", "FID", "FUND", "INDEX", "IDX", "SELECT", "PORT", "POOL", "CL", "TRUST", "TR", "BTC", "THE",
           "ETF", "ENHANCED", "LARGE", "CAP", "SHORT", "TERM", "TOTAL", "CO"}
# Abbreviations in Fidelity's fund descriptions -> what people call them
FUND_SYNONYMS = {"INTL": "international|intl|foreign", "LPATH": "lifepath|target[- ]date|retirement|2060",
                 "GR": "growth", "RUS": "russell", "VANG": "vanguard", "CONTRA": "contra", "TREASURY": "treasury|bond"}


def mentioned_funds(holdings, question: str) -> list[str]:
    """Funds named in a question: by symbol ('vgt'), or, when it talks about funds, by a distinctive word of the
    fund's name ('my semiconductor fund', 'Contrafund', 'the 500 index')."""
    q = question.lower()
    found = [s for s in holdings.funds if re.search(rf"\b{s.lower()}\b", q)]
    if not found and FUND_WORDS.search(question):
        for s, f in sorted(holdings.funds.items(), key=lambda kv: -kv[1]["weight"]):
            words = [w for w in re.findall(r"[A-Z0-9]+", f["description"]) if w not in GENERIC
                     and (len(w) >= 3 or w in FUND_SYNONYMS)]
            pats = [FUND_SYNONYMS.get(w, re.escape(w.lower().rstrip("s"))) for w in words]
            if any(re.search(rf"\b(?:{p})", q) for p in pats):
                found.append(s)
    return found


def fund_lines(fd: FundData, holdings, symbol: str, top: int = 15) -> list[str]:
    """FUND DETAIL for Ask: what the fund is, its largest positions, which of the owner's stocks it holds, its mix."""
    f, filing = holdings.funds[symbol], fd.filing(symbol)
    head = f"FUND DETAIL: {symbol} ({f['description'].title()}): ${f['value']:,.0f}, {f['weight']:.1%} of portfolio"
    if not filing:
        return [head, "No holdings data for this fund (a pooled fund with no public twin, or not yet fetched)."]
    rows = fd.holdings(symbol)
    src = fd.source_of(symbol)
    lines = [head, f"Holdings as of {filing['period']} from the SEC filing of {src}"
             + (f" ({filing['name']}, the public fund with the same strategy; this pooled fund doesn't file)"
                if src != symbol else "") + f": {len(rows)} positions"]
    lines.append("Largest positions (share of fund, the owner's slice): " + "; ".join(
        f"{r['name']}{' (' + r['ticker'] + ')' if r['ticker'] else ''} {r['pct']:.1f}% ${f['value'] * r['pct'] / 100:,.0f}"
        for r in rows[:top]))
    mine = [(r["ticker"], r["pct"]) for r in rows if r["ticker"] in holdings.stocks]
    if mine:
        lines.append("Also held directly by the owner: " + ", ".join(f"{t} {p:.1f}%" for t, p in mine[:12]))
    assets, countries = fd.mix(symbol)
    lines.append("Mix: " + ", ".join(f"{a} {p:.0f}%" for a, p in assets[:4] if p >= 0.5)
                 + ("; countries: " + ", ".join(f"{c} {p:.0f}%" for c, p in countries[:6]) if len(countries) > 1 else ""))
    return lines
