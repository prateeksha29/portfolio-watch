"""Search the archive and ask questions over it."""
import datetime as dt
import json
import re
from collections import Counter

from .addons import advisor
from .askfmt import day, source_header, vs_cost
from .collect import iso, utcnow
from .embed import Embedder
from .filings import form_of, insider, insider_line
from .funds import FundData, exposure_line, fund_lines, mentioned_funds
from .llm import LLM
from .market import Market, companies_in, context_lines, stale
from .returns import context_lines as returns_lines


def search(cfg: dict, store, query: str, k: int = 10, since: str | None = None, ticker: str | None = None,
           embedder: Embedder | None = None) -> list[dict]:
    """Pass a loaded embedder to reuse it (the web app keeps one on CPU); otherwise one is loaded and freed."""
    own = embedder is None
    embedder = embedder or Embedder(cfg)
    q = embedder.queries([query])[0]
    if own:
        embedder.close()
    hits = store.knn(q, k * 20)
    rows = store.articles([aid for aid, _ in hits])
    half_life = cfg["search"]["recency_half_life_days"]
    now = utcnow()
    out = []
    for aid, distance in hits:
        r = rows.get(aid)
        if r is None or (since and r["published"] < since):
            continue
        if ticker and ticker.upper() not in json.loads(r["tickers"] or "[]"):
            continue
        published = dt.datetime.strptime(r["published"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        age_days = max((now - published).total_seconds() / 86400, 0)
        similarity = 1 - distance
        score = similarity * (0.7 + 0.3 * 0.5 ** (age_days / half_life))
        out.append({**dict(r), "similarity": similarity, "score": score})
    out.sort(key=lambda r: -r["score"])
    return out[:k]


def mentioned_tickers(cfg: dict, holdings, question: str) -> list[str]:
    """Holdings named in a question by ticker or company name, case-insensitively ('nvda', 'Nvidia')."""
    ambiguous = set(cfg.get("ambiguous_tickers", []))
    found = []
    for t in holdings.tickers_by_weight():
        names = [n for n in cfg.get("aliases", {}).get(t, []) if len(n) > 3 or n.isupper()]
        ticker_hit = re.search(rf"\b{t}\b", question if t in ambiguous else question.upper())
        if ticker_hit or any(re.search(rf"\b{re.escape(n)}\b", question, re.I) for n in names):
            found.append(t)
    return found


def other_stocks(cfg: dict, store, holdings, question: str, limit: int = 3) -> list[str]:
    """Stocks named in a question that are not holdings ('should I buy Oracle?'), with market data fetched now when
    none is stored from the last 20 hours. Stocks Yahoo has no data for are dropped."""
    market, out = Market(cfg, store), []
    for t in companies_in(cfg, question):
        if t in holdings.stocks or len(out) == limit:
            continue
        if stale(market.latest(t)):
            try:
                market.live(t)
            except Exception:
                continue
        out.append(t)
    return out


def portfolio_context(cfg: dict, store, holdings, tickers: list[str], others: list[str] = (),
                      funds: list[str] = ()) -> str:
    """What the LLM needs to reason about the owner's positions: the portfolio with its biggest real exposures
    (direct + through funds), then for each holding asked about its position and exposure, market data and analyst
    targets, recent news sentiment, summarized stories, company announcements, insider trades and earnings date; for
    other stocks asked about, their market data; for funds asked about, their holdings."""
    names = {t: (a[0] if a else t) for t, a in cfg.get("aliases", {}).items()}
    total = sum(e["value"] for e in (*holdings.stocks.values(), *holdings.funds.values()))
    lines = [f"PORTFOLIO (total ${total:,.0f}; cash in money-market funds excluded)",
             "Stocks (largest first): ticker, name, value, share of portfolio, gain or loss vs cost:"]
    for t in holdings.tickers_by_weight():
        e = holdings.stocks[t]
        gain = vs_cost(e["value"], e["cost"]) if e.get("cost") else "cost unknown"
        lines.append(f"- {t} {names.get(t, '')}: ${e['value']:,.0f}, {e['weight']:.1%}, {gain}")
    fd = FundData(cfg, store)
    lines.append("Funds: symbol, name, theme, share of portfolio, holdings data:")
    for sym, f in sorted(holdings.funds.items(), key=lambda kv: -kv[1]["weight"]):
        filing = fd.filing(sym)
        known = (f"holdings as of {filing['period']}" + (f" via public twin {fd.source_of(sym)}"
                                                         if fd.source_of(sym) != sym else "")) if filing else "no holdings data"
        lines.append(f"- {sym} ({f['description'].title()}; theme: {f['theme']}): {f['weight']:.1%}; {known}")
    ex = fd.exposures(holdings)
    lines.append("Biggest real exposures, ranked (direct shares + the owner's slice of each fund's holdings):")
    for n, (t, d) in enumerate(sorted(ex["by_ticker"].items(), key=lambda kv: -kv[1]["total"])[:10], 1):
        lines.append(f"- #{n} " + exposure_line(t, d, ex["portfolio"]))
    if ex["missing"]:
        lines.append(f"(Not included: {', '.join(ex['missing'])}, {ex['missing_value'] / ex['portfolio']:.1%} of the "
                     "portfolio, whose holdings are unknown.)")

    lines += returns_lines(cfg, store)

    week = iso(utcnow() - dt.timedelta(days=7))
    two_weeks = iso(utcnow() - dt.timedelta(days=14))
    market = Market(cfg, store)
    for t in tickers[:4]:
        lines += ["", f"HOLDING DETAIL: {t} ({names.get(t, t)})"]
        e = holdings.stocks[t]
        if e.get("shares"):
            lines.append(f"Position: {e['shares']:,.3f} shares worth ${e['value']:,.0f}"
                         + (f", cost basis ${e['cost']:,.0f} (now {vs_cost(e['value'], e['cost'])})"
                            if e.get("cost") else ""))
        if t in ex["by_ticker"]:
            lines.append("Real exposure: " + exposure_line(t, ex["by_ticker"][t], ex["portfolio"]))
        q = market.latest(t)
        if q and q.get("price"):
            lines += context_lines(q, market.closes(t))
        lines += advisor.context_lines(store, t) if advisor else []
        rows = store.db.execute(
            "SELECT s.sentiment, s.summary, s.why, s.headline, s.kind, a.title, a.published, a.body, a.source "
            "FROM stories s JOIN articles a ON a.id = s.primary_id WHERE s.relevant = 1 AND s.tickers LIKE ? "
            "AND a.published >= ? ORDER BY a.published DESC", (f'["{t}"%', two_weeks)).fetchall()
        news = [r for r in rows if r["kind"] not in ("filing", "stake") and r["published"] >= week]
        summarized = [r for r in news if r["summary"]]
        counts = Counter(r["sentiment"] for r in summarized)
        lines.append(f"News in the last 7 days: {len(news)} stories, {len(summarized)} summarized; sentiment of the "
                     f"summarized ones: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()) if summarized
                     else f"News in the last 7 days: {len(news)} stories, none summarized.")
        for r in summarized[:8]:
            lines.append(f"- {day(r['published'])} [{r['sentiment']}] {r['title']}: {r['summary']} "
                         f"Why it matters: {r['why']}")
        for r in rows:
            if r["headline"]:
                lines.append(f"- Company filing {day(r['published'])} [{r['sentiment']}]: {r['headline']}. "
                             f"{r['summary']}")
        trades = [insider_line(t, d) for r in rows if r["kind"] == "filing" and form_of(r["title"]) == "4"
                  for d in [insider(r["body"])] if d]
        trades = [x for x in trades if x]
        lines.append("Insider open-market trades (14 days): " + ("; ".join(trades) if trades else "none"))
        e = store.db.execute("SELECT date, detail FROM events WHERE ticker = ? AND kind = 'earnings' AND date >= ? "
                             "ORDER BY date LIMIT 1", (t, dt.date.today().isoformat())).fetchone()
        if e:
            d = json.loads(e["detail"] or "{}")
            eps = f", EPS estimate {d['eps_estimate']}" if d.get("eps_estimate") is not None else ""
            lines.append(f"Next earnings: {day(e['date'])}{eps}")
    for t in others[:3]:
        q = market.latest(t)
        via = ex["by_ticker"].get(t)
        held = (f"Not owned directly, but held through funds: {exposure_line(t, via, ex['portfolio'])}"
                if via and via["total"] else "Not owned directly or through the funds with known holdings.")
        lines += ["", f"OTHER STOCK: {t} ({q['name']})", held, *context_lines(q, market.closes(t)),
                  *(advisor.context_lines(store, t) if advisor else [])]
    for s in funds[:3]:
        lines += ["", *fund_lines(fd, holdings, s)]
    return "\n".join(lines)


def ask(cfg: dict, store, question: str, k: int = 8, embedder: Embedder | None = None) -> tuple[str, list[dict]]:
    from .pipeline import load_portfolio

    holdings = load_portfolio(cfg)
    tickers = mentioned_tickers(cfg, holdings, question)
    others = other_stocks(cfg, store, holdings, question)
    funds = mentioned_funds(holdings, question)
    hits = []
    for t in tickers[:2]:  # articles about the holdings asked about first, then the closest matches overall
        hits += search(cfg, store, question, k=k // 2, ticker=t, embedder=embedder)
    hits += search(cfg, store, question, k=k, embedder=embedder)
    seen, unique = set(), []
    for h in hits:  # the same article often arrives from two feeds
        key = re.sub(r"\W+", " ", h["title"].lower()).strip()
        if h["id"] not in seen and key not in seen:
            seen |= {h["id"], key}
            unique.append(h)
    hits = unique[:k]
    llm = LLM(cfg, purpose="ask")
    if not llm.available:
        return "The LLM endpoint is not available (see the log).", hits
    sources = [f"{source_header(h['title'], h['source'], h['published'])}\n{(h['body'] or h['summary'] or '')[:1500]}"
               for h in hits]
    return llm.answer(question, sources, portfolio_context(cfg, store, holdings, tickers, others, funds)), hits
