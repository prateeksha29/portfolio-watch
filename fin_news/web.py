"""Phone-friendly web app over the digests and the archive.  Run: python -m fin_news.web

Reads what the pipeline wrote; needs no GPU except for Ask (which calls llama-swap). Search embeds the query with
EmbeddingGemma on the CPU, so the web app never takes GPU memory from llama-swap or a pipeline run."""
import contextvars
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import threading
import time
from collections import Counter
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool

from . import discover
from .addons import advisor, optional
from .collect import iso, utcnow
from .config import ROOT, load_config
from .digest import _ago, person_topics
from .discover import both_sources
from .embed import Embedder
from .exportcheck import kind_of_export as _kind_of_export
from .funds import FundData
from .market import (
    EXCHANGE_TZ,
    Market,
    action_line,
    chart_svg,
    counts,
    eps_change,
    find_companies,
    market_open,
    money,
    pct,
    stale,
    upside,
    valid_ticker,
)
from .pipeline import load_portfolio, open_store
from .query import ask, search
from .returns import (
    FRESH_DAYS,
    PERIODS,
    Returns,
    by_month,
    by_stock,
    concentration,
    load_cash,
    months_back,
    summarize,
)
from .users import (
    VALID_ID,
    check_password,
    password_hash,
    set_password,
    user_config,
    user_ids,
)

cfg = load_config()  # shared settings; a signed-in person's view of them is me()
SECRET = os.environ.get("WEB_SECRET", "")
if not SECRET:
    raise SystemExit("Set WEB_SECRET in .env (a long random string)")
COOKIE, COOKIE_DAYS = "fin_news_auth", 180
_me: contextvars.ContextVar[dict | None] = contextvars.ContextVar("me", default=None)


def me() -> dict:
    """The signed-in person's config (users.user_config): their folder, digests and settings."""
    u = _me.get()
    if u is None:
        raise HTTPException(401)
    return u


def uid() -> str:
    return me()["user"]["id"]


def is_admin() -> bool:
    return me()["user"]["admin"]
OPEN_PATHS = {"/login", "/manifest.webmanifest", "/sw.js", "/offline", "/ping"}  # exact paths; no personal data in these
RUN_HOURS = (6, 14, 18)  # must match the run.sh crontab line

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "fin_news" / "static"), name="static")
templates = Environment(loader=FileSystemLoader(ROOT / "fin_news" / "templates"), autoescape=select_autoescape())
# Digest text comes from feed titles and summaries: render Markdown with raw HTML disabled.
# Bottom tab bar icons (24x24 line drawings).
ICONS = {k: Markup(f'<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">{v}</svg>') for k, v in {
    "digest": '<path d="M4 5h13v14H6a2 2 0 0 1-2-2z"/><path d="M17 9h3v8a2 2 0 0 1-2 2h-1"/><path d="M8 9h5M8 13h5M8 16h3"/>',
    "holdings": '<path d="M12 3v9h9"/><path d="M20.5 15A9 9 0 1 1 9 3.5"/>',
    "ideas": '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.7.6 1.1 1.3 1.1 2.2h5c0-.9.4-1.6 1.1-2.2A6 6 0 0 0 12 3z"/>',
    "ask": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
}.items()}
templates.globals.update(money=money, pct=pct, action_line=action_line, eps_change=eps_change, icons=ICONS)
markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
log = logging.getLogger(__name__)
_embedder, _embedder_lock = None, threading.Lock()
_use_lock = threading.Lock()  # one search/ask at a time (the GPU server answers one at a time anyway)


_alert_cache: dict[str, list] = {}  # person -> [checked at, needs attention, an add-on's login expired]


def _refresh_alerts() -> list:
    """Cached for a minute per person. Needs attention: their portfolio file was skipped, or an add-on says so (for
    admins: its login has expired and its data is old). Login expired: an add-on's, admins only."""
    u = me()
    hit = _alert_cache.get(u["user"]["id"])
    if hit is None or time.time() - hit[0] > 60:
        store = open_store(cfg)
        needs, expired = advisor_web.alerts(store, u["user"]["admin"]) if advisor_web else (False, False)
        hit = [time.time(), bool(load_portfolio(u).warning) or needs, expired]
        _alert_cache[u["user"]["id"]] = hit
    return hit


def needs_attention() -> bool:
    """Something on the Status page needs the person: shown as a dot on every page."""
    return _refresh_alerts()[1]


def login_expired() -> bool:
    """An add-on's login has run out: a red ! by Status in the More menu, but no dot on every page until its data is
    old."""
    return _refresh_alerts()[2]


def render(template: str, **ctx) -> HTMLResponse:
    # generated: when the page was made; a saved copy shown offline says how old it is
    u = _me.get()
    extra = {"alert": needs_attention(), "login_expired": login_expired(), "person": u["user"]} if u else {}
    return HTMLResponse(templates.get_template(template).render(
        generated=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **extra, **ctx))


def _sign(who: str, expiry: str) -> str:
    # The password hash is part of what is signed: a new password signs out every other device.
    msg = f"{who}.{expiry}.{password_hash(who)}"
    return hmac.new(SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()


def _signed_in(request: Request) -> str | None:
    """The person whose valid sign-in cookie came with the request."""
    who, _, rest = request.cookies.get(COOKIE, "").partition(".")
    expiry, _, sig = rest.partition(".")
    if not (VALID_ID.match(who) and expiry.isdigit() and sig and int(expiry) > time.time()):
        return None
    if not password_hash(who) or not hmac.compare_digest(sig, _sign(who, expiry)):
        return None
    return who if who in user_ids() else None


WEB = cfg.get("web", {})
PROXY = WEB.get("proxy", "")  # the nginx machine's address: only it may say the request came over https
DIRECT = WEB.get("direct_access", True)  # False: only the proxy (and this machine) may connect


@app.middleware("http")
async def require_login(request: Request, call_next):
    peer = request.client.host if request.client else ""
    if PROXY and peer == PROXY:
        if request.headers.get("x-forwarded-proto") == "https":
            request.scope["scheme"] = "https"  # the login cookie is then marked Secure
    elif not DIRECT and peer not in ("127.0.0.1", "::1"):
        return HTMLResponse("Use the https address.", status_code=403)
    path = request.url.path
    if path in OPEN_PATHS or path.startswith("/static/"):
        return await call_next(request)
    who = _signed_in(request)
    if who is None:
        return RedirectResponse("/login", status_code=303)
    _me.set(user_config(cfg, who))
    return await call_next(request)


@app.get("/login")
def login_form():
    return render("login.html")


@app.post("/login")
def login(request: Request, name: str = Form(""), password: str = Form("")):
    who = name.strip().lower()
    if not (VALID_ID.match(who) and who in user_ids() and check_password(who, password)):
        time.sleep(1)  # slows guessing
        return render("login.html", error=True, name=name)
    expiry = str(int(time.time()) + COOKIE_DAYS * 86400)
    resp = RedirectResponse("/", status_code=303)
    # Secure once served over https (nginx says so in X-Forwarded-Proto), so the phone never sends it unencrypted
    resp.set_cookie(COOKIE, f"{who}.{expiry}.{_sign(who, expiry)}", max_age=COOKIE_DAYS * 86400, httponly=True,
                    samesite="lax", secure=request.url.scheme == "https")
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp


@app.get("/manifest.webmanifest")
def manifest():
    icon = lambda name, size, purpose: {"src": f"/static/{name}", "sizes": f"{size}x{size}", "type": "image/png",
                                        "purpose": purpose}
    return JSONResponse({
        "id": "/", "name": "Portfolio Watch", "short_name": "Portfolio Watch", "start_url": "/", "scope": "/",
        "description": "News, positions and returns for your portfolio", "display": "standalone",
        "orientation": "portrait", "background_color": "#0b1120", "theme_color": "#0f172a",
        "icons": [icon("icon-192.png", 192, "any"), icon("icon-512.png", 512, "any"),
                  icon("icon-maskable-192.png", 192, "maskable"), icon("icon-maskable-512.png", 512, "maskable")],
        "shortcuts": [{"name": "Holdings", "url": "/positions"}, {"name": "Performance", "url": "/returns"},
                      {"name": "Ask", "url": "/ask"}]}, media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    """Served from the root so it controls every page; never cached by the browser, so updates reach the phone."""
    return FileResponse(ROOT / "fin_news" / "static" / "sw.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/offline")
def offline():
    return HTMLResponse(templates.get_template("offline.html").render())


@app.get("/ping")
def ping():
    """For the pages' offline detection: answers only when the server is reachable."""
    return HTMLResponse(status_code=204, headers={"Cache-Control": "no-store"})


@app.get("/offline-urls")
def offline_urls():
    """Every page the service worker saves for offline use: all tabs and their variants, the stock page of each
    holding and of each stock its funds hold, each fund, the kept digests and each holding's news search."""
    u = me()
    portfolio, store, names = load_portfolio(u), open_store(cfg), _names()
    urls = ["/", "/digests", "/holdings", "/positions", "/returns", "/money", "/ideas", "/ideas?view=watching",
            *(advisor_web.OFFLINE_URLS if advisor_web else []), "/status", "/ask", "/search", "/guide", "/files",
            "/account"]
    urls += ["/benchmarks"] if is_admin() else []
    urls += [f"/digests/{p.name}" for p in sorted(u["paths"]["digests"].glob("*.md")) if p.name != "latest.md"]
    urls += [f"/positions?sort={k}" for k in ("value", "gain", "pct", "today")]
    urls += [f"/money?w={k}" for k in MONEY_WINDOWS]
    r = Returns(u, store)
    for k in PERIODS:
        if r.files and r.end_date() and r.available(k):
            start = r.period_start(k)
            urls += [f"/returns?p={k}", f"/money?first={start}&last={r.end_date()}"]
    fd = FundData(cfg, store)
    stocks = list(portfolio.tickers_by_weight())
    stocks += [s for s in r.positions if s not in portfolio.funds]
    stocks += [t for t in fd.exposures(portfolio)["by_ticker"]]
    stocks += [i["ticker"] for i in discover.ideas(store, "status != 'dismissed' AND flagged >= ?",
                                                   ((dt.date.today() - dt.timedelta(days=IDEAS_DAYS)).isoformat(),),
                                                   user=uid())]
    stocks += [i["ticker"] for i in discover.ideas(store, "status = 'watch'", user=uid())]
    for f in portfolio.funds:
        urls.append(f"/fund/{f}")
        stocks += [h["ticker"] for h in fd.holdings(f)[:25] if h.get("ticker")]
    market = Market(cfg, store)  # stocks without stored market data would only give a 404 page
    stocks = [t for t in dict.fromkeys(stocks) if valid_ticker(t) and (t in portfolio.stocks or market.latest(t) is not None)]
    urls += [f"/holding/{t}" for t in stocks]
    urls += [f"/search?q={quote(t)}&articles=1" for t in stocks if t not in portfolio.stocks]  # their search link
    for t in portfolio.tickers_by_weight():  # the "search {ticker}" link on each holding's page
        name = names.get(t, "") or (market.latest(t) or {}).get("name", "")
        urls.append(f"/search?q={quote(name)}&ticker={t}")
    return JSONResponse({"urls": list(dict.fromkeys(urls))}, headers={"Cache-Control": "no-store"})


def _digest_page(path, title: str, active: str):
    html = markdown.render(path.read_text()) if path.exists() else ""
    return render("digest.html", title=title, active=active, html=html)


@app.get("/")
def latest():
    return _digest_page(me()["paths"]["digests"] / "latest.md", "Latest digest", "/")


@app.get("/digests")
def digests():
    files = sorted((p for p in me()["paths"]["digests"].glob("*.md") if p.name != "latest.md"), reverse=True)
    labels = []
    for p in files:
        try:
            labels.append((p.name, dt.datetime.strptime(p.stem, "%Y-%m-%d_%H%M").strftime("%a %b %d, %H:%M")))
        except ValueError:
            labels.append((p.name, p.stem))
    return render("digests.html", title="Past digests", active="/digests", files=labels,
                  days=cfg["retention"]["digest_days"])


@app.get("/digests/{name}")
def digest(name: str):
    folder = me()["paths"]["digests"]
    path = folder / name
    if path.parent != folder or path.suffix != ".md" or not path.exists():
        raise HTTPException(404)
    return _digest_page(path, path.stem, "/digests")


def _names() -> dict[str, str]:
    return {t: (a[0] if a else t) for t, a in me().get("aliases", {}).items()}


def _stories(store, where: str, params=()) -> list[dict]:
    rows = store.db.execute(
        "SELECT s.id, s.tickers, s.summary, s.sentiment, s.why, s.n_sources, s.kind, a.title, a.url, a.source, "
        f"a.published FROM stories s JOIN articles a ON a.id = s.primary_id WHERE {where}", params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["tickers"] = json.loads(d["tickers"] or "[]")
        d["ago"] = _ago(d["published"])
        out.append(d)
    return out


@app.get("/holdings")
def holdings():
    portfolio, names, store = load_portfolio(me()), _names(), open_store(cfg)
    since = iso(utcnow() - dt.timedelta(days=3))
    stories = _stories(store, "s.relevant = 1 AND s.kind NOT IN ('stake', 'filing') AND a.published >= ?", (since,))
    market, rows = Market(cfg, store), []
    for t in portfolio.tickers_by_weight():  # a story counts under the first of its stocks this person holds
        mine = [s for s in stories if [x for x in s["tickers"] if x in portfolio.stocks][:1] == [t]]
        sent = Counter(s["sentiment"] for s in mine if s["sentiment"])
        q = market.latest(t) or {}
        rows.append({"ticker": t, "name": names.get(t, ""), "weight": portfolio.stocks[t]["weight"],
                     "count": len(mine), "pos": sent["positive"], "neg": sent["negative"], "mix": sent["mixed"],
                     "price": q.get("price"), "day": q.get("change_pct"),
                     "month": (q.get("returns") or {}).get("oneMonthReturn"),
                     "upside": upside(q) if q else None})
    price = _price_info(market, portfolio)
    fd = FundData(cfg, store)
    funds = []
    for s, f in sorted(portfolio.funds.items(), key=lambda kv: -kv[1]["weight"]):
        filing = fd.filing(s)
        funds.append({"symbol": s, "name": f["description"].title(), "weight": f["weight"], "theme": f["theme"],
                      "period": filing["period"] if filing else None,
                      "proxy": fd.source_of(s) if fd.source_of(s) != s else None})
    ex = fd.exposures(portfolio)
    exposures = [{"ticker": t, **d, "share": d["total"] / ex["portfolio"], "owned": t in portfolio.stocks}
                 for t, d in sorted(ex["by_ticker"].items(), key=lambda kv: -kv[1]["total"])[:10]]
    return render("holdings.html", title="Holdings", active="/holdings", rows=rows, funds=funds,
                  exposures=exposures, missing=ex["missing"], missing_share=ex["missing_value"] / ex["portfolio"],
                  as_of=price["label"], prices=_prices_status(), auto=_auto(price))


STALE_MIN = 15  # minutes: during the market's hours, older prices than this get refreshed when a page opens


def _price_info(market: Market, portfolio) -> dict:
    """The held stocks' price time for the 'as of' label ('Live · 10:31 am', or 'Nov 4 close · checked 6:13 am') and
    whether they're stale (market open and nothing fetched in the last STALE_MIN minutes)."""
    qs = [q for t in portfolio.stocks for q in [market.latest(t)] if q]
    fetched = [dt.datetime.strptime(q["fetched"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC) for q in qs]
    times = [q["market_time"] for q in qs if q.get("market_time")]
    now, is_open = dt.datetime.now(dt.UTC), market_open()
    stale = is_open and (not fetched or now - max(fetched) > dt.timedelta(minutes=STALE_MIN))
    if not times:
        return {"label": "", "stale": stale}
    clock = lambda t: f"{t:%-I:%M %p}".lower()
    newest = dt.datetime.fromtimestamp(max(times), dt.UTC)
    if is_open and newest.astimezone(EXCHANGE_TZ).date() == now.astimezone(EXCHANGE_TZ).date():
        return {"label": f"Live · {clock(newest.astimezone())}", "stale": stale}
    ny = newest.astimezone(EXCHANGE_TZ)  # a price from before 4 pm New York is a mid-day price, not the close
    label = f"{ny:%b %-d} close" if ny.hour >= 16 else f"{ny:%b %-d}, {clock(newest.astimezone())}"
    if fetched:
        c = max(fetched).astimezone()
        label += f" · checked {'' if c.date() == now.astimezone().date() else f'{c:%b %-d}, '}{clock(c)}"
    return {"label": label, "stale": stale}


def _auto(price: dict) -> bool:
    """Should the page start a price refresh by itself? (Its JavaScript does, so plain GETs never fetch.)"""
    s = _prices_status()
    return bool(price["stale"] and not s["running"] and not s["cooling"])


@app.get("/fund/{symbol}")
def fund_page(symbol: str):
    portfolio, store = load_portfolio(me()), open_store(cfg)
    symbol = symbol.upper()
    if symbol not in portfolio.funds:
        raise HTTPException(404)
    fd, f = FundData(cfg, store), portfolio.funds[symbol]
    rows = fd.holdings(symbol)
    for r in rows:
        r["mine"] = f["value"] * r["pct"] / 100
        r["owned"] = r["ticker"] in portfolio.stocks
    assets, countries = fd.mix(symbol)
    return render("fund.html", title=symbol, active="/holdings", symbol=symbol, fund=f, filing=fd.filing(symbol),
                  proxy=fd.source_of(symbol) if fd.source_of(symbol) != symbol else None, top=rows[:25],
                  overlap=[r for r in rows if r["owned"]], n=len(rows), assets=[a for a in assets if a[1] >= 0.5],
                  countries=countries if len(countries) > 1 else [])


def _returns() -> tuple[Returns, dict]:
    r = Returns(me(), open_store(cfg))
    ctx = {"active": "/holdings", "files": r.files, "covered_from": r.covered_from, "gaps": r.gaps,
           "behind": r.history_behind(), "positions_date": r.positions_date, "ready": bool(r.files and r.end_date()), "names": {**r.names, **_names()}, "fresh_days": FRESH_DAYS}
    return r, ctx


def _dollar_chart(days: list[dict], w: int = 340, h: int = 190) -> dict | None:
    """SVG coordinates for Performance's chart: value, starting value plus money put in (drawn as steps), and the
    same money in the S&P 500."""
    if len(days) < 2:
        return None
    lo = min(min(d["value"], d["put_in"], d["bench"]) for d in days)
    hi = max(max(d["value"], d["put_in"], d["bench"]) for d in days)
    pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 4, w - 52, 8, h - 22
    x = lambda i: left + (right - left) * i / (len(days) - 1)
    y = lambda v: bottom - (bottom - top) * (v - lo) / (hi - lo)
    line = lambda k: " ".join(f"{x(i):.1f},{y(d[k]):.1f}" for i, d in enumerate(days))
    steps = []
    for i, d in enumerate(days):
        if i:
            steps.append(f"{x(i):.1f},{y(days[i - 1]['put_in']):.1f}")
        steps.append(f"{x(i):.1f},{y(d['put_in']):.1f}")
    ticks = [lo + (hi - lo) * f for f in (0.1, 0.5, 0.9)]
    end = days[-1]
    return {"w": w, "h": h, "value": line("value"), "bench": line("bench"), "put_in": " ".join(steps),
            "right": right, "bottom": bottom,
            "ticks": [(y(v), f"${v / 1000:,.0f}k") for v in ticks],
            "dates": [(x(i), days[i]["date"].strftime("%b %-d")) for i in (0, len(days) // 2, len(days) - 1)],
            "end": {k: (y(end[k]), end[k]) for k in ("value", "put_in", "bench")}}


@app.get("/positions")
def positions_page(sort: str = "value"):
    """Holdings -> Positions: Fidelity's positions view for the brokerage accounts at the latest close."""
    r, ctx = _returns()
    portfolio = load_portfolio(me())
    ctx.update(title="Positions", sort=sort, rows=[])
    if r.positions_file and r.end_date():
        rows = r.holdings_now()
        # The day's change is for the newest price day; a holding still at an older day's price (a mutual fund before
        # its price for the day is out) counts as unchanged instead of adding that older day's change.
        day_date = max((h["day_date"] for h in rows if h["day_date"]), default=None)
        for h in rows:
            h["pending"] = bool(h["day_date"] and h["day_date"] < day_date)
        keys = {"value": lambda h: -h["value"], "gain": lambda h: -(h["gain"] or 0),
                "pct": lambda h: -(h["gain_pct"] if h["gain_pct"] is not None else -9),
                "today": lambda h: -(h["day"] or 0) if not h["pending"] else 0}
        rows.sort(key=keys.get(sort, keys["value"]))
        value = sum(h["value"] for h in rows)
        basis = sum(h["basis"] for h in rows if h["basis"] is not None)
        priced = sum(h["value"] for h in rows if h["basis"] is not None)
        day = sum(h["day"] or 0 for h in rows if h["day_date"] == day_date)
        cash, pending_cash = load_cash(r.positions_file, r.accounts)
        ctx.update(rows=rows, value=value, basis=basis, gain=priced - basis, gain_pct=(priced / basis - 1) if basis else None,
                   day=day, day_pct=day / (value - day) if value != day else None, end=r.end_date(), day_date=day_date,
                   pending=[h["symbol"] for h in rows if h["pending"]],
                   cash=cash, pending_cash=pending_cash, positions_date=r.positions_date,
                   no_basis=[h["symbol"] for h in rows if h["basis"] is None],
                   conc=concentration(me(), rows),
                   newer=sorted(h["symbol"] for h in rows if h["newer"]),
                   prices=_prices_status(), funds=set(portfolio.funds))
        price = _price_info(Market(cfg, open_store(cfg)), portfolio)
        ctx.update(as_of=price["label"], auto=_auto(price))
    return render("positions.html", **ctx)


@app.get("/returns")
def returns_page(p: str = "1Y"):
    """Holdings -> Performance: what the owner earned over a period, the chart of their dollars, and each month."""
    r, ctx = _returns()
    ctx.update(title="Performance", periods=[(k, r.available(k)) for k in PERIODS], p=p, x=None)
    if ctx["ready"]:
        if not r.available(p):
            p = next((k for k, ok in ctx["periods"] if ok), None)
        if p:
            x = r.period(p)
            ctx.update(p=p, x=x, chart=_dollar_chart(r.daily(x["start"], x["end"])),
                       months=r.monthly(x["start"], x["end"]) if p != "1M" else [])
    return render("returns.html", **ctx)


MONEY_WINDOWS = {"1": ("Last month", 0, 1), "1-3": ("1–3 mo ago", 1, 3), "3-6": ("3–6 mo ago", 3, 6),
                 "6-12": ("6–12 mo ago", 6, 12), "all": ("All", 0, None)}


@app.get("/money")
def money_page(w: str = "all", first: str = "", last: str = ""):
    """Holdings -> Buys vs S&P: every investment made in a window of buy dates, and how it is doing today."""
    r, ctx = _returns()
    ctx.update(title="Buys vs S&P", w=w, windows=MONEY_WINDOWS, lots=None)
    if not ctx["ready"]:
        return render("money.html", **ctx)
    end, floor = r.end_date(), r.covered_from
    try:
        a, b = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
        w = "custom"
    except ValueError:
        w = w if w in MONEY_WINDOWS else "all"
        _, newer, older = MONEY_WINDOWS[w]
        a = months_back(end, older) + dt.timedelta(days=1) if older else floor
        b = months_back(end, newer) if newer else end
    a, b = max(min(a, b), floor), min(max(a, b), end)
    lots = r.window(a, b)
    groups = by_stock(lots)
    scale = max([abs(g["settled"]["vs"] if g["settled"]["vs"] is not None else g["vs"] or 0) for g in groups] + [0.05])
    ctx.update(w=w, first=a, last=b, end=end, floor=floor, lots=lots, x=summarize(lots), months=by_month(lots),
               groups=groups, scale=min(scale, 0.5), approximate=sorted(r.prices.approximate),
               month_max=max([m["cost"] for m in by_month(lots)] + [1]), funds=set(load_portfolio(me()).funds))
    return render("money.html", **ctx)


IDEAS_DAYS = 14  # how long a new idea stays in New unless watched or dismissed


def _idea_card(i: dict, q: dict, via_funds: float, sp_now: float | None, store) -> dict:
    """An idea with its context: where the price sits, the Street's view and its moves in 30 days, earnings."""
    price, today = q.get("price"), dt.date.today()
    since = (today - dt.timedelta(days=30)).isoformat()
    acts = [a for a in q.get("actions", []) if a["date"] >= since]
    lo, hi = q.get("low_52w"), q.get("high_52w")
    earn = q.get("earnings_date")
    earn_days = (dt.date.fromisoformat(earn) - today).days if earn and earn >= today.isoformat() else None
    return {**i, "q": q, "via_funds": via_funds, "both": both_sources(store, i["ticker"], i["source"], uid()),
            "ret": price / i["price"] - 1 if price and i.get("price") else None,
            "sp_ret": sp_now / i["sp500"] - 1 if sp_now and i.get("sp500") else None,
            "range_pos": (price - lo) / (hi - lo) if price and lo and hi and hi > lo else None,
            "vs200": price / q["avg_200d"] - 1 if price and q.get("avg_200d") else None,
            "upgrades": sum(a["action"] in ("up", "init") for a in acts),
            "downgrades": sum(a["action"] == "down" for a in acts),
            "raised": sum(bool(a.get("target") and a.get("prior") and a["target"] > a["prior"]) for a in acts),
            "cut": sum(bool(a.get("target") and a.get("prior") and a["target"] < a["prior"]) for a in acts),
            "earn_days": earn_days}


@app.get("/ideas")
def ideas_page(view: str = "new"):
    """Discovery: new stock ideas (an add-on's picks, insider buying) with context, the ones being watched, and how each
    source's ideas did over their first 30 days."""
    portfolio, store, who = load_portfolio(me()), open_store(cfg), uid()
    market, ex = Market(cfg, store), FundData(cfg, store).exposures(portfolio)["by_ticker"]
    owned = set(portfolio.stocks) | set(portfolio.funds)
    if view == "watching":
        rows = discover.ideas(store, "status = 'watch'", user=who)
    else:
        view = "new"
        rows = [i for i in discover.ideas(store, "status = 'new' AND flagged >= ?",
                                          ((dt.date.today() - dt.timedelta(days=IDEAS_DAYS)).isoformat(),), user=who)
                if i["ticker"] not in owned]  # held (or bought since): not an idea for this person
    sp_now = discover.sp500_close(store)
    cards = []
    for i in rows:
        via = ex.get(i["ticker"])
        cards.append(_idea_card(i, market.latest(i["ticker"]) or {}, via["total"] - via["direct"] if via else 0,
                                sp_now, store))
    if view == "new":  # both sources first, then an add-on's picks, then the largest insider buying
        cards.sort(key=lambda c: (not c["both"], c["source"] == "insider", -(c.get("value") or 0)))
    n_watch = len(discover.ideas(store, "status = 'watch'", user=who))
    return render("ideas.html", title="Ideas", active="/ideas", seg="Watching" if view == "watching" else "New",
                  view=view, cards=cards, n_watch=n_watch, score=discover.scorecard(store), days=IDEAS_DAYS,
                  ran=store.get_meta("discovery_ran"), usd=discover.usd, sources=discover.SOURCES,
                  icons=discover.ICONS)


@app.post("/ideas/status")
def idea_status(key: str = Form(...), status: str = Form(...), view: str = Form("new")):
    if status not in ("new", "watch", "dismissed"):
        raise HTTPException(400)
    discover.set_status(open_store(cfg), key, status, uid())
    return RedirectResponse("/ideas?view=watching" if view == "watching" else "/ideas", status_code=303)


def _local(ts) -> str:
    """Epoch seconds or an ISO UTC string -> 'Oct 20, 13:00' in local time."""
    if isinstance(ts, str):
        t = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC)
    else:
        t = dt.datetime.fromtimestamp(ts, dt.UTC)
    return f"{t.astimezone():%b %d, %H:%M}"


@app.get("/holding/{ticker}")
def holding(request: Request, ticker: str, error: str = ""):
    """Any ticker: a holding gets its position and news under the market card; another stock is fetched live when
    it was not looked up in the last 20 hours (not when the service worker is saving pages for offline use)."""
    syncing = request.headers.get("X-Offline-Sync") == "1"
    ticker = ticker.upper().lstrip("$")
    if not valid_ticker(ticker):
        raise HTTPException(404)
    portfolio, store = load_portfolio(me()), open_store(cfg)
    owned = ticker in portfolio.stocks
    market = Market(cfg, store)
    q = market.latest(ticker)
    if not syncing and (q is None or (not owned and stale(q))):
        try:
            market.live(ticker)
            q = market.latest(ticker)
        except Exception as e:  # unknown symbol or Yahoo unavailable: show what we have
            error = error or (f"No market data found for {ticker}." if isinstance(e, LookupError)
                              else "Couldn't reach Yahoo Finance just now.")
    if q is None and not owned:
        return HTMLResponse(templates.get_template("holding.html").render(
            title=ticker, active="/search", ticker=ticker, q=None, error=error, owned=False), status_code=404)
    if q and not syncing and len(market.closes(ticker)) < 15:
        with _use_lock:
            market.backfill(ticker)
    closes = market.closes(ticker)
    card = {}
    if q:
        trend = q.get("trend", [])
        card = {"upside": upside(q), "now": counts(q), "before": counts(q, "-3m") or counts(q, "-2m"),
                "chart": Markup(chart_svg(closes, q)), "n_closes": len(closes), "fetched": _local(q["fetched"]),
                "market_time": _local(q["market_time"]) if q.get("market_time") else "",
                "month": (closes[-1][1] / closes[0][1] - 1) if len(closes) >= 15 else None, "trend": trend}
        if q.get("low_52w") and q.get("high_52w") and q["high_52w"] > q["low_52w"]:
            card["pos_52w"] = (q["price"] - q["low_52w"]) / (q["high_52w"] - q["low_52w"])
    ctx = {"title": ticker, "active": "/holdings" if owned else "/search", "ticker": ticker, "q": q, "card": card,
           "error": error, "owned": owned, "name": _names().get(ticker, "") or (q or {}).get("name", "")}
    if advisor_web:
        ctx.update(advisor_web.stock_ctx(store, ticker))
    ex = FundData(cfg, store).exposures(portfolio)
    via = ex["by_ticker"].get(ticker)
    if via and via["via"]:
        ctx["through_funds"] = {"total_via": via["total"] - via["direct"], "total": via["total"],
                                "share": via["total"] / ex["portfolio"],
                                "funds": sorted(via["via"].items(), key=lambda kv: -kv[1])}
    if owned:
        e = portfolio.stocks[ticker]
        ctx["position"] = {**e, "gain": e["value"] / e["cost"] - 1 if e.get("cost") else None}
        since = iso(utcnow() - dt.timedelta(days=7))
        stories = _stories(store, "s.relevant = 1 AND s.kind NOT IN ('stake', 'filing') AND a.published >= ? "
                           "AND s.tickers LIKE ?", (since, f'%"{ticker}"%'))
        stories.sort(key=lambda s: s["published"], reverse=True)
        ctx["summarized"] = [s for s in stories if s["summary"]]
        ctx["others"] = [s for s in stories if not s["summary"]][:40]
        ctx["sentiment"] = Counter(s["sentiment"] for s in ctx["summarized"] if s["sentiment"])
    return render("holding.html", **ctx)


@app.post("/holding/{ticker}/refresh")
def refresh_quote(ticker: str):
    ticker = ticker.upper()
    if not valid_ticker(ticker):
        raise HTTPException(404)
    try:
        Market(cfg, open_store(cfg)).live(ticker)
        return RedirectResponse(f"/holding/{ticker}", status_code=303)
    except Exception:
        return RedirectResponse(f"/holding/{ticker}?error=Refresh+failed%3B+showing+the+last+data.", status_code=303)


# Refresh prices: every stock held directly, one Yahoo page at a time in a background thread, so the page can show
# progress. Funds aren't fetched: their price changes once a day, after the close.
PRICES_COOLDOWN = 600  # seconds after a refresh in which the button doesn't fetch again
_prices = {"running": False, "done": 0, "total": 0, "errors": 0, "finished": 0.0}
_prices_lock = threading.Lock()


def _refresh_prices(tickers: list[str]) -> None:
    try:
        market = Market(cfg, open_store(cfg))
        for i, t in enumerate(tickers):
            if i:
                time.sleep(market.collector.delay_for("finance.yahoo.com"))
            try:
                market.live_price(t)
            except Exception as e:  # one stock failing doesn't stop the rest
                _prices["errors"] += 1
                log.warning("Refresh prices, %s: %s", t, e)
            _prices["done"] = i + 1
    finally:
        _prices.update(running=False, finished=time.time())


def _prices_status() -> dict:
    ago = time.time() - _prices["finished"]
    return {**_prices, "ago_min": round(ago / 60) if _prices["finished"] else None,
            "cooling": bool(_prices["finished"]) and ago < PRICES_COOLDOWN}


@app.post("/prices/refresh")
def refresh_prices(request: Request, next: str = Form("/holdings")):
    with _prices_lock:
        s = _prices_status()
        if not s["running"] and not s["cooling"]:
            tickers = load_portfolio(me()).tickers_by_weight()
            _prices.update(running=True, done=0, total=len(tickers), errors=0)
            threading.Thread(target=_refresh_prices, args=(tickers,), daemon=True).start()
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(_prices_status())
    return RedirectResponse(next if next in ("/holdings", "/positions") else "/holdings", status_code=303)


@app.get("/prices/status")
def prices_status():
    return JSONResponse(_prices_status())


def embedder() -> Embedder:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = Embedder(cfg, wait_seconds=60)  # a person is waiting
        return _embedder


@app.get("/search")
def search_page(q: str = "", ticker: str = "", articles: int = 0):
    """A lone ticker opens its stock page: holdings in any case ('nvda'), other stocks in capitals or with $
    ('AMD', '$amd'). So does a company's exact name ('Oracle'); a partial name ('Advanced Micro') lists matching
    companies above the article results. articles=1 (the stock page's own search link) always lists articles."""
    term, companies = q.strip(), []
    if term and not ticker and not articles:
        sym = term.lstrip("$").upper()
        if " " not in term and valid_ticker(sym) and (sym in load_portfolio(me()).stocks or term.startswith("$")
                                                     or term.isupper()):
            return RedirectResponse(f"/holding/{sym}", status_code=303)
        if len(term.split()) <= 4:
            exact, companies = find_companies(cfg, term)
            if exact:
                return RedirectResponse(f"/holding/{exact[0]}", status_code=303)
    hits = []
    if q.strip():
        e = embedder()
        with _use_lock:
            hits = search(cfg, open_store(cfg), q.strip(), k=20, ticker=ticker or None, embedder=e)
    return render("search.html", title="Search", active="/search", q=q, ticker=ticker, hits=hits, companies=companies,
                  tickers=load_portfolio(me()).tickers_by_weight(), days=cfg["retention"]["article_days"])


@app.get("/ask")
def ask_form():
    return render("ask.html", title="Ask", active="/ask", q="")


@app.post("/ask")
def ask_page(q: str = Form("")):
    answer, hits = ("", [])
    if q.strip():
        e = embedder()
        with _use_lock:
            answer, hits = ask(me(), open_store(cfg), q.strip(), embedder=e)
    return render("ask.html", title="Ask", active="/ask", q=q, answer=answer, hits=hits,
                  answer_html=Markup(markdown.render(answer)))  # the model writes markdown; raw HTML stays off


@app.get("/guide")
def guide():
    """How it works: the run, the schedule and the tabs, with the last run's numbers."""
    store = open_store(cfg)
    row = store.db.execute("SELECT started, stats FROM runs WHERE finished IS NOT NULL ORDER BY started DESC LIMIT 1").fetchone()
    last = json.loads(row["stats"] or "{}") if row else {}
    if row:
        last["when"] = f"{dt.datetime.strptime(row['started'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.UTC).astimezone():%a %b %-d, %-I:%M %p}"
        last["sources"] = len(last.get("health", {}))
    now = dt.datetime.now().astimezone()
    slots = (now.replace(hour=h, minute=0, second=0, microsecond=0) + dt.timedelta(days=d) for d in (0, 1) for h in RUN_HOURS)
    counts = store.db.execute("SELECT (SELECT count(*) FROM articles), (SELECT count(*) FROM stories), "
                              "(SELECT count(*) FROM stories WHERE summary IS NOT NULL)").fetchone()
    portfolio = load_portfolio(me())
    return render("guide.html", title="How it works", active="/guide", last=last, next_run=f"{next(t for t in slots if t > now):%a %-I %p}",
                  articles=counts[0], stories=counts[1], summarized=counts[2], keep=cfg["retention"]["article_days"],
                  n_stocks=len(portfolio.stocks), n_funds=len(portfolio.funds), llm=cfg["llm"], emb=cfg["embedding"],
                  f=cfg["filter"], fetch=cfg["fetch"], n_topics=len(person_topics(me(), portfolio)))


@app.get("/benchmarks")
def benchmarks():
    """What was tested before the November 2026 model and prompt changes, and why they were made. Admins only: the
    examples come from the admin's own portfolio and questions."""
    if not is_admin():
        raise HTTPException(404)
    t, path = None, ROOT / "bench" / "tech.json"  # written by `python -m fin_news.bench tech`
    if path.exists():
        t = json.loads(path.read_text())
        for task in ("stories", "filings", "ask"):
            t[task]["by"] = {f"{r['run']}|{r['set']}": r for r in t[task]["rows"]}
            t[task]["pk"] = {p["key"]: p for p in t[task]["paired"]}
    return render("bench.html", title="Benchmarks", active="/benchmarks", t=t, f=cfg["filter"], llm=cfg["llm"],
                  emb=cfg["embedding"])


@app.get("/status")
def status():
    store = open_store(cfg)
    runs = []
    for r in store.db.execute("SELECT id, started, finished, stats FROM runs ORDER BY started DESC LIMIT 12"):
        st = json.loads(r["stats"] or "{}")
        started = dt.datetime.strptime(r["started"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC).astimezone()
        runs.append({"started": f"{started:%a %b %d %H:%M}", "minutes": round(st.get("seconds", 0) / 60, 1),
                     "new": st.get("new_articles", 0), "summarized": st.get("summarized", 0),
                     "full_text": st.get("full_text", 0),
                     "errors": sum(h.get("errors", 0) for h in st.get("health", {}).values()), "stats": st})
    health = sorted(runs[0]["stats"].get("health", {}).items()) if runs else []
    now = dt.datetime.now().astimezone()
    slots = (now.replace(hour=h, minute=0, second=0, microsecond=0) + dt.timedelta(days=d)
             for d in (0, 1) for h in RUN_HOURS)
    next_run = next(t for t in slots if t > now)
    articles = store.db.execute("SELECT count(*) FROM articles").fetchone()[0]
    portfolio = load_portfolio(me())
    return render("status.html", title="Status", active="/status", runs=runs, health=health,
                  next_run=f"{next_run:%a %H:%M}", articles=articles, keep=cfg["retention"]["article_days"],
                  holdings_file=portfolio.source, holdings_warning=portfolio.warning, admin=is_admin(),
                  **(advisor_web.status_ctx(store, is_admin()) if advisor_web else {}))


# ---------- each person's files and password ----------

MAX_UPLOAD = 10 * 1024 * 1024
EXPORT_DATE = re.compile(r"([A-Z][a-z]{2}-\d{1,2}-\d{4})")


def _save_export(folder, filename: str, data: bytes) -> str:
    """Save an upload under a name of our choosing (never the uploaded path). Returns what happened."""
    if len(data) > MAX_UPLOAD:
        return f"{filename}: too large"
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return f"{filename}: not a text CSV"
    kind = _kind_of_export(text)
    if kind is None:
        return f"{filename}: not a Fidelity positions or activity export"
    m = EXPORT_DATE.search(filename)
    stamp = m.group(1) if m else f"{dt.date.today():%b-%d-%Y}"
    base = f"Portfolio_Positions_{stamp}" if kind == "positions" else f"Accounts_History-{stamp}"
    for n in range(1, 100):
        path = folder / (f"{base}.csv" if n == 1 else f"{base}-{n}.csv")
        if path.exists() and path.read_bytes() == data:
            return f"{filename}: already uploaded as {path.name}"
        if not path.exists():
            tmp = path.with_suffix(".part")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            tmp.replace(path)
            return f"{filename}: saved as {path.name}"
    return f"{filename}: too many copies"


def _files_page(messages=()):
    u = me()
    folder = u["paths"]["home"] / "fidelity"
    files = [{"name": p.name, "kb": round(p.stat().st_size / 1024), "when": _local(p.stat().st_mtime)}
             for p in sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)]
    portfolio = load_portfolio(u)
    return render("files.html", title="Your files", active="/files", files=files, messages=messages,
                  holdings_file=portfolio.source, holdings_warning=portfolio.warning)


@app.get("/files")
def files_page():
    return _files_page()


@app.post("/files")
async def upload_files(files: list[UploadFile] = File(...)):
    folder = me()["paths"]["home"] / "fidelity"
    messages = []
    for f in files[:20]:
        data = await f.read(MAX_UPLOAD + 1)
        name = os.path.basename(f.filename or "upload.csv")[:120]
        messages.append(await run_in_threadpool(_save_export, folder, name, data))
    _alert_cache.pop(uid(), None)
    return _files_page(messages)


@app.get("/account")
def account_page():
    return render("account.html", title="Account", active="/account")


@app.post("/account/password")
def change_password(current: str = Form(""), new: str = Form(""), again: str = Form("")):
    who = uid()
    if not check_password(who, current):
        time.sleep(1)
        return render("account.html", title="Account", active="/account", error="The current password is wrong.")
    if len(new) < 10 or new != again:
        return render("account.html", title="Account", active="/account",
                      error="The new password must be at least 10 characters, typed the same twice.")
    set_password(who, new)
    resp = RedirectResponse("/login", status_code=303)  # every device signs in again with the new password
    resp.delete_cookie(COOKIE)
    return resp


# A private add-on's pages (fin_news/advisor_web.py, git-ignored): loaded last, as they use this module.
advisor_web = optional("advisor_web") if advisor else None
if advisor_web:
    advisor_web.register(app, sys.modules[__name__])  # this module, also when run as __main__


def main() -> None:
    # proxy_headers off: the middleware itself decides whom to believe ([web] proxy) from the real peer address.
    uvicorn.run(app, host=WEB.get("host", "0.0.0.0"), port=WEB.get("port", 8050), log_level="info",
                proxy_headers=False)


if __name__ == "__main__":
    main()
