"""End-to-end test of the whole pipeline in an isolated folder (fresh database, own digests and logs).

    .venv/bin/python scripts/e2e_test.py [--keep]

Reads the people in users/ and their exports; their database, digests, the web app and cron are not touched.
Checks are about the first admin's portfolio (its largest holding and fund), so they work for any portfolio. Uses llama-swap (embedding, then the LLMs) for ~10-15
minutes and one extra round of feed requests. Checks results rather than just "it ran"; exits 1 on any failure.
The SEC window is widened to 10 days so there are 8-K/6-K press releases and Form 4s to process."""
import argparse
import copy
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fin_news.addons import advisor  # noqa: E402
from fin_news.config import load_config  # noqa: E402
from fin_news.filings import form_of, insider  # noqa: E402
from fin_news.llm import grounded  # noqa: E402
from fin_news.pipeline import COMPARISON, QUOTE_PAGE, is_stake_change, open_store, run  # noqa: E402

store_of = open_store
from fin_news.retention import prune  # noqa: E402
from fin_news.users import default_user, user_ids  # noqa: E402

results: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> bool:
    results.append((bool(ok), what))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}", flush=True)
    return bool(ok)


def test_cfg(base: Path) -> dict:
    cfg = copy.deepcopy(load_config())
    for name in ("data", "digests", "logs"):
        (base / name).mkdir(parents=True, exist_ok=True)
        cfg["paths"][name] = base / name
    cfg["paths"]["db"] = base / "data" / "news.db"
    for cached in (ROOT / "data").glob("*.json"):  # SEC ticker map: no need to download it again
        shutil.copy(cached, base / "data")
    cfg["fetch"]["max_age_days"] = 10
    cfg["sec"]["lookback_days"] = 10
    cfg.setdefault("market", {})["refresh_hour"] = 0  # market data on the first run at any hour
    if advisor:
        advisor.e2e_setup(cfg, ROOT, base)
    return cfg


def scrambled_users(base: Path) -> None:
    """Everyone's settings with scrambled copies of their exports (fin_news.exportcheck: fake accounts, amounts scaled
    at random), so the test never reads the real files. Account names in their settings are renamed to match."""
    from fin_news import users
    from fin_news.exportcheck import scramble

    real = users.USERS_DIR
    for uid in users.user_ids():
        home = base / "users" / uid
        (home / "fidelity").mkdir(parents=True)
        src = real / uid / "fidelity"
        accounts = scramble(src, home / "fidelity")["accounts"] if src.is_dir() else {}
        text = (real / uid / "user.toml").read_text()
        for name, fake in sorted(accounts.items(), key=lambda kv: -len(kv[0])):
            text = text.replace(users._toml_str(name), users._toml_str(fake))
        (home / "user.toml").write_text(text)
        if (real / uid / "password").exists():
            shutil.copy(real / uid / "password", home / "password")
    users.USERS_DIR = base / "users"


def load_person(cfg) -> tuple[dict, object]:
    """The first admin (as the command line picks), with their holdings."""
    from fin_news.pipeline import load_portfolio
    person = default_user(cfg)
    return person, load_portfolio(person)


def last_run(db) -> dict:
    return json.loads(db.execute("SELECT stats FROM runs ORDER BY started DESC LIMIT 1").fetchone()[0])


def check_run1(cfg, db, digest: str, seconds: float) -> None:
    st = last_run(db)
    print(f"\nRun 1: {seconds / 60:.1f} min, stats: { {k: v for k, v in st.items() if k != 'health'} }")
    check(st.get("new_articles", 0) > 500, f"collected new articles ({st.get('new_articles')})")
    broken = [n for n, h in st["health"].items() if h["requests"] and h["errors"] == h["requests"]]
    check(not broken, "every source answered at least once (all-failed: "
          + (", ".join(f"{n}: {st['health'][n].get('last_error') or '?'}" for n in broken) or "none") + ")")
    check(st.get("full_text", 0) >= 30, f"full article text fetched ({st.get('full_text')})")
    check(st.get("summarized", 0) >= 40, f"stories summarized by the LLM ({st.get('summarized')})")
    vecs = db.execute("SELECT count(*) FROM vec_articles").fetchone()[0]
    arts = db.execute("SELECT count(*) FROM articles").fetchone()[0]
    check(vecs == arts, f"one search vector per article ({vecs} / {arts})")

    # Filings
    filings = db.execute("SELECT a.title, a.body, s.headline, s.summary FROM articles a JOIN stories s "
                         "ON s.id = a.story_id WHERE a.kind = 'filing'").fetchall()
    releases = [f for f in filings if form_of(f["title"]) in ("8-K", "6-K")]
    form4 = [f for f in filings if form_of(f["title"]) == "4"]
    print(f"  ({len(filings)} filings: {len(releases)} 8-K/6-K, {len(form4)} Form 4)")
    if releases:
        check(all(f["body"] is not None for f in releases), "every 8-K/6-K was read")
        check(sum(bool(f["headline"]) for f in releases) >= 0.8 * len(releases),
              f"8-K/6-K summarized ({sum(bool(f['headline']) for f in releases)} of {len(releases)})")
        check("## Company announcements" in digest, "digest has Company announcements")
    if form4:
        parsed = [insider(f["body"]) for f in form4]
        check(all(p is not None for p in parsed), f"every Form 4 parsed ({len(form4)})")
        if any(p and (p.get("sold") or p.get("bought")) for p in parsed):
            check("## Insider trades" in digest, "digest has Insider trades")

    # Relevance and grouping invariants
    rel = db.execute("SELECT a.title, s.tickers, s.kind FROM stories s JOIN articles a ON a.id = s.primary_id "
                     "WHERE s.relevant = 1").fetchall()
    junk = [r["title"] for r in rel if QUOTE_PAGE.search(r["title"])]
    check(not junk, f"no quote/market-size pages among relevant stories ({junk[:2]})")
    stake_as_news = [r["title"] for r in rel if r["kind"] != "stake" and is_stake_change(r["title"])]
    check(not stake_as_news, f"fund stake changes kept out of news ({stake_as_news[:2]})")
    mixed = []
    for (sid,) in db.execute("SELECT id FROM stories WHERE n_articles > 1"):
        titles = [r[0] for r in db.execute("SELECT title FROM articles WHERE story_id = ?", (sid,))]
        norm = {re.sub(r"\W+", " ", t.lower()).strip() for t in titles}
        if len(norm) > 1 and any(COMPARISON.search(t) for t in norm):
            mixed.append(titles[:2])
    check(not mixed, f"no story merges different 'X vs. Y' headlines ({mixed[:1]})")

    # Grounding: every key number must appear in the text the LLM was given
    ungrounded = []
    for r in db.execute("SELECT s.key_numbers, a.title, coalesce(a.body, a.summary, '') AS text FROM stories s "
                        "JOIN articles a ON a.id = s.primary_id WHERE s.key_numbers IS NOT NULL"):
        source = f"{r['title']}\n{r['text']}"
        ungrounded += [n for n in json.loads(r["key_numbers"]) if not grounded(n, source)]
    check(not ungrounded, f"every summary number appears in its source ({ungrounded[:3]})")

    # Market data (once a day, in the first run)
    n_stocks = db.execute("SELECT count(DISTINCT ticker) FROM quotes").fetchone()[0]
    holdings = len(re.findall(r"^### [A-Z.]+ — ", digest, re.M))
    check(n_stocks >= 20, f"prices and analyst targets fetched for the holdings ({n_stocks})")
    check(db.execute("SELECT count(*) FROM prices").fetchone()[0] >= n_stocks, "a daily close stored per stock")
    check("avg target $" in digest, f"digest holding headers show price and target ({holdings} sections)")

    top = load_person(cfg)[1].tickers_by_weight()[0]
    closes = db.execute("SELECT count(*) FROM prices WHERE ticker = ?", (top,)).fetchone()[0]
    check(closes >= 15, f"a month of closes backfilled ({closes} for {top})")
    eps = db.execute("SELECT count(*) FROM quotes WHERE data LIKE '%\"eps\": {%'").fetchone()[0]
    check(eps >= 15, f"EPS forecasts collected ({eps} stocks)")

    # Fund holdings (SEC N-PORT) and real exposure
    from fin_news.funds import FundData
    person, holdings_now = load_person(cfg)
    from fin_news.holdings import _export_date
    newest = max((person["paths"]["home"] / "fidelity").glob("Portfolio_Positions_*.csv"), key=_export_date).name
    check(holdings_now.source == newest, f"newest portfolio export used ({holdings_now.source})")
    n_funds = db.execute("SELECT count(*) FROM fund_filings").fetchone()[0]
    check(n_funds >= 0.7 * len(holdings_now.funds), f"fund holdings fetched ({n_funds} filings)")
    mapped = db.execute("SELECT sum(pct) FROM fund_holdings WHERE asset = 'EC' AND ticker IS NOT NULL AND country = 'US'"
                        ).fetchone()[0] / db.execute("SELECT sum(pct) FROM fund_holdings WHERE asset = 'EC' AND "
                                                     "country = 'US'").fetchone()[0]
    check(mapped > 0.95, f"US fund holdings matched to tickers ({mapped:.1%})")
    ex = FundData(cfg, store_of(cfg)).exposures(holdings_now)["by_ticker"][top]
    check(ex["total"] >= ex["direct"] > 0, f"{top} real exposure counted (direct and through funds)")

    # Motley Fool: free articles from the news sitemap (with publisher tickers)
    fool_rel = db.execute("SELECT count(*) FROM articles WHERE source = 'Motley Fool' AND url LIKE '%fool.com/%' "
                          "AND relevant = 1 AND hints != '[]'").fetchone()[0]
    check(fool_rel >= 20, f"fool.com sitemap articles tagged to holdings ({fool_rel})")
    if advisor:
        advisor.e2e_pipeline_checks(cfg, db, store_of(cfg), holdings_now, digest, check)

    # Digest
    for section in ("## Heads-up", "## Your holdings", "## Macro"):  # source health is on the Status page
        check(section in digest, f"digest has {section[3:]}")
    check("| CPI inflation |" in digest or "| S&P 500 |" in digest, "macro table has FRED rows")
    check(len(digest) < 120_000, f"digest size reasonable ({len(digest) // 1000} KB)")


def check_run2(db, before: dict) -> None:
    st = last_run(db)
    print(f"\nRun 2 (incremental): { {k: v for k, v in st.items() if k != 'health'} }")
    check(st.get("new_articles", 0) < 0.3 * before["new_articles"],
          f"second run only adds new items ({st.get('new_articles')} vs {before['new_articles']})")
    dup = db.execute("SELECT count(*) FROM (SELECT url FROM articles GROUP BY url HAVING count(*) > 1)").fetchone()[0]
    check(dup == 0, "no duplicate URLs stored")
    orphans = db.execute("SELECT count(*) FROM articles WHERE story_id IS NULL").fetchone()[0]
    check(orphans == 0, f"every article belongs to a story ({orphans} unassigned)")


def check_retention(cfg, base: Path) -> None:
    print("\nRetention")
    folder = load_person(cfg)[0]["paths"]["digests"]
    old = folder / "2000-01-01_0000.md"
    old.write_text("old")
    os.utime(old, (time.time() - 10 * 86400,) * 2)
    c = copy.deepcopy(cfg)
    c["retention"]["article_days"] = 2  # most of the 10-day window is now "old"
    store = open_store(c)
    n0 = store.db.execute("SELECT count(*) FROM articles").fetchone()[0]
    counts = prune(c, store)
    n1 = store.db.execute("SELECT count(*) FROM articles").fetchone()[0]
    vecs = store.db.execute("SELECT count(*) FROM vec_articles").fetchone()[0]
    check(counts["articles_deleted"] > 0 and n1 == n0 - counts["articles_deleted"],
          f"old articles deleted ({counts['articles_deleted']} of {n0})")
    check(vecs == n1, f"their search vectors deleted too ({vecs} vectors / {n1} articles)")
    check(not old.exists() and (folder / "latest.md").exists(), "old digest deleted, latest.md kept")


def check_web(cfg) -> None:
    print("\nWeb app")
    from fastapi.testclient import TestClient

    from fin_news import web

    web.cfg = cfg  # point the app at the test folder
    person, holdings = load_person(cfg)
    top, fund = holdings.tickers_by_weight()[0], max(holdings.funds, key=lambda f: holdings.funds[f]["weight"])
    client = TestClient(web.app)
    check(client.get("/", follow_redirects=False).status_code == 303, "pages need sign-in")
    check("Wrong name or password" in client.post("/login", data={"name": person["user"]["id"], "password": "nope"}).text,
          "wrong password rejected")
    who, expiry = person["user"]["id"], str(int(time.time()) + 3600)  # signed in without knowing their password
    client.cookies.set(web.COOKIE, f"{who}.{expiry}.{web._sign(who, expiry)}")
    for path in ("/", "/holdings", f"/holding/{top}", "/digests", "/status", "/ask", "/search", "/files", "/account"):
        r = client.get(path)
        check(r.status_code == 200, f"GET {path} -> {r.status_code}")
    r = client.get(f"/holding/{top}")
    check("Average target" in r.text and "P/E (trailing)" in r.text and "Your position" in r.text,
          "stock page shows targets, valuation and position")
    if advisor:
        advisor.e2e_page_checks(cfg, client, r.text, check)
    r = client.get("/returns?p=3M")
    check(r.status_code == 200 and ("Earned" in r.text or "Drop Fidelity activity" in r.text),
          "Performance page renders")
    r = client.get("/positions?sort=gain")
    check(r.status_code == 200 and ("Gain since you bought" in r.text or "Needs a Fidelity" in r.text),
          "Positions page renders")
    r = client.get("/money?w=3-6")
    check(r.status_code == 200 and ("You put in" in r.text or "Needs Fidelity" in r.text or "No investments" in r.text),
          "My money page renders")
    check(client.get("/search", params={"q": top.lower()}, follow_redirects=False).headers.get("location")
          == f"/holding/{top}", "searching a ticker opens its stock page")
    check(client.get("/holding/ZZZZQ").status_code == 404, "unknown ticker shows not-found")
    check("Biggest real exposures" in client.get("/holdings").text, "holdings page shows real exposures")
    r = client.get(f"/fund/{fund}")
    check(r.status_code == 200 and "Largest positions" in r.text, "fund page lists holdings")
    check(client.get("/search", params={"q": "Apple"}, follow_redirects=False).headers.get("location")
          == "/holding/AAPL", "searching a company name opens its stock page")
    r = client.get("/search", params={"q": "AI data center demand"})
    check(r.text.count('class="card"') >= 5, "search returns results")
    t = time.time()
    r = client.post("/ask", data={"q": f"Based on how much {top} I hold, should I buy more?"})
    answer = re.search(r'<div class="card md">(.*?)</div>', r.text, re.S)
    text = answer.group(1) if answer else ""
    print(f"  (Ask took {time.time() - t:.0f}s)")
    check(len(text) > 300, "Ask answered")
    pos = holdings.stocks[top]
    check(f"{pos['weight']:.1%}" in text or f"{pos['value']:,.0f}" in text or f"{pos['shares']:.3f}" in text,
          f"Ask used the actual {top} position")
    bad = [n for n in re.findall(r"\[(\d+)\]", text) if not 1 <= int(n) <= 8]
    check(not bad and "[0]" not in text, f"Ask citations point at real sources ({bad})")
    check(re.search(r"target|P/E", text) is not None, "Ask used the market data (targets or P/E)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--keep", action="store_true", help="keep the test folder for inspection")
    args = p.parse_args()
    base = Path(tempfile.mkdtemp(prefix="fin-news-e2e-"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(base / "test.log")])
    print(f"Test folder: {base}")
    cfg = test_cfg(base)
    try:
        scrambled_users(base)
        t = time.time()
        path = run(cfg)
        store = open_store(cfg)
        db = store.db
        db.row_factory = sqlite3.Row
        first = last_run(db)
        mine = load_person(cfg)[0]["paths"]["digests"] / "latest.md"
        check(len(path.splitlines()) == len(user_ids()), "one digest per person")
        check_run1(cfg, db, mine.read_text(), time.time() - t)
        run(cfg)
        check_run2(db, first)
        check_web(cfg)
        check_retention(cfg, base)
    except BaseException:  # a crash is a failure, not "0 failed"
        import traceback
        traceback.print_exc()
        results.append((False, "the test crashed (traceback above)"))
    finally:
        failed = [w for ok, w in results if not ok]
        print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
        for w in failed:
            print(f"  FAIL  {w}")
        if args.keep:
            print(f"Kept: {base}")
        else:
            shutil.rmtree(base, ignore_errors=True)
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
