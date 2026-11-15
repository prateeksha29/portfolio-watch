"""Write each person's Markdown digest for one run."""
import datetime as dt
import json
import re
import shutil
from collections import defaultdict

from .addons import advisor
from .collect import utcnow
from .filings import form_of, insider, insider_line
from .market import Market, action_line, notable, pct, revision_flag, upside

ICON = {"positive": "🟢", "negative": "🔴", "mixed": "🟡", "neutral": "⚪"}
NEWS_KINDS = {"news", "press_release", "transcript", "macro"}


def _ago(ts: str) -> str:
    try:
        t = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return ""
    hours = (utcnow() - t).total_seconds() / 3600
    if hours < 1:
        return "just now"
    return f"{int(hours)}h ago" if hours < 48 else f"{int(hours // 24)}d ago"


def _md(s: str) -> str:
    return (s or "").replace("[", "(").replace("]", ")").replace("\n", " ").strip()


GARBLED = re.compile(r"[a-z%)][A-Z$][a-z]|\d[A-Z][a-z]")  # "StockPrice", "$54152-Week": table text run together


def _short(summary: str | None, title: str, n: int = 240) -> str:
    s = _md(summary or "")
    if not s or s.lower().startswith(title.lower()[:40]) or len(GARBLED.findall(s)) >= 3:
        return ""
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def story_md(s: dict, section_ticker: str | None = None) -> str:
    icon = ICON.get(s.get("sentiment") or "", "")
    head = f"- {icon + ' ' if icon else ''}**[{_md(s['title'])}]({s['url']})** — {s['source']} · {_ago(s['published'])}"
    if s["n_sources"] > 1:
        head += f" · {s['n_sources']} sources"
    if s["created_run"] != s["updated_run"]:
        head += " · developing"
    others = [t for t in s["tickers"] if t != section_ticker]
    if others:
        head += f" · also {', '.join(others)}"
    lines = [head]
    body = _md(s.get("summary") or "") or _short(s.get("art_summary"), s["title"])
    if body:
        lines.append(f"  {body}")
    extra = []
    if s.get("why"):
        extra.append(f"*Why it matters:* {_md(s['why'])}")
    if s.get("key_numbers"):
        extra.append("*Numbers:* " + "; ".join(_md(n) for n in s["key_numbers"]))
    if extra:
        lines.append("  " + " ".join(extra))
    return "\n".join(lines)


# What makes a story worth reading, by headline wording. Weights are small next to relevance (~0.2-0.6).
SIGNALS = [
    (re.compile(r"\b(earnings|results|revenue|sales|guidance|outlook|forecast|profit|beats?|miss(es|ed)?|"
                r"Q[1-4]|quarter(ly)?)\b", re.I), 0.12),
    (re.compile(r"\b(lawsuit|sues|sued|probe|investigation|antitrust|regulators?|ban(s|ned)?|export|tariffs?|"
                r"fined?|recall)\b", re.I), 0.08),
    (re.compile(r"\b(acquir\w*|acquisition|merger|buyout|deal|contract|partnership|orders?)\b", re.I), 0.06),
    (re.compile(r"\b(upgrades?|downgrades?|price target|initiates?|coverage)\b", re.I), 0.06),
    (re.compile(r"\b(layoffs?|lay off|job cuts|CEO|CFO|resigns?|steps down|appoints?)\b", re.I), 0.06),
    (re.compile(r"\b(launch(es|ed)?|unveils?|announces?|introduces?)\b", re.I), 0.03),
]
LOW_VALUE = re.compile(
    r"\bvs\.?\s|Better Buy|Should You Buy|Is It Time to Buy|Buy Before|^Prediction:|Could Be by 20\d\d|"
    r"Here's Why .* (Fell|Rose|Dipped|Jumped|Gained|Outpaced|Lagged|Trailed)|What You Need To Know|"
    r"Moved (Up|Down) by|Trading (Up|Down)|Traders' Radar|pre-market|premarket|top (gainers|movers)|Unusual Volume|"
    r"Class Action|Law Firm|Shareholder Alert|Investors Who Lost|Final Trades|History Says", re.I)
ROUTINE = re.compile(r"Conference Call|to Report|Will Report|to Announce|Webcast|to Present at|to Participate|"
                     r"Fireside|Date for", re.I)


def importance(s, low_sources=()) -> float:
    """Ranking within a digest section: relevance (centered cosine, ~0.1-0.6), a small bonus per independent
    source, headline signals of material news, and a penalty for content farms and user posts."""
    title = s["title"]
    score = (s["relevance"] or 0) + 0.02 * min(s["n_sources"] or 1, 10)
    score += min(0.2, sum(w for pat, w in SIGNALS if pat.search(title)))
    if LOW_VALUE.search(title):
        score -= 0.15
    if any(src in s["source"] for src in low_sources):
        score -= 0.15
    return score


def _top(stories: list[dict], n: int, low=()) -> list[dict]:
    # Summarized stories first among equals: they are the ones a reader can act on.
    return sorted(stories, key=lambda s: -(importance(s, low) + (0.05 if s.get("summary") else 0)))[:n]


def _section(stories: list[dict], cap: int, section_ticker: str | None = None, low=()) -> list[str]:
    lines = [story_md(s, section_ticker) for s in _top(stories, cap, low)]
    if len(stories) > cap:
        lines.append(f"- _+{len(stories) - cap} lower-ranked stories (use `search`)_")
    return lines


def _macro_table(cfg, store) -> list[str]:
    rows = []
    for s in cfg.get("fred", {}).get("series", []):
        obs = store.db.execute("SELECT date, value FROM macro WHERE series_id = ? ORDER BY date DESC LIMIT 15",
                               (s["id"],)).fetchall()
        if not obs:
            continue
        date, v0 = obs[0]["date"], obs[0]["value"]
        if s.get("transform") == "yoy":
            # Same month a year earlier, looked up by date (FRED occasionally skips a month).
            year_ago = {o["date"]: o["value"] for o in obs}.get(f"{int(date[:4]) - 1}{date[4:]}")
            if not year_ago:
                continue
            latest, change = f"{(v0 / year_ago - 1) * 100:.1f}% YoY", ""
        else:
            latest = f"{v0:,.2f}"
            change = f"{v0 - obs[1]['value']:+,.2f}" if len(obs) > 1 else ""
        rows.append(f"| {s['name']} | {latest} | {change} | {date} |")
    return ["| Series | Latest | Change | As of |", "|---|---|---|---|", *rows] if rows else []


def group(stories: list[dict]) -> dict:
    """Split a run's relevant stories into the digest's sections."""
    news = [s for s in stories if s["kind"] in NEWS_KINDS]
    g = {"filings": [s for s in stories if s["kind"] == "filing"],
         "analysis": [s for s in stories if s["kind"] == "analysis"],
         "social": [s for s in stories if s["kind"] == "social"],
         "hot": [s for s in news if s["n_sources"] >= 3 and s["tickers"] and not ROUTINE.search(s["title"])],
         "macro": [s for s in news if s["kind"] == "macro" and not s["tickers"]],
         "by_ticker": defaultdict(list), "by_topic": defaultdict(list), "stakes": defaultdict(int)}
    for s in news:
        if s["tickers"]:
            g["by_ticker"][s["tickers"][0]].append(s)  # the story's best-matching holding
        elif s["topic"] and s["kind"] != "macro":
            g["by_topic"][s["topic"]].append(s)
    for s in stories:
        if s["kind"] == "stake" and s["tickers"]:
            g["stakes"][s["tickers"][0]] += 1
    return g


def person_topics(cfg, holdings) -> set[str]:
    """The themes one person follows: config.toml's topics, their own, and their funds' themes."""
    return set(cfg.get("topics", [])) | set(holdings.themes())


def for_person(stories: list[dict], holdings, topics: set[str]) -> list[dict]:
    """The stories one person sees. A run's stories are about everyone's stocks and themes; this keeps those about a
    stock the person holds (with only their stocks in its ticker list, largest-matching first) and those with none of
    their stocks but on one of their themes or general macro news (then with no tickers, so they file under the theme).
    Other people's stocks never appear in someone's digest."""
    held, out = set(holdings.stocks), []
    for s in stories:
        mine = [t for t in s["tickers"] if t in held]
        if mine:
            out.append({**s, "tickers": mine})
        elif s["kind"] not in ("filing", "stake", "social") and (s["topic"] in topics or s["kind"] == "macro"):
            out.append({**s, "tickers": []})
    return out


def display_order(cfg, stories: list[dict], holdings) -> list[dict]:
    """The stories the digest will show, most prominent first: heads-up, then each holding's top stories by
    position size (one per holding per round), themes, macro, analysis, then the rest. Full text and LLM
    summaries follow this order so the stories a reader sees are the ones that get summarized."""
    d, low = cfg["digest"], cfg.get("low_quality_sources", [])
    g = group(stories)
    rank = {t: sorted(v, key=lambda s: -importance(s, low)) for t, v in g["by_ticker"].items()}
    out = sorted(g["hot"], key=lambda s: -importance(s, low))[:5]
    for r in range(d["max_per_section"]):
        out += [rank[t][r] for t in holdings.tickers_by_weight() if len(rank.get(t, [])) > r]
    for items in g["by_topic"].values():
        out += sorted(items, key=lambda s: -importance(s, low))[: d["max_per_topic"]]
    out += sorted(g["macro"], key=lambda s: -importance(s, low))[: d["max_per_section"]]
    out += sorted(g["analysis"], key=lambda s: -importance(s, low))[: d.get("max_analysis", 8)]
    out += sorted(stories, key=lambda s: -importance(s, low))
    seen, ordered = set(), []
    for s in out:
        if s["id"] not in seen:
            seen.add(s["id"])
            ordered.append(s)
    return ordered


def source_flags(store, stats: dict) -> list[str]:
    """Sources that broke since the previous run: most requests failing now but not then, or nothing returned now
    after at least 5 items then. Chronic trouble (and every source's numbers) is on the Status page instead."""
    row = store.db.execute("SELECT stats FROM runs ORDER BY started DESC LIMIT 1").fetchone()  # this run isn't saved yet
    before = json.loads(row["stats"] or "{}").get("health", {}) if row else {}
    failing = lambda h: bool(h and h["requests"]) and h["errors"] * 2 >= h["requests"]
    flags = []
    for name, h in sorted(stats.get("health", {}).items()):
        if advisor and name == advisor.GROUP and stats.get("advisor_warning"):
            continue  # already said above
        old = before.get(name)
        if failing(h) and not failing(old):
            flags.append(f"{name}: {h['errors']} of {h['requests']} requests failing ({h['last_error'][:120]})")
        elif h["requests"] and not h["items"] and not h["errors"] and not h["not_modified"] and old and old["items"] >= 5:
            flags.append(f"{name} returned nothing (had {old['items']} last run)")
    return flags


def write_digest(cfg, store, holdings, display: dict[str, str], run_id: str, stats: dict):
    """One person's digest (cfg from users.user_config). The run's problems (sources failing, an add-on's login, the
    embedding model) are only listed for admins."""
    cap = cfg["digest"]["max_per_section"]
    low = cfg.get("low_quality_sources", [])
    admin = cfg.get("user", {}).get("admin", True)
    stats = {**stats, **stats.get("users", {}).get(cfg.get("user", {}).get("id"), {})}
    if not admin:
        stats = {k: v for k, v in stats.items() if k not in ("advisor_warning", "embedding_warning")}
    owned = set(holdings.stocks) | set(holdings.funds)
    stats["new_ideas"] = [i for i in stats.get("new_ideas", []) if i["ticker"] not in owned]
    stories = for_person(store.stories_with_primary("s.updated_run = ? AND s.relevant = 1", (run_id,)), holdings,
                         person_topics(cfg, holdings))
    weight = {t: holdings.stocks[t]["weight"] for t in holdings.stocks}
    g = group(stories)
    filings, analysis, social, stakes = g["filings"], g["analysis"], g["social"], g["stakes"]

    now_local = dt.datetime.now().astimezone()
    out = [f"# Portfolio news — {now_local:%a %b %d, %Y %H:%M %Z}", "",
           f"_{stats.get('new_articles', 0)} new articles → {len(stories)} relevant stories · "
           f"{stats.get('summarized', 0)} summarized · {stats.get('seconds', 0)}s_", ""]

    # Heads-up
    heads = []
    if stats.get("holdings_warning"):
        heads.append(f"- ⚠️ {_md(stats['holdings_warning'])}")
    if stats.get("portfolio_changes"):
        heads.append(f"- 🗂️ {_md(stats['portfolio_changes'])}")
    if stats.get("embedding_warning"):
        heads.append(f"- ⚠️ {_md(stats['embedding_warning'])}")
    if stats.get("advisor_warning"):
        heads.append(f"- ⚠️ {_md(stats['advisor_warning'])}")
    if admin:
        heads += [f"- ⚠️ {_md(f)}" for f in source_flags(store, stats)]
    for change in stats.get("advisor_changes", []):
        if any(re.search(rf"\b{re.escape(t)}\b", change) for t in holdings.stocks):  # about one of their stocks
            heads.append(f"- 🃏 {_md(change)}")
    today = dt.date.today()
    horizon = (today + dt.timedelta(days=cfg["digest"]["upcoming_earnings_days"])).isoformat()
    for e in store.db.execute("SELECT * FROM events WHERE kind = 'earnings' AND date >= ? AND date <= ? ORDER BY date",
                              (today.isoformat(), horizon)):
        if e["ticker"] not in holdings.stocks:
            continue
        d = json.loads(e["detail"] or "{}")
        when = {"bmo": "before open", "amc": "after close"}.get(d.get("hour") or "", "")
        eps = f", EPS est. {d['eps_estimate']:.2f}" if isinstance(d.get("eps_estimate"), (int, float)) else ""
        heads.append(f"- 📅 **{e['ticker']}** reports earnings {e['date']}{' ' + when if when else ''}{eps}")
    # Widely covered news about a holding; not theme-only stories, not scheduling notices.
    for s in _top(g["hot"], 5, low):
        heads.append(f"- 🔥 [{_md(s['title'])}]({s['url']}) — {s['n_sources']} sources")
    if heads:
        out += ["## Heads-up", *heads, ""]

    # New ideas (Discovery, once a day): stocks the owner doesn't hold, newsletter picks first
    new_ideas = sorted(stats.get("new_ideas", []), key=lambda i: (not i["both"], i["source"] == "insider"))
    if new_ideas:
        out += ["## New ideas"]
        for i in new_ideas[:5]:
            star = "⭐ " if i["both"] else ""
            why = f" {_md(i['why'])}" if i.get("why") else ""
            out.append(f"- 💡 {star}**[{i['ticker']}](/holding/{i['ticker']})** {_md(i.get('name') or '')} — "
                       f"{_md(i['headline'])}.{why}")
        more = f"{len(new_ideas) - 5} more on the " if len(new_ideas) > 5 else "Context, watch or dismiss: "
        out += [f"- _{more}[Ideas tab](/ideas)_", ""]

    # Filings: company announcements (summarized 8-K/6-K press releases), insider trades, then the rest
    if filings:
        ticker_of = {id(s): s["tickers"][0] if s["tickers"] else "?" for s in filings}
        by_weight = sorted(filings, key=lambda s: (-weight.get(ticker_of[id(s)], 0), s["published"]))
        announced = [s for s in by_weight if s.get("headline")]
        if announced:
            out += ["## Company announcements"]
            for s in announced:
                icon = ICON.get(s.get("sentiment") or "", "")
                out.append(f"- {icon + ' ' if icon else ''}**{ticker_of[id(s)]} · {_md(s['headline'])}** — "
                           f"{form_of(s['title'])} · {_ago(s['published'])} · [filing]({s['url']})")
                if s.get("summary"):
                    out.append(f"  {_md(s['summary'])}")
                if s.get("key_numbers"):
                    out.append("  *Numbers:* " + "; ".join(_md(n) for n in s["key_numbers"]))
            out.append("")
        trades, quiet_form4 = [], defaultdict(int)
        for s in by_weight:
            if form_of(s["title"]) != "4":
                continue
            d = insider(s.get("art_body"))
            line = insider_line(ticker_of[id(s)], d) if d else None
            if line:
                value = sum((d.get(k) or {}).get("value", 0) for k in ("bought", "sold"))
                trades.append((not d.get("bought"), -value, f"- {'🟢 ' if d.get('bought') else ''}{line} · "
                                                              f"{_ago(s['published'])} · [Form 4]({s['url']})"))
            else:
                quiet_form4[ticker_of[id(s)]] += 1
        if trades or quiet_form4:
            out += ["## Insider trades", *[t[2] for t in sorted(trades)]]  # purchases first, then largest
            if quiet_form4:
                out.append("- _No open-market trade (awards, option exercises, tax withholding): "
                           + ", ".join(f"{t} ×{n}" for t, n in quiet_form4.items()) + "_")
            out.append("")
        rest = [s for s in by_weight if not s.get("headline") and form_of(s["title"]) != "4"]
        if rest:
            out += ["## Other SEC filings"]
            out += [f"- **{ticker_of[id(s)]}** [{_md(s['title'].split(' filed ', 1)[-1])}]({s['url']}) · "
                    f"{_ago(s['published'])}" for s in rest]
            out.append("")

    # Analyst moves on holdings: EPS forecast revisions this week, then rating changes, new coverage and moved
    # targets in the last 3 days
    market = Market(cfg, store)
    quotes = {t: market.latest(t) for t in holdings.stocks}
    since = (today - dt.timedelta(days=3)).isoformat()
    moves = [(a["date"], t, a) for t, q in quotes.items() if q for a in q.get("actions", [])
             if a["date"] >= since and notable(a)]
    flags = [(t, revision_flag(q)) for t in holdings.tickers_by_weight() for q in [quotes.get(t)] if q]
    flags = [(t, f) for t, f in flags if f]
    if moves or flags:
        out += ["## Analyst moves", *[f"- **{t}** — {f}" for t, f in flags]]
        for day, t, a in sorted(moves, key=lambda m: (m[0], weight.get(m[1], 0)), reverse=True)[:15]:
            icon = {"up": "⬆️ ", "down": "⬇️ "}.get(a["action"], "")
            out.append(f"- {icon}**{t}** — {_md(action_line(a))} · {dt.date.fromisoformat(day):%b %d}")
        out.append("")

    # Holdings, ordered by position size
    out += ["## Your holdings"]
    by_ticker, quiet = g["by_ticker"], []
    for t in holdings.tickers_by_weight():
        if not by_ticker.get(t):
            quiet.append(t)
            continue
        q = quotes.get(t)
        price = (f" · ${q['price']:,.2f}" + (f", avg target ${q['target_mean']:,.0f} ({pct(upside(q))})"
                                            if q.get("target_mean") else "")) if q and q.get("price") else ""
        price += advisor.digest_tag(store, t) if advisor else ""
        out += [f"### {t} — {display.get(t, t)} ({weight.get(t, 0):.1%} of portfolio){price}",
                *_section(by_ticker[t], cap, t, low)]
        if stakes.get(t):
            out.append(f"- _{stakes[t]} institutional position change(s) reported (13F)_")
        out.append("")
    if quiet:
        out += [f"_No new stories: {', '.join(quiet)}_", ""]

    # Themes and funds
    themes = holdings.themes()
    by_topic = g["by_topic"]
    if by_topic:
        out += ["## Themes & funds"]
        for topic, items in sorted(by_topic.items(), key=lambda kv: -len(kv[1])):
            funds = f" ({', '.join(themes[topic])})" if topic in themes else ""
            out += [f"### {topic}{funds}", *_section(items, cfg["digest"]["max_per_topic"], low=low), ""]

    # Macro
    table = _macro_table(cfg, store)
    macro_stories = g["macro"]
    if table or macro_stories:
        out += ["## Macro", *table, ""]
        out += _section(macro_stories, cap, low=low)
        out.append("")

    if analysis:
        out += ["## Analysis & opinion", *_section(analysis, cfg["digest"].get("max_analysis", 8), low=low), ""]
    if social:
        out += ["## Reddit", *_section(social, cap, low=low), ""]

    path = cfg["paths"]["digests"] / f"{now_local:%Y-%m-%d_%H%M}.md"
    path.write_text("\n".join(out))
    shutil.copyfile(path, cfg["paths"]["digests"] / "latest.md")
    return path
