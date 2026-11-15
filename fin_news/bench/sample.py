"""Draw the benchmark items from the database and freeze them under bench/data/.

Sampling is stratified around the decisions the models actually make (e.g. headline pairs near the duplicate
threshold, articles near the topic cut-off), so easy cases don't dominate. Reference outputs from the current
pipeline go to bench/data/<task>_current.jsonl, kept apart so the annotator labels without seeing them."""
import json
import random
import re

import numpy as np

from ..classify import RULES
from ..pipeline import _normalize, company_names, load_portfolio, open_store
from ..users import default_user
from ..filings import form_of
from .common import DATA, write

rng = random.Random(20261101)
EXTEND = False  # set by main(extend=True): keep the frozen items (and their labels) and append new ones after them


def _start(task: str) -> list[dict]:
    """Items already frozen for this task when extending, else none."""
    from .common import items
    return items(task) if EXTEND else []


def _snip(s: str | None, n: int = 300) -> str:
    return re.sub(r"\s+", " ", s or "").strip()[:n]


def _art(r) -> dict:
    return {"title": r["title"], "source": r["source"], "published": r["published"], "snippet": _snip(r["summary"])}


def dedup(store, n_per_band=(35, 40, 45, 30)) -> None:
    """Headline pairs from the same 36 h window, binned by the current model's centered similarity."""
    mu = np.array(json.loads(store.get_meta("title_mean")), np.float32)
    rows = store.db.execute("SELECT id, title, summary, source, published, story_id, title_vec FROM articles "
                            "WHERE kind NOT IN ('filing', 'social') AND title_vec IS NOT NULL").fetchall()
    rows = rng.sample(rows, min(8000, len(rows)))
    V = _normalize(np.stack([np.frombuffer(r["title_vec"], np.float32) for r in rows]) - mu)
    t = np.array([np.datetime64(r["published"][:19]).astype("int64") for r in rows], dtype=np.int64)
    norm = [re.sub(r"\W+", " ", r["title"].lower()).strip() for r in rows]
    bands = [(0.55, 0.70), (0.70, 0.78), (0.78, 0.86), (0.86, 1.01)]
    pools = {b: [] for b in bands}
    for i in rng.sample(range(len(rows)), 4000):
        s = V @ V[i]
        s[np.abs(t - t[i]) > 36 * 3600] = -1
        s[i] = -1
        for j in np.argsort(-s)[:3]:
            if norm[i] == norm[j]:
                continue  # identical headlines are merged by rule, not by the model
            for b in bands:
                if b[0] <= s[j] < b[1]:
                    pools[b].append((min(i, j), max(i, j), float(s[j])))
    old = _start("dedup")
    norm_t = lambda x: re.sub(r"\W+", " ", x.lower()).strip()
    out, seen = [], {frozenset((norm_t(o["a"]["title"]), norm_t(o["b"]["title"]))) for o in old}
    for b, n in zip(bands, n_per_band):
        cand = list({(i, j): sim for i, j, sim in pools[b]}.items())
        rng.shuffle(cand)
        for (i, j), sim in cand:
            if len([o for o in out if o["band"] == f"{b[0]:.2f}"]) == n:
                break
            key = frozenset((norm[i], norm[j]))
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": f"d{len(old) + len(out):03d}", "band": f"{b[0]:.2f}", "sim_current": round(sim, 3),
                        "a": _art(rows[i]), "b": _art(rows[j]),
                        "same_story_current": rows[i]["story_id"] == rows[j]["story_id"]})
    write(DATA / "dedup.jsonl", old + out)
    print("dedup", len(old), "+", len(out))


def topic(cfg, store, holdings, embedder, n_bands=(50, 60, 40)) -> None:
    """Articles binned by their best topic score under the current model: likely, borderline, unlikely."""
    topics = list(dict.fromkeys(cfg.get("topics", []) + list(holdings.themes())))
    rows = store.db.execute("SELECT id, title, summary, source, published, kind, topic FROM articles "
                            "WHERE kind IN ('news', 'analysis', 'press_release', 'macro')").fetchall()
    rows = rng.sample(rows, min(3000, len(rows)))
    E = embedder.documents([(r["title"], r["summary"]) for r in rows])
    mu = np.array(json.loads(store.get_meta("corpus_mean")), np.float32)
    T = embedder.queries(topics)
    S = _normalize(E - mu) @ _normalize(T - mu).T
    best = S.max(1)
    bands = [(0.30, 9), (0.18, 0.30), (-9, 0.18)]
    old = _start("topic")
    taken = {o["title"] for o in old}
    out = []
    for (lo, hi), n in zip(bands, n_bands):
        idx = [i for i in range(len(rows)) if lo <= best[i] < hi and rows[i]["title"] not in taken]
        for i in rng.sample(idx, min(n, len(idx))):
            r = rows[i]
            out.append({"id": f"t{len(old) + len(out):03d}", **_art(r), "kind": r["kind"],
                        "score_current": round(float(best[i]), 3), "best_current": topics[int(S[i].argmax())],
                        "topic_current": r["topic"]})
    write(DATA / "topic.jsonl", old + out)
    if not old:  # the topic list is part of the frozen labels; keep it when extending
        write(DATA / "topic_list.jsonl", [{"topic": t} for t in topics])
    print("topic", len(old), "+", len(out))


def kind(store, want=None) -> None:
    """Headlines the embedding classifier decides (not caught by the regex rules)."""
    rows = store.db.execute("SELECT id, title, source, kind FROM articles "
                            "WHERE kind IN ('news', 'analysis', 'press_release', 'transcript')").fetchall()
    old = _start("kind")
    taken = {o["title"] for o in old}
    rows = [r for r in rows if not any(rx.search(r["title"]) for _, rx in RULES) and r["title"] not in taken]
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r)
    want = want or {"news": 40, "analysis": 35, "press_release": 20, "transcript": 5}
    out = []
    for k, m in want.items():
        for r in rng.sample(by.get(k, []), min(m, len(by.get(k, [])))):
            out.append({"id": f"k{len(old) + len(out):03d}", "title": r["title"], "source": r["source"], "kind_current": r["kind"]})
    rng.shuffle(out)
    write(DATA / "kind.jsonl", old + out)
    print("kind", len(old), "+", len(out))


def search_corpus(store, n=2500) -> None:
    """A fixed corpus every embedding model indexes: random recent articles plus keyword matches for the queries
    in bench/data/search.jsonl (written by hand), so each query has answers in it."""
    queries = [json.loads(line) for line in (DATA / "search.jsonl").read_text().splitlines() if line.strip()]
    rows = store.db.execute("SELECT id, title, summary, source, published FROM articles "
                            "WHERE kind NOT IN ('filing', 'social', 'stake') AND length(title) > 25").fetchall()
    old = _start("search_corpus")
    from .common import LABELS, read
    judged = {r["qid"] for r in read(LABELS / "search.jsonl")}  # queries whose answers are already in the corpus
    chosen = {} if old else {r["id"]: r for r in rng.sample(rows, n)}
    have = {o["id"] for o in old}
    for q in queries:
        if old and q["id"] in judged:
            continue
        for kw in q["keywords"]:
            hits = [r for r in rows if re.search(kw, r["title"], re.I)]
            for r in rng.sample(hits, min(15, len(hits))):
                chosen[r["id"]] = r
    out = [{"id": f"c{r['id']}", **_art(r)} for r in chosen.values() if f"c{r['id']}" not in have]
    write(DATA / "search_corpus.jsonl", old + out)
    print("search corpus", len(old), "+", len(out))


def stories(cfg, store, holdings, n=60) -> None:
    names = company_names(cfg, holdings.tickers_by_weight(), {})
    rows = store.stories_with_primary(
        "s.relevant + s.off_topic >= 1 AND s.kind NOT IN ('filing', 'social', 'stake') "
        "AND length(coalesce(a.body, a.summary, '')) >= 300 AND s.summary IS NOT NULL")
    rng.shuffle(rows)
    old, taken = _start("stories"), {o["title"] for o in _start("stories")}
    rows = [r for r in rows if r["title"] not in taken]
    out, current, per = [], [], {}
    off = [r for r in rows if r["off_topic"]][:max(6, n // 10)]
    for r in off + [r for r in rows if not r["off_topic"]]:
        key = r["tickers"][0] if r["tickers"] else (r["topic"] or "none")
        if per.get(key, 0) >= 3 or len(out) >= n:
            continue
        per[key] = per.get(key, 0) + 1
        about = ", ".join(f"{(names.get(t) or [t])[0]} ({t})" for t in r["tickers"]) or r["topic"] or "general market"
        sid = f"s{len(old) + len(out):03d}"
        out.append({"id": sid, "title": r["title"], "source": r["source"], "published": r["published"],
                    "about": about, "text": (r["art_body"] or r["art_summary"])[:12000],
                    "full_text": bool(r["art_body"])})
        current.append({"id": sid, "model": r["llm_model"], "summary": r["summary"], "sentiment": r["sentiment"],
                        "why": r["why"], "off_topic": r["off_topic"]})
    from .common import read
    write(DATA / "stories.jsonl", old + out)
    write(DATA / "stories_current.jsonl", (read(DATA / "stories_current.jsonl") if old else []) + current)
    print("stories", len(old), "+", len(out), "keys", len(per))


def filings(cfg, store, holdings, n=15) -> None:
    names = company_names(cfg, holdings.tickers_by_weight(), {})
    rows = store.stories_with_primary("s.kind = 'filing' AND a.body IS NOT NULL AND a.body != ''")
    old = _start("filings")
    taken = {(o["title"], o["published"]) for o in old}
    rows = [r for r in rows if form_of(r["title"]) in ("8-K", "6-K") and (r["title"], r["published"]) not in taken]
    rng.shuffle(rows)
    out, current = [], []
    for r in rows[:n]:
        t = r["tickers"][0] if r["tickers"] else ""
        fid = f"f{len(old) + len(out):03d}"
        out.append({"id": fid, "title": r["title"], "published": r["published"],
                    "company": f"{(names.get(t) or [t])[0]} ({t})", "text": r["art_body"][:40000]})
        current.append({"id": fid, "headline": r["headline"], "summary": r["summary"], "sentiment": r["sentiment"]})
    from .common import read
    write(DATA / "filings.jsonl", old + out)
    write(DATA / "filings_current.jsonl", (read(DATA / "filings_current.jsonl") if old else []) + current)
    print("filings", len(old), "+", len(out))


def ask(cfg, store) -> None:
    """Freeze each hand-written question's full prompt (portfolio context + retrieved sources) as the app builds it
    today, so every model answers from identical input."""
    from ..embed import Embedder
    from ..funds import mentioned_funds
    from ..askfmt import source_header
    from ..query import mentioned_tickers, other_stocks, portfolio_context, search

    holdings, emb = load_portfolio(default_user(cfg)), Embedder(cfg)
    qs = [json.loads(line) for line in (DATA / "ask_questions.jsonl").read_text().splitlines() if line.strip()]
    old = _start("ask")
    qs = [q for q in qs if q["id"] not in {o["id"] for o in old}]
    out = []
    for q in qs:
        question = q["question"]
        tickers = mentioned_tickers(cfg, holdings, question)
        others = other_stocks(cfg, store, holdings, question)
        funds = mentioned_funds(holdings, question)
        hits = []
        for t in tickers[:2]:
            hits += search(cfg, store, question, k=4, ticker=t, embedder=emb)
        hits += search(cfg, store, question, k=8, embedder=emb)
        seen, unique = set(), []
        for h in hits:
            key = re.sub(r"\W+", " ", h["title"].lower()).strip()
            if h["id"] not in seen and key not in seen:
                seen |= {h["id"], key}
                unique.append(h)
        sources = [f"{source_header(h['title'], h['source'], h['published'])}\n{(h['body'] or h['summary'] or '')[:1500]}"
                   for h in unique[:8]]
        out.append({**q, "context": portfolio_context(cfg, store, holdings, tickers, others, funds),
                    "sources": sources})
    write(DATA / "ask.jsonl", old + out)
    print("ask", len(old), "+", len(out))


def main(cfg, what: list[str], extend: bool = False) -> None:
    """extend=True keeps every frozen item and appends a second, larger draw (sizes below) with its own seed."""
    global EXTEND, rng
    from ..embed import Embedder

    EXTEND = extend
    if extend:
        rng = random.Random(20261102)
    store, holdings = open_store(cfg), load_portfolio(default_user(cfg))
    what = what or ["dedup", "topic", "kind", "stories", "filings"]
    if "dedup" in what:
        dedup(store, (80, 95, 105, 70) if extend else (35, 40, 45, 30))
    if "topic" in what:
        topic(cfg, store, holdings, Embedder(cfg), (115, 140, 95) if extend else (50, 60, 40))
    if "kind" in what:
        kind(store, {"news": 80, "analysis": 70, "press_release": 40, "transcript": 10} if extend else None)
    if "stories" in what:
        stories(cfg, store, holdings, 90 if extend else 60)
    if "filings" in what:
        filings(cfg, store, holdings, 25 if extend else 15)
    if "search" in what:
        search_corpus(store)
    if "ask" in what:
        ask(cfg, store)
