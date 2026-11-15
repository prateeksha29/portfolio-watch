"""One pipeline run: collect -> store -> tag -> embed -> classify -> score -> group -> full text -> LLM -> market
data (daily) -> digests.

A run is shared by everyone (users.py): news, filings and prices are collected once for all their stocks and themes,
and each story is summarized once. Which stories get full text and summaries interleaves everyone's digest order, so
each person's largest holdings are covered; then each person gets their own digest."""
import datetime as dt
import json
import logging
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .addons import advisor
from .classify import KindClassifier
from .collect import Collector, iso, utcnow
from .digest import display_order, for_person, person_topics, write_digest
from .discover import Discovery
from .embed import Embedder, EmbeddingUnavailable
from .filings import form_of
from .funds import FundData
from .holdings import Holdings, changes, newest_holdings, snapshot
from .llm import LLM
from .market import Market, refresh_due, refresh_slot
from .retention import prune
from .returns import Returns
from .store import Store, norm_title
from .tagging import Tagger
from .users import everyone

log = logging.getLogger(__name__)

KIND_RANK = {"filing": 0, "press_release": 1, "news": 2, "macro": 2, "transcript": 3, "analysis": 4, "social": 5,
             "stake": 6}
# Sources whose ticker hints are the publisher's own tags of what an article is about (fool.com's news sitemap, an
# add-on's write-ups); they count like a ticker named in the headline.
PUBLISHER_TAGGED = ("Motley Fool", *([advisor.SOURCE] if advisor else []))
NAME_SUFFIX = re.compile(r"[,\s]+(Inc|Corp|Corporation|Co|Company|Ltd|Limited|Plc|N\.?V|S\.?A|SE|AG|Holdings?|Group)\.?$",
                         re.I)


def open_store(cfg: dict) -> Store:
    e = cfg["embedding"]
    return Store(cfg["paths"]["db"], e["dim"], e["model"])


def load_portfolio(cfg: dict) -> Holdings:
    """Positions from the newest Fidelity export matching holdings_csv (a file name or a pattern such as
    fidelity/Portfolio_Positions_*.csv), so a new export is picked up by the next run and the web app at once."""
    path = cfg["paths"].get("home", cfg["paths"]["root"]) / cfg.get("holdings_csv", "fidelity/Portfolio_Positions_*.csv")
    h = newest_holdings(path.parent, path.name, cfg.get("fund_themes", []), cfg.get("tickers", []),
                        set(cfg.get("returns", {}).get("accounts", [])))
    if h.warning and (str(path), h.warning) not in _warned:  # once per process: the web app loads it on every page
        _warned.add((str(path), h.warning))
        log.warning("%s: %s", cfg.get("user", {}).get("id", ""), h.warning)
    return h


_warned: set[tuple[str, str]] = set()


def shared_order(cfg: dict, stories: list[dict], people: list[tuple[dict, Holdings]]) -> list[dict]:
    """Everyone's digest order interleaved (each person's 1st story, then each one's 2nd, ...), then the rest by
    importance: the stories full text and summaries go to, so no one's top stories are crowded out."""
    by_id = {s["id"]: s for s in stories}
    lists = [[s["id"] for s in display_order(u, for_person(stories, h, person_topics(u, h)), h)] for u, h in people]
    seen, out = set(), []
    for k in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if k < len(x) and x[k] not in seen:
                seen.add(x[k])
                out.append(by_id[x[k]])
    return out + [s for s in display_order(cfg, stories, Holdings()) if s["id"] not in seen]


def combined(people: list[tuple[dict, Holdings]]) -> tuple[list[str], list[str], set[str]]:
    """(stocks, funds, stocks priced by snapshot) across everyone: each person's largest first, interleaved."""
    order = [h.tickers_by_weight() for _, h in people]
    stocks = list(dict.fromkeys(t for k in range(max((len(o) for o in order), default=0)) for o in order if k < len(o)
                                for t in [o[k]]))
    funds = list(dict.fromkeys(f for _, h in people for f in sorted(h.funds, key=lambda s: -h.funds[s]["weight"])))
    return stocks, funds, set(stocks)


def company_names(cfg: dict, tickers: list[str], sec_map: dict) -> dict[str, list[str]]:
    """ticker -> [display name, aliases...]"""
    out = {}
    for t in tickers:
        aliases = list(cfg.get("aliases", {}).get(t, []))
        if not aliases and t in sec_map:
            name = sec_map[t][1]
            while NAME_SUFFIX.search(name):
                name = NAME_SUFFIX.sub("", name)
            aliases = [name.title() if name.isupper() else name]
        out[t] = aliases
    return out


def ticker_query(ticker: str, name: str) -> str:
    # Compared on 4,500 real articles: this wording separated headline-named articles from the rest as well as
    # "{name} ({ticker}) stock news" (z AUC 0.971 vs 0.973) with better raw separation; adding a company
    # description made separation worse (0.946-0.961).
    return f"news about {name} ({ticker})"


def source_rank(source: str, priority: list[str]) -> int:
    for i, p in enumerate(priority):
        if source.startswith(p):
            return i
    return len(priority)


OPTION_CONTRACT = re.compile(r"\b[A-Z]{1,6}\d{6}[CP]\d{8}\b|\b\d{4} \d+(?:\.\d+)? (?:call|put)\b"
                             r"|Tokenized Stock|^Form 4 .* By Investing\.com")  # quote/converter pages, not news
# Quote, forum and comparison pages, and paid "market size" reports (EIN News, openPR...).
QUOTE_PAGE = re.compile(r"Stock Price (?:&|and) News|Stock Price, News, Quote|latest stock news and headlines"
                        r"|Quote Comparison|Stock Forum & Discussion|\bMarket (?:Size|Share) (?:Report|Analysis|Forecast)"
                        r"|\bmarket (?:is |was )?(?:projected|expected|poised|set|anticipated) to (?:grow|reach|surpass|hit)"
                        r"|\bCAGR\b", re.I)
# Institutional position changes from 13F filings (MarketBeat, Quiver...): one line per ticker, not stories.
STAKE_VERB = re.compile(r"\b(?:Boosts|Increases|Raises|Lifts|Grows|Lowers|Cuts|Decreases|Trims|Reduces|Sells|Buys|"
                        r"Acquires|Purchases|Takes|Initiates|Establishes|Builds|Adds)\b.{0,40}\b(?:Stake|Position|"
                        r"Holdings|Shares)\b", re.I)
STAKE_ORG = re.compile(r"\b(?:LLC|L\.L\.C|Inc\.?|Ltd|LP|Corp\.?|Co\.?|Trust|Advis[eo]rs?|Partners|Management|Capital|"
                       r"Wealth|Bank|Fund|Financial|Investments?|Associates|Securities)\b")
STAKE_PASSIVE = re.compile(r"\b(?:Holdings|Stake|Position|Shares|Stock)\b (?:Cut|Lifted|Raised|Trimmed|Lowered|Lessened|"
                           r"Increased|Decreased|Boosted|Reduced|Sold|Bought|Acquired|Purchased|Grown)\b.{0,20}\bby\b"
                           r"|\b(?:Bought|Sold|Acquired|Purchased) by\b|^Fund Update:|shares (?:added to|removed from)",
                           re.I)


def is_stake_change(title: str) -> bool:
    return bool(STAKE_PASSIVE.search(title) or (STAKE_VERB.search(title) and STAKE_ORG.search(title)))
AUTHOR_PAGE = re.compile(r"^[A-Z][\w'\-]+(?: [A-Z]\.)?(?: [A-Z][\w'\-]+){1,2}(?:,? [A-Z]{2,5})?(?: - Articles & Analysis)?$")


def is_junk(it: dict) -> bool:
    """Quote, forum and option-contract pages, market-size spam, non-English pages and Motley Fool author pages."""
    title = it["title"]
    if OPTION_CONTRACT.search(title) or QUOTE_PAGE.search(title):
        return True
    letters = [ch for ch in title if ch.isalpha()]
    if letters and sum(not ch.isascii() for ch in letters) > 0.3 * len(letters):
        return True  # Korean, Chinese... pages that Google News returns for US tickers

    return "Motley Fool" in it["source"] and len(title.split()) <= 6 and bool(AUTHOR_PAGE.match(title))


def _zscores(S: np.ndarray) -> np.ndarray:
    return (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True).clip(min=1e-6)


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=-1, keepdims=True).clip(min=1e-12)


def corpus_mean(store: Store, E: np.ndarray, key: str = "corpus_mean") -> np.ndarray:
    """Average article vector. EmbeddingGemma 2 gives every text a large shared component (an unrelated
    cake recipe scores ~0.46 against "Nvidia"); subtracting the corpus mean removes it, so unrelated articles
    score ~0 and relevant ones ~0.3. Blended across runs so one small batch doesn't swing it."""
    stored = store.get_meta(key)
    old = np.array(json.loads(stored), np.float32) if stored else None
    if len(E) >= 200:
        mu = E.mean(0) if old is None else 0.5 * old + 0.5 * E.mean(0)
    else:
        mu = old if old is not None else E.mean(0)
    store.set_meta(key, json.dumps(mu.round(6).tolist()))
    return mu


def score_relevance(arts, E, T, targets, tickers, f, tagger: Tagger, mu: np.ndarray,
                    topic_exclude: re.Pattern | None = None) -> None:
    """Relevance on mean-centered cosine (calibrated on 3,899 articles, 2026-10-15).
    A holding counts only when the article names it: in the headline always; in the summary only if that holding
    is also the article's best match. Embedding similarity alone could not tell which holding an unnamed
    article is about (e.g. 'Nvidia China chip share falls' scored 0.28 for Alibaba)."""
    nt = len(tickers)
    S = _normalize(E - mu) @ _normalize(T - mu).T
    St, Sp = S[:, :nt], S[:, nt:]
    Zp = _zscores(Sp) if Sp.shape[1] > 1 else np.zeros_like(Sp)
    tidx = {t: k for k, t in enumerate(tickers)}
    for row, (_, it) in enumerate(arts):
        if it["kind"] == "filing":
            it.update(topic=None, relevance=1.0, relevant=1)
            continue
        in_title = (set(tagger.tag(it["title"])) | set(it.get("tagged", []))) & set(tidx)
        in_text = set(it["tickers"]) & set(tidx)
        tick = set(in_title)
        best = tickers[int(St[row].argmax())] if nt else None
        if best in in_text - in_title and St[row, tidx[best]] >= f["ticker_text_min"]:
            tick.add(best)
        topic, topic_score = None, 0.0
        excluded = bool(topic_exclude and topic_exclude.search(it["title"]))
        if Sp.shape[1] and it["kind"] not in ("social", "stake") and not excluded:
            k = int(Sp[row].argmax())
            if Sp[row, k] >= f["topic_min"] and Zp[row, k] >= f["topic_min_z"]:
                topic, topic_score = targets[nt + k][0], float(Sp[row, k])
        # Best-matching ticker first: named in the headline, then highest score.
        it["tickers"] = sorted(tick, key=lambda t: (t not in in_title, -St[row, tidx[t]]))
        it["topic"] = topic
        scores = [float(St[row, tidx[t]]) for t in tick] + ([topic_score] if topic else [])
        it["relevance"] = max(scores) if scores else float(St[row].max())
        it["relevant"] = int(bool(tick) or topic is not None or it["kind"] == "macro")


def _epoch(ts: str) -> float:
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


COMPARISON = re.compile(r"\bvs\.?\s|\bversus\b", re.I)

# Analyst actions: "Baird raises Apple target" and "UBS raises Apple target" embed alike but are different news.
ANALYST_ACTION = re.compile(r"price target|\bPT\b|\btarget\b|upgrade|downgrade|initiat|reiterat|maintain|\brating\b|"
                            r"overweight|underweight|outperform|underperform|top pick", re.I)
_FIRM_ALIASES = {
    "JPMorgan": ["JPMorgan", "J.P. Morgan", "JP Morgan"], "Goldman Sachs": ["Goldman"], "Morgan Stanley": ["Morgan Stanley"],
    "Citi": ["Citi", "Citigroup"], "BofA": ["BofA", "Bank of America", "BoA"], "Wells Fargo": ["Wells Fargo"],
    "Barclays": ["Barclays"], "UBS": ["UBS"], "Deutsche Bank": ["Deutsche Bank"], "HSBC": ["HSBC"],
    "BNP Paribas": ["BNP Paribas", "Exane"], "Macquarie": ["Macquarie"], "Nomura": ["Nomura"], "Mizuho": ["Mizuho"],
    "Daiwa": ["Daiwa"], "Jefferies": ["Jefferies"], "Bernstein": ["Bernstein"], "Evercore": ["Evercore"],
    "Cantor": ["Cantor"], "Piper Sandler": ["Piper Sandler"], "Needham": ["Needham"], "Truist": ["Truist"],
    "TD Cowen": ["TD Cowen", "Cowen"], "Raymond James": ["Raymond James"], "Oppenheimer": ["Oppenheimer"],
    "Wedbush": ["Wedbush"], "Baird": ["Baird"], "Loop Capital": ["Loop Capital"], "Rosenblatt": ["Rosenblatt"],
    "Stifel": ["Stifel"], "KeyBanc": ["KeyBanc"], "BMO": ["BMO"], "RBC": ["RBC"], "Benchmark": ["Benchmark Co", "Benchmark analyst"],
    "Susquehanna": ["Susquehanna"], "Argus": ["Argus"], "DA Davidson": ["DA Davidson", "D.A. Davidson"],
    "Guggenheim": ["Guggenheim"], "Melius": ["Melius"], "Redburn": ["Redburn"], "Citizens JMP": ["Citizens JMP", "JMP"],
    "Wolfe": ["Wolfe Research", "Wolfe"], "Seaport": ["Seaport"], "Canaccord": ["Canaccord"], "Northland": ["Northland"],
    "Craig-Hallum": ["Craig-Hallum"], "Arete": ["Arete"], "Tigress": ["Tigress"], "CFRA": ["CFRA"], "Morningstar": ["Morningstar"],
    "Monness": ["Monness"], "Erste": ["Erste"], "Phillip Securities": ["Phillip Securities"], "Itau BBA": ["Itau BBA"],
    "China Renaissance": ["China Renaissance"], "CLSA": ["CLSA"], "Lake Street": ["Lake Street"], "Roth": ["Roth Capital", "Roth MKM"],
    "B. Riley": ["B. Riley", "B Riley"], "William Blair": ["William Blair"], "Societe Generale": ["SocGen", "Societe Generale"],
}
_FIRM_RX = [(name, re.compile(r"(?<![\w.])(" + "|".join(re.escape(a) for a in aliases) + r")(?![\w])"))
            for name, aliases in _FIRM_ALIASES.items()]


def analyst_firms(title: str) -> frozenset:
    """Brokerages named in an analyst-action headline (empty if it isn't one). Case-sensitive: firm names are proper nouns."""
    if not ANALYST_ACTION.search(title or ""):
        return frozenset()
    return frozenset(name for name, rx in _FIRM_RX if rx.search(title))


def same_story_ok(a: tuple, b: tuple) -> bool:
    """Guards on top of headline similarity. a, b = (normalized title, headline tickers).
    Templated headlines ("Broadcom vs. Marvell: Which Stock Is a Better Buy?", "Erste Group Bank's forecast for TSM
    earnings") embed alike although the companies differ: never merge headlines naming different holdings, and
    merge comparison headlines only when they are identical. a[2], b[2] = analyst firms (analyst_firms): two analyst
    actions by different brokerages are separate news."""
    if a[0] == b[0]:
        return True
    if a[1] and b[1] and not set(a[1]) & set(b[1]):
        return False
    if len(a) > 2 and len(b) > 2 and a[2] and b[2] and not a[2] & b[2]:
        return False
    return not (COMPARISON.search(a[0]) or COMPARISON.search(b[0]))


def assign_stories(store: Store, arts, H, f, run_id: str, mu_h: np.ndarray, tagger: Tagger,
                   embedder: Embedder | None = None) -> set[int]:
    """Group duplicates by headline: same story if the mean-centered headline similarity >= dedup_threshold and the
    two were published within the window. Headlines (not title+summary) so a Google News copy with no summary still
    matches the Yahoo copy of the same article. Qwen3-Embedding thresholds from the benchmark (2026-11-02).
    A holding's name can dominate a headline ("Acme Industries (ACME) CEO Maps Out Acquisition Push" vs
    "Acme Industries (ACME) is an Incredible Growth Stock" scored 0.74), so a match must also pass
    dedup_mask_min with the names masked out (Tagger.mask): what the headlines say must be related too."""
    window = f["dedup_window_hours"] * 3600
    since = iso(utcnow() - dt.timedelta(hours=2 * f["dedup_window_hours"]))
    old = store.recent_titles(since, exclude={aid for aid, _ in arts})
    Hc = _normalize(H - mu_h)
    pool = np.zeros((len(old) + len(arts), H.shape[1]), np.float32)
    times = np.zeros(len(old) + len(arts))
    pool_story, pool_key, pool_title = [], [], []

    def key(title):
        return norm_title(title), tuple(tagger.tag(title)), analyst_firms(title)

    for n, r in enumerate(old):
        pool[n] = _normalize(np.frombuffer(r["title_vec"], np.float32) - mu_h)
        times[n] = _epoch(r["published"])
        pool_story.append(r["story_id"])
        pool_key.append(key(r["title"]))
        pool_title.append(r["title"])
    n = len(old)

    # Masked headline vectors: every new headline, and the stored ones that could match one (one batch each).
    mask_min = f.get("dedup_mask_min")
    masked: dict[int, np.ndarray] = {}  # pool index -> centered, normalized vector of the masked headline
    if mask_min is not None and embedder is not None and arts:
        try:
            Mn = embedder.headlines([tagger.mask(it["title"]) for _, it in arts])
            mu_m = corpus_mean(store, Mn, "title_mask_mean")
            new_masked = _normalize(Mn - mu_m)
            need = set()
            for k, (_, it) in enumerate(arts):
                if n and it["kind"] != "filing":
                    sims = pool[:n] @ Hc[k]
                    sims[np.abs(times[:n] - _epoch(it["published"])) > window] = -1
                    need |= {int(j) for j in np.argsort(-sims)[:5] if sims[j] >= f["dedup_threshold"]}
            need = sorted(need)
            if need:
                Mo = _normalize(embedder.headlines([tagger.mask(pool_title[j]) for j in need]) - mu_m)
                masked.update(zip(need, Mo))
        except EmbeddingUnavailable as e:
            log.warning("Masked headline check skipped this run: %s", e)
            mask_min = None
    else:
        mask_min = None

    assign, touched = {}, set()
    for k in sorted(range(len(arts)), key=lambda k: arts[k][1]["published"]):
        aid, it = arts[k]
        story, t, kk = None, _epoch(it["published"]), key(it["title"])
        if it["kind"] != "filing" and n:
            sims = pool[:n] @ Hc[k]
            sims[np.abs(times[:n] - t) > window] = -1
            for j in np.argsort(-sims)[:5]:
                if sims[j] < f["dedup_threshold"]:
                    break
                if not same_story_ok(kk, pool_key[j]):
                    continue
                if (mask_min is not None and kk[0] != pool_key[j][0] and j in masked
                        and float(masked[j] @ new_masked[k]) < mask_min):
                    continue
                story = pool_story[j]
                break
        if story is None:
            story = store.new_story(run_id)
        assign[aid] = story
        touched.add(story)
        if it["kind"] != "filing":
            pool[n], times[n] = Hc[k], t
            pool_story.append(story)
            pool_key.append(kk)
            pool_title.append(it["title"])
            if mask_min is not None:
                masked[n] = new_masked[k]
            n += 1
    store.set_story_ids(assign)
    return touched


def refresh_stories(store: Store, story_ids, run_id: str, collector: Collector, priority: list[str]) -> None:
    off_topic = {r[0] for r in store.db.execute("SELECT id FROM stories WHERE off_topic = 1")}
    for sid in story_ids:
        rows = store.story_members(sid)
        if not rows:
            continue
        primary = min(rows, key=lambda r: (KIND_RANK.get(r["kind"], 9),
                                           0 if (r["body"] or collector.fetchable(r["url"])) else 1,
                                           source_rank(r["source"], priority), r["published"]))
        # Main ticker: the one most member articles match best (the main article breaks ties).
        # Other tickers are kept only if the main article or at least half the members name them.
        member_tickers = [json.loads(r["tickers"] or "[]") for r in rows]
        votes = Counter(ts[0] for ts in member_tickers if ts)
        primary_tickers = json.loads(primary["tickers"] or "[]")
        if primary_tickers:
            votes[primary_tickers[0]] += 0.5
        counts = Counter(t for ts in member_tickers for t in ts)
        main = votes.most_common(1)[0][0] if votes else None
        tickers = ([main] if main else []) + [t for t, c in counts.most_common() if t != main and
                                               (t in primary_tickers or c >= max(2, len(rows) / 2))]
        topics = Counter(r["topic"] for r in rows if r["topic"])
        store.update_story(sid, primary_id=primary["id"], n_articles=len(rows),
                           n_sources=len({r["source"].split(" · ")[-1] for r in rows}),
                           tickers=json.dumps(tickers), topic=topics.most_common(1)[0][0] if topics else None,
                           kind=primary["kind"], relevance=max(r["relevance"] or 0.0 for r in rows),
                           relevant=int(any(r["relevant"] for r in rows) and sid not in off_topic), updated_run=run_id)
        store.set_primary(sid, primary["id"])
    store.commit()


def enrich_stories(cfg: dict, store: Store, run_id: str, names: dict[str, list[str]],
                   people: list[tuple[dict, Holdings]]) -> dict:
    llm = LLM(cfg)
    if not llm.available:
        return {}
    stories = shared_order(cfg, store.stories_with_primary("s.updated_run = ? AND s.relevant = 1", (run_id,)), people)
    # A headline and a one-line feed snippet only yield a summary that restates the headline.
    min_chars = cfg["llm"].get("min_text_chars", 300)
    # New stories, and stories whose main article is now a different one (an earnings preview that became the
    # results story): a summary of the earlier article would sit under the new headline.
    stories = [s for s in stories if s["kind"] not in ("filing", "social", "stake")
               and (s["summary"] is None or s["summary_of"] != norm_title(s["title"]))]
    stale = [s for s in stories if s["summary"] is not None and len(s["art_body"] or s["art_summary"] or "") < min_chars]
    for s in stale:  # nothing to summarize the new main article from: drop the old summary rather than mislabel it
        store.update_story(s["id"], summary=None, sentiment=None, key_numbers=None, why=None, summary_of=None)
    store.commit()
    stories = [s for s in stories if len(s["art_body"] or s["art_summary"] or "") >= min_chars][: cfg["llm"]["max_stories"]]
    log.info("Summarizing %d stories with %s (%d with full article text; %d again because their main article changed, "
             "%d old summaries dropped)", len(stories), llm.model, sum(1 for s in stories if s["art_body"]),
             sum(1 for s in stories if s["summary"] is not None), len(stale))

    def one(s):
        about = ", ".join(f"{names.get(t, [t])[0]} ({t})" for t in s["tickers"]) or s["topic"] or "general market"
        return s, llm.enrich(s["title"], s["art_body"] or s["art_summary"], about)

    # Company filings first: few, and the most reliable facts of the run.
    filings = [s for s in store.stories_with_primary(
        "s.updated_run = ? AND s.kind = 'filing' AND s.headline IS NULL AND a.body IS NOT NULL AND a.body != ''",
        (run_id,)) if form_of(s["title"]) in ("8-K", "6-K")]
    filed = 0
    for s in filings:
        t = s["tickers"][0] if s["tickers"] else ""
        result = llm.enrich_filing(s["title"], s["art_body"], f"{names.get(t, [t])[0]} ({t})")
        if result:
            store.update_story(s["id"], headline=result["headline"].strip(), summary=result["summary"].strip(),
                               sentiment=result["sentiment"], key_numbers=json.dumps(result["key_numbers"]),
                               why=result["why_it_matters"].strip(), llm_model=llm.model)
            store.commit()
            filed += 1
    if filings:
        log.info("Summarized %d of %d company filings", filed, len(filings))

    done = off = 0
    # llama-swap serves several requests at once; a few in flight keeps the GPU busy between responses.
    with ThreadPoolExecutor(max_workers=cfg["llm"].get("parallel", 1)) as pool:
        for s, result in pool.map(one, stories):
            if not result:
                continue
            fields = dict(summary=result["summary"].strip(), sentiment=result["sentiment"],
                          key_numbers=json.dumps(result["key_numbers"]), why=result["why_it_matters"].strip(),
                          llm_model=llm.model, summary_of=norm_title(s["title"]))
            if not result["on_topic"]:
                # Holding only mentioned in passing (an ad, a comparison): keep the summary, drop it from the digest.
                fields.update(off_topic=1, relevant=0)
                off += 1
            store.update_story(s["id"], **fields)
            store.commit()
            done += 1
    return {"summarized": done, "off_topic": off, "filings_summarized": filed}


def run(cfg: dict, use_llm: bool = True, use_fulltext: bool = True):
    started = utcnow()
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    store = open_store(cfg)
    people = [(u, load_portfolio(u)) for u in everyone(cfg)]
    if not people:
        raise SystemExit("No users yet: python -m fin_news adduser <name> --admin")
    tickers, fund_list, held = combined(people)
    shared = {**cfg, "aliases": {k: v for u, _ in people for k, v in u["aliases"].items()}}
    collector = Collector(cfg)
    sec_map = collector.sec_ticker_map(cfg["paths"]["data"])
    names = company_names(shared, tickers, sec_map)
    display = {t: (names[t][0] if names[t] else t) for t in tickers}
    topics = list(dict.fromkeys(t for u, h in people for t in u["topics"] + list(h.themes())))
    log.info("Run %s: %d people, %d tickers, %d funds, %d topics", run_id, len(people), len(tickers), len(fund_list),
             len(topics))
    steps, last = {}, [time.monotonic()]

    def lap(name):  # seconds per step, kept in the run record for the How it works page
        now = time.monotonic()
        steps[name] = round(now - last[0], 1)
        last[0] = now

    # 1. Collect
    results = collector.collect(tickers, display, topics, sec_map, store.feed_cache())
    health = defaultdict(lambda: {"requests": 0, "errors": 0, "not_modified": 0, "items": 0, "last_error": ""})
    items, events, macro = [], [], []
    for r in results:
        job, h = r["job"], health[r["job"]["group"]]
        h["requests"] += 1
        if r.get("error"):
            h["errors"] += 1
            h["last_error"] = r["error"][:200]
        elif r.get("status") == 304:
            h["not_modified"] += 1
        h["items"] += len(r.get("items", []))
        items += r.get("items", [])
        events += r.get("events", [])
        macro += r.get("macro", [])
        if job["fn"] == "feed" and (r.get("etag") or r.get("modified")):
            store.save_feed_cache(job["url"], r.get("etag"), r.get("modified"))
    store.save_events(events)
    store.save_macro(macro)
    store.commit()
    lap("collect")

    # 1b. A private add-on (fin_news/advisor.py, not in the public code): a newsletter read with the owner's login
    advisor_stats = {}
    if advisor:
        advisor_stats, advisor_items = advisor.run_step(cfg, store, collector, tickers, health)
        items += advisor_items
        lap("advisor")

    # 2. Keep recent, unseen, unique URLs
    cutoff = iso(utcnow() - dt.timedelta(days=cfg["fetch"]["max_age_days"]))
    by_url: dict[str, dict] = {}
    junk = 0
    for it in items:
        if it["published"] < cutoff:
            continue
        if is_junk(it):
            junk += 1
            continue
        if it["url"] in by_url:
            prev = by_url[it["url"]]
            prev["hints"] = sorted(set(prev["hints"]) | set(it["hints"]))
        else:
            by_url[it["url"]] = it
    seen = store.seen_urls(by_url)
    new = [it for url, it in by_url.items() if url not in seen]
    ids = store.insert_articles(new, run_id, iso(utcnow()))
    store.set_bodies({aid: it["body"] for aid, it in zip(ids, new) if aid and it.get("body")})  # an add-on's write-ups
    arts = [(aid, it) for aid, it in zip(ids, new) if aid]
    log.info("Collected %d items, %d new (%d junk pages dropped)", len(items), len(arts), junk)
    # Pick up articles a failed earlier run stored but never processed.
    pending = store.db.execute("SELECT * FROM articles WHERE story_id IS NULL AND published >= ? AND run_id != ?",
                               (cutoff, run_id)).fetchall()
    if pending:
        log.info("Resuming %d unprocessed articles from earlier runs", len(pending))
        arts += [(r["id"], {"url": r["url"], "title": r["title"], "summary": r["summary"] or "", "source": r["source"],
                            "kind": r["kind"], "published": r["published"], "hints": json.loads(r["hints"] or "[]")})
                 for r in pending]

    stats = {"collected": len(items), "new_articles": len(arts), "junk_dropped": junk, "health": dict(health),
             **advisor_stats, "users": {}}
    for u, h in people:  # per person: their export and what changed in it (only their own digest says)
        previous = store.get_meta(f"portfolio:{u['user']['id']}")
        mine = {"holdings_file": h.source, "portfolio_changes": changes(json.loads(previous) if previous else None, h)}
        if h.warning:
            mine["holdings_warning"] = h.warning
        stats["users"][u["user"]["id"]] = mine
    embedder = Embedder(cfg)
    if arts:
        try:
            embedder.queries(["ready"])  # waits for llama-swap to load the model, up to [embedding] wait_seconds
        except EmbeddingUnavailable as e:
            log.warning("No embeddings this run; %d articles wait for the next one: %s", len(arts), e)
            stats["embedding_warning"] = f"Embedding model unavailable, so {len(arts)} new articles wait for the next run ({e})"
            arts = []
    lap("prepare")
    tagger = Tagger({t: names[t] for t in tickers}, set(cfg.get("ambiguous_tickers", [])))
    if arts:
        # 3. Tag tickers by name/alias/cashtag
        for _, it in arts:
            it["tickers"] = list(it["hints"]) if it["kind"] == "filing" else tagger.tag(f"{it['title']} {it['summary']}")
            if it["source"] in PUBLISHER_TAGGED:
                it["tagged"] = [t for t in it["hints"] if t in names]
                it["tickers"] = list(dict.fromkeys(it["tagged"] + it["tickers"]))

        # 4. Embed (title + summary for now; relevant primaries are re-embedded with full text below)
        E = embedder.documents([(it["title"], it["summary"]) for _, it in arts])
        lap("embed")

        # 5. Refine story type for mixed news feeds
        mixed = [k for k, (_, it) in enumerate(arts) if it["kind"] == "news"]
        if mixed:
            labels = KindClassifier(embedder).predict([arts[k][1]["title"] for k in mixed],
                                                     [arts[k][1].get("source", "") for k in mixed])
            for k, label in zip(mixed, labels):
                arts[k][1]["kind"] = label
        for _, it in arts:
            if it["kind"] != "filing" and is_stake_change(it["title"]):
                it["kind"] = "stake"
        lap("story_type")

        # 6. Relevance to holdings and topics
        targets = [(t, ticker_query(t, display[t])) for t in tickers] + [(tp, tp) for tp in topics]
        T = embedder.queries([q for _, q in targets])
        exclude = cfg["filter"].get("topic_exclude")
        score_relevance(arts, E, T, targets, tickers, cfg["filter"], tagger, corpus_mean(store, E),
                        re.compile(exclude) if exclude else None)
        store.update_article_scores(arts)
        store.set_vectors([aid for aid, _ in arts], E)
        lap("relevance")

        # 7. Group duplicates into stories, by headline
        H = embedder.headlines([it["title"] for _, it in arts])
        store.set_title_vectors([aid for aid, _ in arts], H)
        touched = assign_stories(store, arts, H, cfg["filter"], run_id, corpus_mean(store, H, "title_mean"), tagger,
                                 embedder)
        priority = cfg.get("source_priority", [])
        refresh_stories(store, touched, run_id, collector, priority)
        stats["stories_touched"] = len(touched)
        lap("group")

        # 7b. SEC filing contents: press releases attached to 8-K/6-K, insider trades in Form 4. Also retries
        #     filings an earlier run could not read (body still NULL).
        pending_filings = store.db.execute("SELECT id, url, title FROM articles WHERE kind = 'filing' AND body IS NULL "
                                           "AND published >= ?", (cutoff,)).fetchall()
        if pending_filings:
            store.set_bodies(collector.sec_details([tuple(r) for r in pending_filings]))
        lap("sec_filings")

        # 8. Full text for the stories the digest will show, in display order (the LLM step uses it too).
        if use_fulltext and cfg["fetch"]["full_text"] and touched:
            candidates = [s for s in shared_order(cfg, store.stories_with_primary(
                "s.updated_run = ? AND s.relevant = 1", (run_id,)), people)
                if s["art_body"] is None and s["kind"] not in ("filing", "social", "stake")]
            wanted = [(s["primary_id"], s["url"]) for s in candidates if collector.fetchable(s["url"])]
            stats["full_text_unfetchable"] = len(candidates) - len(wanted)
            bodies, resolved = collector.fetch_full_texts(wanted[: cfg["fetch"]["max_full_text_per_run"]])
            store.set_bodies(bodies)
            store.set_urls(resolved)
            stats["links_resolved"] = len(resolved)
            stats["full_text"] = len(bodies)
            if bodies:
                fetched = store.articles(bodies)
                ids_with_body = list(bodies)
                E2 = embedder.documents([(fetched[i]["title"], fetched[i]["body"]) for i in ids_with_body])
                store.set_vectors(ids_with_body, E2)
                refresh_stories(store, touched, run_id, collector, priority)
    embedder.close()
    lap("full_text")

    # 9. LLM summaries for top stories
    if use_llm and cfg["llm"].get("enabled", True) and arts:
        stats.update(enrich_stories(shared, store, run_id, names, people))
    lap("summaries")

    # 10. Prices, valuation and analyst targets: twice a day, before the open and after the close ([market]
    #     refresh_hours; ~1.5 min, one page per holding), and for new holdings
    #     in the first run that sees them
    market = Market(cfg, store, collector)
    due = refresh_due(cfg, store)
    wanted = tickers if due else [t for t in tickers if market.latest(t) is None]
    if wanted:
        market_health = market.refresh(wanted)
        stats["health"].update(market_health)
        h = market_health["Yahoo Finance (prices, targets)"]
        stats["market_refreshed"] = h["items"]
        if due and h["items"] >= h["requests"] / 2:  # mostly failed: try again next run
            store.set_meta("market_refreshed", refresh_slot(cfg))
    lap("prices")

    # 10a. Daily closes for the Returns page: a year once per symbol, then the last week, for every security in
    #      anyone's activity exports that the snapshot above doesn't price (funds, sold stocks) and the S&P 500
    if due:
        try:
            full: dict[str, dt.date] = {}  # symbol -> fetch from
            recent: set[str] = set()
            splits_seen: dict[str, tuple[dt.date, dt.date]] = {}  # symbol -> (last split, fetch from)
            for u, _ in people:
                ret = Returns(u, store)
                if not (ret.files and ret.covered_from):
                    continue
                start = ret.covered_from - dt.timedelta(days=10)
                for s in ret.symbols():
                    first = store.db.execute("SELECT MIN(date) FROM prices WHERE ticker = ?", (s,)).fetchone()[0]
                    if not first or first > (ret.covered_from + dt.timedelta(days=5)).isoformat():
                        full[s] = min(full.get(s, start), start)
                    elif s not in held and (ret.shares_on(dt.date.today()).get(s) or s.startswith("^")):
                        recent.add(s)
                for s, splits in ret.splits.items():
                    if s not in splits_seen or start < splits_seen[s][1]:
                        splits_seen[s] = (splits[-1][0], start)
            calls = [market.history(sorted(s for s in full if full[s] == start), start) for start in sorted(set(full.values()))]
            recent -= set(full)
            if recent:
                calls.append(market.history(sorted(recent), dt.date.today() - dt.timedelta(days=7)))
            # After a split Yahoo restates the closes before it, but the ones stored then are pre-split prices
            # (Returns.price scales adjusted closes): fetch the whole history again, once per split.
            for s, (day, start) in splits_seen.items():
                key = f"split_refetched:{s}:{day}"
                if s not in full and not store.get_meta(key):
                    if market.history([s], start)["items"]:
                        store.set_meta(key, dt.date.today().isoformat())
            merged = {k: sum(c.get(k, 0) for c in calls) for k in ("requests", "errors", "items")}
            if merged["requests"]:
                stats["health"]["Yahoo Finance (price history)"] = {**merged, "not_modified": 0, "last_error": next(
                    (c["last_error"] for c in reversed(calls) if c.get("last_error")), "")}
        except Exception as e:  # the digest must not depend on the Returns page
            log.warning("Price history for Returns: %s", e)
    lap("price_history")

    # 10b. Fund holdings (SEC N-PORT): funds not checked in refresh_days, and new funds
    fund_health = FundData(cfg, store, collector).refresh(fund_list)
    if fund_health["requests"]:
        stats["health"]["SEC N-PORT (fund holdings)"] = fund_health
    lap("fund_holdings")

    # 10c. Discovery: new stock ideas once a day (an add-on's picks of stocks not held, insider buying), each with a
    #      Yahoo snapshot
    discovery = Discovery(cfg, store, collector, market)
    if discovery.due():
        try:
            # Not an idea for anyone: stocks everyone (with a portfolio) already holds. Each person's Ideas tab and
            # digest leave out the ones they hold.
            owned = [set(h.stocks) | set(h.funds) for _, h in people if h.source]
            res = discovery.run(set.intersection(*owned) if owned else set(), use_llm)
            stats["health"].update(res["health"])
            if res["new"]:
                stats["new_ideas"] = res["new"]
        except Exception as e:  # the digest must not depend on Discovery
            log.warning("Discovery: %s", e)
    lap("ideas")

    # 11. Digest, run record, housekeeping
    finished = utcnow()
    stats["seconds"] = round((finished - started).total_seconds())
    stats["step_seconds"] = steps
    paths = []
    for u, h in people:
        paths.append(write_digest(u, store, h, display, run_id, stats))
    store.finish_run(run_id, iso(started), iso(finished), stats)
    for u, h in people:
        store.set_meta(f"portfolio:{u['user']['id']}", json.dumps(snapshot(h)))
    prune(cfg, store)
    log.info("Digests written for %s (%ss)", ", ".join(u["user"]["id"] for u, _ in people), stats["seconds"])
    return "\n".join(str(p) for p in paths)
