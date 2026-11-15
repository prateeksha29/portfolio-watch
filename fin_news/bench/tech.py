"""Technical numbers for the Benchmarks page: every metric recomputed from bench/ with 95% bootstrap intervals.

python -m fin_news.bench tech   ->  bench/tech.json (read by the web app; nothing on the page is typed in by hand)

Embedding numbers for the production model come from the live llama-swap server with the app's own text formats
(cached in bench/runs/embed/llama_<dim>/). The other embedding candidates' vectors were deleted after the
benchmark, so their scores come from bench/runs/embed/results.json without intervals. LLM numbers come from the
stored runs (bench/runs/) and the blind grades (bench/grading/)."""
import datetime as dt
import json
import logging
import statistics

import numpy as np

from ..classify import PROTOTYPES, rule_kind
from ..llm import SENTIMENTS
from . import embed_bench as eb
from .common import BENCH, GRADING, LABELS, RUNS, items, labels, read, split
from .grade import invented_numbers, runs

log = logging.getLogger(__name__)
B = 5000  # bootstrap resamples
OUT = BENCH / "tech.json"


def _ci(stat, n, rng, *, paired=None):
    """Percentile bootstrap over items: stat(idx) -> float."""
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = stat(idx)
        if v == v:  # skip NaN draws (e.g. a resample with no positives)
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def est(stat, n, rng):
    v = stat(np.arange(n))
    lo, hi = _ci(stat, n, rng)
    return {"v": round(float(v), 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": int(n)}


def _auc(s, y):
    if y.all() or not y.any():
        return float("nan")
    r = s.argsort().argsort() + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def _prf(pred, y):
    tp = (pred & y).sum()
    p, r = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
    return float(p), float(r), float(2 * p * r / (p + r)) if p + r else 0.0


# ---------------------------------------------------------------- embeddings

def embedding(cfg, rng) -> dict:
    from ..pipeline import company_names, load_portfolio
    from ..users import default_user
    from ..collect import Collector
    from ..tagging import Tagger
    dim = cfg["embedding"]["dim"]
    name = f"llama:{dim}"
    V = eb.embed_all(cfg, name)
    out = {"model": cfg["embedding"]["served_model"], "dim": dim}

    # Duplicate headlines: plain mean-centered cosine, and with holdings' names masked (the app's second check)
    lab, d = labels("dedup"), items("dedup")
    keep = [i for i, r in enumerate(d) if not lab[r["id"]].get("exclude")]
    y = np.array([bool(lab[d[i]["id"]]["same"]) for i in keep])
    sp = np.array([split(d[i]["id"]) for i in keep])
    mu = V["corpus_heads"].mean(0)
    A, Bv = eb._norm(V["dedup_a"] - mu), eb._norm(V["dedup_b"] - mu)
    plain = (A * Bv).sum(1)[keep]
    mpath = RUNS / "embed" / name.replace(":", "_") / "masked.npz"
    if mpath.exists():
        M = dict(np.load(mpath))
    else:
        from ..embed import Embedder
        h = load_portfolio(default_user(cfg))
        tick = h.tickers_by_weight()
        names = company_names(cfg, tick, Collector(cfg).sec_ticker_map(cfg["paths"]["data"]))
        tg = Tagger({t: names[t] for t in tick}, set(cfg.get("ambiguous_tickers", [])))
        e = Embedder(cfg)
        M = {"corpus": e.headlines([tg.mask(r["title"]) for r in items("search_corpus")]),
             "a": e.headlines([tg.mask(r["a"]["title"]) for r in d]), "b": e.headlines([tg.mask(r["b"]["title"]) for r in d])}
        e.close()
        np.savez(mpath, **M)
    mm = M["corpus"].mean(0)
    masked = (eb._norm(M["a"] - mm) * eb._norm(M["b"] - mm)).sum(1)[keep]
    t_now, m_now = cfg["filter"]["dedup_threshold"], cfg["filter"].get("dedup_mask_min")
    sweep = {"thresholds": [round(float(t), 2) for t in np.arange(0.40, 0.92, 0.02)]}
    for part in ("dev", "test", "all"):
        k = np.ones(len(y), bool) if part == "all" else sp == part
        rows = {}
        for label, rule in (("plain", lambda t: plain[k] >= t),
                            ("masked", lambda t: (plain[k] >= t) & (masked[k] >= (m_now or 0)))):
            rows[label] = [dict(zip(("p", "r", "f1"), (round(x, 3) for x in _prf(rule(t), y[k])))) for t in sweep["thresholds"]]
        sweep[part] = {**rows, "n": int(k.sum()), "pos": int(y[k].sum())}
    ops = []
    for label, pred in (("old: ≥0.78", plain >= 0.78), (f"now: ≥{t_now}", plain >= t_now),
                        (f"now + masked ≥{m_now}", (plain >= t_now) & (masked >= (m_now or 0)))):
        row = {"rule": label}
        for part in ("dev", "test"):
            k = sp == part
            p, r, f = _prf(pred[k], y[k])
            row[part] = {"p": round(p, 3), "r": round(r, 3), "f1": round(f, 3),
                         "fp": int((pred[k] & ~y[k]).sum()), "fn": int((~pred[k] & y[k]).sum())}
        ops.append(row)
    dedup = {"sweep": sweep, "ops": ops, "threshold": t_now, "mask_min": m_now}
    for part in ("dev", "test", "all"):
        k = np.ones(len(y), bool) if part == "all" else sp == part
        s_, y_ = plain[k], y[k]
        dedup[f"auc_{part}"] = est(lambda i: _auc(s_[i], y_[i]), len(y_), rng)
    out["dedup"] = dedup

    # Story type: nearest class centroid of the prototype headlines (with the classification instruction)
    labk, kinds = labels("kind"), items("kind")
    cls = list(PROTOTYPES)
    protos = [k for k, ts in PROTOTYPES.items() for _ in ts]
    C = eb._norm(np.stack([V["protos"][[j for j, k in enumerate(protos) if k == c]].mean(0) for c in cls]))
    rows = [i for i, r in enumerate(kinds) if not labk[r["id"]].get("exclude")]
    pred = np.array([cls[int((V["kind"][i] @ C.T).argmax())] for i in rows])
    gold = np.array([labk[kinds[i]["id"]]["kind"] for i in rows])
    ksp = np.array([split(kinds[i]["id"]) for i in rows])
    cm = [[int(((gold == g) & (pred == p)).sum()) for p in cls] for g in cls]
    per = {}
    for c in cls:
        p, r, f = _prf(pred == c, gold == c)
        per[c] = {"p": round(p, 3), "r": round(r, 3), "f1": round(f, 3), "support": int((gold == c).sum())}

    def macro(i):
        g, p = gold[i], pred[i]
        f = [_prf(p == c, g == c)[2] for c in cls if (g == c).any()]
        return float(np.mean(f))
    kind = {"classes": cls, "confusion": cm, "per_class": per,
            "accuracy": est(lambda i: float((gold[i] == pred[i]).mean()), len(gold), rng),
            "macro_f1": est(macro, len(gold), rng),
            "accuracy_test": est(lambda i: float((gold[ksp == "test"][i] == pred[ksp == "test"][i]).mean()),
                                 int((ksp == "test").sum()), rng)}

    # The app's real path: a feed's own label (PR Newswire = press release), then the title/source rules, then the
    # embedder. The items were sampled from headlines the old title rules missed, so before 2026-11-04 the rules
    # changed nothing here. "Gemma" is the label the app stored at sampling time (EmbeddingGemma, same path).
    feed_kind = {f["name"]: f.get("kind", "news") for v in cfg["sources"].values() if isinstance(v, list)
                 for f in v if isinstance(f, dict) and "name" in f}
    fk = [feed_kind.get(kinds[i]["source"], "news") for i in rows]
    paths = {
        "gemma": np.array([kinds[i]["kind_current"] for i in rows]),
        "qwen": np.array([f if f != "news" else p for f, p in zip(fk, pred)]),
        "qwen_rules": np.array([f if f != "news" else (rule_kind(kinds[i]["title"], kinds[i]["source"]) or p)
                                for f, i, p in zip(fk, rows, pred)]),
    }
    app, rk = {}, np.random.default_rng(1)  # own generator: the other sections' intervals stay as published
    for name, pp in paths.items():
        pr, gp = pp == "press_release", gold == "press_release"
        own = np.array([f == "news" for f in fk])  # not labelled by its feed
        app[name] = {"pr_found": int((pr & gp).sum()), "pr_support": int(gp.sum()),
                     "pr_found_other": int((pr & gp & own).sum()), "pr_support_other": int((gp & own).sum()),
                     "pr_false": int((pr & ~gp).sum()), "pr_f1": round(_prf(pr, gp)[2], 3),
                     "accuracy": est(lambda i, pp=pp: float((gold[i] == pp[i]).mean()), len(gold), rk),
                     "accuracy_test": est(lambda i, pp=pp: float((gold[ksp == "test"][i] == pp[ksp == "test"][i]).mean()),
                                          int((ksp == "test").sum()), rk)}
    app["paired"] = est(lambda i: float((gold[i] == paths["qwen_rules"][i]).mean() - (gold[i] == paths["gemma"][i]).mean()),
                        len(gold), rk)
    app["feed_labelled"] = int(sum(f != "news" for f in fk))
    kind["app"] = app
    out["kind"] = kind

    # Search: per-query nDCG@10 and recall@10 over the judged pools (unjudged results count as not relevant)
    judged = {}
    for r in read(LABELS / "search.jsonl"):
        judged.setdefault(r["qid"], {})[r["cid"]] = r["rel"]
    nd, rc, qsp, unjudged = [], [], [], 0
    for qid, top in eb.search_rank(V).items():
        rel = judged.get(qid, {})
        good = {c for c, v in rel.items() if v > 0}
        if not good:
            continue
        unjudged += sum(c not in rel for c in top)
        dcg = sum(rel.get(c, 0) / np.log2(i + 2) for i, c in enumerate(top))
        ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]
        nd.append(dcg / sum(g / np.log2(i + 2) for i, g in enumerate(ideal)))
        rc.append(len(good & set(top)) / min(len(good), 10))
        qsp.append(split(qid))
    nd, rc = np.array(nd), np.array(rc)
    out["search"] = {"ndcg10": est(lambda i: nd[i].mean(), len(nd), rng), "recall10": est(lambda i: rc[i].mean(), len(rc), rng),
                     "unjudged": unjudged, "judged_top10": 10 * len(nd) - unjudged}

    # Theme detection AUC: best theme score, article has a labeled theme vs not
    labt, top_items = labels("topic"), items("topic")
    mu_d = V["corpus_docs"].mean(0)
    S = eb._norm(V["topic_docs"] - mu_d) @ eb._norm(V["topic_queries"] - mu_d).T
    trow = [i for i, r in enumerate(top_items) if not labt[r["id"]].get("exclude")]
    mx = S.max(1)[trow]
    has = np.array([bool(labt[top_items[i]["id"]]["topics"]) for i in trow])
    out["topic"] = {**eb.score_topic(V), "detect_auc": est(lambda i: _auc(mx[i], has[i]), len(has), rng)}

    # All candidates (point estimates; their vectors are gone)
    res = json.loads((RUNS / "embed" / "results.json").read_text())
    cands = []
    for n_, r in res.items():
        row = {"name": n_}
        for part in ("dev", "test", "all"):
            a = r[part]
            row[part] = {"dedup_auc": a["dedup"]["auc"], "topic_auc": a["topic"]["detect_auc"], "topic_f1": a["topic"]["best_f1"],
                         "kind_acc": a["kind"]["accuracy"], "kind_f1": a["kind"]["macro_f1"], "ndcg10": a["search"]["ndcg@10"],
                         "recall10": a["search"]["recall@10"], "n_dedup": a["dedup"]["n"], "n_search": a["search"]["n"]}
        cands.append(row)
    served = {"name": f"served {cfg['embedding']['served_model']} ({dim})"}
    for part in ("dev", "test", "all"):
        ids = None if part == "all" else {r["id"] for t in ("dedup", "topic", "kind", "search") for r in items(t)
                                          if split(r["id"]) == part}
        dd, tp, kd, se = eb.score_dedup(V, ids), eb.score_topic(V, ids), eb.score_kind(V, ids), eb.score_search(V, ids)
        served[part] = {"dedup_auc": dd["auc"], "topic_auc": tp["detect_auc"], "topic_f1": tp["best_f1"],
                        "kind_acc": kd["accuracy"], "kind_f1": kd["macro_f1"], "ndcg10": se["ndcg@10"],
                        "recall10": se["recall@10"], "n_dedup": dd["n"], "n_search": se["n"]}
    cands.append(served)
    out["candidates"] = cands
    return out


# ---------------------------------------------------------------- LLM tasks

def _grades(task):
    key = json.loads((GRADING / f"{task}_key.json").read_text())
    lab = labels(task)
    g = {}
    for row in read(GRADING / f"{task}_grades.jsonl"):
        n = key.get(row["id"], {}).get(row["letter"])
        total = len(lab[row["id"]].get("facts") or lab[row["id"]].get("must") or [])
        if n and total:
            if row["id"] in g.get(n, {}):  # graded again in a later round: kept apart (grader consistency)
                n += "#2"
            g.setdefault(n, {})[row["id"]] = {"cov": len(set(row.get("hit", []))) / total, "wrong": row.get("wrong", 0),
                                              "viol": len(row.get("violations", []))}
    return g


def _sec(rows, ids):
    s = sorted(r["seconds"] for k, r in rows.items() if k in ids and "seconds" in r and not r.get("error")
               and not r.get("contended"))
    if not s:
        return None
    return {"avg": round(statistics.mean(s), 1), "p50": round(s[len(s) // 2], 1), "p90": round(s[int(0.9 * (len(s) - 1))], 1)}


def _summary_rows(task, g, R, sets, rng):
    lab, src = labels(task), {it["id"]: it for it in items(task)}
    out = []
    for name in sorted(g):
        for set_name, ids in sets.items():
            ids_ = sorted(set(g[name]) & ids)
            if len(ids_) < 5:
                continue
            cov = np.array([g[name][i]["cov"] for i in ids_])
            wrong = np.array([g[name][i]["wrong"] for i in ids_])
            sent, inv, words = [], 0, []
            for i in ids_:
                o = R[name.split("#")[0]].get(i, {}).get("output") or {}
                s = o.get("sentiment") if o.get("sentiment") in SENTIMENTS else "neutral"
                sent.append(s == lab[i].get("sentiment") or s in lab[i].get("accept", []))
                words.append(len((o.get("summary") or "").split()))
                inv += invented_numbers(f"{o.get('headline', '')} {o.get('summary', '')} {o.get('why_it_matters', '')}",
                                        f"{src[i]['title']}\n{src[i]['text']}")
            sent = np.array(sent)
            out.append({"run": name, "set": set_name, "n": len(ids_), "coverage": est(lambda i: cov[i].mean(), len(cov), rng),
                        "wrong": int(wrong.sum()), "sentiment": est(lambda i: sent[i].mean(), len(sent), rng),
                        "invented": inv, "words": round(statistics.mean(words)) if words else None, "sec": _sec(R[name.split("#")[0]], set(ids_))})
    return out


def _paired(g, a, b, ids, rng, field="cov"):
    common = sorted(set(g.get(a, {})) & set(g.get(b, {})) & ids)
    if len(common) < 5:
        return None
    d = np.array([g[a][i][field] - g[b][i][field] for i in common], float)
    agg = (lambda i: d[i].mean()) if field == "cov" else (lambda i: d[i].sum())
    r = est(agg, len(d), rng)
    r.update(a=a, b=b, field=field, wins=int((d > 0).sum()), losses=int((d < 0).sum()))
    return r


def llm_tasks(rng) -> dict:
    out = {}
    # Stories and filings
    for task in ("stories", "filings"):
        g, R = _grades(task), runs(task)
        all_ids = {it["id"] for it in items(task)}
        sets = {"all": all_ids, "dev": {i for i in all_ids if split(i) == "dev"}, "test": {i for i in all_ids if split(i) == "test"}}
        out[task] = {"rows": _summary_rows(task, g, R, sets, rng), "sets": {k: len(v) for k, v in sets.items()}}
    gs = _grades("stories")
    base, qwen, s1, s2 = "unsloth__gemma-4-26B-A4B", "unsloth__Qwen3.8-27B", "unsloth__Qwen3.8-27B__s1", "unsloth__Qwen3.8-27B__s2"
    st_ids = {it["id"] for it in items("stories")}
    dev, test = {i for i in st_ids if split(i) == "dev"}, {i for i in st_ids if split(i) == "test"}
    out["stories"]["paired"] = [p for p in (
        dict(_paired(gs, qwen, base, st_ids, rng), label="Qwen3.8 vs Gemma-4-26B (model round, all 150)", key="model"),
        dict(_paired(gs, "atomic__Qwen3.6-35B-A3B-MTP", base, st_ids, rng), label="Qwen3.6-35B vs Gemma-4-26B (all 150)", key="q36"),
        dict(_paired(gs, s1, qwen + "#2", dev, rng), label="3 sentences vs 2 (prompt round, tuning)", key="prompt_dev"),
        dict(_paired(gs, s2, s1, dev, rng), label="3 sentences + figure rule vs 3 sentences (tuning)", key="s2_dev"),
        dict(_paired(gs, s1, qwen + "#2", test, rng), label="3 sentences vs 2 (held-out)", key="prompt_test"),
        dict(_paired(gs, s1, base, test, rng), label="Now vs before: Qwen3.8 s1 vs Gemma (held-out)", key="now_test")) if p]
    # Grader consistency: the same Qwen3.8 outputs were graded blind twice, in two rounds a day apart
    a, b = gs.get(qwen, {}), gs.get(qwen + "#2", {})
    common = sorted(set(a) & set(b))
    if common:
        x, y = np.array([a[i]["cov"] for i in common]), np.array([b[i]["cov"] for i in common])
        out["stories"]["regrade"] = {"n": len(common), "same": int((x == y).sum()), "mean_abs_diff": round(float(np.abs(x - y).mean()), 3),
                                     "mean_1": round(float(x.mean()), 3), "mean_2": round(float(y.mean()), 3),
                                     "corr": round(float(np.corrcoef(x, y)[0, 1]), 3),
                                     "wrong_1": int(sum(a[i]["wrong"] for i in common)), "wrong_2": int(sum(b[i]["wrong"] for i in common))}
    gf = _grades("filings")
    f_ids = {it["id"] for it in items("filings")}
    out["filings"]["paired"] = [dict(_paired(gf, qwen, base, f_ids, rng), label="Qwen3.8 vs Gemma-4-26B (all filings)", key="model")]

    # Ask: tuned on 29 of the first 40 questions; 15 questions written later are the held-out set
    ga, RA = _grades("ask"), runs("ask")
    hold = set(json.loads((LABELS / "ask_holdout.json").read_text()))
    tune = set(ga.get("unsloth__Qwen3.8-27B__c2p1", {}))
    sets = {"tune": tune, "holdout": hold}
    rows = []
    for name in sorted(ga):
        for set_name, ids in sets.items():
            ids_ = sorted(set(ga[name]) & ids)
            if len(ids_) < 5:
                continue
            cov = np.array([ga[name][i]["cov"] for i in ids_])
            wrong = np.array([ga[name][i]["wrong"] for i in ids_])
            words = [len((RA[name.split("#")[0]].get(i, {}).get("output") or "").split()) for i in ids_]
            rows.append({"run": name, "set": set_name, "n": len(ids_), "coverage": est(lambda i: cov[i].mean(), len(cov), rng),
                         "wrong": est(lambda i: float(wrong[i].sum()), len(wrong), rng),
                         "answers_with_error": int((wrong > 0).sum()),
                         "violations": int(sum(ga[name][i]["viol"] for i in ids_)), "words": round(statistics.mean(words)),
                         "sec": _sec(RA[name.split("#")[0]], set(ids_))})
    q8, q6 = "unsloth__Qwen3.8-27B", "atomic__Qwen3.6-35B-A3B-MTP"
    paired = [
        dict(_paired(ga, f"{q8}__c2p2", q8, hold, rng, "wrong"), label="Qwen3.8 tuned vs original: wrong facts (held-out 15)", key="wrong_hold"),
        dict(_paired(ga, f"{q8}__c2p2", q8, hold, rng), label="Qwen3.8 tuned vs original: coverage (held-out 15)", key="cov_hold"),
        dict(_paired(ga, f"{q6}__c2p2", f"{q8}__c2p2", hold, rng, "wrong"), label="Qwen3.6 vs Qwen3.8, both tuned: wrong facts (held-out 15)", key="q36_hold"),
        dict(_paired(ga, f"{q8}__c2p1", q8, tune, rng, "wrong"), label="Qwen3.8 tuned vs original: wrong facts (tuning 29)", key="wrong_tune"),
    ]
    out["ask"] = {"rows": rows, "paired": [p for p in paired if p.get("n")], "sets": {k: len(v) for k, v in sets.items()}}
    return out


def build(cfg) -> dict:
    rng = np.random.default_rng(0)
    data = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "bootstrap": B,
            "embedding": embedding(cfg, rng), **llm_tasks(rng),
            "labels": {t: len(read(LABELS / f"{t}.jsonl")) for t in ("dedup", "topic", "kind", "search", "stories", "filings", "ask")},
            "items": {t: len(items(t)) for t in ("dedup", "topic", "kind", "search", "stories", "filings", "ask", "search_corpus")}}
    OUT.write_text(json.dumps(data, indent=1, default=float))
    log.info("wrote %s", OUT)
    return data
