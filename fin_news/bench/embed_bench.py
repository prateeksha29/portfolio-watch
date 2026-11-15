"""Embedding benchmark: duplicate headlines, topic matching, story type and search, per model.

Backends:
  llama:<dim>        the production model (EmbeddingGemma 2 on llama-swap), truncated to <dim>, with the
                     pipeline's own text formats and prompts
  st:<hf model id>   a sentence-transformers model on the CPU (benchmark only; production needs it on llama-swap)
Vectors are cached in bench/runs/embed/<backend>/ so re-scoring is free."""
import copy
import json
import logging

import numpy as np

from ..classify import PROTOTYPES
from .common import DATA, RUNS, items, labels, read, slug, write

log = logging.getLogger(__name__)
DEFAULT = ["llama:256", "llama:768", "st:Qwen/Qwen3-Embedding-0.6B", "st:BAAI/bge-m3",
           "st:Snowflake/snowflake-arctic-embed-l-v2.0"]


def _norm(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True).clip(min=1e-12)


class Llama:
    def __init__(self, cfg, dim):
        from ..embed import Embedder
        c = copy.deepcopy(cfg)
        c["embedding"]["dim"] = dim
        self.e = Embedder(c)

    def docs(self, pairs):
        return self.e.documents(pairs)

    def headlines(self, titles):
        return self.e.headlines(titles)

    def queries(self, texts):
        return self.e.queries(texts)

    def classify(self, texts):
        return self.e.classify(texts)


class ST:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(16)
        self.m = SentenceTransformer(name, device="cpu", trust_remote_code=False,
                                     model_kwargs={"dtype": torch.float32})
        self.m.max_seq_length = 256

    def _enc(self, texts, kind):
        fn = {"q": self.m.encode_query, "d": self.m.encode_document}.get(kind, self.m.encode)
        return np.asarray(fn(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False), np.float32)

    def docs(self, pairs):
        return self._enc([f"{t}. {s}" if s else t for t, s in pairs], "d")

    def headlines(self, titles):
        return self._enc(titles, "s")

    def queries(self, texts):
        return self._enc(texts, "q")

    def classify(self, texts):
        return self._enc(texts, "s")


def backend(cfg, name):
    kind, _, arg = name.partition(":")
    return Llama(cfg, int(arg)) if kind == "llama" else ST(arg)


def embed_all(cfg, name) -> dict:
    """All vectors one backend needs, cached."""
    out_dir = RUNS / "embed" / slug(name)
    path = out_dir / "vectors.npz"
    if path.exists():
        return dict(np.load(path))
    b = backend(cfg, name)
    corpus = items("search_corpus")
    dedup = items("dedup")
    topic = items("topic")
    topics = [r["topic"] for r in read(DATA / "topic_list.jsonl")]
    kinds = items("kind")
    protos = [(k, t) for k, ts in PROTOTYPES.items() for t in ts]
    log.info("%s: embedding %d corpus docs and the task items", name, len(corpus))
    V = {
        "corpus_docs": b.docs([(r["title"], r["snippet"]) for r in corpus]),
        "corpus_heads": b.headlines([r["title"] for r in corpus]),
        "dedup_a": b.headlines([r["a"]["title"] for r in dedup]),
        "dedup_b": b.headlines([r["b"]["title"] for r in dedup]),
        "topic_docs": b.docs([(r["title"], r["snippet"]) for r in topic]),
        "topic_queries": b.queries(topics),
        "kind": b.classify([r["title"] for r in kinds]),
        "protos": b.classify([t for _, t in protos]),
        "search_queries": b.queries([q["query"] for q in items("search")]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, **V)
    return V


def _auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _best_f1(scores, y):
    best = (0.0, None)
    for t in np.unique(scores):
        p = scores >= t
        tp = (p & y).sum()
        if tp == 0:
            continue
        f1 = 2 * tp / (p.sum() + y.sum())
        best = max(best, (float(f1), float(t)))
    return best


def score_dedup(V, split_ids=None):
    lab = labels("dedup")
    rows = [(i, r) for i, r in enumerate(items("dedup")) if not lab[r["id"]].get("exclude")
            and (split_ids is None or r["id"] in split_ids)]
    mu = V["corpus_heads"].mean(0)
    A, B = _norm(V["dedup_a"] - mu), _norm(V["dedup_b"] - mu)
    s = np.array([float(A[i] @ B[i]) for i, _ in rows])
    y = np.array([lab[r["id"]]["same"] for _, r in rows])
    f1, t = _best_f1(s, y)
    return {"n": len(rows), "auc": _auc(s[y], s[~y]), "best_f1": f1, "best_threshold": t, "scores": s, "y": y}


def score_topic(V, split_ids=None):
    lab = labels("topic")
    topics = [r["topic"] for r in read(DATA / "topic_list.jsonl")]
    rows = [(i, r) for i, r in enumerate(items("topic")) if not lab[r["id"]].get("exclude")
            and (split_ids is None or r["id"] in split_ids)]
    mu = V["corpus_docs"].mean(0)
    S = _norm(V["topic_docs"] - mu) @ _norm(V["topic_queries"] - mu).T
    best = S.max(1)
    has = np.array([bool(lab[r["id"]]["topics"]) for _, r in rows])
    mx = np.array([best[i] for i, _ in rows])
    right = np.array([topics[int(S[i].argmax())] in lab[r["id"]]["topics"] for i, r in rows])
    # End to end: assign the best topic when its score clears a threshold; best F1 over thresholds.
    bestf = (0.0, None)
    for t in np.unique(mx):
        pred = mx >= t
        tp = (pred & right).sum()
        if tp:
            prec, rec = tp / pred.sum(), tp / has.sum()
            bestf = max(bestf, (float(2 * prec * rec / (prec + rec)), float(t)))
    return {"n": len(rows), "detect_auc": _auc(mx[has], mx[~has]),
            "top1_when_topic": float(right[has].mean()) if has.any() else float("nan"),
            "best_f1": bestf[0], "best_threshold": bestf[1]}


def score_kind(V, split_ids=None):
    lab = labels("kind")
    names = list(PROTOTYPES)
    protos = [k for k, ts in PROTOTYPES.items() for _ in ts]
    P = V["protos"]
    C = _norm(np.stack([P[[j for j, k in enumerate(protos) if k == n]].mean(0) for n in names]))
    rows = [(i, r) for i, r in enumerate(items("kind")) if not lab[r["id"]].get("exclude")
            and (split_ids is None or r["id"] in split_ids)]
    pred = [names[int((V["kind"][i] @ C.T).argmax())] for i, _ in rows]
    gold = [lab[r["id"]]["kind"] for _, r in rows]
    f1s = []
    for n in names:
        tp = sum(p == g == n for p, g in zip(pred, gold))
        fp = sum(p == n != g for p, g in zip(pred, gold))
        fn = sum(g == n != p for p, g in zip(pred, gold))
        if tp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"n": len(rows), "accuracy": float(np.mean([p == g for p, g in zip(pred, gold)])),
            "macro_f1": float(np.mean(f1s))}


def search_rank(V, k=10):
    corpus = items("search_corpus")
    S = _norm(V["search_queries"]) @ _norm(V["corpus_docs"]).T
    return {q["id"]: [corpus[j]["id"] for j in np.argsort(-S[n])[:k]] for n, q in enumerate(items("search"))}


def score_search(V, split_ids=None):
    judged = {}
    for r in read(DATA.parent / "labels" / "search.jsonl"):
        judged.setdefault(r["qid"], {})[r["cid"]] = r["rel"]
    ranks = search_rank(V)
    rec, ndcg, mrr, n = [], [], [], 0
    for qid, top in ranks.items():
        if split_ids is not None and qid not in split_ids:
            continue
        rel = judged.get(qid, {})
        good = {c for c, v in rel.items() if v > 0}
        if not good:
            continue
        n += 1
        unjudged = sum(c not in rel for c in top)
        gains = [rel.get(c, 0) for c in top]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:10]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
        rec.append(len(good & set(top)) / min(len(good), 10))
        ndcg.append(dcg / idcg)
        mrr.append(next((1 / (i + 1) for i, c in enumerate(top) if rel.get(c, 0) > 0), 0.0))
        if unjudged:
            log.info("search %s: %d of the top 10 unjudged (counted as not relevant)", qid, unjudged)
    return {"n": n, "recall@10": float(np.mean(rec)) if rec else float("nan"),
            "ndcg@10": float(np.mean(ndcg)) if ndcg else float("nan"), "mrr": float(np.mean(mrr)) if mrr else float("nan")}


def pool(cfg, names, depth=10) -> None:
    """Union of each backend's top results per query, for the annotator to judge (bench/data/search_pool.jsonl)."""
    corpus = {r["id"]: r for r in items("search_corpus")}
    already = {(r["qid"], r["cid"]) for r in read(DATA.parent / "labels" / "search.jsonl")}
    want = {}
    for name in names:
        for qid, top in search_rank(embed_all(cfg, name)).items():
            for c in top[:depth]:
                want.setdefault((qid, c), []).append(name)
    rows = [{"qid": q, "cid": c, "title": corpus[c]["title"], "snippet": corpus[c]["snippet"][:200]}
            for (q, c) in sorted(want) if (q, c) not in already]
    write(DATA / "search_pool.jsonl", rows)
    print(f"{len(rows)} query-article pairs to judge -> bench/data/search_pool.jsonl")


def run(cfg, names) -> dict:
    from .common import split
    results = {}
    for name in names:
        V = embed_all(cfg, name)
        res = {}
        for sp in ("dev", "test", "all"):
            ids = None if sp == "all" else {r["id"] for t in ("dedup", "topic", "kind", "search") for r in items(t)
                                            if split(r["id"]) == sp}
            d = score_dedup(V, ids)
            res[sp] = {"dedup": {k: v for k, v in d.items() if k not in ("scores", "y")},
                       "topic": score_topic(V, ids), "kind": score_kind(V, ids), "search": score_search(V, ids)}
        results[name] = res
        a = res["all"]
        print(f"{name:45} dedup AUC {a['dedup']['auc']:.3f} F1 {a['dedup']['best_f1']:.3f} | "
              f"topic AUC {a['topic']['detect_auc']:.3f} F1 {a['topic']['best_f1']:.3f} | "
              f"kind acc {a['kind']['accuracy']:.2f} | search nDCG {a['search']['ndcg@10']:.3f}")
    out = RUNS / "embed" / "results.json"
    out.write_text(json.dumps(results, indent=1))
    return results


def current_rules(cfg) -> None:
    """The production model with the pipeline's own thresholds, as a reference line."""
    V = embed_all(cfg, "llama:256")
    d = score_dedup(V)
    t = cfg["filter"]["dedup_threshold"]
    p = d["scores"] >= t
    tp = (p & d["y"]).sum()
    print(f"production dedup threshold {t}: precision {tp / max(p.sum(), 1):.2f} recall {tp / d['y'].sum():.2f}")

