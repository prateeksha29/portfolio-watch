"""Scoring: automatic checks for every LLM run, blind grading packets for free text, and the report.

Automatic: valid JSON, errors, seconds per item, sentiment and on-topic agreement with the labels, figures the
model reported that the source text doesn't contain, and for Ask, citations to sources that don't exist.
Blind grading: bench/grading/<task>_packet.jsonl shows each item's reference facts (or Ask checklist) with every
run's answer under a random letter; the letter-to-run key is kept in <task>_key.json. Grades go in
<task>_grades.jsonl: {"id", "letter", "hit": [indices of facts/checklist points covered], "wrong": n,
"violations": [indices of must-not points], "note"}."""
import datetime as dt
import json
import random
import re
import statistics

from ..llm import SENTIMENTS, grounded
from .common import GRADING, REPORTS, RUNS, items, labels, read, split, write


def runs(task: str) -> dict[str, dict[str, dict]]:
    """run name (model[__prompt]) -> item id -> output row"""
    out = {}
    for p in sorted((RUNS / task).glob("*.jsonl")):
        out[p.stem] = {r["id"]: r for r in read(p)}
    return out


def _nums(s: str) -> list[str]:
    return re.findall(r"\d[\d,]*(?:\.\d+)?", s or "")


def invented_numbers(text: str, source: str) -> int:
    """Figures in the model's text that appear nowhere in the source (years and small counts ignored)."""
    have = {n.replace(",", "") for n in _nums(source)}
    bad = 0
    for n in _nums(text):
        v = n.replace(",", "")
        if v in have or (v.isdigit() and (int(v) < 10 or 1990 <= int(v) <= 2035)):
            continue
        bad += 1
    return bad


def auto(task: str, ids: set[str] | None = None) -> dict[str, dict]:
    lab = labels(task)
    src = {it["id"]: it for it in items(task)}
    res = {}
    for name, rows in runs(task).items():
        rows = {k: v for k, v in rows.items() if ids is None or k in ids}
        if not rows:
            continue
        secs = [r["seconds"] for r in rows.values() if "seconds" in r and not r.get("error") and not r.get("contended")]
        m = {"n": len(rows), "errors": sum(1 for r in rows.values() if r.get("error")),
             "invalid_json": sum(1 for r in rows.values() if r.get("invalid_json")),
             "sec_avg": round(statistics.mean(secs), 1) if secs else None,
             "sec_p90": round(sorted(secs)[int(0.9 * (len(secs) - 1))], 1) if secs else None}
        outs = {k: r["output"] for k, r in rows.items() if r.get("output")}
        if task in ("stories", "filings"):
            sent_ok, topic_ok, topic_n, kn, kn_bad, inv = 0, 0, 0, 0, 0, 0
            for k, o in outs.items():
                g = lab.get(k, {})
                s = o.get("sentiment") if o.get("sentiment") in SENTIMENTS else "neutral"
                sent_ok += s == g.get("sentiment") or s in g.get("accept", [])
                if task == "stories" and not g.get("ambiguous_topic"):
                    topic_n += 1
                    topic_ok += bool(o.get("on_topic")) == g.get("on_topic")
                text = f"{src[k]['title']}\n{src[k]['text']}"
                for n in o.get("key_numbers", []):
                    kn += 1
                    kn_bad += not grounded(n, text)
                inv += invented_numbers(f"{o.get('headline', '')} {o.get('summary', '')} {o.get('why_it_matters', '')}",
                                        text)
            m.update(sentiment_acc=round(sent_ok / max(len(outs), 1), 3), key_numbers=kn, key_numbers_ungrounded=kn_bad,
                     invented_numbers_per_item=round(inv / max(len(outs), 1), 2))
            if task == "stories":
                m["on_topic_acc"] = round(topic_ok / max(topic_n, 1), 3)
        if task == "ask":
            bad_cites = 0
            for k, o in outs.items():
                n_src = len(src[k]["sources"])
                bad_cites += sum(1 for c in re.findall(r"\[(\d+)\]", o) if not 1 <= int(c) <= n_src)
            m.update(bad_citations=bad_cites, words_avg=round(statistics.mean(len(o.split()) for o in outs.values()))
                     if outs else None)
        res[name] = m
    return res


def packets(task: str, names: list[str] | None = None, seed: int = 7) -> None:
    """Blind packet of every run's answers per item, for grading."""
    all_runs = runs(task)
    names = names or sorted(all_runs)
    lab = labels(task)
    rng = random.Random(seed)
    key, rows = {}, []
    for it in items(task):
        answers = [(n, all_runs[n].get(it["id"], {}).get("output")) for n in names]
        answers = [(n, o) for n, o in answers if o]
        rng.shuffle(answers)
        letters = "ABCDEFGHIJKLMNOP"
        key[it["id"]] = {letters[i]: n for i, (n, _) in enumerate(answers)}
        shown = {}
        for i, (n, o) in enumerate(answers):
            if task == "ask":
                shown[letters[i]] = o
            elif task == "stories":
                shown[letters[i]] = f"{o.get('summary', '')} || why: {o.get('why_it_matters', '')}"
            else:
                shown[letters[i]] = f"{o.get('headline', '')} || {o.get('summary', '')} || why: {o.get('why_it_matters', '')}"
        ref = lab.get(it["id"], {})
        rows.append({"id": it["id"], "title": it.get("title") or it.get("question"),
                     "reference": ref.get("facts") or {"must": ref.get("must", []), "must_not": ref.get("must_not", [])},
                     "answers": shown})
    write(GRADING / f"{task}_packet.jsonl", rows)
    (GRADING / f"{task}_key.json").write_text(json.dumps(key, indent=1))
    print(f"{task}: {len(rows)} items x {len(names)} runs -> bench/grading/{task}_packet.jsonl")


def graded(task: str, ids: set[str] | None = None) -> dict[str, dict]:
    """Per run: share of reference facts / checklist points covered, wrong statements and violations per item."""
    key = json.loads((GRADING / f"{task}_key.json").read_text()) if (GRADING / f"{task}_key.json").exists() else {}
    lab = labels(task)
    acc = {}
    for g in read(GRADING / f"{task}_grades.jsonl"):
        if ids is not None and g["id"] not in ids:
            continue
        name = key.get(g["id"], {}).get(g["letter"])
        if not name:
            continue
        ref = lab[g["id"]]
        total = len(ref.get("facts") or ref.get("must") or [])
        a = acc.setdefault(name, {"items": 0, "coverage": [], "wrong": 0, "violations": 0})
        a["items"] += 1
        if total:
            a["coverage"].append(len(set(g.get("hit", []))) / total)
        a["wrong"] += g.get("wrong", 0)
        a["violations"] += len(g.get("violations", []))
    def margin(xs):  # 95% normal-approximation margin on the mean coverage
        return round(1.96 * statistics.stdev(xs) / len(xs) ** 0.5, 3) if len(xs) > 1 else None

    return {n: {"items": a["items"], "coverage": round(statistics.mean(a["coverage"]), 3) if a["coverage"] else None,
                "coverage_pm": margin(a["coverage"]),
                "wrong_per_item": round(a["wrong"] / a["items"], 2), "violations_per_item": round(a["violations"] / a["items"], 2)}
            for n, a in acc.items()}


PRODUCTION = "unsloth__gemma-4-26B-A4B"


def item_coverage(task: str) -> dict[str, dict[str, float]]:
    """Per run, per item: share of the reference facts / checklist points the graded answer covered."""
    key = json.loads((GRADING / f"{task}_key.json").read_text()) if (GRADING / f"{task}_key.json").exists() else {}
    lab, out = labels(task), {}
    for g in read(GRADING / f"{task}_grades.jsonl"):
        name = key.get(g["id"], {}).get(g["letter"])
        total = len(lab.get(g["id"], {}).get("facts") or lab.get(g["id"], {}).get("must") or [])
        if name and total:
            out.setdefault(name, {})[g["id"]] = len(set(g.get("hit", []))) / total
    return out


def paired_vs(task: str, base: str = PRODUCTION) -> list[str]:
    """Coverage difference vs the base run on the items both were graded on, with a 95% margin (paired)."""
    cov = item_coverage(task)
    if base not in cov:
        return []
    lines = []
    for name, c in sorted(cov.items()):
        common = sorted(set(c) & set(cov[base]))
        if name == base or len(common) < 2:
            continue
        d = [c[i] - cov[base][i] for i in common]
        m, pm = statistics.mean(d), 1.96 * statistics.stdev(d) / len(d) ** 0.5
        lines.append(f"- `{name}` vs `{base}`: {m:+.3f} ± {pm:.3f} on {len(d)} shared items")
    return lines


def _cov(g: dict) -> str:
    if g.get("coverage") is None:
        return "–"
    return f"{g['coverage']} ± {g['coverage_pm']}" if g.get("coverage_pm") is not None else str(g["coverage"])


def report(embed_results: dict | None = None) -> str:
    today = dt.date.today().isoformat()
    lines = [f"# Model benchmark {today}", "",
             "Lean set, labeled by Claude (see bench/README.md). dev = tuning split, test = held-out third.", ""]
    if embed_results:
        lines += ["## Embedding", "",
                  "| Model | Dedup AUC | Dedup best F1 | Topic AUC | Topic best F1 | Kind accuracy | Search nDCG@10 | Search recall@10 |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in embed_results.items():
            a = r["all"]
            lines.append(f"| `{name}` | {a['dedup']['auc']:.3f} | {a['dedup']['best_f1']:.3f} | {a['topic']['detect_auc']:.3f} | "
                         f"{a['topic']['best_f1']:.3f} | {a['kind']['accuracy']:.2f} | {a['search']['ndcg@10']:.3f} | "
                         f"{a['search']['recall@10']:.3f} |")
        lines.append("")
    for task in ("stories", "filings", "ask"):
        au, gr = auto(task), graded(task)
        if not au:
            continue
        lines += [f"## {task.title()}", ""]
        if task == "ask":
            lines += ["| Run | Items | Checklist coverage | Wrong facts/answer | Violations/answer | Bad citations | Words | Sec avg | Sec p90 | Errors |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for n, m in sorted(au.items(), key=lambda kv: -(gr.get(kv[0], {}).get("coverage") or 0)):
                g = gr.get(n, {})
                lines.append(f"| `{n}` | {m['n']} | {_cov(g)} | {g.get('wrong_per_item', '–')} | {g.get('violations_per_item', '–')} | "
                             f"{m['bad_citations']} | {m['words_avg']} | {m['sec_avg']} | {m['sec_p90']} | {m['errors']} |")
        else:
            topic = "On-topic acc | " if task == "stories" else ""
            lines += [f"| Run | Items | Fact coverage | Wrong/item | Sentiment acc | {topic}Invented numbers/item | Key numbers ungrounded | Sec avg | Sec p90 | Errors+bad JSON |",
                      "|---|---|---|---|---|" + ("---|" if topic else "") + "---|---|---|---|---|"]
            for n, m in sorted(au.items(), key=lambda kv: -(gr.get(kv[0], {}).get("coverage") or 0)):
                g = gr.get(n, {})
                t = f"{m['on_topic_acc']} | " if task == "stories" else ""
                lines.append(f"| `{n}` | {m['n']} | {_cov(g)} | {g.get('wrong_per_item', '–')} | {m['sentiment_acc']} | {t}"
                             f"{m['invented_numbers_per_item']} | {m['key_numbers_ungrounded']}/{m['key_numbers']} | "
                             f"{m['sec_avg']} | {m['sec_p90']} | {m['errors'] + m['invalid_json']} |")
        pv = paired_vs(task)
        if pv:
            lines += ["", "Coverage vs production (paired, same items):", *pv]
        lines.append("")
    text = "\n".join(lines)
    (REPORTS / f"bench-{today}.md").write_text(text)
    return text


def split_ids(task: str, which: str) -> set[str]:
    return {it["id"] for it in items(task) if split(it["id"]) == which}
