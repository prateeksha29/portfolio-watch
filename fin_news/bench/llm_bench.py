"""LLM benchmark: story summaries, filing summaries and Ask answers, per model and prompt.

Each model sees exactly the pipeline's messages (same system prompt, JSON schema, temperature 0); Ask uses the
prompts frozen in bench/data/ask.jsonl. A model name ending in "+think" runs with thinking on. Outputs go to
bench/runs/<task>/<model>[__<prompt>].jsonl, one line per item with timing; a rerun only fills in missing items."""
import copy
import json
import logging
import re
import time

from .. import llm as L
from .common import PROMPTS, RUNS, items, read, slug, write

log = logging.getLogger(__name__)
SUMMARY_MODELS = ["unsloth/gemma-4-26B-A4B", "turboquant/gemma-4-26B-A4B", "unsloth/gemma-4-E4B-MTP",
                  "atomic/Qwen3.6-35B-A3B-MTP", "turboderp/Qwen3.8-Flash-Next", "unsloth/Qwen3.8-27B",
                  "prism-ml/Ternary-Bonsai-2-27B"]
ASK_MODELS = ["unsloth/Qwen3.8-27B", "unsloth/gemma-4-26B-A4B", "atomic/Qwen3.6-35B-A3B-MTP",
              "turboderp/Qwen3.8-Flash-Next", "prism-ml/Ternary-Bonsai-2-27B"]
SYSTEM = {"stories": "ENRICH_SYSTEM", "filings": "FILING_SYSTEM", "ask": "ASK_SYSTEM"}


def system_prompt(task: str, prompt: str | None) -> str:
    if not prompt:
        return getattr(L, SYSTEM[task])
    return (PROMPTS / task / f"{prompt}.txt").read_text().strip()


def out_path(task: str, model: str, prompt: str | None):
    return RUNS / task / (slug(model) + (f"__{prompt}" if prompt else "") + ".jsonl")


def client(cfg: dict, model: str) -> L.LLM:
    c = copy.deepcopy(cfg)
    think = model.endswith("+think")
    c["llm"]["thinking"] = think
    c["llm"]["timeout_seconds"] = 600
    client = L.LLM(c)
    client.model = model.removesuffix("+think")
    return client


def _messages(task: str, it: dict, system: str, max_chars: dict) -> tuple[list[dict], dict | None, int]:
    if task == "stories":
        text = it["text"][: max_chars["article"]]
        return ([{"role": "system", "content": system},
                 {"role": "user", "content": f"Holdings: {it['about']}\n\nTITLE: {it['title']}\n\nARTICLE:\n{text}"}],
                L.ENRICH_SCHEMA, 600)
    if task == "filings":
        text = it["text"][: max_chars["filing"]]
        return ([{"role": "system", "content": system},
                 {"role": "user", "content": f"Company: {it['company']}\nFILING: {it['title']}\n\nTEXT:\n{text}"}],
                L.FILING_SCHEMA, 800)
    context, sources = it["context"], it["sources"]
    if it.get("_c2"):  # prompt variants named c2*: the frozen facts in the current context wording (bench/ctxfmt.py)
        context, sources = it["_c2"]
    numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))
    return ([{"role": "system", "content": system},
             {"role": "user", "content": f"{context}\n\nSOURCES:\n{numbered}\n\nQUESTION: {it['question']}"}],
            None, 1500)


def near_pipeline_run() -> bool:
    """The pipeline runs at 6am, 2pm and 6pm (crontab) and summarizes with the GPU: keep out from 20 min before to
    25 after."""
    import datetime as dt
    now = dt.datetime.now()
    minutes = now.hour * 60 + now.minute
    return any(h * 60 - 20 <= minutes < h * 60 + 25 for h in (6, 14, 18))


def others_running(cfg: dict, mine: str) -> list[str]:
    """Models other people have loaded on the shared GPU (llama-swap /running), besides ours and the embedder."""
    import requests
    base = cfg["llm"]["base_url"].rstrip("/").removesuffix("/v1")
    try:
        running = requests.get(f"{base}/running", timeout=10).json().get("running", [])
    except Exception:
        return []
    ours = {mine, cfg["embedding"].get("served_model"), *SUMMARY_MODELS, *ASK_MODELS}  # our own leftovers swap out freely
    return [m["model"] for m in running if m["model"] not in ours]


def wait_for_gpu(cfg: dict, mine: str, quiet_s: int = 300, max_wait_s: int = 4 * 3600) -> bool:
    """Wait until no one else has had a model loaded for quiet_s seconds, so our requests don't force swaps."""
    start, quiet_since = time.time(), None
    while time.time() - start < max_wait_s:
        busy = others_running(cfg, mine)
        if busy:
            if quiet_since is not None or time.time() - start < 1:
                log.info("GPU in use by %s; waiting", ", ".join(busy))
            quiet_since = None
        else:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= quiet_s or (time.time() - start < 1):
                return True
        time.sleep(30)
    return False


def run(cfg: dict, task: str, models: list[str], prompt: str | None = None, ids: set[str] | None = None) -> None:
    system = system_prompt(task, prompt)
    max_chars = {"article": cfg["llm"]["max_article_chars"], "filing": cfg["llm"].get("max_filing_chars", 40000)}
    todo_items = [it for it in items(task) if ids is None or it["id"] in ids]
    c2 = task == "ask" and bool(prompt) and prompt.startswith("c2")
    if c2:
        from .ctxfmt import Converter
        conv = Converter()
        todo_items = [{**it, "_c2": (conv.context(it["context"]), conv.sources(it["sources"]))} for it in todo_items]
    for model in models:
        path = out_path(task, model, prompt)
        done = {r["id"]: r for r in read(path)}
        todo = [it for it in todo_items if it["id"] not in done or done[it["id"]].get("error")]
        if not todo:
            continue
        if near_pipeline_run():
            log.warning("pipeline run due; not starting %s (rerun to resume)", model)
            return
        llm = client(cfg, model)
        base_model = model.removesuffix("+think")
        if not wait_for_gpu(cfg, base_model):
            log.warning("GPU stayed busy; %s skipped", model)
            continue
        think = model.endswith("+think")
        log.info("%s / %s%s: %d items", task, model, f" / prompt {prompt}" if prompt else "", len(todo))
        t0 = time.time()
        try:  # load the model first so the first item's time isn't the model swap
            llm._chat([{"role": "user", "content": "Reply with OK."}], max_tokens=5)
        except Exception as e:
            log.warning("%s not answering (%s); skipped", model, e)
            continue
        log.info("%s loaded in %.0f s", model, time.time() - t0)
        for it in todo:
            if near_pipeline_run():
                log.warning("pipeline run due; stopping (rerun to resume)")
                break
            if others_running(cfg, base_model):  # someone else needs the GPU: step aside until it's quiet again
                if not wait_for_gpu(cfg, base_model):
                    log.warning("GPU stayed busy; stopping %s", model)
                    break
            messages, schema, max_tokens = _messages(task, it, system, max_chars)
            if think:
                max_tokens *= 4  # thinking spends tokens before the answer
            row = {"id": it["id"], "model": model, "prompt": prompt or "default"}
            t = time.time()
            try:
                raw = llm._chat(messages, schema, max_tokens=max_tokens)
                row["seconds"] = round(time.time() - t, 2)
                if schema:
                    try:
                        row["output"] = json.loads(raw)
                    except json.JSONDecodeError:
                        row.update(output=None, invalid_json=True, raw=raw[:2000])
                else:
                    row["output"] = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
                    if c2:  # the app drops citations to sources that don't exist (askfmt.clean_citations)
                        from ..askfmt import clean_citations
                        row["output"], row["dropped_citations"] = clean_citations(row["output"], len(it["sources"]))
            except Exception as e:
                row.update(error=f"{type(e).__name__}: {e}"[:300], seconds=round(time.time() - t, 2))
            if others_running(cfg, base_model):
                row["contended"] = True  # a swap may have happened mid-request: its time is left out of the speed figures
            done[it["id"]] = row
            write(path, [done[k] for k in sorted(done)])
        log.info("%s done in %.0f s", model, time.time() - t0)
