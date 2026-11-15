"""python -m fin_news.bench <step>: sample | embed | llm | grade | score | report (see bench/README.md)."""
import argparse
import logging

from ..config import load_config


def main() -> None:
    p = argparse.ArgumentParser(prog="fin_news.bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="freeze benchmark items from the database")
    s.add_argument("what", nargs="*", help="dedup topic kind stories filings search ask (default: the first five)")
    s.add_argument("--extend", action="store_true", help="keep the frozen items and append a second draw")
    e = sub.add_parser("embed", help="embedding benchmark")
    e.add_argument("--models", nargs="*", help="backends, e.g. llama:256 st:BAAI/bge-m3 (default: all candidates)")
    e.add_argument("--pool", action="store_true", help="write the search results to judge, then stop")
    lm = sub.add_parser("llm", help="run LLM candidates on a task")
    lm.add_argument("task", choices=["stories", "filings", "ask"])
    lm.add_argument("--models", nargs="*", help="llama-swap model ids; append +think for thinking on")
    lm.add_argument("--prompt", help="system prompt variant: bench/prompts/<task>/<name>.txt (default: the app's)")
    lm.add_argument("--split", choices=["dev", "test"], help="only this split")
    lm.add_argument("--ids", help="only these items: a list and/or ranges, e.g. a00-a39,a45")
    g = sub.add_parser("grade", help="write blind grading packets for a task")
    g.add_argument("task", choices=["stories", "filings", "ask"])
    g.add_argument("--runs", nargs="*", help="run names to include (default: all)")
    sub.add_parser("report", help="score everything and write reports/bench-<date>.md")
    sub.add_parser("tech", help="recompute the Benchmarks page's technical numbers into bench/tech.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("urllib3", "httpx", "sentence_transformers", "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config()
    if args.cmd == "sample":
        from .sample import main as sample
        sample(cfg, args.what, args.extend)
    elif args.cmd == "embed":
        from . import embed_bench as eb
        names = args.models or eb.DEFAULT
        if args.pool:
            eb.pool(cfg, names)
        else:
            eb.run(cfg, names)
            if "llama:256" in names:
                eb.current_rules(cfg)
    elif args.cmd == "llm":
        from . import llm_bench as lb
        from .grade import split_ids
        default = lb.ASK_MODELS if args.task == "ask" else lb.SUMMARY_MODELS
        ids = split_ids(args.task, args.split) if args.split else None
        if args.ids:
            ids = set()
            for part in args.ids.split(","):
                a, _, b = part.partition("-")
                if not b:
                    ids.add(a)
                    continue
                pre, w = a.rstrip("0123456789"), len(a) - len(a.rstrip("0123456789"))
                ids |= {f"{pre}{i:0{w}d}" for i in range(int(a[len(pre):]), int(b[len(pre):]) + 1)}
        lb.run(cfg, args.task, args.models or default, args.prompt, ids)
    elif args.cmd == "grade":
        from .grade import packets
        packets(args.task, args.runs)
    elif args.cmd == "tech":
        from .tech import build
        build(cfg)
    elif args.cmd == "report":
        import json
        from .common import RUNS
        from .grade import report
        path = RUNS / "embed" / "results.json"
        print(report(json.loads(path.read_text()) if path.exists() else None))


if __name__ == "__main__":
    main()
