"""Command line: run | search | ask | prune | market | funds | holdings | adduser | passwd"""
import argparse
import fcntl
import getpass
import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import load_config


def setup_logging(cfg: dict) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stderr),
                    RotatingFileHandler(cfg["paths"]["logs"] / "run.log", maxBytes=5_000_000, backupCount=3)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for noisy in ("urllib3", "httpx", "sentence_transformers", "transformers", "trafilatura", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    p = argparse.ArgumentParser(prog="fin_news", description="Local portfolio news pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="collect, process and write a digest")
    r.add_argument("--no-llm", action="store_true", help="skip LLM summaries")
    r.add_argument("--no-fulltext", action="store_true", help="skip downloading article pages")
    s = sub.add_parser("search", help="semantic search over stored articles")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--since", help="YYYY-MM-DD")
    s.add_argument("--ticker")
    a = sub.add_parser("ask", help="answer a question from stored articles, with citations")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=8)
    sub.add_parser("prune", help="apply the retention policy now")
    m = sub.add_parser("market", help="fetch prices, valuation and analyst targets now (holdings, or given tickers)")
    m.add_argument("tickers", nargs="*")
    fu = sub.add_parser("funds", help="fetch fund holdings (SEC N-PORT) and show your real exposure per stock")
    fu.add_argument("--force", action="store_true", help="check every fund for a newer filing now")
    sub.add_parser("holdings", help="show the tickers and fund themes read from the CSV")
    au = sub.add_parser("adduser", help="add a person (asks for their password)")
    au.add_argument("name", help="sign-in name: lowercase letters, digits, - and _")
    au.add_argument("--display", default="", help="name shown in the app")
    au.add_argument("--admin", action="store_true", help="sees source problems, Benchmarks and add-on logins")
    pw = sub.add_parser("passwd", help="set a person's password")
    pw.add_argument("name")
    fl = sub.add_parser("files", help="look at the Fidelity exports without showing their contents")
    fl.add_argument("action", choices=("check", "scramble"),
                    help="check: columns, kinds of values, accounts as A/B/C, rows the app can't read; "
                         "scramble: a copy with fake accounts and amounts in data/scrambled/<name>/")
    p.add_argument("--user", help="whose portfolio: search, ask, market, funds, holdings (default: the first admin); "
                                  "files (default: everyone)")
    for sp in sub.choices.values():  # --user before or after the command
        sp.add_argument("--user", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    args = p.parse_args()

    cfg = load_config()
    if args.cmd in ("adduser", "passwd"):
        from .users import add_user, set_password, user_ids

        if args.cmd == "adduser":
            try:
                home = add_user(args.name, args.display, args.admin)
            except ValueError as e:
                sys.exit(str(e))
        elif args.name not in user_ids():
            sys.exit(f"No user {args.name!r}")
        password = getpass.getpass(f"Password for {args.name} (10+ characters): ")
        if len(password) < 10 or password != getpass.getpass("Again: "):
            sys.exit("Not set: the passwords differ or are shorter than 10 characters." +
                     (f" Set one with: python -m fin_news passwd {args.name}" if args.cmd == "adduser" else ""))
        set_password(args.name, password)
        print(f"{args.name}: password set" + (f"; their exports go in {home / 'fidelity'} (or upload them on the "
                                               "Files page)" if args.cmd == "adduser" else ""))
        return
    if args.cmd == "files":  # never prints a value from the files: see exportcheck.py
        from .exportcheck import check, scramble
        from .users import default_user, user_ids

        labels: dict = {}
        for uid in [args.user] if args.user else user_ids():
            person = default_user(cfg, uid)
            folder = person["paths"]["home"] / "fidelity"
            if args.action == "check":
                print(f"== {uid}: " + "\n".join(check(folder, person.get("returns", {}).get("accounts", []), labels)))
            else:
                dst = cfg["paths"]["data"] / "scrambled" / uid
                n = scramble(folder, dst)["files"] if folder.is_dir() else 0
                print(f"{uid}: {n} scrambled copies in {dst.relative_to(cfg['paths']['root'])}/ (accounts renamed, "
                      "amounts and quantities scaled at random per stock; prices, dates and symbols kept)")
        return
    setup_logging(cfg)
    from . import pipeline  # heavy imports after argument parsing
    if args.cmd not in ("run", "prune"):  # shared: everyone's run, retention
        from .users import default_user

        cfg = default_user(cfg, args.user)

    if args.cmd == "run":
        lock = open(cfg["paths"]["data"] / "run.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("Another run is in progress.")
        print(pipeline.run(cfg, use_llm=not args.no_llm, use_fulltext=not args.no_fulltext))
    elif args.cmd == "search":
        from .query import search

        since = f"{args.since}T00:00:00Z" if args.since else None
        for i, h in enumerate(search(cfg, pipeline.open_store(cfg), args.query, args.k, since, args.ticker), 1):
            print(f"{i:2}. [{h['similarity']:.2f}] {h['published'][:10]}  {h['title']}\n    {h['source']} · {h['url']}")
    elif args.cmd == "ask":
        from .query import ask

        answer, hits = ask(cfg, pipeline.open_store(cfg), args.question, args.k)
        print(answer, "\n\nSources:")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h['title']} — {h['source']}, {h['published'][:10]}\n    {h['url']}")
    elif args.cmd == "prune":
        from .retention import prune

        print(prune(cfg, pipeline.open_store(cfg)))
    elif args.cmd == "market":
        from .market import Market, pct, upside

        store = pipeline.open_store(cfg)
        tickers = [t.upper() for t in args.tickers] or pipeline.load_portfolio(cfg).tickers_by_weight()
        market = Market(cfg, store)
        for name, h in market.refresh(tickers).items():
            print(f"{name}: {h['items']} of {h['requests']} ok{'; last error: ' + h['last_error'] if h['errors'] else ''}")
        for t in tickers:
            q = market.latest(t)
            if q and q.get("price"):
                target = f"avg target ${q['target_mean']:,.2f} ({pct(upside(q))})" if q.get("target_mean") else "no target"
                print(f"{t:6} ${q['price']:>9,.2f}  P/E {q.get('pe') or 0:6.1f}  {target}  {len(market.closes(t))} closes")
    elif args.cmd == "funds":
        from .funds import FundData, exposure_line

        store, h = pipeline.open_store(cfg), pipeline.load_portfolio(cfg)
        fd = FundData(cfg, store)
        st = fd.refresh(list(h.funds), force=args.force)
        print(f"Funds checked: {st['requests']}, new filings: {st['items']}, unchanged/not filed: {st['not_modified']}, "
              f"errors: {st['errors']} {st['last_error']}")
        for s in sorted(h.funds, key=lambda s: -h.funds[s]["weight"]):
            f = fd.filing(s)
            src = f" via {fd.source_of(s)}" if fd.source_of(s) != s else ""
            print(f"  {s:10} {h.funds[s]['weight']:5.1%}  " + (f"{len(fd.holdings(s))} positions as of {f['period']}{src}"
                                                              if f else "no holdings data"))
        ex = fd.exposures(h)
        print("\nBiggest real exposures:")
        for t, d in sorted(ex["by_ticker"].items(), key=lambda kv: -kv[1]["total"])[:12]:
            print("  " + exposure_line(t, d, ex["portfolio"]))
        if ex["missing"]:
            print(f"No holdings data for {', '.join(ex['missing'])} ({ex['missing_value'] / ex['portfolio']:.1%} of portfolio)")
    elif args.cmd == "holdings":
        h = pipeline.load_portfolio(cfg)
        print("Tickers (largest position first):", ", ".join(h.tickers_by_weight()))
        for theme, funds in h.themes().items():
            print(f"Theme '{theme}': {', '.join(funds)}")


if __name__ == "__main__":
    main()
