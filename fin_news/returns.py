"""How the owner's own investing did over a period: what they earned, and how each new investment fared against the
same dollars put into the S&P 500 on the same days.

Inputs: Fidelity's activity exports (fidelity/Accounts_History*.csv, 90 days each; overlapping files are merged), the
newest positions export, and daily closes from Yahoo Finance's history page (split-adjusted, not dividend-adjusted).
Only the accounts in the person's [returns] accounts count (all accounts if not set), and money-market funds are cash.

Earned in a period = value at the end - value at the start - money moved in + dividends and interest
(money moved in: buys, ESPP purchases, vested shares at their value, minus sale proceeds). Shares at any date come from
the positions export and the transactions in between. Sales use first-in, first-out (Fidelity's default for stocks)."""
import bisect
import calendar
import csv
import datetime as dt
import io
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .holdings import _export_date

BENCHMARK = "^GSPC"
DEFAULT_ACCOUNTS: list[str] = []  # empty: every account in the exports


def counted(account: str | None, accounts) -> bool:
    return not accounts or (account or "").strip() in accounts
CASH_SYMBOLS = {"SPAXX", "FZFXX", "FDRXX", "FCASH", "CORE", "FDLXX", "SPRXX", "FTEXX"}
PERIODS = {"1M": "1 month", "3M": "3 months", "6M": "6 months", "YTD": "this year", "1Y": "1 year"}
# Action text -> kind. The first match wins.
KINDS = [
    ("YOU BOUGHT ESPP", "espp"), ("YOU BOUGHT", "buy"), ("YOU SOLD", "sell"), ("REDEMPTION PAYOUT", "sell"),
    ("REINVESTMENT", "reinvest"),
    ("CONVERSION SHARES DEPOSITED", "vest"), ("DISTRIBUTION", "received"),
    ("DIVIDEND RECEIVED", "income"), ("LONG-TERM CAP GAIN", "income"), ("SHORT-TERM CAP GAIN", "income"),
    ("INTEREST", "income"), ("FOREIGN TAX PAID", "income"), ("FEE CHARGED", "income"),
    ("TRANSFERRED", "transfer"), ("ELECTRONIC FUNDS TRANSFER", "bank"), ("JOURNALED SPP", "espp_cash"),
]
NEW_MONEY = {"buy": "buy", "espp": "ESPP", "vest": "vest", "received": "received", "transfer": "transferred in"}
FRESH_DAYS = 30  # buys younger than this are shown but not judged


@dataclass
class Tx:
    date: dt.date
    account: str
    kind: str
    symbol: str
    qty: float
    amount: float
    action: str


@dataclass
class Lot:
    date: dt.date
    symbol: str
    kind: str
    shares: float
    cost: float
    sold: list = field(default_factory=list)  # (date, shares, proceeds)


def _num(s: str | None) -> float:
    s = (s or "").replace("$", "").replace(",", "").replace("+", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def is_cash(symbol: str) -> bool:
    return not symbol or symbol.endswith("**") or symbol.rstrip("*") in CASH_SYMBOLS or symbol == "Pending activity"


def kind_of(action: str) -> str:
    a = action.upper()
    return next((k for prefix, k in KINDS if a.startswith(prefix)), "other")


def _rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Run Date")), None)
    if start is None:
        return []
    return [{k: (v or "").strip() for k, v in r.items() if k}
            for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))
            if re.match(r"\d\d/\d\d/\d{4}$", (r.get("Run Date") or "").strip())]


def load_history(folder: Path, pattern: str, accounts) -> tuple[list[Tx], list[dict]]:
    """Transactions in the given accounts from every activity export, each counted once even when exports overlap
    (a row repeated within one file, like two identical buys, is kept as often as that file has it)."""
    best: dict[tuple, int] = {}
    files = []
    for path in sorted(folder.glob(pattern)):
        rows = _rows(path)
        if not rows:
            continue
        dates = [dt.datetime.strptime(r["Run Date"], "%m/%d/%Y").date() for r in rows]
        m = re.search(r"Date downloaded (\d\d/\d\d/\d{4})", path.read_text(encoding="utf-8-sig", errors="replace"))
        files.append({"name": path.name, "first": min(dates), "last": max(dates), "rows": len(rows),
                      "downloaded": dt.datetime.strptime(m.group(1), "%m/%d/%Y").date() if m else None})
        counts: dict[tuple, int] = defaultdict(int)
        for r in rows:
            counts[tuple(sorted(r.items()))] += 1
        for k, n in counts.items():
            best[k] = max(best.get(k, 0), n)
    txs = []
    for k, n in best.items():
        r = dict(k)
        if not counted(r.get("Account"), accounts):
            continue
        tx = Tx(dt.datetime.strptime(r["Run Date"], "%m/%d/%Y").date(), r["Account"], kind_of(r.get("Action", "")),
                r.get("Symbol", "").strip(), _num(r.get("Quantity")), _num(r.get("Amount ($)")), r.get("Action", ""))
        if is_cash(tx.symbol) and tx.kind in ("buy", "sell", "reinvest"):
            tx.kind = "cash"  # money-market purchases and reinvested interest stay cash
        if tx.kind == "sell" and tx.qty > 0:
            tx.qty = -tx.qty  # a T-bill's redemption at maturity is a sale; its quantity's sign isn't guaranteed
        txs += [tx] * n
    txs.sort(key=lambda t: t.date)
    return txs, sorted(files, key=lambda f: f["first"])


def coverage(files: list[dict], max_gap_days: int = 7) -> tuple[dt.date | None, list[tuple[dt.date, dt.date]]]:
    """(the earliest date from which the history is continuous up to the newest file, gaps found). An export only
    shows days with activity, so a gap counts only when the files leave more than max_gap_days uncovered."""
    if not files:
        return None, []
    spans = sorted((f["first"], f["last"]) for f in files)
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if (a - merged[-1][1]).days > max_gap_days:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
    return merged[-1][0], gaps


def load_positions(path: Path, accounts) -> dict[str, dict]:
    """symbol -> {shares, price, value, basis, name} over the given accounts (cash and money-market funds excluded).
    basis is Fidelity's cost basis total, None when a row has none. price is per share: value / shares, since a bond's
    or T-bill's "Last price" is per $100 of face value (97.99 for a bill worth $0.9799)."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not counted(r.get("Account name"), accounts):
                continue
            sym = (r.get("Symbol") or "").strip()
            if is_cash(sym):
                continue
            d = out.setdefault(sym, {"shares": 0.0, "price": _num(r.get("Last price")), "value": 0.0, "basis": 0.0,
                                     "name": (r.get("Description") or "").strip().title()})
            d["shares"] += _num(r.get("Quantity"))
            d["value"] += _num(r.get("Current value"))
            basis = (r.get("Cost basis total") or "").strip()
            d["basis"] = d["basis"] + _num(basis) if d["basis"] is not None and basis not in ("", "--") else None
    for d in out.values():
        if d["shares"] and d["value"]:
            d["price"] = d["value"] / d["shares"]
    return out


def load_cash(path: Path, accounts) -> tuple[float, float]:
    """(cash, pending): cash and money-market balances in the given accounts, after "Pending activity" (trades not yet
    settled: their shares are already in the positions, their cash not yet taken out), and that pending amount."""
    cash = pending = 0.0
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sym = (r.get("Symbol") or "").strip()
            if counted(r.get("Account name"), accounts) and sym and is_cash(sym):
                cash += _num(r.get("Current value"))
                pending += _num(r.get("Current value")) if sym == "Pending activity" else 0.0
    return cash, pending


def net_transfers(txs: list[Tx]) -> list[Tx]:
    """Share moves between the counted accounts cancel out; what remains of a day's transfers of one symbol came
    from or went to an account outside them. Cash transfers are dropped (cash is not tracked)."""
    out, moves = [], defaultdict(list)
    for t in txs:
        if t.kind == "transfer":
            if not is_cash(t.symbol) and t.qty:
                moves[(t.date, t.symbol)].append(t)
        else:
            out.append(t)
    for (day, sym), ts in moves.items():
        qty = sum(t.qty for t in ts)
        if abs(qty) > 1e-6:
            value = abs(sum(t.amount for t in ts if (t.qty > 0) == (qty > 0)))
            out.append(Tx(day, ts[0].account, "transfer", sym, qty, value if qty > 0 else -value, ts[0].action))
    return sorted(out, key=lambda t: t.date)


def share_change(t: Tx) -> float:
    return t.qty if t.kind in ("buy", "espp", "sell", "reinvest", "vest", "received", "split", "transfer") else 0.0


def months_back(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.year * 12 + d.month - 1 - n, 12)
    return dt.date(y, m + 1, min(d.day, calendar.monthrange(y, m + 1)[1]))


def period_start(end: dt.date, period: str) -> dt.date:
    return {"1M": months_back(end, 1), "3M": months_back(end, 3), "6M": months_back(end, 6),
            "YTD": dt.date(end.year - 1, 12, 31), "1Y": months_back(end, 12)}[period]


class Prices:
    """Stored daily closes. A fund Yahoo doesn't list falls back to the prices in the owner's own trades and the
    positions export (nearest one, earlier preferred); such symbols are reported as approximate."""

    def __init__(self, store, fallback: dict[str, list[tuple[str, float]]]):
        self.store, self._cache, self._days, self.fallback, self.approximate = store, {}, {}, fallback, set()

    def series(self, sym: str) -> list[tuple[str, float]]:
        if sym not in self._cache:
            rows = [(r[0], r[1]) for r in self.store.db.execute(
                "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date", (sym,))]
            self._cache[sym] = rows
            self._days[sym] = [r[0] for r in rows]
        return self._cache[sym]

    def on(self, sym: str, d: dt.date, slack_days: int = 6) -> float | None:
        """Close on d, or on the last trading day before it (within slack_days)."""
        s, key = self.series(sym), d.isoformat()
        i = bisect.bisect_right(self._days[sym], key)
        best = s[i - 1] if i else None
        if best and (d - dt.date.fromisoformat(best[0])).days <= slack_days:
            return best[1]
        if not s and self.fallback.get(sym):
            points = sorted(self.fallback[sym])
            earlier = [c for day, c in points if day <= key]
            self.approximate.add(sym)
            return earlier[-1] if earlier else points[0][1]
        return None

    def date_on(self, sym: str, d: dt.date, slack_days: int = 6) -> dt.date | None:
        """The day of the close on() uses for d (None when it uses none, or a fallback price)."""
        s = self.series(sym)
        i = bisect.bisect_right(self._days[sym], d.isoformat())
        day = dt.date.fromisoformat(s[i - 1][0]) if i else None
        return day if day and (d - day).days <= slack_days else None

    def last_date(self, sym: str) -> dt.date | None:
        s = self.series(sym)
        return dt.date.fromisoformat(s[-1][0]) if s else None

    def before(self, sym: str, d: dt.date) -> float | None:
        """The close of the trading day before d's close (for a day's change)."""
        s = self.series(sym)
        i = bisect.bisect_right(self._days[sym], d.isoformat())
        return s[i - 2][1] if i >= 2 else None

    def days(self, sym: str, start: dt.date, end: dt.date) -> list[dt.date]:
        return [dt.date.fromisoformat(day) for day, _ in self.series(sym) if start.isoformat() <= day <= end.isoformat()]


class Returns:
    """Everything the page needs, computed from the files and stored closes."""

    def __init__(self, cfg: dict, store):
        r = cfg.get("returns", {})
        self.cfg, self.store = cfg, store
        self.accounts = set(r.get("accounts", DEFAULT_ACCOUNTS))
        root = cfg["paths"].get("home", cfg["paths"]["root"])  # the person's folder
        pattern = r.get("history_csv", "fidelity/Accounts_History*.csv")
        folder = (root / pattern).parent
        raw, self.files = load_history(folder, Path(pattern).name, self.accounts)
        self.txs = net_transfers(raw)
        self.covered_from, self.gaps = coverage(self.files)
        pos_pattern = cfg.get("holdings_csv", "fidelity/Portfolio_Positions_*.csv")
        files = sorted((root / pos_pattern).parent.glob(Path(pos_pattern).name),
                       key=lambda p: (_export_date(p) or dt.date.min, p.stat().st_mtime))
        self.positions_file = files[-1] if files else None
        self.positions_date = (_export_date(self.positions_file) if self.positions_file else None)
        self.positions = load_positions(self.positions_file, self.accounts) if self.positions_file else {}
        self.splits: dict[str, list[tuple[dt.date, float]]] = defaultdict(list)  # symbol -> (day, new/old shares)
        for t in self.txs:  # Fidelity books a stock split as a DISTRIBUTION of the extra shares, valued at market
            held = self.shares_on(t.date - dt.timedelta(days=1)).get(t.symbol, 0) if t.kind == "received" else 0
            if held > 1e-6:
                t.kind = "split"
                self.splits[t.symbol].append((t.date, (held + t.qty) / held))
        fallback = defaultdict(list)
        for t in self.txs:
            if t.kind in ("buy", "sell", "reinvest") and t.qty and t.amount:
                fallback[t.symbol].append((t.date.isoformat(), abs(t.amount / t.qty)))
        for sym, v in self.positions.items():
            if v["price"] and self.positions_date:
                fallback[sym].append((self.positions_date.isoformat(), v["price"]))
        self.prices = Prices(store, fallback)
        self.names = {s: v["name"] for s, v in self.positions.items() if v.get("name")}

    def symbols(self) -> list[str]:
        """Every security held or traded in the counted accounts, plus the benchmark."""
        syms = {t.symbol for t in self.txs if t.symbol and not is_cash(t.symbol) and share_change(t)} | set(self.positions)
        return sorted(syms) + [BENCHMARK]

    def shares_on(self, d: dt.date) -> dict[str, float]:
        """Shares held at the close of d. The positions export is taken to include its own day's activity."""
        p = self.positions_date
        out = defaultdict(float, {s: v["shares"] for s, v in self.positions.items()})
        for t in self.txs:
            q = share_change(t)
            if q and d < t.date <= p:
                out[t.symbol] -= q
            elif q and p < t.date <= d:
                out[t.symbol] += q
        return {s: q for s, q in out.items() if abs(q) > 1e-6}

    def period_start(self, period: str) -> dt.date:
        return period_start(self.end_date(), period)

    def history_behind(self) -> dt.date | None:
        """The day the activity history runs to, when the positions export is newer (its later buys and sales are
        missing). An export lists only days with activity, so a file counts up to the day it was downloaded."""
        if not self.files or not self.positions_date:
            return None
        through = max(max(f["last"], f["downloaded"] or f["last"]) for f in self.files)
        return through if through < self.positions_date else None

    def end_date(self) -> dt.date | None:
        return self.prices.last_date(BENCHMARK)

    def available(self, period: str) -> bool:
        end = self.end_date()
        return bool(end and self.covered_from and period_start(end, period) >= self.covered_from - dt.timedelta(days=1))

    def price(self, sym: str, d: dt.date) -> float | None:
        """The price of one share as held on d. Yahoo's closes are split-adjusted (CRWD's before its 4-for-1 split are
        a quarter of what they traded at), so a close from before a split is scaled back up to match the share count
        of that day. Fallback prices (the owner's trades, the export) are real prices already."""
        px = self.prices.on(sym, d)
        if px is None or not self.prices.series(sym):
            return px
        for day, ratio in self.splits.get(sym, ()):
            if d < day:
                px *= ratio
        return px

    def value(self, shares: dict[str, float], d: dt.date) -> tuple[float, list[str]]:
        total, missing = 0.0, []
        for s, q in shares.items():
            px = self.price(s, d)
            if px is None:
                missing.append(s)
            else:
                total += q * px
        return total, missing

    def _flow(self, t: Tx) -> float | None:
        """Money moved into the investments by a transaction (negative: taken out), None if it moves none."""
        if t.kind in ("buy", "espp", "reinvest", "sell"):
            return -t.amount
        if t.kind in ("vest", "received"):
            return t.amount
        if t.kind == "transfer":
            value = abs(t.amount) or abs(t.qty) * (self.price(t.symbol, t.date) or 0)
            return value if t.qty > 0 else -value
        return None

    def span(self, start: dt.date, end: dt.date) -> dict:
        """Value at both ends, money moved in, dividends and what was earned between the closes of start and end."""
        v0, miss0 = self.value(self.shares_on(start), start)
        v1, miss1 = self.value(self.shares_on(end), end)
        flows, income, money_in = [], 0.0, defaultdict(float)
        for t in (t for t in self.txs if start < t.date <= end):
            f = self._flow(t)
            if f is not None:
                flows.append((t.date, f))
            elif t.kind == "income":
                income += t.amount
            if t.kind == "espp":
                money_in["ESPP"] += -t.amount
            elif t.kind in ("vest", "received"):
                money_in["vested shares"] += t.amount
            elif t.kind == "bank" and t.amount > 0:
                money_in["from your bank"] += t.amount
            elif t.kind == "bank":
                money_in["to your bank"] += t.amount
        net_flow = sum(f for _, f in flows)
        earned = v1 - v0 - net_flow + income
        days = max((end - start).days, 1)
        base = v0 + sum(f * (end - d).days / days for d, f in flows)  # Modified Dietz
        bench = (self.prices.on(BENCHMARK, end) or 0) / (self.prices.on(BENCHMARK, start) or 1) - 1
        return {"start": start, "end": end, "v0": v0, "v1": v1, "net_flow": net_flow, "income": income,
                "earned": earned, "ret": earned / base if base > 0 else None, "bench": bench,
                "money_in": dict(money_in), "missing": sorted(set(miss0 + miss1)),
                "bought": sum(f for _, f in flows if f > 0), "sold": -sum(f for _, f in flows if f < 0)}

    def period(self, period: str) -> dict:
        end = self.end_date()
        x = self.span(period_start(end, period), end)
        lots = self.new_money(x["start"], end)
        x.update(period=period, label=PERIODS[period], lots=lots, approximate=sorted(self.prices.approximate),
                 **summarize(lots))
        x["from_new"] = x["gain"]
        x["from_existing"] = x["earned"] - x["gain"] - x["income"]
        x["invested"], x["now"] = x["cost"], x["value"]
        return x

    def monthly(self, start: dt.date, end: dt.date) -> list[dict]:
        """span() for each calendar month between start and end (the first and last may be partial), newest first."""
        month_end = lambda d: dt.date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
        out, a = [], start
        while a < end:
            b = min(end, month_end(a + dt.timedelta(days=1)))  # from a month-end close to the next month end
            x = self.span(a, b)
            x["label"] = b.strftime("%b %Y")
            out.append(x)
            a = b
        return out[::-1]

    def daily(self, start: dt.date, end: dt.date) -> list[dict]:
        """Each trading day: value of the investments, starting value plus money put in since, and the same money in
        the S&P 500 (bought and sold on the same days)."""
        days = [start] + self.prices.days(BENCHMARK, start + dt.timedelta(days=1), end)  # start: its (last) close
        if len(days) < 2:
            return []
        shares = defaultdict(float, self.shares_on(days[0]))
        later = [t for t in self.txs if days[0] < t.date <= end]
        v0 = self.value(shares, days[0])[0]
        b0 = self.prices.on(BENCHMARK, days[0])
        put_in, units, i, out = v0, v0 / b0, 0, []
        for d in days:
            while i < len(later) and later[i].date <= d:
                t = later[i]
                shares[t.symbol] += share_change(t)
                f = self._flow(t)
                if f:
                    put_in += f
                    units += f / (self.prices.on(BENCHMARK, t.date) or b0)
                i += 1
            out.append({"date": d, "value": self.value(shares, d)[0], "put_in": put_in,
                        "bench": units * self.prices.on(BENCHMARK, d)})
        return out

    def holdings_now(self) -> list[dict]:
        """Fidelity's positions view for the counted accounts at each holding's newest stored price (the latest close,
        or a later price fetched with a stock page's Refresh): value, the change since the close before it (day_date: the
        day of that price, older for a fund whose newest price isn't out yet), and gain vs cost basis (the export's basis, moved by any buys and sales after it; sales take average cost)."""
        end, p = self.end_date(), self.positions_date
        basis = {s: v["basis"] for s, v in self.positions.items()}
        shares = defaultdict(float, {s: v["shares"] for s, v in self.positions.items()})
        for t in self.txs:
            if not (p < t.date <= end) or not share_change(t):
                continue
            b = basis.get(t.symbol, 0.0)
            if b is not None and t.qty > 0:
                basis[t.symbol] = b + (self._flow(t) or 0.0)
            elif b is not None and shares[t.symbol] > 0:
                basis[t.symbol] = b * (1 + t.qty / shares[t.symbol])
            shares[t.symbol] += t.qty
        out = []
        for s, q in shares.items():
            if q <= 1e-6:
                continue
            last = self.prices.last_date(s)
            when = last if last and last > end else end
            px = self.prices.on(s, when) or self.positions.get(s, {}).get("price")
            day_date = self.prices.date_on(s, when)
            prev = self.prices.before(s, when) if day_date else None
            value = q * px if px else self.positions.get(s, {}).get("value", 0.0)
            day = q * (px - prev) if px and prev else None
            if day is not None:  # shares bought that day gained or lost from what was paid, not from the close before
                new = [(t.qty, self._flow(t)) for t in self.txs
                       if t.symbol == s and t.date == day_date and t.qty > 0 and self._flow(t)]
                new_q, new_cost = sum(n for n, _ in new), sum(c for _, c in new)
                if new_q > q:  # some of them were sold again the same day
                    new_q, new_cost = q, new_cost * q / new_q
                day = (q - new_q) * (px - prev) + new_q * px - new_cost
            b = basis.get(s)
            out.append({"symbol": s, "name": self.names.get(s, ""), "shares": q, "price": px, "value": value,
                        "day": day,
                        "day_pct": px / prev - 1 if px and prev else None, "day_date": day_date if prev else None,
                        "basis": b, "avg_cost": b / q if b else None,
                        "gain": value - b if b is not None else None,
                        "gain_pct": value / b - 1 if b else None,
                        "approximate": not self.prices.series(s), "newer": when if when > end else None})
        total = sum(r["value"] for r in out) or 1
        for r in out:
            r["weight"] = r["value"] / total
        return out

    def window(self, first: dt.date, last: dt.date) -> list[dict]:
        """Every investment made between first and last (inclusive), valued at the latest close."""
        start = max(first - dt.timedelta(days=1), self.covered_from - dt.timedelta(days=1))
        return [lot for lot in self.new_money(start, self.end_date()) if lot["date"] <= last]

    def new_money(self, start: dt.date, end: dt.date) -> list[dict]:
        """Each investment made in (start, end]: buys, ESPP purchases, vested or received shares. Later sales come
        out of the oldest shares first (shares held before the period, then lots in date order)."""
        queues: dict[str, list[Lot]] = defaultdict(list)
        for s, q in self.shares_on(start).items():
            queues[s].append(Lot(start, s, "held", q, 0.0))
        lots = []
        for t in (t for t in self.txs if start < t.date <= end):
            if t.kind in ("buy", "espp", "reinvest", "vest", "received") or (t.kind == "transfer" and t.qty > 0):
                cost = -t.amount if t.kind in ("buy", "espp", "reinvest") else abs(t.amount)
                if t.kind == "transfer" and not cost:
                    cost = t.qty * (self.price(t.symbol, t.date) or 0)
                lot = Lot(t.date, t.symbol, t.kind, t.qty, cost)
                queues[t.symbol].append(lot)
                if t.kind != "reinvest":
                    lots.append(lot)
            elif t.kind == "split":  # more shares, same cost: grow each open lot, and restate earlier sales in new shares
                held = sum(lot.shares - sum(x[1] for x in lot.sold) for lot in queues[t.symbol])
                ratio = (held + t.qty) / held if held > 1e-9 else 1.0
                for lot in queues[t.symbol]:
                    lot.shares *= ratio
                    lot.sold = [(d, q * ratio, v) for d, q, v in lot.sold]
            elif t.kind == "sell" or (t.kind == "transfer" and t.qty < 0):
                left = -t.qty
                per_share = abs(t.amount) / -t.qty if t.qty else 0
                if t.kind == "transfer" and not per_share:
                    per_share = self.price(t.symbol, t.date) or 0
                for lot in queues[t.symbol]:
                    have = lot.shares - sum(x[1] for x in lot.sold)
                    take = min(have, left)
                    if take > 1e-9:
                        lot.sold.append((t.date, take, take * per_share))
                        left -= take
                    if left <= 1e-9:
                        break
        b_end = self.prices.on(BENCHMARK, end)
        out = []
        for lot in lots:
            px = self.prices.on(lot.symbol, end)
            sold_shares = sum(x[1] for x in lot.sold)
            remaining = lot.shares - sold_shares
            value = sum(x[2] for x in lot.sold) + (remaining * px if px is not None else 0)
            b_start = self.prices.on(BENCHMARK, lot.date)
            bench_value = None
            if b_start and b_end and lot.shares:
                units = lot.cost / b_start
                bench_value = sum(units * x[1] / lot.shares * (self.prices.on(BENCHMARK, x[0]) or b_end)
                                  for x in lot.sold) + units * remaining / lot.shares * b_end
            gain = value - lot.cost
            pct = gain / lot.cost if lot.cost else None
            bench_pct = bench_value / lot.cost - 1 if bench_value is not None and lot.cost else None
            out.append({"date": lot.date, "symbol": lot.symbol, "kind": NEW_MONEY.get(lot.kind, lot.kind),
                        "shares": lot.shares, "cost": lot.cost, "value": value, "gain": gain, "pct": pct,
                        "sold_pct": sold_shares / lot.shares if lot.shares else 0, "priced": px is not None,
                        "bench_pct": bench_pct,
                        "bench_gain": (bench_value - lot.cost) if bench_value is not None else 0.0,
                        "vs": (pct - bench_pct) if pct is not None and bench_pct is not None else None,
                        "days": (end - lot.date).days})
        return out


    def check_against(self, path: Path) -> list[str]:
        """Shares rebuilt for an older positions export vs what it says: mismatches mean a missing or misread
        transaction."""
        d = _export_date(path)
        old = load_positions(path, self.accounts)
        rebuilt = self.shares_on(d)
        return [f"{s}: export {old.get(s, {}).get('shares', 0):g}, rebuilt {rebuilt.get(s, 0):g}"
                for s in sorted(set(old) | set(rebuilt))
                if abs(old.get(s, {}).get("shares", 0) - rebuilt.get(s, 0)) > 1e-3]


def by_stock(lots: list[dict]) -> list[dict]:
    """New investments grouped per stock (same-day partial fills merged into one buy), best vs the S&P first."""
    groups: dict[str, dict] = {}
    for l in lots:
        g = groups.setdefault(l["symbol"], {"symbol": l["symbol"], "cost": 0.0, "value": 0.0, "bench_gain": 0.0,
                                            "buys": {}, "priced": True, "lots": []})
        g["lots"].append(l)
        g["cost"] += l["cost"]
        g["value"] += l["value"]
        g["bench_gain"] += l["bench_gain"]
        g["priced"] &= l["priced"]
        b = g["buys"].setdefault((l["date"], l["kind"]), {"date": l["date"], "kind": l["kind"], "cost": 0.0,
                                                           "value": 0.0, "bench_gain": 0.0, "days": l["days"]})
        for k in ("cost", "value", "bench_gain"):
            b[k] += l[k]
    out = []
    for g in groups.values():
        buys = sorted(g.pop("buys").values(), key=lambda b: b["date"], reverse=True)
        for b in buys:
            _rates(b)
        s = summarize(g.pop("lots"))
        g.update(buys=buys, n=len(buys), newest_days=min(b["days"] for b in buys), settled=s["settled"],
                 fresh=s["fresh"])
        out.append(_rates(g))
    key = lambda g: g["settled"]["vs"] if g["settled"]["vs"] is not None else (g["vs"] if g["vs"] is not None else -9)
    return sorted(out, key=lambda g: (g["settled"]["n"] == 0, -key(g)))


def _rates(d: dict) -> dict:
    d["gain"] = d["value"] - d["cost"]
    d["pct"] = d["gain"] / d["cost"] if d["cost"] else None
    d["bench_pct"] = d["bench_gain"] / d["cost"] if d["cost"] else None
    d["vs"] = d["pct"] - d["bench_pct"] if d["pct"] is not None and d["bench_pct"] is not None else None
    return d


def summarize(lots: list[dict]) -> dict:
    """Totals over lots, plus the same for the settled ones only (at least FRESH_DAYS old): their % and S&P
    comparison aren't pulled toward zero by money invested days ago."""
    def total(ls):
        return _rates({"cost": sum(l["cost"] for l in ls), "value": sum(l["value"] for l in ls),
                       "bench_gain": sum(l["bench_gain"] for l in ls), "n": len(ls),
                       "beat": sum(1 for l in ls if l["vs"] is not None and l["vs"] > 0)})
    settled = [l for l in lots if l["days"] >= FRESH_DAYS]
    fresh = [l for l in lots if l["days"] < FRESH_DAYS]
    return {**total(lots), "settled": total(settled), "fresh": total(fresh)}


def by_month(lots: list[dict]) -> list[dict]:
    """Investments grouped by the month they were made, oldest first."""
    groups = defaultdict(list)
    for l in lots:
        groups[l["date"].replace(day=1)].append(l)
    return [{"month": m, "label": m.strftime("%b %Y"), **summarize(ls)} for m, ls in sorted(groups.items())]


def concentration(cfg: dict, rows: list[dict]) -> dict:
    """Largest holding and each [returns] groups entry as a share of the counted accounts."""
    total = sum(r["value"] for r in rows) or 1
    value = {r["symbol"]: r["value"] for r in rows}
    top = max(rows, key=lambda r: r["value"], default=None)
    groups = [(name, sum(value.get(s, 0.0) for s in syms) / total)
              for name, syms in cfg.get("returns", {}).get("groups", {}).items() if not top or syms != [top["symbol"]]]
    return {"top": (top["symbol"], top["value"] / total) if top else None,
            "groups": sorted(((n, w) for n, w in groups if w > 0), key=lambda g: -g[1])}


def context_lines(cfg: dict, store) -> list[str]:
    """For Ask: what the owner earned and how their new investments did, over 3 months and 1 year."""
    from .askfmt import returns_line

    try:
        r = Returns(cfg, store)
        if not r.files or not r.end_date():
            return []
    except Exception:
        return []
    lines = ["YOUR INVESTING (the accounts counted for returns; price gains plus dividends):"]
    for per in ("1M", "3M", "YTD", "1Y"):
        if not r.available(per):
            continue
        x = r.period(per)
        lines.append(returns_line(x["label"], x["start"], x["end"], x))
        if per == "3M":
            groups = by_stock(x["lots"])
            fmt = lambda g: (f"{g['symbol']} {g['pct']:+.0%} vs S&P {g['bench_pct']:+.0%} "
                             f"(${g['cost']:,.0f} invested)")
            head = "  New investments of the last 3 months, {} vs the same dollars in the S&P 500: "
            lines.append(head.format("best") + "; ".join(fmt(g) for g in groups[:3] if g["vs"] is not None))
            lines.append(head.format("worst") + "; ".join(fmt(g) for g in groups[::-1][:3]
                                                           if g["vs"] is not None))
    return lines if len(lines) > 1 else []
