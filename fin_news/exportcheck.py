"""Look at Fidelity exports without showing what is in them.

check():    the structure of each export (columns, kinds of values, accounts as A/B/C, symbols, rows the app can't
            read) and nothing else: no amounts, quantities, account names or numbers.
scramble(): a copy in the same format with account names and numbers replaced and every quantity and dollar amount
            multiplied by a random factor per stock (so totals mean nothing); prices, dates and symbols are kept, so a
            parsing bug can be reproduced on it. A leak check refuses to write a copy in which an original amount or
            account still appears."""
import csv
import io
import random
import re
import string
from collections import Counter
from pathlib import Path

from .holdings import MONEY_MARKET, TICKER, _is_fund, load_holdings
from .returns import KINDS, _rows, counted, is_cash, kind_of

NUMBER = re.compile(r"^(?P<s1>[+-]?)(?P<dollar>\$?)(?P<s2>[+-]?)(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)(?P<pct>%?)$")
DATE = re.compile(r"^\d\d/\d\d/\d{4}$")
APP_COLUMNS = {  # exact names holdings.py and returns.py read
    "positions": ("Account name", "Symbol", "Description", "Quantity", "Last price", "Current value", "Cost basis total"),
    "history": ("Run Date", "Account", "Action", "Symbol", "Quantity", "Amount ($)"),
}
NUMERIC = {"quantity", "last price", "last price change", "current value", "today's gain/loss dollar",
           "total gain/loss dollar", "cost basis total", "average cost basis", "price ($)", "commission ($)",
           "fees ($)", "accrued interest ($)", "amount ($)", "cash balance ($)", "exchange quantity"}
PRIVATE_KINDS = {"transfer", "bank", "espp_cash", "other"}
KEEP_DIGITS = re.compile(r"\d\d/\d\d/\d{4}|\d{1,2}:\d\d(?::\d\d)?")  # dates and times inside text


def kind_of_export(text: str) -> str | None:
    """'positions' or 'history' from the header row (Fidelity's Positions and Activity & Orders CSV exports)."""
    return _header(text.splitlines())[0]


def _header(lines: list[str]) -> tuple[str | None, int]:
    for i, line in enumerate(lines[:10]):
        cols = {c.strip().strip('"').lstrip("\ufeff") for c in line.split(",")}
        if {"Account Name", "Symbol", "Quantity"} <= {c.title() for c in cols} and any(c.lower() == "current value" for c in cols):
            return "positions", i
        if {"Run Date", "Action", "Symbol"} <= cols:
            return "history", i
    return None, -1


def _shape(v: str) -> str:
    v = v.strip()
    if not v:
        return "blank"
    if v == "--":
        return "--"
    if DATE.match(v):
        return "date"
    m = NUMBER.match(v)
    if m:
        return "percent" if m["pct"] else "money" if m["dollar"] else "number"
    return "text"


def _mask(v: str, n: int = 24) -> str:
    """The look of a text value without its content: letters -> a, digits -> 9."""
    s = re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", v.strip()))
    return s[:n] + ("…" if len(s) > n else "")


def _is_data(kind: str, row: dict) -> bool:
    if kind == "history":
        return bool(DATE.match((row.get("Run Date") or "").strip()))
    return sum(1 for v in row.values() if isinstance(v, str) and v.strip()) > 3


def _col(row: dict, name: str) -> str:
    """A column by name, ignoring case (Fidelity has used both 'Account Name' and 'Account name')."""
    for k, v in row.items():
        if k and k.strip().lower() == name:
            return (v or "").strip()
    return ""


def _symbol_group(sym: str, description: str) -> str:
    s = sym.rstrip("*").upper()
    if not sym:
        return "blank"
    if sym == "Pending activity":
        return "pending"
    if is_cash(sym) or s in MONEY_MARKET or "MONEY MARKET" in description.upper():
        return "cash"
    if _is_fund(s, description.upper()):
        return "fund"
    if TICKER.match(s):
        return "stock"
    return "other"


# ---------------------------------------------------------------------------------------------------- check


def check(folder: Path, accounts=(), labels: dict | None = None) -> list[str]:
    """Report lines for every file in a person's export folder. accounts: their settings (which accounts count)."""
    labels = {} if labels is None else labels
    if not folder.is_dir():
        return [f"{folder.name}/: no folder"]
    files = sorted(p for p in folder.iterdir() if p.is_file())
    out = [f"{len(files)} file(s)"]
    for p in files:
        out += _check_file(p, accounts, labels)
    return out


def _label(labels: dict, key: tuple) -> str:
    if key not in labels:
        n = len(labels)
        labels[key] = string.ascii_uppercase[n] if n < 26 else f"#{n + 1}"
    return labels[key]


def _check_file(path: Path, accounts, labels: dict) -> list[str]:
    head = f"\n{path.name}"
    if path.suffix.lower() != ".csv":
        return [f"{head}: not a CSV (ignored by the app)"]
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return [f"{head}: not UTF-8 text (the app reads it with replacement characters)"]
    lines = text.splitlines()
    kind, at = _header(lines)
    if kind is None:
        return [f"{head}: no Fidelity header in the first 10 lines (the app skips it)", f"  {len(lines)} lines"]
    reader = csv.DictReader(io.StringIO("\n".join(lines[at:])))
    cols = [c for c in (reader.fieldnames or []) if c]
    rows, other = [], 0
    for r in reader:
        if _is_data(kind, r):
            rows.append(r)
        else:
            other += 1
    out = [f"{head}: {kind} export",
           (f"  header on line {at + 1}, {len(cols)} columns; {len(rows)} data rows, {other} other lines "
            f"(notes, blank lines), {at} lines before the header")]

    # Accounts, as letters
    name_col = "account name" if kind == "positions" else "account"
    per: Counter = Counter()
    for r in rows:
        per[(_col(r, name_col), _col(r, "account number"))] += 1
    out.append("  accounts: " + (", ".join(
        f"{_label(labels, k)} {n} rows ({'counted' if counted(k[0], accounts) else 'NOT counted'} under your settings)"
        for k, n in per.items()) or "none"))

    # Symbols
    groups: dict[str, list[str]] = {}
    for r in rows:
        sym, desc = _col(r, "symbol"), _col(r, "description")
        groups.setdefault(_symbol_group(sym, desc), []).append(sym)
    parts = [f"{len(set(v))} {g}" for g, v in groups.items() if g not in ("other",)]
    out.append("  symbols: " + ", ".join(parts))
    if groups.get("other"):
        out.append("  unusual symbols (the app skips these rows): " + ", ".join(sorted({_mask(s) for s in groups["other"]})))

    # Columns: what kinds of values each holds
    problems = []
    for c in cols:
        shapes = Counter(_shape(r.get(c) or "") for r in rows)
        out.append(f"  {c}: " + ", ".join(f"{k} {n}" for k, n in shapes.most_common()))
        numeric = sum(shapes[k] for k in ("number", "money", "percent"))
        if numeric and shapes["text"] and c.strip().lower() in NUMERIC:
            bad = [(i + 1, _col(r, "symbol") or "no symbol", _mask(r.get(c) or ""))
                   for i, r in enumerate(rows) if _shape(r.get(c) or "") == "text"]
            looks = sorted({m for _, _, m in bad})
            problems.append(f"{c}: text where numbers are expected (read as 0) in {len(bad)} rows: "
                            + ", ".join(f"row {i} ({sym})" for i, sym, _ in bad[:8]) + (" …" if len(bad) > 8 else "")
                            + "; it looks like " + ", ".join(f"'{m}'" for m in looks[:5]))
    lower = {x.strip().lower(): x for x in cols}
    for c in APP_COLUMNS[kind]:  # the parsers look columns up by exact name
        if c in cols:
            continue
        problems.append(f"the app reads column '{c}', but this file " +
                        (f"spells it '{lower[c.lower()]}' (read as empty)" if c.lower() in lower else "lacks it"))

    # What the app makes of it
    if kind == "positions":
        h = load_holdings(path, [])
        used = any(e["value"] for e in (*h.stocks.values(), *h.funds.values()))
        out.append(f"  the app reads: {len(h.stocks)} stocks, {len(h.funds)} funds; "
                   + ("has positions" if used else "NO positions with a value (the app skips this file)"))
    else:
        actions = Counter(kind_of(r.get("Action", "")) for r in _rows(path))
        out.append("  actions: " + ", ".join(f"{k} {n}" for k, n in actions.most_common()))
        names = sorted(((k[0], labels[k]) for k in per if k[0]), key=lambda x: -len(x[0]))

        def hide(action: str) -> str:
            """Only the first two words: the rest can name a person, a merchant or an account."""
            for name, letter in names:
                action = action.replace(name, f"Account {letter}")
            words = re.sub(r"\d", "#", action).split()
            return " ".join(words[:2]) + (" …" if len(words) > 2 else "")

        unknown = sorted({hide(_col(r, "action")) for r in rows if kind_of(_col(r, "action")) == "other"})
        if unknown:
            out.append("  actions the app doesn't know (counted as 'other'):")
            out += [f"    {a}" for a in unknown[:15]] + ([f"    … and {len(unknown) - 15} more"] if len(unknown) > 15 else [])
        dates = sorted(_col(r, "run date") for r in rows)
        if dates:
            out.append(f"  dates: {len(set(dates))} different days" + ("" if "Date downloaded" in text else
                                                                       "; no 'Date downloaded' line"))
    out.append("  problems: " + ("none" if not problems else ""))
    out += [f"    {p}" for p in problems[:20]] + ([f"    … and {len(problems) - 20} more"] if len(problems) > 20 else [])
    return out


# ------------------------------------------------------------------------------------------------- scramble


class Leak(RuntimeError):
    pass


def _norm(num: str) -> str:
    """'$1,234.50' -> '1234.5': how an amount would be recognised, whatever its formatting."""
    s = num.replace(",", "").lstrip("0") or "0"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _sig(norm: str) -> int:
    return len(norm.replace(".", "").lstrip("0"))


class _Scrambler:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.factors: dict[str, float] = {}
        self.names: dict[str, str] = {}    # real account name -> "Account A"
        self.numbers: dict[str, str] = {}  # real account number -> fake
        self.secrets: set[str] = set()     # normalised original amounts that must not appear in the copy
        self.keep: set[str] = set()        # normalised numbers that are allowed to (prices, dates)
        self.symbols: set[str] = set()
        self.unchanged = 0                 # amounts the scaling left as they were (must stay 0)
        self.free_text: list[str] = []     # scrambled free text, checked for original amounts

    def factor(self, key: str) -> float:
        if key not in self.factors:  # never close to 1, so no amount stays near its real value
            self.factors[key] = self.rng.choice((self.rng.uniform(0.15, 0.6), self.rng.uniform(1.7, 6.0)))
        return self.factors[key]

    def account_name(self, v: str) -> str:
        if v and v not in self.names:
            self.names[v] = f"Account {string.ascii_uppercase[len(self.names) % 26]}"
        return self.names.get(v, v)

    def account_number(self, v: str) -> str:
        if v and v not in self.numbers:
            self.numbers[v] = re.sub(r"\d", lambda _: str(self.rng.randrange(10)), v)
        return self.numbers.get(v, v)

    def scale(self, v: str, f: float, min_decimals: int = 0) -> str:
        """v times f, with v's formatting. min_decimals: share counts get 4, so rounding the scaled value (1 share
        x 2.3 -> "2") doesn't break the sums the share rebuild checks."""
        m = NUMBER.match(v.strip())
        if not m:
            return self.text(v)
        num = m["num"]
        if _sig(_norm(num)) >= 3:
            self.secrets.add(_norm(num))
        decimals = max(len(num.split(".")[1]) if "." in num else 0, min_decimals)
        new = float(num.replace(",", "")) * f
        body = f"{new:,.{decimals}f}" if "," in num else f"{new:.{decimals}f}"
        if _sig(_norm(num)) >= 3 and _norm(body) == _norm(num):
            self.unchanged += 1
        return f"{m['s1']}{m['dollar']}{m['s2']}{body}{m['pct']}"

    def kept(self, v: str) -> str:
        for n in re.findall(r"\d[\d,]*(?:\.\d+)?", v):
            self.keep.add(_norm(n))
        return v

    def action(self, v: str) -> str:
        """Trades and income name only public securities and keep their wording. Transfers, bank and card lines and
        unknown actions can name people, merchants or accounts: only the words the app recognises them by are kept."""
        k = kind_of(v)
        if k not in PRIVATE_KINDS:
            return self.text(v)
        lead = next((p for p, kk in KINDS if v.upper().startswith(p)), None)
        words = len(lead.split()) if lead else 2
        kept = " ".join(v.split()[:words])
        return f"{kept} REDACTED" + (" (Cash)" if v.rstrip().endswith("(Cash)") else "")

    def accounts_out(self, v: str) -> str:
        for real, fake in sorted(self.names.items(), key=lambda kv: -len(kv[0])):
            v = v.replace(real, fake)
        for real, fake in self.numbers.items():
            v = v.replace(real, fake)
        return v

    def text(self, v: str) -> str:
        """Free text (descriptions, actions, notes): account names and numbers replaced, and any other long number
        (5+ digits: an account or reference number) randomised; dates, times, symbols and short numbers in names
        ("500 INDEX") are kept."""
        v = self.accounts_out(v)
        v = re.sub(r"\$\d[\d,]*(?:\.\d+)?", lambda m: re.sub(r"\d", lambda _: str(self.rng.randrange(10)), m.group()), v)
        out, last = [], 0
        for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9.-]*", v):
            out.append(v[last:m.start()])
            tok = m.group()
            safe = (tok in self.symbols or any(tok.startswith(f) for f in self.numbers.values())
                    or sum(c.isdigit() for c in tok) < 5
                    or re.search(r"\d\.\d", tok))  # rates and prices ("0.00000%")
            out.append(tok if safe else re.sub(r"\d", lambda _: str(self.rng.randrange(10)), tok))
            last = m.end()
        out.append(v[last:])
        s = "".join(out)
        for m in KEEP_DIGITS.finditer(v):  # put dates and times back
            s = s[:m.start()] + m.group() + s[m.end():]
        self.free_text.append(s)
        return s

    def row(self, header: list[str], cells: list[str], kind: str) -> list[str]:
        names = [h.strip().lower() for h in header]
        rec = dict(zip(names, cells))
        sym = (rec.get("symbol") or "").strip()
        key = "cash" if is_cash(sym) else sym
        f = self.factor(key)
        out = []
        for name, v in zip(names, cells):
            if not v.strip() or v.strip() == "--":
                out.append(v)
            elif name == "account number":
                out.append(self.account_number(v))
            elif name in ("account name", "account"):
                out.append(self.account_name(v))
            elif name == "symbol":
                out.append(v)
            elif name == "description":  # the security's name: public, and fund themes are matched on it
                out.append(self.accounts_out(v) if sym or kind == "positions" else "No Description")
            elif name == "action":
                out.append(self.action(v))
            elif "date" in name or "price" in name or "average cost basis" in name or "exchange rate" in name:
                out.append(self.kept(v))  # per-share prices and dates: public or harmless, needed for debugging
            elif name == "percent of account":
                out.append(self.scale(v, self.rng.uniform(0.5, 1.5)))
            elif "percent" in name:
                out.append(self.kept(v))  # gain % is unchanged when value and cost scale together
            elif name.startswith("cash balance"):
                out.append(self.scale(v, self.factor("cash")))
            elif NUMBER.match(v.strip()):
                out.append(self.scale(v, f, 4 if "quantity" in name else 0))
            else:
                out.append(self.text(v))
        out += cells[len(names):]
        return out


def scramble(src: Path, dst: Path, rng: random.Random | None = None) -> dict:
    """Scrambled copies of every CSV in src, written to dst (replacing what was there). Returns
    {"files": n, "accounts": {real name: fake name}} — the mapping is for tests, never print it."""
    rng = rng or random.SystemRandom()
    files = sorted(src.glob("*.csv"))
    for attempt in range(10):
        s = _Scrambler(rng)
        texts = {}
        for p in files:  # symbols and account names first, so free text in any file can be cleaned
            texts[p] = p.read_bytes().decode("utf-8", errors="replace")  # BOM and line endings kept as they are
            lines = texts[p].splitlines()
            kind, at = _header(lines)
            if kind is None:
                continue
            for r in csv.DictReader(io.StringIO("\n".join(lines[at:]).lstrip("\ufeff"))):
                if _is_data(kind, r):
                    s.symbols.add(_col(r, "symbol"))
                    s.account_name(_col(r, "account name" if kind == "positions" else "account"))
                    s.account_number(_col(r, "account number"))
        s.symbols.discard("")
        out = {p: _scramble_text(texts[p], s) for p in files}
        try:
            _leak_check(s, out)
        except Leak:
            if attempt == 9:
                raise
            continue
        dst.mkdir(parents=True, exist_ok=True)
        dst.chmod(0o700)
        for old in dst.glob("*.csv"):
            old.unlink()
        for p, text in out.items():
            (dst / p.name).write_bytes(text.encode("utf-8"))
            (dst / p.name).chmod(0o600)
        return {"files": len(out), "accounts": dict(s.names)}
    raise Leak("unreachable")


def _scramble_text(text: str, s: _Scrambler) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    kind, at = _header(lines)
    out = []
    header: list[str] = []
    for i, line in enumerate(lines):
        if kind is None or i < at or not line.strip():
            out.append(s.text(line) if line.strip() else line)
            continue
        parsed = _split(line)
        cells = [v for v, _ in parsed]
        if i == at:
            header = [c.lstrip("\ufeff") for c in cells]
            out.append(line)
            continue
        row = dict(zip(header, cells))
        if not _is_data(kind, row):
            out.append(s.text(line))
            continue
        new = s.row(header, cells, kind)
        out.append(",".join(_quote(v, q) for v, (_, q) in zip(new, parsed)))
    return newline.join(out) + (newline if text.endswith(("\n", "\r")) else "")


def _split(line: str) -> list[tuple[str, bool]]:
    """CSV cells of one line, each with whether it was quoted, so the copy keeps the original quoting."""
    cells, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            j, buf = i + 1, []
            while j < n:
                if line[j] == '"' and j + 1 < n and line[j + 1] == '"':
                    buf.append('"')
                    j += 2
                elif line[j] == '"':
                    break
                else:
                    buf.append(line[j])
                    j += 1
            cells.append(("".join(buf), True))
            k = line.find(",", j)
            if k < 0:
                return cells
            i = k + 1
        else:
            k = line.find(",", i)
            if k < 0:
                cells.append((line[i:], False))
                return cells
            cells.append((line[i:k], False))
            i = k + 1


def _quote(v: str, quoted: bool) -> str:
    return '"' + v.replace('"', '""') + '"' if quoted or any(c in v for c in ',"') else v


def _leak_check(s: _Scrambler, out: dict) -> None:
    """Refuse the copy if an amount was left as it was, an original amount (4+ digits) shows up in free text, or an
    account name or number appears anywhere. (A scaled amount that happens to equal some other original amount is a
    coincidence, not a leak: with thousands of numbers that happens.)"""
    if s.unchanged:
        raise Leak("an amount was left unchanged")
    risky = {n for n in s.secrets - s.keep - {"0"} if _sig(n) >= 4}
    for text in s.free_text:
        if {_norm(n) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)} & risky:
            raise Leak("an original amount appears in free text")
    for text in out.values():
        for real in (*s.names, *s.numbers):
            if real and len(real) >= 3 and real in text:
                raise Leak("an account name or number appears in the copy")
