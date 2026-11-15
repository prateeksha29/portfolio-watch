"""Rewrite a frozen Ask item (context + sources, captured with the old wording) into the current wording, so prompt
variants can be compared on identical facts. Uses the same helpers as the app (fin_news.askfmt); `check` confirms that
converting a context captured with the old code gives the text the new code produces."""
import re
import sqlite3

from ..askfmt import day, returns_line, source_header, vs_cost, vs_cost_pct

STOCK = re.compile(r"^- (\S+) (.*): \$([\d,]+), ([\d.]+%), (?:([+-]\d+)%|n/a)$")
POSITION = re.compile(r"^(Position: .* worth \$([\d,]+)), cost basis \$([\d,]+)$")
MARKET = re.compile(r"^Market data \(Yahoo Finance, as of (\S+)\): price \$([\d,.]+)(?:, ([+-][\d.]+%) on the day)?$")
RETURNS = re.compile(
    r"^- (.+?) \((\S+) to (\S+)\): earned \$(-?[\d,]+)(?: \(([+-][\d.]+)% vs S&P 500 ([+-][\d.]+)%\))?: "
    r"\$(-?[\d,]+) from holdings already owned, \$(-?[\d,]+) from new investments, \$(-?[\d,]+) dividends/interest\. "
    r"New investments: \$([\d,]+) -> \$([\d,]+); the same dollars in the S&P 500 on the same days would have gained "
    r"\$(-?[\d,]+)\.$")
SOURCE = re.compile(r"^(.*) \((.+), (\d{4}-\d\d-\d\d)\)$")


def _n(s: str) -> float:
    return float(s.replace(",", ""))


class Converter:
    def __init__(self, db_path: str = "data/news.db"):
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    def trading_day(self, ticker: str, price: float, fetched: str) -> str:
        """The close the quoted price matches (as the app does), else the fetch date."""
        r = self.db.execute("SELECT date FROM prices WHERE ticker = ? AND date <= ? AND abs(close - ?) < 0.006 "
                            "ORDER BY date DESC LIMIT 1", (ticker, fetched, price)).fetchone()
        return f"close on {day(r[0])}" if r else f"latest trade, fetched {day(fetched)}"

    def context(self, text: str) -> str:
        out, ticker, section = [], None, None
        for line in text.split("\n"):
            if line.startswith("Stocks: ticker, name"):
                line, section = "Stocks (largest first): ticker, name, value, share of portfolio, gain or loss vs cost:", "stocks"
            elif line.startswith("Biggest real exposures (direct"):
                line, section, rank = ("Biggest real exposures, ranked (direct shares + the owner's slice of each fund's "
                                       "holdings):"), "exposures", 0
            elif not line.startswith("- "):
                section = None
            if section == "stocks" and (m := STOCK.match(line)):
                g = vs_cost_pct(int(m.group(5)) / 100) if m.group(5) else "cost unknown"
                line = f"- {m.group(1)} {m.group(2)}: ${m.group(3)}, {m.group(4)}, {g}"
            elif section == "exposures" and line.startswith("- "):
                rank += 1
                line = f"- #{rank} " + line[2:]
            if m := re.match(r"^(?:HOLDING DETAIL|OTHER STOCK): (\S+) \(", line):
                ticker = m.group(1)
            if m := POSITION.match(line):
                line = f"{m.group(1)}, cost basis ${m.group(3)} (now {vs_cost(_n(m.group(2)), _n(m.group(3)))})"
            elif m := MARKET.match(line):
                price = _n(m.group(2))
                line = (f"Market data (Yahoo Finance): price ${price:,.2f} "
                        f"({self.trading_day(ticker, price, m.group(1))})"
                        + (f", {m.group(3)} that day" if m.group(3) else ""))
            elif m := RETURNS.match(line):
                g = m.groups()
                x = {"earned": _n(g[3]), "ret": _n(g[4]) / 100 if g[4] else None,
                     "bench": _n(g[5]) / 100 if g[5] else None, "from_existing": _n(g[6]), "from_new": _n(g[7]),
                     "income": _n(g[8]), "invested": _n(g[9]), "now": _n(g[10]), "bench_gain": _n(g[11])}
                line = returns_line(g[0], g[1], g[2], x)
            if m := re.match(r"^  (Best|Worst) vs S&P \(3 months\): (.*)$", line):
                line = (f"  New investments of the last 3 months, {m.group(1).lower()} vs the same dollars in the "
                        f"S&P 500: " + re.sub(r"\(\$([\d,]+)\)", r"($\1 invested)", m.group(2)))
            line = re.sub(r"^- (\d{4}-\d\d-\d\d) \[", lambda m: f"- {day(m.group(1))} [", line)
            line = re.sub(r"^- Company filing (\d{4}-\d\d-\d\d) \[", lambda m: f"- Company filing {day(m.group(1))} [", line)
            line = re.sub(r"^Next earnings: (\d{4}-\d\d-\d\d)", lambda m: f"Next earnings: {day(m.group(1))}", line)
            out.append(line)
        return "\n".join(out)

    def sources(self, sources: list[str]) -> list[str]:
        out = []
        for s in sources:
            head, _, rest = s.partition("\n")
            if m := SOURCE.match(head):
                head = source_header(m.group(1), m.group(2), m.group(3))
            out.append(head + ("\n" + rest if _ else ""))
        return out


def check(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Contexts captured with the old and the new code at the same moment: report every line the converter gets wrong."""
    c, bad = Converter(), []
    for k in old:
        got, want = c.context(old[k]).split("\n"), new[k].split("\n")
        bad += [f"{k}: {g!r} != {w!r}" for g, w in zip(got, want) if g != w]
        if len(got) != len(want):
            bad.append(f"{k}: {len(got)} lines vs {len(want)}")
    return bad
