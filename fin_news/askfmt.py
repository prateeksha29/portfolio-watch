"""Wording for the Ask context. Spells out what the model otherwise has to work out itself, and got wrong in the
November 2026 benchmark: the weekday of each date, gain vs loss, ahead of vs behind the S&P 500, and ranks."""
import datetime as dt
import re


def day(d) -> str:
    """'2026-11-02' (or a timestamp) -> 'Mon 2026-11-02'."""
    if not d:
        return ""
    s = str(d)[:10]
    try:
        return f"{dt.date.fromisoformat(s):%a} {s}"
    except ValueError:
        return str(d)


def vs_cost_pct(g: float) -> str:
    """0.28 -> '+28% above cost'; -0.04 -> '-4% below cost'."""
    if round(g * 100) == 0:
        return "about at cost"
    return f"{g:+.0%} {'above' if g > 0 else 'below'} cost"


def vs_cost(value: float, cost: float) -> str:
    return vs_cost_pct(value / cost - 1)


def earned(v: float) -> str:
    """1234 -> 'earned $1,234'; -5678 -> 'lost $5,678'."""
    return f"earned ${v:,.0f}" if round(v) >= 0 else f"lost ${-v:,.0f}"


def gained(v: float) -> str:
    return f"gained ${v:,.0f}" if round(v) >= 0 else f"lost ${-v:,.0f}"


def gap(ret: float, bench: float) -> str:
    """Compares the returns as shown (rounded to 0.1%), so the words match the numbers next to them."""
    d = round(ret * 100, 1) - round(bench * 100, 1)
    if abs(d) < 0.05:
        return "level with the S&P 500"
    return f"{'ahead of' if d > 0 else 'behind'} the S&P 500 by {abs(d):.1f} points"


def money_gap(mine: float, bench: float) -> str:
    """New investments vs the same dollars in the S&P 500, from the dollar figures as shown."""
    d = round(mine) - round(bench)
    if d == 0:
        return "the same as the S&P 500"
    return f"${abs(d):,.0f} {'better' if d > 0 else 'worse'} than the S&P 500"


def returns_line(label: str, start: str, end: str, x: dict) -> str:
    """One period of YOUR INVESTING."""
    ret = (f" ({x['ret']:+.1%} vs S&P 500 {x['bench']:+.1%}: {gap(x['ret'], x['bench'])})"
           if x.get("ret") is not None else "")
    return (f"- {label} ({day(start)} to {day(end)}): {earned(x['earned'])}{ret}: "
            f"${x['from_existing']:,.0f} from holdings already owned, ${x['from_new']:,.0f} from new investments, "
            f"${x['income']:,.0f} dividends/interest. New investments: ${x['invested']:,.0f} -> ${x['now']:,.0f} "
            f"({gained(x['from_new'])}); the same dollars in the S&P 500 on the same days would have "
            f"{gained(x['bench_gain'])}, so new investments did {money_gap(x['from_new'], x['bench_gain'])}.")


def source_header(title: str, source: str, published: str) -> str:
    return f"{title} ({source}, published {day(published)})"


# Section names and dates the models sometimes put in brackets as if they were citations ([HOLDING DETAIL],
# [News 2026-10-31], [Sun 2026-11-01], [Market Data]); the prompt forbids them but they still appear.
LABEL = re.compile(r"\[(?=[^\]\n]{2,60}\](?!\())(?=[^\]\n]*(?:holding detail|other stock|portfolio|your investing|fund detail"
                   r"|sources|market data|news|company filing|insider|write-up|\d{4}-\d\d-\d\d))[^\]\n]*\]", re.I)


def clean_citations(text: str, n_sources: int) -> tuple[str, int]:
    """Drop citation markers that point at no source ([0], a number above the count, or a section name or date in
    brackets). Returns the text and how many were dropped."""
    dropped = 0

    def fix(m):
        nonlocal dropped
        if 1 <= int(m.group(1)) <= n_sources:
            return m.group(0)
        dropped += 1
        return ""

    out = re.sub(r"\[(\d+)\]", fix, text)
    out, n = LABEL.subn("", out)
    dropped += n
    return (re.sub(r"[ \t]+([.,;:)])", r"\1", re.sub(r"(\S)[ \t]{2,}", r"\1 ", out)) if dropped else out), dropped
