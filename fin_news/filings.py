"""Plain-language views of SEC filings: the form of a filing article, and insider trades from Form 4 data."""
import json

from .collect import FILING_FORM, FORM4_CODES


def form_of(title: str) -> str:
    m = FILING_FORM.search(title)
    return m.group(1) if m else ""


def money(v: float) -> str:
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v / 1e3:.0f}K" if v >= 1e3 else f"${v:,.0f}"


def insider(body: str | None) -> dict | None:
    """Form 4 JSON -> {owner, role, plan_10b5_1, bought, sold, other}; bought/sold hold shares, value, price range.
    Only open-market purchases (P) and sales (S) count as trades; awards, option exercises, tax withholding and
    gifts are listed as 'other'."""
    try:
        data = json.loads(body or "")
    except ValueError:
        return None
    out = {k: data.get(k) for k in ("owner", "role", "plan_10b5_1")}
    other = set()
    for code, key in (("P", "bought"), ("S", "sold")):
        ts = [t for t in data.get("trades", []) if t["code"] == code and t["shares"] > 0]
        if ts:
            prices = [t["price"] for t in ts if t["price"]]
            out[key] = {"shares": sum(t["shares"] for t in ts), "value": sum(t["shares"] * t["price"] for t in ts),
                        "low": min(prices, default=0), "high": max(prices, default=0)}
    for t in data.get("trades", []):
        if t["code"] not in ("P", "S"):
            other.add(FORM4_CODES.get(t["code"], f"code {t['code']}"))
    out["other"] = sorted(other)
    return out


def insider_line(ticker: str, d: dict) -> str | None:
    """'AAPL — Ben Borders (Principal Accounting Officer) sold 1,280 shares at $343-346 (~$440K), pre-planned
    (10b5-1)'; None when the filing has no open-market trade."""
    parts = []
    for key in ("bought", "sold"):
        t = d.get(key)
        if not t:
            continue
        price = f"${t['low']:,.2f}" if t["low"] == t["high"] else f"${t['low']:,.0f}-{t['high']:,.0f}"
        parts.append(f"{key} {t['shares']:,.0f} shares at {price} (~{money(t['value'])})")
    if not parts:
        return None
    who = d.get("owner") or "An insider"
    if d.get("role"):
        who += f" ({d['role']})"
    line = f"{ticker} — {who} {' and '.join(parts)}"
    if d.get("sold") and d.get("plan_10b5_1"):
        line += ", pre-planned (10b5-1)"
    return line
