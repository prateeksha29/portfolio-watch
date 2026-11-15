"""Parse a Fidelity Positions CSV into stocks and fund themes. Values stay local: they order the digest and give
the local LLM position sizes for Ask."""
import csv
import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

MONEY_MARKET = {"SPAXX", "FDRXX", "FZFXX", "SPRXX", "FCASH", "CORE"}
CUSIP = re.compile(r"^(?=.*\d)[0-9A-Z]{9}$")
TICKER = re.compile(r"^[A-Z][A-Z.]{0,5}$")
FUND_WORDS = re.compile(r"\b(FUND|ETF|TRUST|TR|POOL|PORT|INDEX|IDX)\b")
DEFAULT_THEME = "US stock market"  # funds no fund_themes rule matches


@dataclass
class Holdings:
    stocks: dict[str, dict] = field(default_factory=dict)  # ticker -> {description, value, cost, shares, weight}
    source: str = ""   # the CSV the positions came from
    warning: str = ""  # set when the newest CSV could not be used
    funds: dict[str, dict] = field(default_factory=dict)   # symbol -> {description, value, weight, theme}

    def tickers_by_weight(self) -> list[str]:
        return sorted(self.stocks, key=lambda t: -self.stocks[t]["weight"])

    def themes(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for symbol, fund in self.funds.items():
            out.setdefault(fund["theme"], []).append(symbol)
        return out


def _money(s: str | None) -> float:
    try:
        return float((s or "").replace("$", "").replace(",", "").strip())
    except ValueError:
        return 0.0


def _is_fund(symbol: str, description: str) -> bool:
    if CUSIP.match(symbol):
        return True
    if len(symbol) == 5 and symbol.endswith("X"):
        return True
    return bool(FUND_WORDS.search(description))


def _theme(description: str, rules: list[dict]) -> str:
    for rule in rules:
        if re.search(rule["match"], description):
            return rule["theme"]
    return DEFAULT_THEME


def load_holdings(path: Path | None, fund_themes: list[dict], extra_tickers=(), accounts=()) -> Holdings:
    """accounts: the [returns] accounts (the brokerage ones; not a 401(k) or cash management account), the same
    ones the Positions and Performance pages count; empty: every account in the export."""
    h = Holdings()
    if path and path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if accounts and (row.get("Account name") or "").strip() not in accounts:
                    continue
                symbol = (row.get("Symbol") or "").strip().rstrip("*").upper()
                description = (row.get("Description") or "").strip().upper()
                if not symbol or symbol in MONEY_MARKET or "MONEY MARKET" in description:
                    continue
                if _is_fund(symbol, description):
                    entry = h.funds.setdefault(symbol, {"description": description, "value": 0.0,
                                                        "theme": _theme(description, fund_themes)})
                elif TICKER.match(symbol):
                    entry = h.stocks.setdefault(symbol, {"description": description, "value": 0.0, "shares": 0.0})
                    entry["shares"] += _money(row.get("Quantity"))
                else:
                    continue  # "Pending activity" and other non-positions
                entry["value"] += _money(row.get("Current value"))
                entry["cost"] = entry.get("cost", 0.0) + _money(row.get("Cost basis total"))
    for t in extra_tickers:
        h.stocks.setdefault(t.upper(), {"description": "", "value": 0.0})
    total = sum(e["value"] for e in (*h.stocks.values(), *h.funds.values())) or 1.0
    for e in (*h.stocks.values(), *h.funds.values()):
        e["weight"] = e["value"] / total
    return h


def _export_date(path: Path) -> dt.date | None:
    """'Portfolio_Positions_Oct-20-2026.csv' -> 2026-10-20 (the date Fidelity puts in the file name)."""
    m = re.search(r"([A-Z][a-z]{2})-(\d{1,2})-(\d{4})", path.name)
    try:
        return dt.datetime.strptime("-".join(m.groups()), "%b-%d-%Y").date() if m else None
    except ValueError:
        return None


def newest_holdings(folder: Path, pattern: str, fund_themes: list[dict], extra_tickers=(), accounts=()) -> Holdings:
    """The newest export in the folder (by the date in its name, else modification time) that has positions. A file
    that can't be read or has none (a different export, a half-saved download) is skipped with a warning."""
    files = sorted(folder.glob(pattern), key=lambda p: (_export_date(p) or dt.date.min, p.stat().st_mtime), reverse=True)
    skipped = []
    for path in files:
        try:
            h = load_holdings(path, fund_themes, extra_tickers, accounts)
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            skipped.append(f"{path.name} ({type(e).__name__})")
            continue
        if any(e["value"] for e in (*h.stocks.values(), *h.funds.values())):
            h.source = path.name
            if skipped:
                h.warning = f"Skipped {', '.join(skipped)}; using {path.name}"
            return h
        skipped.append(f"{path.name} (no positions)")
    h = load_holdings(None, fund_themes, extra_tickers)
    h.warning = f"No usable holdings file matching {folder.name}/{pattern}" + (f"; skipped {', '.join(skipped)}" if skipped else "")
    return h


def snapshot(h: Holdings) -> dict:
    """What a portfolio change note compares: shares per stock, the set of funds."""
    return {"source": h.source, "stocks": {t: round(e.get("shares", 0), 4) for t, e in h.stocks.items()},
            "funds": sorted(h.funds)}


def changes(before: dict | None, h: Holdings) -> str:
    """'Portfolio updated from Portfolio_Positions_Oct-20-2026.csv: added AAPL, AMZN; funds added VTSAX; more shares
    of NVDA, GOOGL' or '' when nothing changed (or there is nothing to compare with)."""
    if not before or before.get("source") == h.source:
        return ""
    now = snapshot(h)
    old, new = before.get("stocks", {}), now["stocks"]
    parts = []
    for label, items in (("added", [t for t in new if t not in old]), ("sold", [t for t in old if t not in new]),
                         ("more shares of", [t for t in new if t in old and new[t] > old[t] + 1e-4]),
                         ("fewer shares of", [t for t in new if t in old and new[t] < old[t] - 1e-4])):
        if items:
            parts.append(f"{label} {', '.join(sorted(items))}")
    added_funds = sorted(set(now["funds"]) - set(before.get("funds", [])))
    removed_funds = sorted(set(before.get("funds", [])) - set(now["funds"]))
    if added_funds:
        parts.append("funds added " + ", ".join(added_funds))
    if removed_funds:
        parts.append("funds removed " + ", ".join(removed_funds))
    unthemed = [s for s in added_funds if h.funds[s]["theme"] == DEFAULT_THEME]
    if unthemed:
        parts.append(f"no fund_themes rule matched {', '.join(unthemed)} (using '{DEFAULT_THEME}')")
    return f"Portfolio updated from {h.source}: " + "; ".join(parts) if parts else ""
