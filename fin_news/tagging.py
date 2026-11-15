"""Map article text to tickers using company names, aliases and cashtags."""
import re

# Tickers that are also words (MU, ON, META...) count bare only next to stock vocabulary.
STOCK_CONTEXT = re.compile(r"\b(stocks?|shares|earnings|NASDAQ|NYSE|Nasdaq)\b", re.I)
# Ticker marks of any company: "(ACME)", "(NASDAQ:ACME)", "NYSE: ACME", "$MU".
TICKER_MARK = re.compile(r"\((?:(?:NASDAQ|NYSE|NasdaqGS|NYSEARCA)\s*:\s*)?[A-Z]{1,5}\)|\b(?:NASDAQ|NYSE|NasdaqGS|NYSEARCA)\s*:\s*[A-Z]{1,5}\b|\$[A-Z]{1,5}\b")


class Tagger:
    def __init__(self, names: dict[str, list[str]], ambiguous: set[str]):
        self.patterns, self.bare = {}, {}
        for ticker, aliases in names.items():
            t = re.escape(ticker)
            # Not after a dot: "ETLR.MU" is a Munich listing, not Micron.
            bare = rf"(?<![A-Za-z.]){t}(?![A-Za-z])"
            parts = [rf"\${t}\b", rf"\({t}\)", rf"\b(?:NASDAQ|NYSE|Nasdaq|NYSEARCA)\s*:\s*{t}\b"]
            if ticker in ambiguous:
                self.bare[ticker] = re.compile(bare)
            else:
                parts.append(bare)
            parts += [rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])" for a in aliases]
            self.patterns[ticker] = re.compile("|".join(parts))

    def tag(self, text: str) -> list[str]:
        found = {t for t, p in self.patterns.items() if p.search(text)}
        if STOCK_CONTEXT.search(text):
            found |= {t for t, p in self.bare.items() if p.search(text)}
        return sorted(found)

    def mask(self, text: str) -> str:
        """The headline with the holdings' names and any ticker marks replaced by "Company": what it says, not
        whom it's about ("Acme Industries (ACME) CEO Maps Out Acquisition Push" -> "Company CEO Maps Out
        Acquisition Push")."""
        text = TICKER_MARK.sub(" ", text)
        for p in list(self.patterns.values()) + list(self.bare.values()):
            text = p.sub("Company", text)
        text = re.sub(r"(Company[’']?s?\s*)+", "Company ", text)
        return re.sub(r"\s+", " ", text).strip()
