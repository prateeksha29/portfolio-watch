"""Label items from mixed news feeds as news, analysis/opinion, press release or transcript."""
import re

import numpy as np

RULES = [
    ("transcript", re.compile(r"(earnings|conference) call transcript", re.I)),
    ("press_release", re.compile(r"\b(announces|to present at|declares (quarterly )?dividend|press release|"
                                 r"to host .*(conference call|webcast))\b", re.I)),
    ("analysis", re.compile(r"(\?\s*$|^(is|should|could|can|will|why|what) |prediction:|here's why|better (stock|buy)|"
                            r"billionaire|i'm buying|my top|forever stock|stocks? to (buy|own|watch|hold)|"
                            r"\bbuy (now|before)\b|no-brainer|\bvs\.? )", re.I)),
    # Added 2026-11-04: the Qwen embedder files these under "news" (benchmark: 1 of 8 found outside the PR Newswire
    # feed). After the analysis rule so "FedEx Names A New CFO, Is The Stock Still Undervalued?" stays analysis.
    ("press_release", re.compile(
        r"\bappoints\b|\bnames\b.{0,60}\b(as |new )?(chief \w+ officer|CEO|CFO|COO|CTO|president|chair(man|woman)?|"
        r"board of directors)\b"
        r"|\b(reports|releases|announces)( its)? (record )?(first|second|third|fourth|full[- ]year|fiscal)\b.{0,40}\bresults\b"
        r"|securities (fraud|class action|claims|litigation)|class action (lawsuit|filed|deadline)|investor class action"
        r"|investors? (have|has) (the )?opportunity to lead|deadline (alert|news|reminder)|shareholder (alert|investigation)"
        r"|investor alert|lawsuit reminder|legal notice|\b(reminds|notifies|alerts|urges|encourages)\b.{0,60}\b(investors|shareholders)\b"
        r"|law offices of|\b(rosen law|levi & korsinsky|pomerantz|bragar eagel|faruqi|glancy prongay|bernstein liebhard|"
        r"robbins (llp|geller)|kessler topaz|schall law|portnoy law|gross law|bfa law|kaplan fox|suewallst)\b", re.I)),
]

PROTOTYPES = {
    "news": [
        "Nvidia shares fall after report of new chip export curbs",
        "Microsoft beats quarterly revenue estimates on cloud growth",
        "Fed holds rates steady, signals two cuts this year",
        "Intel to cut 15% of workforce in restructuring",
        "Apple faces EU antitrust fine over App Store rules",
        "SK Hynix raises guidance as memory prices climb",
    ],
    "analysis": [
        "Is Oracle stock a buy after its recent pullback?",
        "3 reasons I'm buying Amazon stock and never selling",
        "Prediction: this AI stock will be worth $5 trillion by 2030",
        "Coca-Cola vs. PepsiCo: which dividend stock is the better buy?",
        "Why Boeing could be the next big turnaround stock",
        "Valuation check: is the market too optimistic about Meta?",
    ],
    "press_release": [
        "Walmart announces date of third-quarter 2026 financial results conference call",
        "Coca-Cola declares quarterly dividend",
        "Ford Motor Company reports results for the third quarter",
        "Caterpillar to present at upcoming investor conference",
        "Starbucks announces new board appointment",
        "Verizon announces pricing of senior notes offering",
    ],
    "transcript": [
        "NVIDIA (NVDA) Q3 2026 earnings call transcript",
        "Texas Instruments Q3 2026 earnings conference call transcript",
        "Johnson & Johnson Q3 2026 earnings call transcript",
    ],
}


# Articles a Google News or Finnhub feed carries from a press-release wire ("Google News · GlobeNewswire").
WIRE_SOURCE = re.compile(r"· (PR Newswire|GlobeNewswire|Business ?Wire|EIN Presswire|ACCESS ?Newswire|Accesswire|"
                         r"Newsfile|PRWeb)$", re.I)


def rule_kind(title: str, source: str = "") -> str | None:
    """Story type from the source and title rules alone; None leaves it to the embedder."""
    if WIRE_SOURCE.search(source or ""):
        return "press_release"
    return next((label for label, rx in RULES if rx.search(title)), None)


class KindClassifier:
    def __init__(self, embedder):
        self.embedder = embedder
        self.labels = list(PROTOTYPES)
        centroids = []
        for label in self.labels:
            c = embedder.classify(PROTOTYPES[label]).mean(axis=0)
            centroids.append(c / np.linalg.norm(c))
        self.centroids = np.stack(centroids)

    def predict(self, titles: list[str], sources: list[str] | None = None) -> list[str]:
        out: list[str | None] = []
        for t, src in zip(titles, sources or [""] * len(titles)):
            out.append(rule_kind(t, src))
        todo = [i for i, label in enumerate(out) if label is None]
        if todo:
            sims = self.embedder.classify([titles[i] for i in todo]) @ self.centroids.T
            for i, row in zip(todo, sims):
                out[i] = self.labels[int(row.argmax())]
        return out
