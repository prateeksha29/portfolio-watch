"""Local LLM via llama-swap's OpenAI-compatible API: story enrichment and grounded Q&A."""
import json
import logging
import os
import re

import requests

from .addons import advisor

log = logging.getLogger(__name__)

SENTIMENTS = ["positive", "negative", "neutral", "mixed"]
ENRICH_SCHEMA = {
    "type": "object",
    "properties": {
        "on_topic": {"type": "boolean"},
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": SENTIMENTS},
        "key_numbers": {"type": "array", "items": {"type": "string"}},
        "why_it_matters": {"type": "string"},
    },
    "required": ["on_topic", "summary", "sentiment", "key_numbers", "why_it_matters"],
}
ENRICH_SYSTEM = """You are a careful financial news analyst writing for a long-term individual investor.
Use ONLY the article text you are given. Never add facts, numbers or context from memory.
Return JSON with:
- on_topic: true if the article is mainly about the named holdings (or the named topic); false if they are only
  mentioned in passing, as one of many names in a list, as a comparison in another company's press release, or in an ad.
- summary: at most 3 sentences on what happened. Be specific: who, what, how much, and the figures that
  explain it (amounts, growth, guidance).
- sentiment: the effect of the reported facts on a shareholder of the named holdings.
  positive: good news for the holding (results beat, guidance raised, new business, upgrade, stock rose on company news).
  negative: bad news (results missed, guidance cut, lost market share, downgrade, lawsuit, stock fell on company news).
  mixed: clear good and bad points of similar weight.
  neutral: routine or no clear effect (scheduled announcements, fund position changes, general commentary,
  an opinion piece that argues a view without reporting new facts).
  Judge what happened, not the author's tone. If the article reports the stock fell, the answer is negative or mixed,
  never positive, even if an analyst rating in it is favourable.
- key_numbers: up to 4 important figures copied exactly as written in the article (e.g. "$35.1 billion revenue"). Empty if none.
- why_it_matters: one sentence on why this matters to a shareholder of the named holdings. If it is not material, say so.
If the text is only a headline and a short snippet, keep the summary to those facts and prefer neutral."""
FILING_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": SENTIMENTS},
        "key_numbers": {"type": "array", "items": {"type": "string"}},
        "why_it_matters": {"type": "string"},
    },
    "required": ["headline", "summary", "sentiment", "key_numbers", "why_it_matters"],
}
FILING_SYSTEM = """You read a company's SEC filing (usually the attached press release) for a long-term shareholder.
Use ONLY the text given. Never add facts or numbers from memory.
Return JSON with:
- headline: one line, at most 15 words: the main news with its most important figure
  (e.g. "Q3 sales EUR 11.7B, Q4 guided to EUR 14.2-15.2B").
- summary: 2-3 sentences. For results: revenue, profit or EPS, margin, growth versus a year ago, and guidance
  (compared with earlier guidance if the text states it). For other events: what was decided, the amounts and dates.
- sentiment: the effect on a shareholder: positive, negative, neutral (routine or procedural) or mixed.
- key_numbers: up to 6 important figures copied exactly as written.
- why_it_matters: one sentence for a shareholder."""
ASK_SYSTEM = """You are a research assistant for the owner of the portfolio described below, answering from:
- PORTFOLIO: their positions (value, share of the portfolio, gain since purchase), their funds, and their biggest
  real exposures: direct shares plus their slice of each fund's holdings.
- HOLDING DETAIL: for the holdings the question is about: the position, market data (price, returns vs the
  S&P 500, valuation, analyst consensus and price targets, latest analyst actions), recent news sentiment,
  summarized stories, company announcements, insider trades and upcoming earnings{detail}.
- OTHER STOCK: the same market data for stocks they don't own directly, and any exposure through their funds.
- YOUR INVESTING: what the owner earned over recent periods in their brokerage accounts (split into holdings
  they already owned, new investments and dividends) and how their new investments did against the same dollars
  put into the S&P 500 on the same days. Use it for questions about their own returns or decisions.
- FUND DETAIL: for funds the question names: largest positions, which of their stocks the fund also holds, its mix.
- SOURCES: numbered articles. Cite them inline like [1] or [2][3] for facts taken from them: at most two per
  sentence, only sources that state that fact. Facts from PORTFOLIO, HOLDING DETAIL, OTHER STOCK or FUND DETAIL get
  no citation and no label; never cite [0], a section name, or a number that is not in SOURCES.
Prefer the market data for the current price, valuation and analyst targets; articles may quote older figures.
Analyst targets are opinions, not forecasts to rely on: weigh them alongside valuation and the news.{rule}
Use only this information; never invent prices, figures or events. If it doesn't answer part of the question,
say so rather than filling the gap.
Read the figures exactly as given and don't recompute them into a different claim: ranks (#1, #2, ...), "above cost"
or "below cost", "ahead of" or "behind" the S&P 500, and dollar amounts. Keep each figure tied to what the text says it
measures (sales guidance, a buyback, a price target and a fair-value estimate are different things).
Dates show their weekday. A price move belongs to the trading day named in Market data; an article's date is when it
was published, often a day or more after the event it describes. US markets are closed on Saturday and Sunday, so
never place a market move on a weekend. Fund holdings come from each fund's latest SEC
filing (a few months old; say "as of" the date when it matters); for pooled funds they come from a public fund with
the same strategy, so call those approximate. For a fund with no holdings data, don't guess its weights.
When position size matters, use the real exposure (direct + through funds), not only the direct position.
For questions about buying, selling or position size, give a reasoned view: the current position size and
concentration, what recent news and sentiment suggest, upcoming catalysts and the main risks, and what would change
the view. Mention once, briefly, that you can't see their goals, taxes or money held elsewhere. Be concise."""
ASK_SYSTEM = ASK_SYSTEM.format(detail=getattr(advisor, "ASK_DETAIL", ""), rule=getattr(advisor, "ASK_RULE", ""))
REASON_SYSTEM = """You read an investment newsletter's write-up recommending a stock. In ONE plain sentence of at most
30 words, say the main reason the writers give for buying it. Use only the write-up; no figures that are not in it.
Reply with the sentence only."""


def _numbers(s: str) -> list[str]:
    return [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", s)]


def grounded(figure: str, source: str) -> bool:
    """A figure is kept only if every number in it also appears in the source text."""
    nums = _numbers(figure)
    if not nums:
        return False
    available = set(_numbers(source))
    return all(n in available for n in nums)


def _closed(schema):
    """The schema with additionalProperties: false on every object (Claude's structured outputs require it)."""
    if isinstance(schema, dict):
        out = {k: _closed(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(schema, list):
        return [_closed(v) for v in schema]
    return schema


class LLM:
    def __init__(self, cfg: dict, purpose: str = "summaries"):
        """purpose: 'summaries' (pipeline) or 'ask' (questions; uses ask_model when configured)."""
        self.c = cfg["llm"]
        self.candidates = [self.c["model"], self.c.get("fallback_model")]
        if purpose == "ask" and self.c.get("ask_model"):
            self.candidates = [self.c["ask_model"], self.c.get("ask_fallback_model"), *self.candidates]
        self.purpose = purpose
        # "openai": any OpenAI-compatible server (llama-swap, llama.cpp, vLLM, Ollama, api.openai.com);
        # "anthropic": Claude through Anthropic's own Python library (pip install anthropic)
        self.provider = self.c.get("provider", "openai")
        self.local = self.c.get("local", self.provider == "openai")  # a llama.cpp-style server on your own GPU
        self.base = self.c["base_url"].rstrip("/") if self.provider == "openai" else "api.anthropic.com"
        self.headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.c.get("api_key_env", ""), "")
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = None
        if self.provider == "anthropic":
            import anthropic  # only needed for Claude
            # No api_key_env: the library finds ANTHROPIC_API_KEY (or an `ant auth login` profile) itself.
            self.client = anthropic.Anthropic(api_key=api_key or None, timeout=self.c.get("timeout_seconds", 300))
        self.model = self._choose()

    def _served(self) -> set[str]:
        if self.client:
            return {m.id for m in self.client.models.list()}
        r = requests.get(f"{self.base}/models", headers=self.headers, timeout=20)
        r.raise_for_status()
        return {m["id"] for m in r.json().get("data", [])}

    def _choose(self) -> str | None:
        try:
            served = self._served()
        except Exception as e:  # noqa: BLE001 -- unreachable, a wrong key, no network: the run goes on without summaries
            log.warning("LLM endpoint %s not reachable (%s); skipping LLM steps", self.base, e)
            return None
        for model in self.candidates:
            if model and model in served:
                log.info("LLM: %s via %s", model, self.base)
                return model
        log.warning("None of %s is served at %s", [m for m in self.candidates if m], self.base)
        return None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 600) -> str:
        if self.client:
            return self._claude(messages, schema, max_tokens)
        body = {"model": self.model, "messages": messages}
        if self.local:
            body.update(temperature=0, max_tokens=max_tokens,
                        # Gemma 4 reasons before answering by default; that spends the token budget before any JSON.
                        chat_template_kwargs={"enable_thinking": bool(self.c.get("thinking", False))})
        else:  # a hosted API: newer models reject temperature 0 and max_tokens, and their reasoning counts as output
            body["max_completion_tokens"] = max(max_tokens, self.c.get("min_output_tokens", 16000))
            if self.c.get("temperature") is not None:
                body["temperature"] = self.c["temperature"]
        if schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "result", "strict": True, "schema": schema}}
        r = requests.post(f"{self.base}/chat/completions", headers=self.headers, json=body,
                          timeout=self.c["timeout_seconds"])
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

    def _claude(self, messages: list[dict], schema: dict | None, max_tokens: int) -> str:
        """The same request through Anthropic's Messages API. Claude's thinking can't be turned off, so its depth is
        set with effort ([llm] effort for summaries, ask_effort for Ask) and max_tokens has room for it. A request
        Claude declines is retried on the model Anthropic picks (server-side fallback). Errors come back as
        requests exceptions, which every caller already handles."""
        import anthropic
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        effort = self.c.get("ask_effort", "high") if self.purpose == "ask" else self.c.get("effort", "low")
        output_config = {"effort": effort}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": _closed(schema)}
        try:
            r = self.client.beta.messages.create(
                model=self.model, max_tokens=max(max_tokens, self.c.get("min_output_tokens", 16000)),
                system=system or anthropic.NOT_GIVEN, output_config=output_config,
                messages=[m for m in messages if m["role"] != "system"],
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        except anthropic.APIError as e:
            raise requests.RequestException(f"Claude: {e}") from e
        if r.stop_reason == "refusal":
            raise requests.RequestException("Claude declined this request")
        return "".join(b.text for b in r.content if b.type == "text")

    def enrich(self, title: str, text: str, about: str) -> dict | None:
        text = text[: self.c["max_article_chars"]]
        messages = [{"role": "system", "content": ENRICH_SYSTEM},
                    {"role": "user", "content": f"Holdings: {about}\n\nTITLE: {title}\n\nARTICLE:\n{text}"}]
        for attempt in range(2):
            try:
                data = json.loads(self._chat(messages, ENRICH_SCHEMA))
                break
            except (json.JSONDecodeError, KeyError, IndexError, requests.RequestException) as e:
                log.warning("LLM enrich failed (attempt %d): %s", attempt + 1, e)
        else:
            return None
        data["on_topic"] = bool(data.get("on_topic", True))
        if data.get("sentiment") not in SENTIMENTS:
            data["sentiment"] = "neutral"
        source = f"{title}\n{text}"
        data["key_numbers"] = [n for n in data.get("key_numbers", [])[:4] if grounded(n, source)]
        return data

    def enrich_filing(self, title: str, text: str, company: str) -> dict | None:
        text = text[: self.c.get("max_filing_chars", 40000)]
        messages = [{"role": "system", "content": FILING_SYSTEM},
                    {"role": "user", "content": f"Company: {company}\nFILING: {title}\n\nTEXT:\n{text}"}]
        for attempt in range(2):
            try:
                data = json.loads(self._chat(messages, FILING_SCHEMA, max_tokens=800))
                break
            except (json.JSONDecodeError, KeyError, IndexError, requests.RequestException) as e:
                log.warning("LLM filing summary failed (attempt %d): %s", attempt + 1, e)
        else:
            return None
        if data.get("sentiment") not in SENTIMENTS:
            data["sentiment"] = "neutral"
        data["key_numbers"] = [n for n in data.get("key_numbers", [])[:6] if grounded(n, text)]
        return data

    def pick_reason(self, company: str, text: str) -> str:
        """One sentence on why a newsletter recommends a stock, from its write-up (Discovery)."""
        out = self._chat([{"role": "system", "content": REASON_SYSTEM},
                          {"role": "user", "content": f"Company: {company}\n\nWRITE-UP:\n{text[:6000]}"}], max_tokens=120)
        return re.sub(r"\s+", " ", out).strip().strip('"')[:300]

    def answer(self, question: str, sources: list[str], context: str = "") -> str:
        numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))
        from .askfmt import clean_citations
        out = self._chat([{"role": "system", "content": ASK_SYSTEM},
                          {"role": "user", "content": f"{context}\n\nSOURCES:\n{numbered}\n\nQUESTION: {question}"}],
                         max_tokens=1500)
        return clean_citations(out, len(sources))[0]
