"""Embeddings served by llama-swap (OpenAI-compatible /embeddings): Matryoshka-truncated and re-normalized.
Each model has its own text formats and instructions (FORMATS); the server applies none, so they are added here.

llama-swap loads the model on demand and swaps it with whatever else holds the shared GPU, so this module never
touches the GPU itself. When the server can't answer (busy GPU, model loading, network), requests are retried until
[embedding] wait_seconds have passed; then EmbeddingUnavailable is raised and the run leaves its articles for the
next one. Served vs sentence-transformers on identical texts: EmbeddingGemma 2 cosine >= 0.9995 (2026-10-21); Qwen3-
Embedding-0.6B >= 0.9977, mean 0.9991 (q8 weights; 2026-11-02), with the same benchmark scores."""
import logging
import os
import re
import time

import numpy as np
import requests

log = logging.getLogger(__name__)

# Per model: document format, headline format, and the instruction put before search queries and story-type labels.
# EmbeddingGemma 2's come from its sentence-transformers config. Qwen3-Embedding: documents and headlines plain,
# queries with its standard retrieval instruction; the story-type instruction was chosen on the benchmark's dev split
# (accuracy 0.668 vs 0.620 with none, 2026-11-02).
FORMATS = {
    "google/embeddinggemma-2": {
        "doc": lambda title, text: f"title: {title or 'none'} | text: {text or title}",
        "headline": lambda title: f"title: {title or 'none'} | text: {title}",
        "SearchQuery": "task: search result | query: ",
        "Classification": "task: classification | query: ",
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "doc": lambda title, text: f"{title}\n{text}" if text and text != title else (title or text or ""),
        "headline": lambda title: title,
        "SearchQuery": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "Classification": ("Instruct: Classify the type of this financial headline: news report, analysis or opinion, "
                           "company press release, or earnings call transcript\nQuery:"),
    },
}


class EmbeddingUnavailable(RuntimeError):
    pass


class Embedder:
    def __init__(self, cfg: dict, wait_seconds: float | None = None):
        e, c = cfg["embedding"], cfg["llm"]
        self.name, self.dim, self.batch = e["model"], e["dim"], e["batch_size"]
        self.served = e.get("served_model", self.name)
        self.max_chars = e.get("max_chars", 6000)
        if self.name not in FORMATS:
            raise SystemExit(f"[embedding] model {self.name!r} has no text formats in fin_news/embed.py FORMATS")
        self.fmt = FORMATS[self.name]
        self.wait = wait_seconds if wait_seconds is not None else e.get("wait_seconds", 900)
        self.url = re.sub(r"/+$", "", e.get("base_url") or c["base_url"]) + "/embeddings"
        self.headers = {"Content-Type": "application/json"}
        # Its own server ([embedding] base_url) gets its own key; the [llm] key only goes to the [llm] server.
        key_env = e.get("api_key_env", "") if e.get("base_url") else c.get("api_key_env", "")
        api_key = os.environ.get(key_env, "")
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, texts: list[str], deadline: float) -> np.ndarray:
        delay = 5
        while True:
            try:
                r = requests.post(self.url, headers=self.headers, json={"model": self.served, "input": texts},
                                  timeout=(10, max(30, min(300, deadline - time.time()))))
                if r.status_code == 200:
                    data = sorted(r.json()["data"], key=lambda d: d["index"])
                    return np.array([d["embedding"] for d in data], dtype=np.float32)
                problem = f"HTTP {r.status_code}: {r.text[:200]}"
                if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                    raise EmbeddingUnavailable(f"{self.served}: {problem}")  # a bad request won't get better
            except (requests.RequestException, ValueError, KeyError) as e:
                problem = f"{type(e).__name__}: {e}"[:200]
            if time.time() + delay > deadline:
                raise EmbeddingUnavailable(f"{self.served} not available after {self.wait}s ({problem})")
            log.warning("Embedding server not ready (%s); retrying in %ds", problem, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)

    def _encode(self, texts: list[str], prompt_name: str | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.fmt[prompt_name] if prompt_name else ""
        deadline = time.time() + self.wait  # the wait budget covers the whole call, not each batch
        parts = [self._post([prefix + t for t in texts[i:i + self.batch]], deadline)
                 for i in range(0, len(texts), self.batch)]
        arr = np.concatenate(parts)[:, : self.dim]
        arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)  # re-normalize after truncation
        return arr.astype(np.float32)

    def documents(self, pairs: list[tuple[str, str | None]]) -> np.ndarray:
        """Embed (title, text) pairs in the model's document format."""
        return self._encode([self.fmt["doc"](title, (text or "")[: self.max_chars]) for title, text in pairs])

    def headlines(self, titles: list[str]) -> np.ndarray:
        """Headlines alone, for duplicate grouping."""
        return self._encode([self.fmt["headline"](t) for t in titles])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "SearchQuery")

    def classify(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "Classification")

    def close(self) -> None:
        """Nothing to free: the model lives in llama-swap."""
