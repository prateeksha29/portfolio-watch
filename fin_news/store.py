"""SQLite + sqlite-vec storage."""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS articles(
  id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, title TEXT NOT NULL, summary TEXT, body TEXT,
  source TEXT, kind TEXT, published TEXT, fetched TEXT, run_id TEXT,
  hints TEXT, tickers TEXT, topic TEXT, relevance REAL, relevant INTEGER DEFAULT 0,
  story_id INTEGER, is_primary INTEGER DEFAULT 0, title_vec BLOB);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published);
CREATE INDEX IF NOT EXISTS idx_articles_story ON articles(story_id);
CREATE TABLE IF NOT EXISTS stories(
  id INTEGER PRIMARY KEY, created_run TEXT, updated_run TEXT, primary_id INTEGER,
  n_articles INTEGER DEFAULT 0, n_sources INTEGER DEFAULT 0, tickers TEXT, topic TEXT, kind TEXT,
  relevance REAL, relevant INTEGER DEFAULT 0,
  summary TEXT, sentiment TEXT, key_numbers TEXT, why TEXT, llm_model TEXT, off_topic INTEGER DEFAULT 0,
  headline TEXT, summary_of TEXT);
CREATE INDEX IF NOT EXISTS idx_stories_updated ON stories(updated_run);
CREATE TABLE IF NOT EXISTS feed_cache(url TEXT PRIMARY KEY, etag TEXT, modified TEXT);
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, started TEXT, finished TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS macro(series_id TEXT, date TEXT, value REAL, PRIMARY KEY(series_id, date));
CREATE TABLE IF NOT EXISTS events(ticker TEXT, date TEXT, kind TEXT, detail TEXT, PRIMARY KEY(ticker, date, kind));
CREATE TABLE IF NOT EXISTS quotes(ticker TEXT, date TEXT, fetched TEXT, data TEXT, PRIMARY KEY(ticker, date));
CREATE TABLE IF NOT EXISTS prices(ticker TEXT, date TEXT, close REAL, PRIMARY KEY(ticker, date));
CREATE TABLE IF NOT EXISTS fund_filings(fund TEXT PRIMARY KEY, accession TEXT, period TEXT, filed TEXT, net_assets REAL,
  name TEXT, checked TEXT);
CREATE TABLE IF NOT EXISTS fund_holdings(fund TEXT, name TEXT, ticker TEXT, cusip TEXT, pct REAL, asset TEXT, country TEXT);
CREATE INDEX IF NOT EXISTS idx_fund_holdings ON fund_holdings(fund);
"""


def _chunks(seq, n=500):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def norm_title(t: str) -> str:
    """Headline compared without case or punctuation: copies of one article on different sites match."""
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


class Store:
    def __init__(self, path: Path, dim: int, model: str):
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        # WAL: the web app can read while a pipeline run writes.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.executescript(SCHEMA)
        if "title_vec" not in {r[1] for r in self.db.execute("PRAGMA table_info(articles)")}:
            self.db.execute("ALTER TABLE articles ADD COLUMN title_vec BLOB")  # databases created before 2026-10-15
        if "off_topic" not in {r[1] for r in self.db.execute("PRAGMA table_info(stories)")}:
            self.db.execute("ALTER TABLE stories ADD COLUMN off_topic INTEGER DEFAULT 0")  # before 2026-10-19
        if "headline" not in {r[1] for r in self.db.execute("PRAGMA table_info(stories)")}:
            self.db.execute("ALTER TABLE stories ADD COLUMN headline TEXT")  # filing one-liners, before 2026-10-20
        if "summary_of" not in {r[1] for r in self.db.execute("PRAGMA table_info(stories)")}:
            # The main article's headline (normalized) a summary was written from, so a story is summarized again
            # when a different article becomes its main one (added 2026-11-03; existing summaries count as current).
            self.db.execute("ALTER TABLE stories ADD COLUMN summary_of TEXT")
            rows = self.db.execute("SELECT s.id, a.title FROM stories s JOIN articles a ON a.id = s.primary_id "
                                   "WHERE s.summary IS NOT NULL").fetchall()
            self.db.executemany("UPDATE stories SET summary_of = ? WHERE id = ?", [(norm_title(t), i) for i, t in rows])
            self.db.commit()
        self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_articles USING vec0("
                        f"embedding float[{dim}] distance_metric=cosine)")
        stored = (self.get_meta("embed_model"), self.get_meta("embed_dim"))
        if stored == (None, None):
            self.set_meta("embed_model", model)
            self.set_meta("embed_dim", str(dim))
        elif stored != (model, str(dim)):
            raise SystemExit(f"{path} was built with {stored[0]} at {stored[1]} dims, but config says {model} at "
                             f"{dim}. Change the config back, or move the database aside to start fresh.")

    # ---------- meta ----------

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.db.commit()

    # ---------- feeds, events, macro, runs ----------

    def feed_cache(self) -> dict[str, tuple[str | None, str | None]]:
        return {r["url"]: (r["etag"], r["modified"]) for r in self.db.execute("SELECT * FROM feed_cache")}

    def save_feed_cache(self, url: str, etag: str | None, modified: str | None) -> None:
        self.db.execute("INSERT OR REPLACE INTO feed_cache(url, etag, modified) VALUES (?, ?, ?)", (url, etag, modified))

    def save_events(self, events) -> None:
        self.db.executemany("INSERT OR REPLACE INTO events(ticker, date, kind, detail) VALUES (?, ?, ?, ?)", events)

    def save_macro(self, rows) -> None:
        self.db.executemany("INSERT OR REPLACE INTO macro(series_id, date, value) VALUES (?, ?, ?)", rows)

    def finish_run(self, run_id: str, started: str, finished: str, stats: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO runs(id, started, finished, stats) VALUES (?, ?, ?, ?)",
                        (run_id, started, finished, json.dumps(stats)))
        self.db.commit()

    # ---------- articles ----------

    def seen_urls(self, urls) -> set[str]:
        seen = set()
        for chunk in _chunks(urls):
            q = f"SELECT url FROM articles WHERE url IN ({','.join('?' * len(chunk))})"
            seen.update(r[0] for r in self.db.execute(q, chunk))
        return seen

    def insert_articles(self, items: list[dict], run_id: str, fetched: str) -> list[int | None]:
        ids = []
        for it in items:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO articles(url, title, summary, source, kind, published, fetched, run_id, hints) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (it["url"], it["title"], it["summary"], it["source"], it["kind"], it["published"], fetched, run_id,
                 json.dumps(it["hints"])))
            ids.append(cur.lastrowid if cur.rowcount else None)
        self.db.commit()
        return ids

    def update_article_scores(self, arts: list[tuple[int, dict]]) -> None:
        self.db.executemany(
            "UPDATE articles SET kind = ?, tickers = ?, topic = ?, relevance = ?, relevant = ? WHERE id = ?",
            [(it["kind"], json.dumps(it["tickers"]), it["topic"], it["relevance"], it["relevant"], aid)
             for aid, it in arts])
        self.db.commit()

    def set_bodies(self, bodies: dict[int, str]) -> None:
        self.db.executemany("UPDATE articles SET body = ? WHERE id = ?", [(b, i) for i, b in bodies.items()])
        self.db.commit()

    def set_urls(self, urls: dict[int, str]) -> None:
        """Replace redirect links with the publisher URL (skipped if that URL is already stored)."""
        self.db.executemany("UPDATE OR IGNORE articles SET url = ? WHERE id = ?", [(u, i) for i, u in urls.items()])
        self.db.commit()

    def articles(self, ids) -> dict[int, sqlite3.Row]:
        out = {}
        for chunk in _chunks(ids):
            q = f"SELECT * FROM articles WHERE id IN ({','.join('?' * len(chunk))})"
            out.update({r["id"]: r for r in self.db.execute(q, chunk)})
        return out

    # ---------- vectors ----------

    def set_vectors(self, ids, embs: np.ndarray) -> None:
        rows = [(int(i), e.astype(np.float32).tobytes()) for i, e in zip(ids, embs)]
        self.db.executemany("DELETE FROM vec_articles WHERE rowid = ?", [(i,) for i, _ in rows])
        self.db.executemany("INSERT INTO vec_articles(rowid, embedding) VALUES (?, ?)", rows)
        self.db.commit()

    def set_title_vectors(self, ids, embs: np.ndarray) -> None:
        self.db.executemany("UPDATE articles SET title_vec = ? WHERE id = ?",
                            [(e.astype(np.float32).tobytes(), int(i)) for i, e in zip(ids, embs)])
        self.db.commit()

    def recent_titles(self, since: str, exclude: set[int]) -> list[sqlite3.Row]:
        """Recent grouped articles with headline vectors, for duplicate grouping."""
        return [r for r in self.db.execute(
            "SELECT id, story_id, title, published, title_vec FROM articles WHERE published >= ? AND kind != 'filing' "
            "AND story_id IS NOT NULL AND title_vec IS NOT NULL", (since,)) if r["id"] not in exclude]

    def knn(self, emb: np.ndarray, k: int) -> list[tuple[int, float]]:
        rows = self.db.execute(
            "SELECT rowid, distance FROM vec_articles WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (emb.astype(np.float32).tobytes(), min(k, 4096)))
        return [(r[0], r[1]) for r in rows]

    def delete_articles(self, ids) -> None:
        for chunk in _chunks(ids):
            marks = ",".join("?" * len(chunk))
            self.db.execute(f"DELETE FROM articles WHERE id IN ({marks})", chunk)
            self.db.executemany("DELETE FROM vec_articles WHERE rowid = ?", [(int(i),) for i in chunk])
        self.db.commit()

    # ---------- stories ----------

    def new_story(self, run_id: str) -> int:
        return self.db.execute("INSERT INTO stories(created_run, updated_run) VALUES (?, ?)", (run_id, run_id)).lastrowid

    def set_story_ids(self, assign: dict[int, int]) -> None:
        self.db.executemany("UPDATE articles SET story_id = ? WHERE id = ?", [(s, a) for a, s in assign.items()])
        self.db.commit()

    def story_members(self, story_id: int) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM articles WHERE story_id = ?", (story_id,)).fetchall()

    def update_story(self, story_id: int, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE stories SET {cols} WHERE id = ?", (*fields.values(), story_id))

    def set_primary(self, story_id: int, primary_id: int) -> None:
        self.db.execute("UPDATE articles SET is_primary = (id = ?) WHERE story_id = ?", (primary_id, story_id))

    def stories_with_primary(self, where: str, params=()) -> list[dict]:
        q = ("SELECT s.*, a.title, a.url, a.source, a.published, a.summary AS art_summary, a.body AS art_body "
             f"FROM stories s JOIN articles a ON a.id = s.primary_id WHERE {where}")
        out = []
        for r in self.db.execute(q, params):
            d = dict(r)
            d["tickers"] = json.loads(d["tickers"] or "[]")
            d["key_numbers"] = json.loads(d["key_numbers"] or "[]")
            out.append(d)
        return out

    def commit(self) -> None:
        self.db.commit()
