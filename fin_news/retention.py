"""Retention: articles and stories for two weeks, digests for three days, market data for four months."""
import datetime as dt
import logging

from .collect import iso, utcnow

log = logging.getLogger(__name__)


def prune(cfg: dict, store) -> dict:
    """Delete articles (with their vectors), stories, events and digests past their retention. Runs after every
    pipeline run; cheap when there is nothing to delete."""
    r, db, now = cfg["retention"], store.db, utcnow()

    def before(days):
        return iso(now - dt.timedelta(days=days))

    old = [row[0] for row in db.execute("SELECT id FROM articles WHERE coalesce(published, fetched) < ?",
                                        (before(r["article_days"]),))]
    store.delete_articles(old)
    counts = {"articles_deleted": len(old)}
    counts["stories_deleted"] = db.execute(
        "DELETE FROM stories WHERE id NOT IN (SELECT DISTINCT story_id FROM articles WHERE story_id IS NOT NULL)"
    ).rowcount
    db.execute("DELETE FROM events WHERE date < ?", (before(r["article_days"])[:10],))
    db.execute("DELETE FROM runs WHERE started < ?", (before(r["run_log_days"]),))
    market_days = cfg.get("market", {}).get("keep_days", 120)
    db.execute("DELETE FROM quotes WHERE date < ?", (before(market_days)[:10],))
    # Daily closes back the Returns page (a year of the owner's activity), so they are kept longer.
    price_days = max(market_days, cfg.get("returns", {}).get("keep_days", 430))
    db.execute("DELETE FROM prices WHERE date < ?", (before(price_days)[:10],))
    db.commit()

    cutoff = (now - dt.timedelta(days=r["digest_days"])).timestamp()
    digests = [p for p in cfg["paths"]["digests"].glob("*/*.md")  # one folder per person
               if p.name != "latest.md" and p.stat().st_mtime < cutoff]
    for p in digests:
        p.unlink()
    counts["digests_deleted"] = len(digests)

    # Reclaim disk space weekly; VACUUM briefly locks the database.
    last = store.get_meta("last_vacuum")
    if last is None or last < before(7):
        db.execute("VACUUM")
        store.set_meta("last_vacuum", iso(now))
    log.info("Pruned: %s", counts)
    return counts
