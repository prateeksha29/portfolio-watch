"""Paths, JSONL helpers and the dev/test split shared by the benchmark steps."""
import hashlib
import json
from pathlib import Path

from ..config import ROOT

BENCH = ROOT / "bench"
DATA = BENCH / "data"        # frozen items (copied out of the database, so retention never deletes them)
LABELS = BENCH / "labels"    # reference answers, one file per task, keyed by item id
RUNS = BENCH / "runs"        # model outputs: runs/<task>/<model slug>.jsonl
GRADING = BENCH / "grading"  # blind grading packets and grades for free-text outputs
PROMPTS = BENCH / "prompts"  # candidate system prompts for tuning (default = the one in fin_news/llm.py)
REPORTS = ROOT / "reports"

TASKS = ["dedup", "topic", "kind", "search", "stories", "filings", "ask"]


def split(item_id: str) -> str:
    """Stable split: a third of every task is held out ('test') and only used to confirm a final choice."""
    return "test" if int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 3 == 0 else "dev"


def slug(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def items(task: str) -> list[dict]:
    return read(DATA / f"{task}.jsonl")


def labels(task: str) -> dict[str, dict]:
    return {r["id"]: r for r in read(LABELS / f"{task}.jsonl")}
