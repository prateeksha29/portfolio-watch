"""Config and secrets loading."""
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def env_file_value(key: str) -> str:
    """A value as it is in .env now (os.environ keeps what it was when the process started, e.g. the web app)."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, value = line.strip().partition("=")
            if sep and k.strip() == key:
                return value.strip().strip("\"'")
    return os.environ.get(key, "")


def load_config(path: Path | None = None) -> dict:
    path = path or ROOT / "config.toml"
    if not path.exists():
        raise SystemExit("No config.toml: copy config.example.toml to config.toml and edit it (see the README)")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    _load_env(ROOT / ".env")
    cfg["secrets"] = {
        "fred": os.environ.get("FRED_API_KEY", ""),
        "finnhub": os.environ.get("FINNHUB_API_KEY", ""),
        "sec_user_agent": os.environ.get("SEC_USER_AGENT", ""),
    }
    from .addons import advisor  # here, not at the top: add-ons import this module
    if advisor:
        cfg["secrets"].update(advisor.secrets())
    paths = {name: ROOT / name for name in ("data", "digests", "logs")}
    for p in paths.values():
        p.mkdir(exist_ok=True)
    paths["root"] = ROOT
    paths["db"] = paths["data"] / "news.db"
    cfg["paths"] = paths
    return cfg
