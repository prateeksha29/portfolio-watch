"""People who use the app. Each has a folder users/<id>/ with:

    user.toml   their settings (below); everything else comes from config.toml, shared by everyone
    password    their password's scrypt hash (written by `python -m fin_news adduser` / `passwd`, or the Account page)
    fidelity/   their Fidelity exports (Portfolio_Positions_*.csv, Accounts_History*.csv), uploaded on the Files page

user.toml keys (all optional):

    name = "Display name"
    admin = true            # sees source problems in the digest, the Benchmarks page and add-on logins
    tickers = ["AMD"]       # stocks to follow that aren't in the positions export
    topics = ["..."]        # themes to watch, on top of config.toml's topics and the person's fund themes
    holdings_csv = "fidelity/Portfolio_Positions_*.csv"   # relative to the person's folder
    [aliases]               # company names for tagging, added to config.toml's [aliases]
    [returns]               # accounts = [...] (default: every account in the exports), history_csv, groups

One pipeline run collects news for everyone's stocks and themes together and summarizes each story once; each person
gets their own digest, positions, returns, ideas and Ask answers."""
import copy
import hashlib
import hmac
import os
import re
import secrets
import tomllib
from pathlib import Path

from .config import ROOT

USERS_DIR = ROOT / "users"
VALID_ID = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
PER_USER_RETURNS = ("accounts", "history_csv", "groups")
_cache: dict[str, tuple[float, dict]] = {}


def user_ids() -> list[str]:
    if not USERS_DIR.is_dir():
        return []
    return sorted(p.name for p in USERS_DIR.iterdir() if VALID_ID.match(p.name) and (p / "user.toml").exists())


def settings(uid: str) -> dict:
    path = USERS_DIR / uid / "user.toml"
    mtime = path.stat().st_mtime
    hit = _cache.get(uid)
    if hit and hit[0] == mtime:
        return hit[1]
    with open(path, "rb") as f:
        data = tomllib.load(f)
    _cache[uid] = (mtime, data)
    return data


def user_config(cfg: dict, uid: str) -> dict:
    """cfg as one person sees it: their folder, digests, settings and topics on top of the shared config."""
    if not VALID_ID.match(uid) or not (USERS_DIR / uid / "user.toml").exists():
        raise KeyError(uid)
    u = settings(uid)
    c = dict(cfg)
    home = USERS_DIR / uid
    (home / "fidelity").mkdir(mode=0o700, parents=True, exist_ok=True)
    digests = cfg["paths"]["digests"] / uid
    digests.mkdir(parents=True, exist_ok=True)
    c["paths"] = {**cfg["paths"], "home": home, "digests": digests}
    c["user"] = {"id": uid, "name": u.get("name") or uid.title(), "admin": bool(u.get("admin"))}
    c["holdings_csv"] = u.get("holdings_csv", cfg.get("holdings_csv", "fidelity/Portfolio_Positions_*.csv"))
    c["tickers"] = list(dict.fromkeys(cfg.get("tickers", []) + u.get("tickers", [])))
    c["topics"] = list(dict.fromkeys(cfg.get("topics", []) + u.get("topics", [])))
    c["aliases"] = {**cfg.get("aliases", {}), **u.get("aliases", {})}
    ret = copy.deepcopy(cfg.get("returns", {}))
    for k in PER_USER_RETURNS:
        ret.pop(k, None)
    ret.update({k: v for k, v in u.get("returns", {}).items() if k in PER_USER_RETURNS})
    c["returns"] = ret
    return c


def everyone(cfg: dict) -> list[dict]:
    return [user_config(cfg, uid) for uid in user_ids()]


def default_user(cfg: dict, uid: str | None = None) -> dict:
    """The command line's person: --user, else the first admin, else the first person."""
    ids = user_ids()
    if not ids:
        raise SystemExit("No users yet: python -m fin_news adduser <name> --admin")
    if uid:
        if uid not in ids:
            raise SystemExit(f"No user {uid!r} (users: {', '.join(ids)})")
        return user_config(cfg, uid)
    admins = [i for i in ids if settings(i).get("admin")]
    return user_config(cfg, (admins or ids)[0])


# ---------- passwords ----------

N, R, P = 2 ** 14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, dklen=32)
    return f"scrypt${N}${R}${P}${salt.hex()}${h.hex()}"


def password_hash(uid: str) -> str:
    path = USERS_DIR / uid / "password"
    return path.read_text().strip() if VALID_ID.match(uid) and path.exists() else ""


def check_password(uid: str, password: str) -> bool:
    stored = password_hash(uid)
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=len(h) // 2)
    except ValueError:
        hashlib.scrypt(b"x", salt=b"0" * 16, n=N, r=R, p=P, dklen=32)  # same time whether or not the user exists
        return False
    return hmac.compare_digest(got.hex(), h)


def set_password(uid: str, password: str) -> None:
    path = USERS_DIR / uid / "password"
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(hash_password(password) + "\n")
    tmp.replace(path)


def add_user(uid: str, name: str = "", admin: bool = False) -> Path:
    if not VALID_ID.match(uid):
        raise ValueError("user names: lowercase letters, digits, - and _, starting with a letter (max 32)")
    home = USERS_DIR / uid
    if (home / "user.toml").exists():
        raise ValueError(f"user {uid!r} already exists")
    (home / "fidelity").mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    lines = [f"name = {_toml_str(name or uid.title())}"] + (["admin = true"] if admin else [])
    lines += ["", "# Optional: tickers = [...], topics = [...], [returns] accounts = [...] (see fin_news/users.py)", ""]
    (home / "user.toml").write_text("\n".join(lines))
    os.chmod(home / "user.toml", 0o600)
    return home


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
